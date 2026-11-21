"""GenD teacher (Yermakov et al., "Deepfake Detection that Generalizes Across Benchmarks",
WACV 2026; code: https://github.com/yermandy/GenD, weights: ``yermandy/GenD_CLIP_L_14``).

GenD is used as a *cached* teacher: its logits and the pooled 1024-d feature its head reads are
computed once over the training frames (``tools/cache_gend.py``) and read during training.

Nothing here re-implements GenD: the official model class is imported from a clone of the GenD
repository (``--gend_repo``), so the network and its preprocessing are exactly the authors'.
The model is built as config -> ``GenD(config)`` -> released weights, instead of
``GenD.from_pretrained()``, because transformers >= 5 refuses the ``from_pretrained()`` call
nested inside GenD's constructor; the result is identical with transformers 4.x and 5.x.
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np
import torch
import torch.nn as nn

from ..metrics import roc_auc

__all__ = ['GEND_MODELS', 'build_gend_teacher', 'GenDTeacher', 'CachedTeacher']

GEND_MODELS = {
    'clip': 'yermandy/GenD_CLIP_L_14',
    'pe': 'yermandy/GenD_PE_L',
    'dinov3': 'yermandy/GenD_DINOv3_L',
}


class GenDTeacher(nn.Module):
    """The official GenD model behind one interface.

    ``preprocess(list of PIL)`` -> batch tensor (GenD's own resize / crop / normalisation);
    ``forward(x)`` -> (logits (B, 2) with column 1 = fake, pooled feature (B, D)).
    """

    def __init__(self, gend_model, fake_index=1):
        super().__init__()
        self.m = gend_model
        self.fake_index = int(fake_index)
        fe = getattr(gend_model, 'feature_extractor', None)
        if fe is None or not hasattr(fe, 'preprocess'):
            raise RuntimeError('this GenD build has no feature_extractor.preprocess(); '
                               'update the GenD clone (git pull)')
        self._prep = fe.preprocess

    def preprocess(self, pil_images):
        out = []
        for im in pil_images:
            t = self._prep(im)
            if hasattr(t, 'pixel_values'):                       # a HF BatchFeature
                t = t['pixel_values']
            t = torch.as_tensor(t)
            if t.dim() == 4 and t.shape[0] == 1:
                t = t[0]
            out.append(t)
        return torch.stack(out)

    def forward(self, x):
        feats = self.m.feature_extractor(x)                      # pooled CLIP feature
        out = self.m.model(feats)                                # GenD.forward = head(features)
        if hasattr(out, 'logits'):
            out = out.logits
        elif isinstance(out, (tuple, list)):
            out = out[0]
        lg = out.float()
        if lg.dim() == 1 or lg.shape[-1] == 1:                   # a single logit for index 1
            z = lg.reshape(-1)
            lg = torch.stack([-0.5 * z, 0.5 * z], dim=1)
        if lg.shape[-1] != 2:
            raise RuntimeError(f'GenD returned {tuple(lg.shape)}; expected two classes')
        if self.fake_index == 0:
            lg = lg.flip(1)
        return lg, feats.float()


def _hub_file(model_id, name):
    if os.path.isdir(model_id):
        p = os.path.join(model_id, name)
        return p if os.path.isfile(p) else None
    from huggingface_hub import hf_hub_download
    try:
        return hf_hub_download(model_id, name)
    except Exception as e:
        if type(e).__name__ in ('EntryNotFoundError', 'RemoteEntryNotFoundError') \
                or 'not found' in str(e).lower() or '404' in str(e):
            return None
        raise


def _gend_state_dict(model_id):
    from safetensors.torch import load_file
    p = _hub_file(model_id, 'model.safetensors')
    if p is not None:
        return load_file(p)
    idx = _hub_file(model_id, 'model.safetensors.index.json')
    if idx is not None:
        with open(idx, 'r', encoding='utf-8') as f:
            shards = sorted(set(json.load(f)['weight_map'].values()))
        sd = {}
        for sh in shards:
            sd.update(load_file(_hub_file(model_id, sh)))
        return sd
    p = _hub_file(model_id, 'pytorch_model.bin')
    if p is not None:
        return torch.load(p, map_location='cpu', weights_only=True)
    raise FileNotFoundError(f'{model_id}: no model.safetensors / pytorch_model.bin found')


def build_gend_teacher(model='clip', repo_dir=None, device='cuda', dtype=torch.float16,
                       fake_index=1, verbose=True) -> GenDTeacher:
    model_id = GEND_MODELS.get(model.lower(), model)
    if repo_dir:
        repo_dir = os.path.abspath(repo_dir)
        if not os.path.isdir(os.path.join(repo_dir, 'src')):
            raise FileNotFoundError(f'--gend_repo {repo_dir} is not a GenD clone (no src/). '
                                    f'git clone https://github.com/yermandy/GenD')
        if repo_dir not in sys.path:
            sys.path.insert(0, repo_dir)
    try:
        from src.hf.modeling_gend import GenD
    except ImportError as e:
        raise ImportError('cannot import GenD (src/hf/modeling_gend.py): clone '
                          'https://github.com/yermandy/GenD and pass --gend_repo') from e
    config = GenD.config_class.from_pretrained(model_id)
    gend = GenD(config)
    sd = _gend_state_dict(model_id)
    msg = gend.load_state_dict(sd, strict=False)
    params = {n for n, _ in gend.named_parameters()}
    missing = [k for k in msg.missing_keys if k in params]
    if missing:
        raise RuntimeError(f'{model_id}: {len(missing)} GenD parameters are not in the released '
                           f'weights, e.g. {missing[:5]}; update the GenD clone')
    gend = gend.to(device=device, dtype=dtype).eval()
    for p in gend.parameters():
        p.requires_grad_(False)
    if verbose:
        n = sum(p.numel() for p in gend.parameters()) / 1e6
        print(f'  [gend] {model_id}: {n:.1f} M parameters, backbone {config.backbone}')
    return GenDTeacher(gend, fake_index=fake_index)


class CachedTeacher:
    """Teacher outputs cached over the training frames (``tools/cache_gend.py``).

    Rows are matched to the training frames by their path relative to the training folder.
    Logits use column 1 = fake. Lives in the main process; DataLoader workers never see it.
    """

    kind = 'cache'
    has_tokens = False

    def __init__(self, path, rel_paths, labels, name='gend'):
        z = np.load(path, allow_pickle=True)
        meta = json.loads(str(z['meta'])) if 'meta' in z.files else {}
        self.name = name
        pos = {str(p).replace('\\', '/'): i for i, p in enumerate(z['paths'])}
        want = [str(p).replace('\\', '/') for p in rel_paths]
        missing = [p for p in want if p not in pos]
        if missing:
            raise SystemExit(f'{path}: {len(missing)} training frames have no cached teacher '
                             f'output, e.g. {missing[:3]}. Re-run tools/cache_gend.py on the same '
                             f'training folder.')
        order = np.array([pos[p] for p in want], dtype=np.int64)
        self.logits = torch.from_numpy(z['logits'][order].astype(np.float32))
        feat = z['feat'] if 'feat' in z.files else None
        self.has_feat = feat is not None and feat.ndim == 2 and feat.shape[1] > 1
        self.feat = torch.from_numpy(feat[order].astype(np.float16)) if self.has_feat else None
        self.dim = int(feat.shape[1]) if self.has_feat else 0
        p_fake = torch.softmax(self.logits, 1)[:, 1].numpy()
        self.train_auc = roc_auc(np.asarray(labels), p_fake)
        if not self.train_auc >= 50.0:
            raise SystemExit(f'{path}: the cached teacher is below chance on its own training '
                             f'frames (AUC {self.train_auc:.2f}): its label columns are swapped. '
                             f'Re-cache with --fake_index 0.')
        self.info = (f'{name}: cache {os.path.basename(path)} ({meta.get("teacher", "?")}) | '
                     f'train-frame AUC {self.train_auc:.2f} | '
                     + (f'{self.dim}-d features' if self.has_feat else 'logits only'))

    def get(self, idx):
        idx = idx.long().cpu()
        return self.logits[idx], (self.feat[idx] if self.has_feat else None)
