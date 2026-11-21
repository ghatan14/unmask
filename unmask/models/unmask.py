"""UNMASK: Unified Network for Multi-prior Adaptation and Specialized Knowledge transfer.

A frame-level deepfake detector built on two frozen foundation priors:

    face branch   FS-VFM ViT-S/16  (self-supervised face-security prior, Wang et al. 2025)
    clip branch   PE-Core S/16 (Base) or T/16 (Mid)  (contrastive vision-language prior)

Each frozen ViT is adapted with LoRA (+ its LayerNorms) and read at four levels: gated
attention pooling of the patch tokens after blocks 4, 8 and 12, plus one summary of the last
block. A multi-level squeeze-and-excitation block re-weights the eight level vectors before the
fused head; one auxiliary head per branch provides deep supervision. UNMASK-Lite keeps only the
face branch.

Class order of the logits: ``[real, fake]``, i.e. ``softmax(logits)[:, 1]`` is p(fake).

Inputs are face crops normalised with ImageNet statistics. Every branch converts them to the
statistics of its own pre-training and resizes them to its own input size internally, so one
data pipeline can feed every prior.
"""

from __future__ import annotations

import copy
import os
from typing import Dict, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint

try:
    import timm
except ImportError as e:  # pragma: no cover
    raise ImportError('timm >= 1.0.18 is required: pip install -U "timm>=1.0.18"') from e

from ..utils import IMAGENET_MEAN, IMAGENET_STD, torch_load
from .fsvfm import (FSVFM_ARCHS, build_fsvfm_vit, load_fsvfm_encoder, read_mean_std,
                    resize_pos_embed)
from .layers import GatedAttentionPool, LoRALinear, MultiLevelSE, apply_lora, fold_lora

__all__ = ['PriorBranch', 'UNMASK', 'VARIANTS', 'CLASS_NAMES', 'FAKE_INDEX', 'unmask_config',
           'build_unmask', 'load_unmask', 'load_weights', 'save_unmask']

BRANCH_ORDER = ('face', 'clip')
CLASS_NAMES = ('real', 'fake')
FAKE_INDEX = 1

PE_CORE_S = 'vit_pe_core_small_patch16_384.fb'
PE_CORE_T = 'vit_pe_core_tiny_patch16_384.fb'

VARIANTS = {
    'base': dict(branches=('face', 'clip'), clip_model=PE_CORE_S),
    'mid': dict(branches=('face', 'clip'), clip_model=PE_CORE_T),
    'lite': dict(branches=('face',), clip_model=None),
}


# ==========================================================================
class PriorBranch(nn.Module):
    """One frozen foundation ViT with LoRA, read at several depths.

    family ``'fsvfm'``  an FS-VFM ViT (12-head ViT-S, sin-cos position embedding). Its summary
                        level is ``LayerNorm(mean of patch tokens)``, the read-out of the
                        FS-VFM fine-tuning head, initialised from the pre-trained final norm.
    family ``'timm'``   any timm ViT / Eva. For PE-Core (Eva with 2-D RoPE) the summary level
                        is the model's own attention-pool embedding.

    ``forward_levels(x)`` -> (list of level vectors (B, d), last-block patch tokens or None).
    """

    def __init__(self, family: str, model_name: str, img_size: int = 224, pretrained=True,
                 ckpt: Optional[str] = None, mean=None, std=None, taps=(3, 7, 11),
                 lora_r: int = 16, lora_alpha: float = 32, lora_dropout: float = 0.0,
                 lora_targets: str = 'attn+mlp', tune: str = 'lora+ln',
                 drop_path_rate: float = 0.0, grad_ckpt: bool = False, verbose: bool = True):
        super().__init__()
        self.family = family
        self.model_name = model_name
        self.img_size = img_size
        self.load_report = ''
        if family == 'timm':
            base = model_name.split('.')[0]
            if not timm.list_models(base):
                raise RuntimeError(f'timm {timm.__version__} does not know "{base}" '
                                   f'(PE-Core needs timm >= 1.0.18)')
            self.vit = timm.create_model(model_name, pretrained=pretrained, img_size=img_size,
                                         num_classes=0, drop_path_rate=drop_path_rate)
            cfg = getattr(self.vit, 'pretrained_cfg', {}) or {}
            self.pixel_mean = tuple(mean or cfg.get('mean', IMAGENET_MEAN))
            self.pixel_std = tuple(std or cfg.get('std', IMAGENET_STD))
        elif family == 'fsvfm':
            if model_name not in FSVFM_ARCHS:
                raise ValueError(f'unknown FS-VFM arch {model_name}; one of {list(FSVFM_ARCHS)}')
            self.vit = build_fsvfm_vit(model_name, img_size=img_size,
                                       drop_path_rate=drop_path_rate)
            dmean, dstd = read_mean_std(None)
            self.pixel_mean = tuple(mean or dmean)
            self.pixel_std = tuple(std or dstd)
            if pretrained:
                if not ckpt or not os.path.isfile(ckpt):
                    raise FileNotFoundError(
                        f'face branch: FS-VFM checkpoint not found: {ckpt!r}. Download it with '
                        f'tools/download_weights.py')
                self._load_fsvfm(ckpt, verbose)
        else:
            raise ValueError(f'unknown branch family {family!r}')

        v = self.vit
        self.is_eva = type(v).__name__ == 'Eva'
        if self.is_eva and getattr(v, 'rope_mixed', False):
            raise NotImplementedError(f'{model_name}: mixed-mode RoPE is not supported')
        self.embed_dim = v.embed_dim
        self.n_prefix = getattr(v, 'num_prefix_tokens', 1)
        self.grid = v.patch_embed.grid_size[0]
        self.depth = len(v.blocks)
        self.taps = tuple(t if t >= 0 else self.depth + t for t in taps)
        if max(self.taps) >= self.depth:
            raise ValueError(f'taps {self.taps} exceed depth {self.depth} of {model_name}')
        self.grad_ckpt = grad_ckpt
        self.tune = tune

        if family == 'fsvfm':
            self.final_norm = nn.LayerNorm(self.embed_dim, eps=1e-6)
            with torch.no_grad():
                self.final_norm.weight.copy_(v.norm.weight)
                self.final_norm.bias.copy_(v.norm.bias)
        else:
            self.final_norm = None

        for p in v.parameters():
            p.requires_grad_(False)
        self.n_lora = 0
        if 'lora' in tune:
            self.n_lora = apply_lora(v, lora_r, lora_alpha, dropout=lora_dropout,
                                     targets=lora_targets)
            if self.n_lora == 0:
                raise RuntimeError(f'{model_name}: no attention / MLP Linear found for LoRA')
        if 'ln' in tune:
            for m in v.modules():
                if isinstance(m, nn.LayerNorm):
                    for p in m.parameters():
                        p.requires_grad_(True)
        if tune == 'full':
            for p in v.parameters():
                p.requires_grad_(True)

        d = self.embed_dim
        self.tap_norms = nn.ModuleList([nn.LayerNorm(d) for _ in self.taps])
        self.tap_pools = nn.ModuleList([GatedAttentionPool(d) for _ in self.taps])
        self.level_dims = [d] * (len(self.taps) + 1)
        self.out_dim = sum(self.level_dims)

    # ------------------------------------------------------------------
    def _load_fsvfm(self, ckpt, verbose):
        enc, info = load_fsvfm_encoder(ckpt)
        want = FSVFM_ARCHS[self.model_name]
        if info['embed_dim'] != want['embed_dim'] or info['depth'] != want['depth']:
            raise RuntimeError(f'{ckpt}: encoder is {info["embed_dim"]}-d x {info["depth"]} '
                               f'blocks, but the face branch is {self.model_name}')
        own = self.vit.state_dict()
        if 'pos_embed' in enc and enc['pos_embed'].shape != own['pos_embed'].shape:
            enc['pos_embed'] = resize_pos_embed(enc['pos_embed'], 1,
                                                self.vit.patch_embed.grid_size[0])
        load = {k: v for k, v in enc.items() if k in own and own[k].shape == v.shape}
        msg = self.vit.load_state_dict(load, strict=False)
        missing = [k for k in msg.missing_keys if not k.startswith(('head.', 'fc_norm.'))]
        if missing:
            raise RuntimeError(f'{ckpt}: {len(missing)} encoder tensors missing, e.g. {missing[:5]}')
        if 'norm.weight' not in load and 'fc_norm.weight' in enc:
            # a fine-tuned detector stores fc_norm instead of the final norm
            with torch.no_grad():
                self.vit.norm.weight.copy_(enc['fc_norm.weight'])
                self.vit.norm.bias.copy_(enc['fc_norm.bias'])
        self.load_report = (f'FS-VFM encoder {info["embed_dim"]}-d x {info["depth"]} from '
                            f'{os.path.basename(ckpt)} (epoch {info["epoch"]}): {len(load)} tensors')
        if verbose:
            print('  [face] ' + self.load_report)

    def _run_block(self, blk, h, rope):
        fn = (lambda t: blk(t, rope=rope)) if self.is_eva else blk
        if self.grad_ckpt and self.training and torch.is_grad_enabled():
            return torch.utils.checkpoint.checkpoint(fn, h, use_reentrant=False)
        return fn(h)

    def forward_levels(self, x, want_tokens: bool = False):
        v = self.vit
        h = v.patch_embed(x)
        rope = None
        if self.is_eva:
            h, rope = v._pos_embed(h)
        else:
            h = v._pos_embed(h)
            if hasattr(v, 'patch_drop'):
                h = v.patch_drop(h)
        h = v.norm_pre(h)
        taps = []
        for i, blk in enumerate(v.blocks):
            h = self._run_block(blk, h, rope)
            if i in self.taps:
                taps.append(h)
        if self.family == 'fsvfm':
            final = self.final_norm(h[:, self.n_prefix:].mean(dim=1))
            hn = v.norm(h) if want_tokens else None
        else:
            hn = v.norm(h)
            if self.is_eva and getattr(v, 'attn_pool', None) is not None:
                final = v.attn_pool(hn)                  # PE-Core: CLIP embedding (pre-projection)
            else:
                final = hn[:, 0]                         # plain ViT: final CLS token
        levels = []
        for t, norm, pool in zip(taps, self.tap_norms, self.tap_pools):
            p, _ = pool(norm(t)[:, self.n_prefix:])
            levels.append(p)
        levels.append(final)
        tokens = hn[:, self.n_prefix:] if want_tokens else None
        return levels, tokens


# ==========================================================================
class UNMASK(nn.Module):
    """The UNMASK detector.

    forward(x)                                -> (B, 2) logits, columns [real, fake]
    forward(x, return_all=True, want_tokens=) -> dict with 'fused', '<branch>' (auxiliary
                                                 branch heads), 'feat', 'feat_<branch>',
                                                 'tok_<branch>' and the SE 'gates'
    readouts(x)                               -> {'fused', 'face', 'clip', 'mean'} logits
    """

    def __init__(self, branch_cfgs: Dict[str, dict], num_classes: int = 2, head_drop: float = 0.1,
                 branch_drop: float = 0.0, pretrained=True, verbose: bool = True):
        super().__init__()
        branches = tuple(b for b in BRANCH_ORDER if b in branch_cfgs)
        if not branches:
            raise ValueError(f'need at least one branch of {BRANCH_ORDER}')
        unknown = set(branch_cfgs) - set(BRANCH_ORDER)
        if unknown:
            raise ValueError(f'unknown branch(es) {sorted(unknown)}')
        self.branch_names = branches
        self.num_classes = num_classes
        self.branch_drop = branch_drop
        self.infer_mode = 'fused'
        in_mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        in_std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)

        self.branches = nn.ModuleDict()
        self.branch_sizes = {}
        for b in branches:
            c = dict(branch_cfgs[b])
            pt = pretrained.get(b, True) if isinstance(pretrained, dict) else bool(pretrained)
            br = PriorBranch(c.pop('family'), c.pop('model_name'), pretrained=pt,
                             verbose=verbose, **c)
            self.branches[b] = br
            self.branch_sizes[b] = br.img_size
            bm = torch.tensor(br.pixel_mean).view(1, 3, 1, 1)
            bs = torch.tensor(br.pixel_std).view(1, 3, 1, 1)
            # x_b = x * in_std / b_std + (in_mean - b_mean) / b_std
            self.register_buffer(f'_scale_{b}', in_std / bs, persistent=False)
            self.register_buffer(f'_shift_{b}', (in_mean - bm) / bs, persistent=False)

        multi = len(branches) > 1
        self.branch_norms = nn.ModuleDict({b: nn.LayerNorm(self.branches[b].out_dim)
                                           for b in branches} if multi else {})
        self.branch_heads = nn.ModuleDict({b: nn.Linear(self.branches[b].out_dim, num_classes)
                                           for b in branches} if multi else {})
        dims = [d for b in branches for d in self.branches[b].level_dims]
        self.level_owner = [b for b in branches for _ in self.branches[b].level_dims]
        self.level_names = [f'{b}/{n}' for b in branches
                            for n in [f'block{t + 1}' for t in self.branches[b].taps] + ['summary']]
        self.se = MultiLevelSE(dims)
        self.fused_dim = sum(dims)
        self.fc_norm = nn.LayerNorm(self.fused_dim)
        self.head_drop = nn.Dropout(head_drop)
        self.head = nn.Linear(self.fused_dim, num_classes)
        for lin in [self.head, *self.branch_heads.values()]:
            nn.init.trunc_normal_(lin.weight, std=0.01)
            nn.init.zeros_(lin.bias)

    # ------------------------------------------------------------------
    def _branch_input(self, x, b):
        xb = x * getattr(self, f'_scale_{b}') + getattr(self, f'_shift_{b}')
        s = self.branch_sizes[b]
        if xb.shape[-1] != s or xb.shape[-2] != s:
            xb = F.interpolate(xb, size=(s, s), mode='bicubic', align_corners=False)
        return xb

    def _drop_mask(self, batch, device):
        """(B, n_branches) keep-mask: during training at most one branch is blanked per sample."""
        nb = len(self.branch_names)
        keep = torch.ones(batch, nb, device=device)
        if not (self.training and self.branch_drop > 0 and nb > 1):
            return keep
        drop = torch.rand(batch, device=device) < self.branch_drop
        which = torch.randint(0, nb, (batch,), device=device)
        keep[drop, which[drop]] = 0.0
        return keep

    def forward(self, x, return_all: bool = False, want_tokens: Sequence[str] = ()):
        levels_all, out = [], {}
        for b in self.branch_names:
            lv, tok = self.branches[b].forward_levels(self._branch_input(x, b),
                                                      want_tokens=b in want_tokens)
            if tok is not None:
                out[f'tok_{b}'] = tok
            if b in self.branch_heads:
                feat = self.branch_norms[b](torch.cat(lv, dim=1))
                out[b] = self.branch_heads[b](self.head_drop(feat))
                out[f'feat_{b}'] = feat
            levels_all.extend(lv)
        levels_all, gates = self.se(levels_all)
        keep = self._drop_mask(x.shape[0], x.device)
        if keep.min() < 1:
            idx = {b: i for i, b in enumerate(self.branch_names)}
            levels_all = [v * keep[:, idx[o]:idx[o] + 1]
                          for v, o in zip(levels_all, self.level_owner)]
        fused_feat = self.fc_norm(torch.cat(levels_all, dim=1))
        out['fused'] = self.head(self.head_drop(fused_feat))
        out['feat'] = fused_feat
        if len(self.branch_names) == 1:
            out[f'feat_{self.branch_names[0]}'] = fused_feat
        out['gates'] = gates
        if return_all:
            return out
        return self._select(out)

    @staticmethod
    def _mean_of_heads(heads):
        """Average of the heads' probabilities, returned as log-probabilities."""
        p = torch.stack([F.softmax(h.float(), dim=1) for h in heads]).mean(0)
        return torch.log(p.clamp_min(1e-12))

    def _select(self, out):
        mode = self.infer_mode
        if mode == 'fused':
            return out['fused']
        if mode in self.branch_names:
            return out.get(mode, out['fused'])
        if mode == 'mean':
            if len(self.branch_heads) == 0:
                return out['fused']
            return self._mean_of_heads([out['fused']] + [out[b] for b in self.branch_names])
        raise ValueError(f'unknown infer_mode {mode!r}')

    @torch.no_grad()
    def readouts(self, x):
        """Every read-out of one forward pass: the fused head (the reported one), and for
        two-branch models each auxiliary branch head and the mean of the three heads."""
        out = self.forward(x, return_all=True)
        res = {'fused': out['fused']}
        if len(self.branch_heads) > 1:
            for b in self.branch_names:
                res[b] = out[b]
            res['mean'] = self._mean_of_heads([out['fused']] + [out[b] for b in self.branch_names])
        return res

    @torch.no_grad()
    def predict_proba(self, x):
        """p(fake) of the fused head, shape (B,)."""
        return F.softmax(self.forward(x).float(), dim=1)[:, FAKE_INDEX]

    # ------------------------------------------------------------------
    def param_report(self):
        total = sum(p.numel() for p in self.parameters())
        train = sum(p.numel() for p in self.parameters() if p.requires_grad)
        lora = sum(p.numel() for m in self.modules() if isinstance(m, LoRALinear)
                   for p in (m.A, m.B))
        per = {b: dict(total=sum(p.numel() for p in self.branches[b].parameters()),
                       lora_layers=self.branches[b].n_lora) for b in self.branch_names}
        return dict(total=total, trainable=train, lora=lora, inference=total - lora, branches=per)

    def describe(self):
        r = self.param_report()
        parts = []
        for b in self.branch_names:
            br = self.branches[b]
            parts.append(f'{b}={br.model_name}@{br.img_size}px taps {tuple(t + 1 for t in br.taps)}'
                         + (f' ({br.n_lora} LoRA layers)' if br.n_lora else ''))
        has_lora = any(self.branches[b].n_lora for b in self.branch_names)
        size = (f'{r["inference"] / 1e6:.2f} M parameters at inference (LoRA folded), '
                f'{r["trainable"] / 1e6:.3f} M trainable' if has_lora
                else f'{r["inference"] / 1e6:.2f} M parameters (LoRA folded)')
        return f'{size} | multi-level SE over {len(self.level_owner)} levels | ' + ' ; '.join(parts)

    @torch.no_grad()
    def fold_lora(self) -> int:
        """Fold every LoRA adapter into its frozen weight (inference / export / INT8)."""
        n = fold_lora(self)
        for b in self.branch_names:
            self.branches[b].n_lora = 0
        return n

    @torch.jit.ignore
    def no_weight_decay(self):
        return {n for n, p in self.named_parameters()
                if p.ndim <= 1 or 'pos_embed' in n or 'cls_token' in n or n.endswith('latent')}


# ==========================================================================
# Configuration: everything needed to rebuild a model is stored in each checkpoint
# (tap indices cannot be read off a state dict; pixel statistics must not depend on a
# file being present at evaluation time).
# ==========================================================================
def unmask_config(variant: str = 'base', *, face_arch: str = 'vit_small_patch16',
                  face_mean=None, face_std=None, clip_model: Optional[str] = None,
                  img_size: int = 224, taps=(3, 7, 11), lora_r: int = 16, lora_alpha: float = 32,
                  lora_dropout: float = 0.05, lora_targets: str = 'attn+mlp',
                  tune: str = 'lora+ln') -> dict:
    """JSON-safe description of an UNMASK variant (``base`` | ``mid`` | ``lite``).

    ``taps`` are 0-based block indices: (3, 7, 11) = after blocks 4, 8 and 12.
    """
    if variant not in VARIANTS:
        raise ValueError(f'unknown variant {variant!r}; one of {list(VARIANTS)}')
    spec = VARIANTS[variant]
    common = dict(img_size=img_size, taps=list(taps), lora_r=lora_r, lora_alpha=lora_alpha,
                  lora_dropout=lora_dropout, lora_targets=lora_targets, tune=tune)
    mean, std = read_mean_std(None)
    branches = {'face': dict(family='fsvfm', model_name=face_arch,
                             mean=list(face_mean or mean), std=list(face_std or std), **common)}
    if 'clip' in spec['branches']:
        branches['clip'] = dict(family='timm', model_name=clip_model or spec['clip_model'],
                                mean=None, std=None, **common)
    return dict(variant=variant, branches=branches, num_classes=2, class_names=list(CLASS_NAMES),
                input_size=img_size, input_mean=list(IMAGENET_MEAN), input_std=list(IMAGENET_STD),
                lora_folded=False)


def _branch_kwargs(cfg, face_ckpt=None, folded=False, grad_ckpt=False, drop_path_rate=0.0):
    out = {}
    for b, c in cfg['branches'].items():
        c = dict(c)
        c['taps'] = tuple(c['taps'])
        for k in ('mean', 'std'):
            if c.get(k) is None:
                c.pop(k, None)
        if folded:
            c['tune'] = 'ln'
        if c['family'] == 'fsvfm':
            c['ckpt'] = face_ckpt
        c['grad_ckpt'] = grad_ckpt
        c['drop_path_rate'] = drop_path_rate
        out[b] = c
    return out


def build_unmask(cfg: dict, pretrained=False, face_ckpt: Optional[str] = None,
                 head_drop: float = 0.0, branch_drop: float = 0.0, grad_ckpt: bool = False,
                 drop_path_rate: float = 0.0, verbose: bool = False) -> UNMASK:
    """Build an UNMASK model from a config (see :func:`unmask_config`).

    ``pretrained=True`` loads the foundation priors (FS-VFM from ``face_ckpt``, PE-Core from
    the timm hub) -- needed for training only; released checkpoints contain every weight.
    """
    folded = bool(cfg.get('lora_folded', False))
    model = UNMASK(_branch_kwargs(cfg, face_ckpt, folded, grad_ckpt, drop_path_rate),
                   num_classes=int(cfg.get('num_classes', 2)), head_drop=head_drop,
                   branch_drop=branch_drop, pretrained=pretrained, verbose=verbose)
    # record the pixel statistics each branch actually uses
    for b in model.branch_names:
        br = model.branches[b]
        cfg['branches'][b]['mean'] = list(br.pixel_mean)
        cfg['branches'][b]['std'] = list(br.pixel_std)
    return model


def load_unmask(src, device='cpu', fold: bool = False, verbose: bool = True):
    """Load an UNMASK checkpoint (path or loaded dict) -> (model in eval mode, config)."""
    ck = torch_load(src) if isinstance(src, (str, os.PathLike)) else src
    if not (isinstance(ck, dict) and 'model' in ck and 'config' in ck):
        raise RuntimeError(f'{src}: not an UNMASK checkpoint (expected the keys "model" and '
                           f'"config")')
    cfg = copy.deepcopy(ck['config'])
    sd = ck['model']
    cfg['lora_folded'] = bool(cfg.get('lora_folded')) or not any(k.endswith('.A') for k in sd)
    model = build_unmask(cfg, pretrained=False)
    msg = model.load_state_dict(sd, strict=False)
    if msg.missing_keys:
        raise RuntimeError(f'{src}: {len(msg.missing_keys)} missing keys, '
                           f'e.g. {msg.missing_keys[:6]}')
    if msg.unexpected_keys:
        raise RuntimeError(f'{src}: {len(msg.unexpected_keys)} unexpected keys, '
                           f'e.g. {msg.unexpected_keys[:6]}')
    if fold and not cfg['lora_folded']:
        model.fold_lora()
        cfg['lora_folded'] = True
    if verbose:
        print(f'  UNMASK-{cfg.get("variant", "?").capitalize()} | {model.describe()}')
    return model.to(device).eval(), cfg


@torch.no_grad()
def load_weights(model: UNMASK, sd: dict):
    """Load a state dict into ``model`` whether or not either side has LoRA folded.

    A folded source (``...qkv.weight``) initialises the frozen base of a LoRA layer
    (``...qkv.base.weight``); the LoRA factors keep their initialisation (``B = 0``), so the
    model starts from exactly the source function. Used to fine-tune a released checkpoint."""
    own = model.state_dict()
    src = dict(sd)
    for k in list(own):
        if k not in src and '.base.' in k:
            alt = k.replace('.base.', '.')
            if alt in src:
                src[k] = src.pop(alt)
    lora_keys = {k for k in own if k.endswith(('.A', '.B'))}
    missing = [k for k in own if k not in src and k not in lora_keys]
    unexpected = [k for k in src if k not in own]
    if missing or unexpected:
        raise RuntimeError(f'weights do not fit the model: {len(missing)} missing (e.g. '
                           f'{missing[:4]}), {len(unexpected)} unexpected (e.g. {unexpected[:4]})')
    model.load_state_dict(src, strict=False)


def save_unmask(path, model: UNMASK, cfg: dict, **extra):
    """Write a self-contained checkpoint: ``{'model', 'config', ...extra}``."""
    cfg = copy.deepcopy(cfg)
    cfg['lora_folded'] = not any(isinstance(m, LoRALinear) for m in model.modules())
    blob = {'model': {k: v.detach().cpu() for k, v in model.state_dict().items()},
            'config': cfg}
    blob.update(extra)
    os.makedirs(os.path.dirname(os.path.abspath(path)) or '.', exist_ok=True)
    torch.save(blob, path)
