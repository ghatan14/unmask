"""FS-VFM / FSFM vision transformers: construction and checkpoint loading.

FS-VFM (Wang et al., 2025) releases its encoders as plain ViTs in the ``models_vit.py``
format of the FSFM code base. The ViT-S/16 uses **12 attention heads** (MoCo v3 / DINO
convention), not the 6 heads of timm's ``vit_small_patch16_224``. The weights have the same
shapes either way, so loading them into a 6-head model would silently compute a different
function; the configurations below are therefore spelled out explicitly.
"""

from __future__ import annotations

import json
import math
import os
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.vision_transformer import VisionTransformer

from ..utils import torch_load

__all__ = ['FSVFM_ARCHS', 'ARCH_BY_DIM', 'VGGFACE2_MEAN', 'VGGFACE2_STD', 'read_mean_std',
           'build_fsvfm_vit', 'load_fsvfm_encoder', 'resize_pos_embed']

FSVFM_ARCHS = {
    'vit_small_patch16': dict(patch_size=16, embed_dim=384, depth=12, num_heads=12),
    'vit_base_patch16': dict(patch_size=16, embed_dim=768, depth=12, num_heads=12),
    'vit_large_patch16': dict(patch_size=16, embed_dim=1024, depth=24, num_heads=16),
}
ARCH_BY_DIM = {384: 'vit_small_patch16', 768: 'vit_base_patch16', 1024: 'vit_large_patch16'}

# VGGFace2 statistics of the FSFM / FS-VFM pre-training (pretrain_ds_mean_std.txt).
VGGFACE2_MEAN = (0.5482207536697388, 0.42340534925460815, 0.3654651641845703)
VGGFACE2_STD = (0.2789176106452942, 0.2438540756702423, 0.23493893444538116)


def read_mean_std(path=None):
    """``pretrain_ds_mean_std.txt`` (first line is JSON) -> (mean, std).

    Falls back to the VGGFace2 statistics of FSFM / FS-VFM when no file is given.
    """
    if path and os.path.isfile(path):
        with open(path, 'r', encoding='utf-8') as f:
            st = json.loads(f.readline().strip())
        return tuple(float(v) for v in st['mean']), tuple(float(v) for v in st['std'])
    return VGGFACE2_MEAN, VGGFACE2_STD


def build_fsvfm_vit(arch='vit_small_patch16', img_size=224, num_classes=0, global_pool='token',
                    fc_norm=None, drop_path_rate=0.0) -> VisionTransformer:
    """An FS-VFM ViT as a timm ``VisionTransformer`` (pre-norm, LayerNorm eps 1e-6)."""
    return VisionTransformer(img_size=img_size, mlp_ratio=4, qkv_bias=True,
                             norm_layer=partial(nn.LayerNorm, eps=1e-6),
                             num_classes=num_classes, global_pool=global_pool, fc_norm=fc_norm,
                             drop_path_rate=drop_path_rate, **FSVFM_ARCHS[arch])


def resize_pos_embed(pe, n_prefix, new_grid):
    """(1, n_prefix + g*g, D) position embedding -> ``new_grid`` x ``new_grid`` (bicubic)."""
    old = pe.shape[1] - n_prefix
    g = int(round(math.sqrt(old)))
    if g * g != old or g == new_grid:
        return pe
    prefix, grid = pe[:, :n_prefix], pe[:, n_prefix:]
    grid = grid.reshape(1, g, g, -1).permute(0, 3, 1, 2).float()
    grid = F.interpolate(grid, size=(new_grid, new_grid), mode='bicubic', align_corners=False)
    grid = grid.permute(0, 2, 3, 1).reshape(1, new_grid * new_grid, -1).to(pe.dtype)
    return torch.cat([prefix, grid], dim=1)


def load_fsvfm_encoder(ckpt_path):
    """The ViT encoder of an FS-VFM / FSFM checkpoint -> (state_dict, info).

    Works for pre-training checkpoints (encoder + decoders + projectors; only the encoder
    is kept) and for fine-tuned detectors (encoder + ``fc_norm`` + ``head``).
    """
    ck = torch_load(ckpt_path)
    sd = ck.get('model', ck.get('state_dict', ck)) if isinstance(ck, dict) else ck
    for pre in ('module.', 'backbone.', 'encoder.'):
        if sd and all(k.startswith(pre) for k in sd):
            sd = {k[len(pre):]: v for k, v in sd.items()}
    enc = {k: v for k, v in sd.items()
           if k in ('cls_token', 'pos_embed')
           or k.startswith(('patch_embed.', 'blocks.', 'norm.', 'fc_norm.', 'head.'))}
    if 'cls_token' not in enc or not any(k.startswith('blocks.') for k in enc):
        raise RuntimeError(f'{ckpt_path}: no ViT encoder found (keys e.g. {list(sd)[:6]})')
    dim = int(enc['cls_token'].shape[-1])
    depth = 1 + max(int(k.split('.')[1]) for k in enc if k.startswith('blocks.'))
    info = dict(embed_dim=dim, depth=depth,
                epoch=ck.get('epoch') if isinstance(ck, dict) else None,
                n_keys=len(sd), n_encoder_keys=len(enc))
    return enc, info
