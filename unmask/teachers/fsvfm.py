"""Online FS-VFM teacher: an FF++-fine-tuned FS-VFM / FSFM detector (ViT-S, B or L).

The teacher runs on exactly the view the student sees (same crop, degradation and mixup),
"consistent teaching" in the sense of Beyer et al. (CVPR 2022). It provides the logits, the
pooled feature its head reads (``fc_norm`` output) and its last-block patch tokens.
"""

from __future__ import annotations

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..models.fsvfm import (ARCH_BY_DIM, FSVFM_ARCHS, build_fsvfm_vit, load_fsvfm_encoder,
                            read_mean_std, resize_pos_embed)
from ..utils import IMAGENET_MEAN, IMAGENET_STD

__all__ = ['Renorm', 'FSVFMTeacher']


class Renorm(nn.Module):
    """ImageNet-normalised input -> another normalisation (+ bicubic resize)."""

    def __init__(self, mean, std, size):
        super().__init__()
        in_m = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
        in_s = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
        m = torch.tensor(mean).view(1, 3, 1, 1)
        s = torch.tensor(std).view(1, 3, 1, 1)
        self.register_buffer('scale', in_s / s, persistent=False)
        self.register_buffer('shift', (in_m - m) / s, persistent=False)
        self.size = size

    def forward(self, x):
        x = x.float() * self.scale + self.shift
        if x.shape[-1] != self.size or x.shape[-2] != self.size:
            x = F.interpolate(x, size=(self.size, self.size), mode='bicubic', align_corners=False)
        return x


class FSVFMTeacher(nn.Module):
    """FS-VFM detector: blocks -> mean of patch tokens -> fc_norm -> head.

    ``forward(x)`` -> dict(logits (B, 2) float32 with column 1 = fake once ``set_fake_col`` has
    been called, feat (B, D) float32, tokens (B, N, D) or None).
    """

    kind = 'online'
    has_tokens = True

    def __init__(self, ckpt, mean_std=None, img_size=224, device='cuda', dtype=torch.float16,
                 name='fsvfm'):
        super().__init__()
        enc, info = load_fsvfm_encoder(ckpt)
        dim, depth = info['embed_dim'], info['depth']
        arch = ARCH_BY_DIM.get(dim)
        if arch is None or FSVFM_ARCHS[arch]['depth'] != depth:
            raise RuntimeError(f'{ckpt}: {dim}-d x {depth} is not an FS-VFM ViT-S/B/L')
        if 'head.weight' not in enc or 'fc_norm.weight' not in enc:
            raise RuntimeError(f'{ckpt}: no fc_norm / head; this is a pre-training checkpoint, '
                               f'not an FF++-fine-tuned detector')
        if int(enc['head.weight'].shape[0]) != 2:
            raise RuntimeError(f'{ckpt}: the head has {enc["head.weight"].shape[0]} outputs, '
                               f'expected 2')
        self.vit = build_fsvfm_vit(arch, img_size=img_size, num_classes=2, global_pool='avg',
                                   fc_norm=True)
        own = self.vit.state_dict()
        if enc['pos_embed'].shape != own['pos_embed'].shape:
            enc['pos_embed'] = resize_pos_embed(enc['pos_embed'], 1,
                                                self.vit.patch_embed.grid_size[0])
        load = {k: v for k, v in enc.items() if k in own and own[k].shape == v.shape}
        msg = self.vit.load_state_dict(load, strict=False)
        if msg.missing_keys:
            raise RuntimeError(f'{ckpt}: {len(msg.missing_keys)} tensors missing, '
                               f'e.g. {msg.missing_keys[:5]}')
        mean, std = read_mean_std(mean_std)
        self.renorm = Renorm(mean, std, img_size)
        self.name = name
        self.arch = arch
        self.dim = dim
        self.grid = self.vit.patch_embed.grid_size[0]
        self.fake_col = None
        n_par = sum(p.numel() for p in self.vit.parameters()) / 1e6
        src = os.path.basename(mean_std) if mean_std and os.path.isfile(mean_std) else 'VGGFace2 default'
        self.info = (f'{name}: FS-VFM {arch} ({n_par:.1f} M) from {os.path.basename(ckpt)} '
                     f'(epoch {info["epoch"]}), pixel statistics: {src}')
        self.to(device=device)
        self.vit.to(dtype=dtype)
        self.dtype = dtype
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)

    def set_fake_col(self, col):
        self.fake_col = int(col)

    @torch.no_grad()
    def forward(self, x, want_tokens=True):
        v = self.vit
        h = v.patch_embed(self.renorm(x).to(self.dtype))
        h = v._pos_embed(h)
        h = v.norm_pre(h)
        for blk in v.blocks:
            h = blk(h)
        tok = h[:, v.num_prefix_tokens:]
        feat = v.fc_norm(tok.mean(dim=1))
        lg = v.head(feat).float()
        if self.fake_col == 0:
            lg = lg.flip(1)
        return dict(logits=lg, feat=feat.float(), tokens=tok if want_tokens else None)
