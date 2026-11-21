"""Routed multi-teacher distillation: training-only projectors, target standardisation and
the teacher orientation check.

Each student branch is routed to the teacher that shares its prior:

    face branch  <-  FS-VFM ViT-L/16 (online): logits, pooled feature, 196 patch tokens
    clip branch  <-  GenD CLIP ViT-L/14 (cached): logits, pooled feature
    fused head   <-  1:1 mixture of both teachers' probabilities

(UNMASK-Lite has no clip branch: its face branch learns from both teachers.) None of the modules
here exists at inference; they are not part of the detector's state dict.
"""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..metrics import roc_auc

__all__ = ['FeatStandardizer', 'KDHeads', 'cos_dist', 'match_grid', 'decide_fake_col',
           'default_routes']


class FeatStandardizer(nn.Module):
    """Per-channel running mean / variance of a teacher target (EMA, momentum 0.01).

    ViT features carry a few channels of very large magnitude that would dominate a cosine
    loss; standardising lets every channel count."""

    def __init__(self, dim, momentum=0.01, eps=1e-4):
        super().__init__()
        self.momentum, self.eps = momentum, eps
        self.register_buffer('mean', torch.zeros(dim))
        self.register_buffer('var', torch.ones(dim))
        self.register_buffer('count', torch.zeros((), dtype=torch.long))

    @torch.no_grad()
    def update(self, t):
        t = t.detach().reshape(-1, t.shape[-1]).float()
        m, v = t.mean(0), t.var(0, unbiased=False)
        if int(self.count) == 0:
            self.mean.copy_(m)
            self.var.copy_(v)
        else:
            self.mean.mul_(1 - self.momentum).add_(m, alpha=self.momentum)
            self.var.mul_(1 - self.momentum).add_(v, alpha=self.momentum)
        self.count += 1

    def forward(self, t):
        return (t.float() - self.mean) / torch.sqrt(self.var + self.eps)


def _mlp(d_in, hidden, d_out):
    return nn.Sequential(nn.LayerNorm(d_in), nn.Linear(d_in, hidden), nn.GELU(),
                         nn.Linear(hidden, d_out))


class KDHeads(nn.Module):
    """Projectors from the student into each teacher's space.

    ``sfd_routes`` / ``dfd_routes``: lists of ``(branch, teacher, d_student, d_teacher)`` for the
    summary-feature (MLP) and dense-token (LayerNorm + linear) distillation."""

    def __init__(self, sfd_routes, dfd_routes, hidden=1024):
        super().__init__()
        self.sfd_routes = [(b, k) for b, k, _, _ in sfd_routes]
        self.dfd_routes = [(b, k) for b, k, _, _ in dfd_routes]
        self.sfd = nn.ModuleDict({f'{b}__{k}': _mlp(ds, hidden, dt) for b, k, ds, dt in sfd_routes})
        self.dfd = nn.ModuleDict({f'{b}__{k}': nn.Sequential(nn.LayerNorm(ds), nn.Linear(ds, dt))
                                  for b, k, ds, dt in dfd_routes})
        t_dims = {k: dt for _, k, _, dt in sfd_routes}
        tok_dims = {k: dt for _, k, _, dt in dfd_routes}
        self.std_feat = nn.ModuleDict({k: FeatStandardizer(d) for k, d in t_dims.items()})
        self.std_tok = nn.ModuleDict({k: FeatStandardizer(d) for k, d in tok_dims.items()})

    def describe(self):
        n = sum(p.numel() for p in self.parameters())
        return (f'distillation projectors (training only, {n / 1e6:.2f} M): summary '
                + ', '.join(f'{b}->{k}' for b, k in self.sfd_routes)
                + (' | dense ' + ', '.join(f'{b}->{k}' for b, k in self.dfd_routes)
                   if self.dfd_routes else ''))


def cos_dist(a, b):
    return (1.0 - F.cosine_similarity(a.float(), b.float(), dim=-1)).mean()


def match_grid(t_tok, n_student):
    """Teacher patch tokens (B, Nt, D) -> (B, Ns, D) by bilinear resampling of the square token
    grid (only needed when the patch grids differ)."""
    B, Nt, D = t_tok.shape
    if Nt == n_student:
        return t_tok
    gt, gs = int(round(math.sqrt(Nt))), int(round(math.sqrt(n_student)))
    if gt * gt != Nt or gs * gs != n_student:
        raise ValueError(f'cannot match token grids {Nt} -> {n_student}')
    g = t_tok.float().transpose(1, 2).reshape(B, D, gt, gt)
    g = F.interpolate(g, size=(gs, gs), mode='bilinear', align_corners=False)
    return g.reshape(B, D, gs * gs).transpose(1, 2)


@torch.no_grad()
def decide_fake_col(teacher, loader, device, max_frames=1536):
    """Which teacher output means FAKE, decided on validation frames of the training domain
    (never on a test set). Returns (fake column, frame AUC after orientation)."""
    ps, ys, n = [], [], 0
    for x, y, _ in loader:
        out = teacher(x.to(device, non_blocking=True), want_tokens=False)
        ps.append(torch.softmax(out['logits'].float(), 1)[:, 1].cpu().numpy())
        ys.append(y.numpy().astype(np.int64))
        n += len(y)
        if n >= max_frames:
            break
    a1 = roc_auc(np.concatenate(ys), np.concatenate(ps))
    col = 1 if a1 >= 50 else 0
    return col, (a1 if col == 1 else 100.0 - a1)


def default_routes(branch_names, teachers, override=None):
    """Which teacher(s) teach which branch: face <- fsvfm, clip <- gend; a single-branch model
    learns from every teacher. ``override`` e.g. ``"face=fsvfm+gend,clip=gend"``."""
    routes = {}
    if override:
        for part in override.split(','):
            b, _, ks = part.partition('=')
            routes[b.strip()] = [k.strip() for k in ks.split('+') if k.strip() in teachers]
    for b in branch_names:
        if b in routes:
            continue
        if b == 'face':
            ks = ['fsvfm'] + ([] if 'clip' in branch_names else ['gend'])
        else:
            ks = ['gend'] if 'gend' in teachers else ['fsvfm']
        routes[b] = [k for k in ks if k in teachers] or list(teachers)
    return routes
