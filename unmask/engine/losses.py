"""Mixup / CutMix and the classification and distillation losses."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

__all__ = ['mix_batch', 'soft_ce', 'kd_kl']


def mix_batch(x, y, rs, mix_prob=0.5, mixup_alpha=0.8, cutmix_alpha=1.0, cutmix_switch=0.5):
    """With probability ``mix_prob`` mix the batch with a permutation of itself (CutMix with
    probability ``cutmix_switch``, otherwise Mixup). ``rs`` is a ``numpy.random.RandomState``.

    Returns ``(x, y_a, y_b, lam, perm or None, kind)``.
    """
    B = x.shape[0]
    if mix_prob <= 0 or rs.rand() > mix_prob or B < 2:
        return x, y, y, 1.0, None, 'none'
    perm = torch.from_numpy(rs.permutation(B)).to(x.device)
    use_cut = cutmix_alpha > 0 and (mixup_alpha <= 0 or rs.rand() < cutmix_switch)
    if use_cut:
        lam = float(rs.beta(cutmix_alpha, cutmix_alpha))
        H, W = x.shape[-2:]
        cut = math.sqrt(1.0 - lam)
        ch, cw = int(H * cut), int(W * cut)
        cy, cx = rs.randint(H), rs.randint(W)
        y1, y2 = max(cy - ch // 2, 0), min(cy + ch // 2, H)
        x1, x2 = max(cx - cw // 2, 0), min(cx + cw // 2, W)
        x = x.clone()
        x[:, :, y1:y2, x1:x2] = x[perm, :, y1:y2, x1:x2]
        lam = 1.0 - (y2 - y1) * (x2 - x1) / float(H * W)
        kind = 'cutmix'
    else:
        lam = float(rs.beta(mixup_alpha, mixup_alpha))
        x = lam * x + (1.0 - lam) * x[perm]
        kind = 'mixup'
    return x, y, y[perm], lam, perm, kind


def soft_ce(logits, y_a, y_b, lam, n_cls, smoothing):
    """Cross-entropy against the (mixed, label-smoothed) one-hot targets."""
    t = lam * F.one_hot(y_a, n_cls).float() + (1.0 - lam) * F.one_hot(y_b, n_cls).float()
    t = t * (1.0 - smoothing) + smoothing / n_cls
    return -(t * F.log_softmax(logits.float(), dim=1)).sum(1).mean()


def kd_kl(s_logits, t_prob, T):
    """Hinton distillation: KL(teacher || student) at temperature ``T``, scaled by T^2."""
    return F.kl_div(F.log_softmax(s_logits.float() / T, dim=1), t_prob,
                    reduction='batchmean') * (T * T)
