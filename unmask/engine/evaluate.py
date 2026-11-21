"""Validation during training: video AUC of every read-out in one pass."""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F

from ..metrics import roc_auc, video_scores
from ..models.unmask import FAKE_INDEX
from ..utils import autocast

__all__ = ['evaluate']


@torch.no_grad()
def evaluate(model, loader, device, amp=True, want_gates=False):
    """-> {'fused': video AUC, '<branch>': video AUC, ['gates': mean SE gates]}"""
    model.eval()
    scores, labels, vids, gates = defaultdict(list), [], [], []
    for x, y, idx in loader:
        x = x.to(device, non_blocking=True)
        with autocast(amp and device.type == 'cuda'):
            out = model(x, return_all=True)
        heads = {'fused': out['fused']}
        for b in model.branch_heads:
            heads[b] = out[b]
        for k, v in heads.items():
            scores[k].append(F.softmax(v.float(), 1)[:, FAKE_INDEX].cpu().numpy())
        if want_gates and out.get('gates') is not None:
            gates.append(out['gates'].float().cpu().numpy())
        labels.append(y.numpy().astype(np.int64))
        vids.extend([loader.dataset.video_id(int(i)) for i in idx])
    y = np.concatenate(labels)
    res = {}
    for k, v in scores.items():
        vy, vs, _ = video_scores(np.concatenate(v), y, vids)
        res[k] = roc_auc(vy, vs)
    if gates:
        res['gates'] = [round(float(g), 4) for g in np.concatenate(gates).mean(0)]
    return res
