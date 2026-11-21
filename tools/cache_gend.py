#!/usr/bin/env python
"""Cache the GenD teacher (logits + pooled 1024-d feature) over the training frames.

GenD runs once per training frame on the clean image with its own preprocessing, averaged
over the upright and mirrored views (probabilities for the logits, mean for the features).
Training then reads the cache instead of running a second ViT-L next to the online FS-VFM
teacher. Before caching, ``--val`` measures GenD on the validation split of the same domain and
stops if its frame AUC is below ``--gate`` (the teacher gate of the paper: 85 for FF++,
75 for HydraFake; it is never decided on a test set).

    git clone https://github.com/yermandy/GenD third_party/GenD
    python tools/cache_gend.py --gend_repo third_party/GenD \
        --data data/FaceForensics++/c23/train --val data/FaceForensics++/c23/val --gate 85 \
        --out pretrained/gend_cache_ffpp_train.npz
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unmask.data import scan_frames  # noqa: E402
from unmask.metrics import roc_auc  # noqa: E402
from unmask.teachers import build_gend_teacher  # noqa: E402


class PILFrames(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        with Image.open(p) as im:
            return im.convert('RGB'), int(y), i


def collate(batch):
    imgs, ys, idx = zip(*batch)
    return list(imgs), torch.as_tensor(ys), torch.as_tensor(idx)


@torch.no_grad()
def run(teacher, items, device, batch_size, workers, tta, want_feat=True, tag=''):
    dl = DataLoader(PILFrames(items), batch_size=batch_size, shuffle=False, num_workers=workers,
                    collate_fn=collate)
    n = len(items)
    logits = np.zeros((n, 2), dtype=np.float16)
    feats = None
    dtype = next(teacher.parameters()).dtype
    t0 = time.time()
    for bi, (imgs, y, idx) in enumerate(dl):
        views = [imgs] + ([[im.transpose(Image.FLIP_LEFT_RIGHT) for im in imgs]] if tta else [])
        outs = [teacher(teacher.preprocess(v).to(device=device, dtype=dtype)) for v in views]
        lg, ft = outs[0]
        if len(outs) > 1:
            p = 0.5 * (F.softmax(lg.float(), 1) + F.softmax(outs[1][0].float(), 1))
            lg = torch.log(p.clamp_min(1e-8))
            ft = 0.5 * (ft.float() + outs[1][1].float())
        i = idx.numpy()
        logits[i] = lg.float().cpu().numpy().astype(np.float16)
        if want_feat:
            if feats is None:
                feats = np.zeros((n, ft.shape[1]), dtype=np.float16)
            feats[i] = ft.float().cpu().numpy().astype(np.float16)
        if bi % 50 == 0 or bi == len(dl) - 1:
            d = (bi + 1) / len(dl)
            print(f'  {tag} {d * 100:5.1f}%  {(time.time() - t0) / 60:.1f} min elapsed, '
                  f'{(time.time() - t0) / max(d, 1e-9) * (1 - d) / 60:.1f} min left', flush=True)
    return logits, feats


def main():
    ap = argparse.ArgumentParser(description='Cache GenD logits and features over training frames')
    ap.add_argument('--data', required=True, help='training split (class sub-folders)')
    ap.add_argument('--out', required=True, help='output .npz')
    ap.add_argument('--gend_repo', required=True, help='clone of https://github.com/yermandy/GenD')
    ap.add_argument('--gend_model', default='clip', help='clip | pe | dinov3 | a Hugging Face id')
    ap.add_argument('--fake_index', type=int, default=1, choices=[0, 1],
                    help='which GenD output is "fake" (1 for the released GenD_CLIP_L_14)')
    ap.add_argument('--val', default=None, help='validation split for the teacher gate')
    ap.add_argument('--gate', type=float, default=0.0, help='minimum validation frame AUC')
    ap.add_argument('--no_tta', action='store_true', help='skip the mirrored view')
    ap.add_argument('--real_hint', default='real')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    dtype = torch.float16 if device.type == 'cuda' else torch.float32
    teacher = build_gend_teacher(args.gend_model, repo_dir=args.gend_repo, device=device, dtype=dtype,
                                 fake_index=args.fake_index)
    tta = not args.no_tta
    val_auc = None
    if args.val:
        val_items, _ = scan_frames(args.val, args.real_hint)
        lg, _ = run(teacher, val_items, device, args.batch_size, args.workers, tta, False, 'val')
        p_fake = F.softmax(torch.from_numpy(lg.astype(np.float32)), 1)[:, 1].numpy()
        val_auc = roc_auc([y for _, y in val_items], p_fake)
        print(f'  GenD validation frame AUC {val_auc:.2f} (gate {args.gate:g})')
        if val_auc < args.gate:
            raise SystemExit('GenD does not fit these crops; the paper trains without it in that case')

    items, root = scan_frames(args.data, args.real_hint)
    print(f'  caching {len(items)} frames from {root}')
    logits, feats = run(teacher, items, device, args.batch_size, args.workers, tta, True, 'train')
    labels = np.array([y for _, y in items], dtype=np.int64)
    train_auc = roc_auc(labels, F.softmax(torch.from_numpy(logits.astype(np.float32)), 1)[:, 1].numpy())
    print(f'  GenD training-frame AUC {train_auc:.2f}')
    if train_auc < 50:
        raise SystemExit('below chance on the training frames: rerun with --fake_index 0')
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    np.savez_compressed(
        args.out, logits=logits, feat=feats, labels=labels,
        paths=np.array([os.path.relpath(p, root).replace('\\', '/') for p, _ in items], dtype=object),
        meta=np.array(json.dumps(dict(teacher=args.gend_model, fake_index=args.fake_index, tta=tta,
                                      root=root, train_auc=train_auc, val_auc=val_auc)), dtype=object))
    print(f'  written: {args.out} ({os.path.getsize(args.out) / 1e6:.0f} MB)')


if __name__ == '__main__':
    main()
