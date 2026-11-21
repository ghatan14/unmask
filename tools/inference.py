#!/usr/bin/env python
"""Score face crops with a released UNMASK checkpoint.

Inputs are face crops (as produced by the FSFM / DeepfakeBench face pre-processing); each crop is
resized to 224 x 224. With ``--video_sep``, frames whose file names share the stem before the
separator are also aggregated into one video score (mean p(fake)).

    python tools/inference.py --ckpt checkpoints/unmask_base_ffpp.pth --input face.png
    python tools/inference.py --ckpt checkpoints/unmask_base_ffpp.pth --input frames/ \
        --video_sep _frame_ --csv scores.csv
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import OrderedDict

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unmask.data import IMG_EXT, eval_transform, video_stem  # noqa: E402
from unmask.models import load_unmask  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description='UNMASK inference on face crops')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--input', nargs='+', required=True, help='image files and/or folders')
    ap.add_argument('--video_sep', default=None, help='e.g. _frame_ to aggregate frames per video')
    ap.add_argument('--threshold', type=float, default=0.5)
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--csv', default=None)
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, cfg = load_unmask(args.ckpt, device=device, fold=True)
    tf = eval_transform(size=cfg['input_size'], mean=cfg['input_mean'], std=cfg['input_std'])
    files = []
    for p in args.input:
        if os.path.isdir(p):
            for dp, _, names in os.walk(p):
                files += [os.path.join(dp, n) for n in sorted(names) if n.lower().endswith(IMG_EXT)]
        elif os.path.isfile(p):
            files.append(p)
    if not files:
        raise SystemExit('no images found')
    scores = []
    for i in range(0, len(files), args.batch_size):
        batch = []
        for f in files[i:i + args.batch_size]:
            with Image.open(f) as im:
                batch.append(tf(im.convert('RGB')))
        scores += model.predict_proba(torch.stack(batch).to(device)).float().cpu().tolist()
    rows = [(f, s) for f, s in zip(files, scores)]
    for f, s in rows:
        print(f'{s:.4f}  {"FAKE" if s >= args.threshold else "real"}  {f}')
    if args.video_sep:
        per = OrderedDict()
        for f, s in rows:
            per.setdefault(os.path.join(os.path.dirname(f), video_stem(f, args.video_sep)), []).append(s)
        print('\nvideos (mean p(fake) over frames):')
        for v, ss in per.items():
            m = float(np.mean(ss))
            print(f'{m:.4f}  {"FAKE" if m >= args.threshold else "real"}  {v}  ({len(ss)} frames)')
    if args.csv:
        with open(args.csv, 'w', newline='', encoding='utf-8') as fh:
            w = csv.writer(fh)
            w.writerow(['path', 'p_fake'])
            w.writerows([(f, f'{s:.6f}') for f, s in rows])
        print(f'written: {args.csv}')


if __name__ == '__main__':
    main()
