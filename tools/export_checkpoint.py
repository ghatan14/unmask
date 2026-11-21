#!/usr/bin/env python
"""Turn a training checkpoint into a compact release checkpoint.

Keeps only the model weights and the configuration (drops the optimiser, EMA, SWA state and the
training-only distillation projectors). The LoRA adapters are kept as they are, so the exported model
gives exactly the scores of the training checkpoint, also under fp16. ``--fold`` folds them into the
frozen weights instead (a plain model with the inference parameter count of the paper, slightly
faster); fp16 rounding then changes the last digits of the scores, which can flip a few images that
lie right at the decision threshold.

    python tools/export_checkpoint.py runs/unmask_base_ffpp/checkpoint-last.pth \
        checkpoints/unmask_base_ffpp.pth --name UNMASK-Base --trained_on "FF++ c23 (w/o NT)"
"""

from __future__ import annotations

import argparse
import hashlib
import os
import sys

import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unmask import __version__  # noqa: E402
from unmask.models import load_unmask, save_unmask  # noqa: E402
from unmask.utils import torch_load  # noqa: E402


def sha256(path, n=1 << 22):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(n), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description='Export a compact UNMASK release checkpoint')
    ap.add_argument('src', help='checkpoint-last.pth (or -ema / -swa / -best) of a training run')
    ap.add_argument('dst', help='output .pth')
    ap.add_argument('--fold', action='store_true', help='fold LoRA into the frozen weights (deployment)')
    ap.add_argument('--name', default=None, help='e.g. UNMASK-Base')
    ap.add_argument('--trained_on', default=None, help='e.g. "FF++ c23 (w/o NeuralTextures)"')
    args = ap.parse_args()

    ck = torch_load(args.src)
    model, cfg = load_unmask(ck, fold=args.fold)
    x = torch.randn(2, 3, cfg['input_size'], cfg['input_size'])
    ref, _ = load_unmask(ck, verbose=False)
    with torch.no_grad():
        d = float((model.predict_proba(x) - ref.predict_proba(x)).abs().max())
    meta = dict(name=args.name or f'UNMASK-{cfg.get("variant", "?").capitalize()}', trained_on=args.trained_on,
                epoch=ck.get('epoch'), source=os.path.basename(args.src), unmask_version=__version__,
                lora_folded=args.fold)
    save_unmask(args.dst, model, cfg, meta=meta)
    r = model.param_report()
    print(f'  {args.src} -> {args.dst}: {r["total"] / 1e6:.2f} M parameters, '
          f'{os.path.getsize(args.dst) / 2 ** 20:.0f} MB, max |dp(fake)| vs source {d:.1e}')
    print(f'  sha256 {sha256(args.dst)}')


if __name__ == '__main__':
    main()
