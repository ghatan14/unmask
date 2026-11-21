#!/usr/bin/env python
"""Parameters, GMACs and (optionally) CPU latency of the UNMASK variants.

Parameters are counted at inference (LoRA folded); GMACs with ``torch.utils.flop_counter`` for
one 224 x 224 frame (MACs = FLOPs / 2). The paper counts the convolution and linear layers
(``GMACs``); ``GMACs+attn`` adds the attention products (QK^T and AV) on top. ``--latency``
times batch-1 inference on the CPU, and
``--int8`` additionally applies post-training dynamic INT8 quantisation to every ``nn.Linear``
(not part of the results reported in the paper).

    python tools/complexity.py                         # all variants, randomly initialised
    python tools/complexity.py --ckpt checkpoints/unmask_mid_ffpp.pth --latency --int8
"""

from __future__ import annotations

import argparse
import copy
import io
import os
import sys
import time

import torch
import torch.nn as nn

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unmask.models import build_unmask, load_unmask, unmask_config  # noqa: E402


def gmacs(model, size):
    """-> (GMACs of convolution + linear layers, GMACs including the attention products)."""
    from torch.utils.flop_counter import FlopCounterMode
    x = torch.randn(1, 3, size, size)
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(x)
    counts = fc.get_flop_counts()['Global']
    total = sum(counts.values())
    attn = sum(c for op, c in counts.items() if 'scaled_dot_product' in str(op))
    return (total - attn) / 2e9, total / 2e9


def size_mb(model):
    buf = io.BytesIO()
    torch.save(model.state_dict(), buf)
    return buf.tell() / 2 ** 20


@torch.no_grad()
def latency_ms(model, size, iters=30, warmup=5):
    x = torch.randn(1, 3, size, size)
    for _ in range(warmup):
        model(x)
    t0 = time.perf_counter()
    for _ in range(iters):
        model(x)
    return (time.perf_counter() - t0) / iters * 1000


def quantize_dynamic(model):
    import warnings
    try:
        from torch.ao.quantization import quantize_dynamic as qd
    except ImportError:                                           # older torch
        from torch.quantization import quantize_dynamic as qd
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        return qd(model, {nn.Linear}, dtype=torch.qint8)


def main():
    ap = argparse.ArgumentParser(description='UNMASK complexity')
    ap.add_argument('--ckpt', nargs='*', default=None, help='checkpoint(s); default: the three variants')
    ap.add_argument('--latency', action='store_true', help='CPU latency at batch 1')
    ap.add_argument('--int8', action='store_true', help='also dynamic INT8 (size, latency)')
    ap.add_argument('--threads', type=int, default=4)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    models = []
    if args.ckpt:
        for c in args.ckpt:
            m, cfg = load_unmask(c, fold=True, verbose=False)
            models.append((os.path.basename(c), m, cfg))
    else:
        for v in ('lite', 'mid', 'base'):
            cfg = unmask_config(v)
            m = build_unmask(cfg).eval()
            trainable = m.param_report()['trainable']
            m.fold_lora()
            models.append((f'UNMASK-{v.capitalize()}', m, dict(cfg, trainable=trainable)))
    print('| Model | Params (M) | Trainable (M) | GMACs | GMACs+attn | FP32 (MB)'
          + (' | CPU ms/frame' if args.latency else '') + (' | INT8 (MB) | INT8 ms/frame' if args.int8 else '') + ' |')
    for name, m, cfg in models:
        m.eval()
        size = cfg['input_size']
        tr = cfg.get('trainable')
        g, ga = gmacs(m, size)
        row = [name, f'{sum(p.numel() for p in m.parameters()) / 1e6:.2f}',
               f'{tr / 1e6:.2f}' if tr else '-', f'{g:.2f}', f'{ga:.2f}', f'{size_mb(m):.0f}']
        if args.latency:
            row.append(f'{latency_ms(m, size):.1f}')
        if args.int8:
            q = quantize_dynamic(copy.deepcopy(m)).eval()
            row += [f'{size_mb(q):.0f}', f'{latency_ms(q, size):.1f}' if args.latency else '-']
        print('| ' + ' | '.join(row) + ' |')


if __name__ == '__main__':
    main()
