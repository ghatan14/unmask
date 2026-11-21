#!/usr/bin/env python
"""Cross-dataset evaluation: frame- and video-level AUC on any frame folders.

Protocol of the paper: every face crop is resized directly to 224 x 224 (bicubic, no centre
crop), normalised with ImageNet statistics, no test-time augmentation; the video score is the
mean p(fake) over its frames.

    python tools/eval_cross_dataset.py \
        --ckpt checkpoints/unmask_mid_ffpp.pth checkpoints/unmask_base_ffpp.pth \
        --data CDF=data/Celeb-DF-v2 WDF=data/WildDeepfake \
        --all_readouts --bootstrap 1000 --out results/cross_dataset.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unmask.data import FrameDataset, eval_transform, scan_frames  # noqa: E402
from unmask.metrics import bootstrap_auc, eer, roc_auc, video_scores  # noqa: E402
from unmask.models import FAKE_INDEX, load_unmask  # noqa: E402
from unmask.utils import autocast, utf8_console  # noqa: E402


def get_args():
    ap = argparse.ArgumentParser(description='UNMASK cross-dataset evaluation')
    ap.add_argument('--ckpt', nargs='+', required=True, help='UNMASK checkpoint(s)')
    ap.add_argument('--names', nargs='*', default=None, help='row names (default: file names)')
    ap.add_argument('--data', nargs='+', required=True, metavar='NAME=PATH',
                    help='test sets, e.g. CDF=data/Celeb-DF-v2 WDF=data/WildDeepfake')
    ap.add_argument('--all_readouts', action='store_true',
                    help='also score the auxiliary branch heads and their mean (ablation rows)')
    ap.add_argument('--tta', action='store_true', help='average with the mirrored image (not used in the paper)')
    ap.add_argument('--topk', type=float, default=0.25,
                    help='also report the video AUC over the top-k fraction of frames')
    ap.add_argument('--ensemble', action='store_true', help='add a row averaging the fused p(fake) of all checkpoints')
    ap.add_argument('--bootstrap', type=int, default=0, help='video-level bootstrap resamples for 95%% CIs (0 = off)')
    ap.add_argument('--frame_ci', action='store_true', help='bootstrap frame AUC as well (slower)')
    ap.add_argument('--ref', default=None, help='row that every other row is compared to in the bootstrap')
    ap.add_argument('--real_hint', default='real')
    ap.add_argument('--video_sep', default='_frame_')
    ap.add_argument('--batch_size', type=int, default=64)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--out', default=None, help='JSON file with every metric')
    ap.add_argument('--save_scores', default=None, help='.npz with every frame probability')
    return ap.parse_args()


@torch.no_grad()
def score(model, loader, device, all_readouts=False, tta=False):
    """-> ({readout: p(fake) (N,)}, labels (N,), video ids)"""
    probs, labels, vids = OrderedDict(), [], []
    amp = device.type == 'cuda'
    t0, total = time.time(), len(loader)
    for bi, (x, y, idx) in enumerate(loader):
        x = x.to(device, non_blocking=True)
        with autocast(amp):
            outs = model.readouts(x) if all_readouts else {'fused': model(x)}
            p = {k: F.softmax(v.float(), 1)[:, FAKE_INDEX] for k, v in outs.items()}
            if tta:
                xf = torch.flip(x, dims=[3])
                outs2 = model.readouts(xf) if all_readouts else {'fused': model(xf)}
                p = {k: 0.5 * (p[k] + F.softmax(outs2[k].float(), 1)[:, FAKE_INDEX]) for k in p}
        for k, v in p.items():
            probs.setdefault(k, []).append(v.cpu().numpy())
        labels.append(y.numpy())
        vids.extend(loader.dataset.video_id(int(i)) for i in idx)
        if bi % 50 == 0 or bi == total - 1:
            d = (bi + 1) / total
            print(f'      {d * 100:5.1f}%  eta {(time.time() - t0) / max(d, 1e-9) * (1 - d) / 60:.1f} min', flush=True)
    return {k: np.concatenate(v) for k, v in probs.items()}, np.concatenate(labels), vids


def metrics(p, y, vids, topk):
    vy, vs, _ = video_scores(p, y, vids)
    out = dict(n_frames=int(len(y)), n_videos=int(len(vy)), frame_auc=roc_auc(y, p), frame_eer=eer(y, p),
               video_auc=roc_auc(vy, vs), video_eer=eer(vy, vs))
    if topk:
        vy2, vs2, _ = video_scores(p, y, vids, topk_frac=topk)
        out[f'video_auc_top{int(round(topk * 100))}'] = roc_auc(vy2, vs2)
    return out


def main():
    utf8_console()
    args = get_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    sets = OrderedDict()
    for spec in args.data:
        name, _, path = spec.partition('=')
        if not path:
            raise SystemExit(f'--data expects NAME=PATH, got {spec!r}')
        items, root = scan_frames(path, args.real_hint)
        n_fake = sum(y for _, y in items)
        ds = FrameDataset(items, None, sep=args.video_sep)
        n_vid = len({ds.video_id(i) for i in range(len(ds))})
        print(f'  {name}: {root} | {len(items)} frames ({len(items) - n_fake} real, {n_fake} fake), {n_vid} videos')
        if len(items) / max(n_vid, 1) < 1.5:
            print(f'  [warn] {name}: ~1 frame per video; check --video_sep (video AUC = frame AUC)')
        sets[name] = ds
    names = args.names or [os.path.splitext(os.path.basename(c))[0] for c in args.ckpt]
    if len(names) != len(args.ckpt):
        raise SystemExit('--names needs one entry per --ckpt')

    results, dump, ref_lab = OrderedDict(), {}, {}
    fused = {n: {} for n in sets}
    for ckpt, tag in zip(args.ckpt, names):
        print(f'\n=== {tag} ===')
        model, cfg = load_unmask(ckpt, device=device)
        tf = eval_transform(size=cfg['input_size'], mean=cfg['input_mean'], std=cfg['input_std'])
        for name, ds in sets.items():
            ds.transform = tf
            dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers,
                            pin_memory=device.type == 'cuda')
            probs, y, vids = score(model, dl, device, args.all_readouts, args.tta)
            ref_lab[name] = (y, vids)
            for k, p in probs.items():
                row = tag if k == 'fused' else f'{tag}[{k}]'
                m = metrics(p, y, vids, args.topk)
                results.setdefault(row, OrderedDict())[name] = m
                dump[f'{row}|{name}'] = p.astype(np.float32)
                print(f'    {name:>6} {"" if k == "fused" else "[" + k + "]":>7}: frame AUC {m["frame_auc"]:.2f} | '
                      f'video AUC {m["video_auc"]:.2f}')
            fused[name][tag] = probs['fused']
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    if args.ensemble and len(names) > 1:
        row = 'ENS(' + '+'.join(names) + ')'
        for name in sets:
            y, vids = ref_lab[name]
            p = np.mean([fused[name][t] for t in names], 0)
            results.setdefault(row, OrderedDict())[name] = metrics(p, y, vids, args.topk)
            dump[f'{row}|{name}'] = p.astype(np.float32)

    boot = None
    if args.bootstrap > 0:
        boot = OrderedDict()
        for name in sets:
            y, vids = ref_lab[name]
            scores = OrderedDict((r.split('|')[0], dump[r]) for r in dump if r.endswith('|' + name))
            boot[name] = bootstrap_auc(scores, y, vids, n_boot=args.bootstrap, ref=args.ref,
                                       frame_level=args.frame_ci)

    # ---------------------------------------------------------------- report
    cols = list(sets)
    print('\n| Model | ' + ' | '.join(f'{c} frame AUC | {c} video AUC' for c in cols) + ' |')
    print('|---|' + '---|---|' * len(cols))
    for row, res in results.items():
        cells = []
        for c in cols:
            m = res.get(c)
            cells += [f'{m["frame_auc"]:.2f}', f'{m["video_auc"]:.2f}'] if m else ['-', '-']
        print(f'| {row} | ' + ' | '.join(cells) + ' |')
    if boot:
        print('\nBootstrap 95% CI of the video AUC (resampling videos within each class):')
        for c in cols:
            for row, b in boot[c].items():
                line = f'  {c:>6} {row}: [{b["video_auc_ci"][0]:.2f}, {b["video_auc_ci"][1]:.2f}]'
                d = b.get(f'video_auc_diff_vs_{args.ref}')
                if d:
                    line += f'   diff vs {args.ref}: {d[0]:+.2f} [{d[1]:+.2f}, {d[2]:+.2f}]'
                print(line)
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
        with open(args.out, 'w', encoding='utf-8') as f:
            json.dump(dict(protocol='direct resize to input size, ImageNet normalisation, '
                                    + ('hflip TTA' if args.tta else 'no TTA'),
                           checkpoints=dict(zip(names, args.ckpt)), results=results, bootstrap=boot),
                      f, indent=2)
        print(f'\nwritten: {args.out}')
    if args.save_scores:
        for name in sets:
            y, vids = ref_lab[name]
            dump[f'labels|{name}'] = y
            dump[f'vids|{name}'] = np.array(vids)
        np.savez_compressed(args.save_scores, **dump)
        print(f'frame scores: {args.save_scores}')


if __name__ == '__main__':
    main()
