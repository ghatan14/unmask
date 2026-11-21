#!/usr/bin/env python
"""HydraFake evaluation with the protocol of Tan et al. (ICLR 2026).

* 4 levels, 23 subsets (In-Domain 5, Cross-Model 6, Cross-Forgery 6, Cross-Domain 6), in the
  column order of their Table 1; official image lists from ``jsons/test`` when downloaded;
* metric: accuracy with fake as the positive class and the argmax decision (p(fake) >= 0.5,
  no threshold tuned on test data), plus precision / recall and AUC as extras;
* images resized directly to 224 x 224, no test-time augmentation;
* ``Avg`` exactly as Table 1: mean of 19 numbers = the In-Domain mean plus the 18 out-of-domain
  subset accuracies (``Avg OOD`` = the mean of the 18 OOD subsets).

    python tools/eval_hydrafake.py --check --root data/HydraFake
    python tools/eval_hydrafake.py --root data/HydraFake --ckpt checkpoints/unmask_base_hydrafake.pth \
        --bootstrap 1000 --out results/hydrafake_base
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unmask.data.hydrafake import (SPLIT_SHORT, SPLITS, describe, json_status,  # noqa: E402
                                   load_all)
from unmask.metrics import accuracy_metrics  # noqa: E402
from unmask.utils import utf8_console  # noqa: E402

# Accuracy (%) published in Table 1 of Tan et al. (ICLR 2026); id = In-Domain mean,
# cm / cf / cd = the six subsets of each level in the column order of SPLITS.
PAPER_TABLE1 = OrderedDict([
    ('F3Net', (85.3,
        [86.7, 87.8, 78.6, 85.0, 86.0, 82.9],
        [41.3, 48.9, 71.9, 84.9, 85.5, 72.6],
        [57.7, 78.5, 55.6, 68.6, 66.2, 66.4],
        73.2)),
    ('UniFD', (82.7,
        [90.7, 93.8, 82.5, 73.0, 94.4, 90.7],
        [61.8, 81.9, 75.4, 73.7, 68.1, 81.3],
        [67.4, 67.3, 80.5, 75.2, 73.3, 67.5],
        78.0)),
    ('IID', (83.4,
        [83.3, 82.8, 80.0, 80.2, 81.1, 82.2],
        [41.4, 53.3, 79.7, 81.8, 81.8, 73.7],
        [65.2, 69.9, 63.8, 63.3, 63.8, 64.2],
        72.4)),
    ('FreqNet', (66.8,
        [60.3, 76.7, 59.0, 69.2, 77.1, 75.1],
        [33.1, 73.1, 70.3, 72.8, 77.4, 67.7],
        [50.6, 67.0, 62.1, 59.3, 58.3, 51.2],
        64.6)),
    ('ProDet', (90.5,
        [92.6, 94.2, 88.2, 91.9, 93.8, 93.1],
        [56.3, 58.6, 80.8, 88.1, 91.0, 83.3],
        [58.1, 82.9, 71.3, 75.6, 66.3, 74.1],
        80.6)),
    ('NPR', (75.6,
        [68.8, 91.2, 59.5, 82.6, 91.3, 84.0],
        [47.7, 67.8, 60.6, 79.8, 89.0, 67.7],
        [52.6, 73.0, 76.6, 62.3, 50.2, 46.0],
        69.8)),
    ('AIDE', (80.4,
        [68.8, 86.3, 64.0, 88.9, 95.4, 76.0],
        [56.7, 79.2, 86.1, 74.2, 62.4, 75.7],
        [59.7, 67.9, 49.7, 58.0, 51.9, 59.2],
        70.6)),
    ('Co-SPY', (86.3,
        [93.5, 95.5, 85.3, 93.3, 96.6, 95.3],
        [77.0, 92.5, 88.6, 90.6, 79.1, 87.3],
        [67.6, 80.0, 82.5, 74.0, 79.5, 64.3],
        84.7)),
    ('D3', (87.3,
        [93.6, 95.6, 91.3, 90.7, 95.8, 95.5],
        [62.4, 71.6, 82.9, 80.0, 82.4, 73.7],
        [69.7, 74.6, 78.1, 70.9, 80.8, 64.3],
        81.1)),
    ('Effort', (94.7,
        [82.8, 96.5, 78.0, 90.5, 97.8, 98.3],
        [64.7, 94.8, 89.7, 89.5, 92.9, 88.0],
        [64.8, 82.2, 61.5, 66.4, 53.8, 74.0],
        82.2)),
    ('Veritas (cold-start)', (96.8,
        [79.5, 99.6, 96.0, 99.9, 99.7, 99.9],
        [84.0, 65.3, 94.8, 86.2, 93.4, 86.7],
        [55.9, 73.5, 93.7, 89.3, 88.1, 76.4],
        87.3)),
    ('Veritas', (97.3,
        [94.8, 99.8, 97.0, 99.9, 99.9, 99.9],
        [90.3, 75.7, 97.0, 91.8, 95.1, 91.7],
        [58.6, 84.1, 92.3, 90.2, 89.2, 78.5],
        90.7)),
])


def get_args():
    ap = argparse.ArgumentParser(description='UNMASK evaluation on HydraFake (Tan et al., ICLR 2026)')
    ap.add_argument('--root', default='data/HydraFake', help='HydraFake folder (test/, jsons/)')
    ap.add_argument('--ckpt', nargs='*', default=[])
    ap.add_argument('--names', nargs='*', default=None)
    ap.add_argument('--lists', choices=['auto', 'json', 'folder'], default='auto',
                    help='auto = official jsons/test lists when downloaded, else folder scan')
    ap.add_argument('--all_readouts', action='store_true', help='also score the auxiliary heads (ablation rows)')
    ap.add_argument('--with_paper_rows', action='store_true', help='add the published rows of Table 1')
    ap.add_argument('--bootstrap', type=int, default=0, help='paired bootstrap resamples (0 = off)')
    ap.add_argument('--ref', default=None, help='row that every other row is compared to in the bootstrap')
    ap.add_argument('--batch_size', type=int, default=32)
    ap.add_argument('--workers', type=int, default=6)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--save_scores', default=None, help='.npz with every image probability')
    ap.add_argument('--from_scores', nargs='*', default=None, help='re-make the tables from saved .npz files')
    ap.add_argument('--out', default=None, help='base name of the .json / .md outputs')
    ap.add_argument('--limit_per_subset', type=int, default=0, help='smoke test: N images per subset')
    ap.add_argument('--check', action='store_true', help='only check the dataset and the official lists')
    return ap.parse_args()


# ==========================================================================
def aggregate(per_subset, field='acc'):
    out = OrderedDict()
    for split, subs in SPLITS.items():
        vals = [per_subset[(split, s[0])][field] for s in subs if (split, s[0]) in per_subset]
        ok = len(vals) == len(subs) and all(np.isfinite(vals))
        out[split] = float(np.mean(vals)) if ok else float('nan')
    ood = [per_subset[(sp, s[0])][field] for sp in ('cm', 'cf', 'cd') for s in SPLITS[sp] if (sp, s[0]) in per_subset]
    ok = len(ood) == 18 and all(np.isfinite(ood))
    out['avg_ood'] = float(np.mean(ood)) if ok else float('nan')
    out['avg'] = float((sum(ood) + out['id']) / 19.0) if ok and np.isfinite(out['id']) else float('nan')
    return out


def compute_all(rows, data):
    res = OrderedDict()
    y_all, si, ok = data['labels'], data['subset_index'], data['ok']
    for name, p in rows.items():
        per = OrderedDict()
        for k, key in enumerate(data['keys']):
            m = (si == k) & ok
            per[key] = accuracy_metrics(p[m], y_all[m])
        res[name] = dict(subsets=per, acc=aggregate(per, 'acc'), auc=aggregate(per, 'auc'))
    return res


def bootstrap(rows, data, B=1000, seed=0, ref=None):
    """Percentile 95% CIs of every level mean / Avg; images are resampled within each subset and
    class with the same draws for every row (paired). Test-set uncertainty only."""
    rng = np.random.default_rng(seed)
    y_all, si, ok = data['labels'], data['subset_index'], data['ok']
    accb = {n: {} for n in rows}
    for k, key in enumerate(data['keys']):
        m = np.where((si == k) & ok)[0]
        if len(m) == 0:
            continue
        yr = y_all[m]
        draws = [idx[rng.integers(0, len(idx), size=(B, len(idx)))]
                 for idx in (np.where(yr == 0)[0], np.where(yr == 1)[0]) if len(idx)]
        for n in rows:
            corr = ((rows[n][m] >= 0.5).astype(np.int64) == yr).astype(np.float32)
            accb[n][key] = 100.0 * sum(corr[d].sum(1) for d in draws) / len(m)

    def agg(n):
        per, a = accb[n], OrderedDict()
        for split, subs in SPLITS.items():
            vs = [per[(split, s[0])] for s in subs if (split, s[0]) in per]
            a[split] = np.mean(vs, 0) if len(vs) == len(subs) else None
        ood = [per[(sp, s[0])] for sp in ('cm', 'cf', 'cd') for s in SPLITS[sp] if (sp, s[0]) in per]
        a['avg_ood'] = np.mean(ood, 0) if len(ood) == 18 else None
        a['avg'] = (np.sum(ood, 0) + a['id']) / 19.0 if len(ood) == 18 and a['id'] is not None else None
        return a

    ab = {n: agg(n) for n in rows}
    ci = lambda v: None if v is None else (float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5)))
    out = OrderedDict()
    for n in rows:
        out[n] = {k: ci(v) for k, v in ab[n].items()}
        if ref and ref in ab and n != ref:
            out[n]['diff_vs_' + ref] = {k: (None if v is None or ab[ref][k] is None else
                                            (float(np.mean(v - ab[ref][k])),) + ci(v - ab[ref][k]))
                                        for k, v in ab[n].items()}
    return out


# ==========================================================================
def fmt(v, nd=1):
    return '-' if v is None or (isinstance(v, float) and not math.isfinite(v)) else f'{v:.{nd}f}'


def table1(res, src_flags, with_paper):
    cols = [s[1] for sp in ('cm', 'cf', 'cd') for s in SPLITS[sp]]
    hdr = ['Method', 'ID'] + cols + ['Avg', 'Avg OOD']
    lines = ['Columns 3-8: Cross-Model, 9-14: Cross-Forgery, 15-20: Cross-Domain (order of Table 1).', '',
             '| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for n, r in res.items():
        cells = [fmt(r['acc']['id'])]
        for sp in ('cm', 'cf', 'cd'):
            for s in SPLITS[sp]:
                c = fmt(r['subsets'][(sp, s[0])]['acc'])
                cells.append(c + ('*' if src_flags.get((sp, s[0])) == 'folder+approx' else ''))
        lines.append(f'| **{n}** | ' + ' | '.join(cells)
                     + f' | **{fmt(r["acc"]["avg"])}** | {fmt(r["acc"]["avg_ood"])} |')
    if with_paper:
        for n, (idv, cm, cf, cd, avg) in PAPER_TABLE1.items():
            ood = cm + cf + cd
            lines.append(f'| {n} (Tan et al.) | {fmt(idv)} | ' + ' | '.join(fmt(v) for v in ood)
                         + f' | {fmt(avg)} | {fmt(sum(ood) / 18)} |')
    return '\n'.join(lines)


def level_table(res):
    hdr = ['Model', 'ID', 'CM', 'CF', 'CD', 'Avg', 'Avg OOD', 'AUC ID', 'AUC CM', 'AUC CF', 'AUC CD', 'AUC OOD']
    lines = ['| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
    for n, r in res.items():
        a, u = r['acc'], r['auc']
        lines.append(f'| {n} | ' + ' | '.join(fmt(a[k]) for k in ('id', 'cm', 'cf', 'cd', 'avg', 'avg_ood'))
                     + ' | ' + ' | '.join(fmt(u[k]) for k in ('id', 'cm', 'cf', 'cd', 'avg_ood')) + ' |')
    return '\n'.join(lines)


def per_subset_table(res):
    out = []
    for split, subs in SPLITS.items():
        hdr = ['Model'] + [f'{s[1]} Acc / P / R / AUC' for s in subs]
        lines = [f'**{SPLIT_SHORT[split]}**', '', '| ' + ' | '.join(hdr) + ' |', '|' + '---|' * len(hdr)]
        for n, r in res.items():
            cells = []
            for s in subs:
                m = r['subsets'][(split, s[0])]
                cells.append(f'{fmt(m["acc"])} / {fmt(m["prec"])} / {fmt(m["rec"])} / {fmt(m["auc"])}')
            lines.append(f'| {n} | ' + ' | '.join(cells) + ' |')
        out.append('\n'.join(lines))
    return '\n\n'.join(out)


def boot_table(boot, ref):
    keys = ('id', 'cm', 'cf', 'cd', 'avg', 'avg_ood')
    lines = ['| Model | ' + ' | '.join(k.upper() for k in keys) + ' |', '|' + '---|' * (len(keys) + 1)]
    for n, b in boot.items():
        lines.append(f'| {n} | ' + ' | '.join('-' if b.get(k) is None else f'[{b[k][0]:.1f}, {b[k][1]:.1f}]'
                                               for k in keys) + ' |')
    if ref:
        lines += ['', f'Paired difference to **{ref}** (mean [95% CI]):', '',
                  '| Model | ' + ' | '.join(k.upper() for k in keys) + ' |', '|' + '---|' * (len(keys) + 1)]
        for n, b in boot.items():
            d = b.get('diff_vs_' + ref)
            if d:
                lines.append(f'| {n} - {ref} | ' + ' | '.join(
                    '-' if d.get(k) is None else f'{d[k][0]:+.1f} [{d[k][1]:+.1f}, {d[k][2]:+.1f}]'
                    for k in keys) + ' |')
    return '\n'.join(lines)


# ==========================================================================
def score_items(items, specs, names, args, device):
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader
    from unmask.data import ImageList, eval_transform
    from unmask.models import FAKE_INDEX, load_unmask
    from unmask.utils import autocast

    rows, ok, params = OrderedDict(), np.ones(len(items), dtype=bool), {}
    for spec, name in zip(specs, names):
        print(f'  loading {name}: {spec}', flush=True)
        model, cfg = load_unmask(spec, device=device)
        params[name] = model.param_report()['inference'] / 1e6
        tf = eval_transform(size=cfg['input_size'], mean=cfg['input_mean'], std=cfg['input_std'])
        dl = DataLoader(ImageList(items, tf, size=cfg['input_size']), batch_size=args.batch_size, shuffle=False,
                        num_workers=args.workers, pin_memory=device.type == 'cuda')
        t0, total = time.time(), len(dl)
        with torch.no_grad():
            for bi, (x, y, idx, good) in enumerate(dl):
                idx = idx.numpy()
                ok[idx] &= good.numpy().astype(bool)
                with autocast(device.type == 'cuda'):
                    x = x.to(device)
                    outs = model.readouts(x) if args.all_readouts else {'fused': model(x)}
                for k, v in outs.items():
                    rn = name if k == 'fused' else f'{name}[{k}]'
                    rows.setdefault(rn, np.full(len(items), np.nan, dtype=np.float32))[idx] = \
                        F.softmax(v.float(), 1)[:, FAKE_INDEX].cpu().numpy()
                if bi % 50 == 0 or bi == total - 1:
                    d = (bi + 1) / total
                    eta = (time.time() - t0) / max(d, 1e-9) * (1 - d) / 60
                    print(f'    {d * 100:5.1f}%  eta {eta:.1f} min', flush=True)
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()
    return rows, ok, params


def save_scores(path, rows, data, params):
    blob = dict(keys=np.array(['/'.join(k) for k in data['keys']], dtype=object),
                sources=np.array(data['sources'], dtype=object), subset_index=data['subset_index'],
                labels=data['labels'], ok=data['ok'], paths=np.array(data['paths'], dtype=object),
                row_names=np.array(list(rows), dtype=object), params=np.array(json.dumps(params)),
                describe=np.array(data['describe']))
    for i, p in enumerate(rows.values()):
        blob[f'row_{i}'] = p
    np.savez_compressed(path, **blob)


def load_scores(paths):
    """Merge saved score files; images are aligned by path."""
    rows, params, data = OrderedDict(), {}, None
    for path in paths:
        z = np.load(path, allow_pickle=True)
        d = dict(keys=[tuple(str(k).split('/')) for k in z['keys']], sources=[str(s) for s in z['sources']],
                 subset_index=z['subset_index'], labels=z['labels'], ok=z['ok'],
                 paths=[str(p) for p in z['paths']], describe=str(z['describe']))
        pos = None
        if data is None:
            data = d
        else:
            if d['keys'] != data['keys']:
                raise SystemExit(f'{path}: different subset list, cannot merge')
            where = {p: i for i, p in enumerate(d['paths'])}
            if any(p not in where for p in data['paths']):
                raise SystemExit(f'{path}: the score files were made on different image lists')
            pos = np.array([where[p] for p in data['paths']])
            data['ok'] = data['ok'] & d['ok'][pos]
        for i, n in enumerate(str(n) for n in z['row_names']):
            v = z[f'row_{i}']
            rows[n if n not in rows else f'{n}@{os.path.basename(path)}'] = v if pos is None else v[pos]
        params.update(json.loads(str(z['params'])))
    return rows, data, params


# ==========================================================================
def main():
    utf8_console()
    args = get_args()
    root = os.path.abspath(args.root)
    if args.check:
        ok, ptr, absent = json_status(root)
        print(f'HydraFake root: {root}\nofficial test lists: {ok} downloaded, {ptr} Git-LFS pointers, '
              f'{absent} missing (of 23)')
        if ptr or absent:
            print('  -> cd <HydraFake> && git lfs install && git lfs pull --include="jsons/test/**,jsons/val/**"')
        print(describe(load_all(root, mode=args.lists)))
        return

    if args.from_scores:
        rows, data, params = load_scores(args.from_scores)
    else:
        if not args.ckpt:
            raise SystemExit('give --ckpt (or --from_scores, or --check)')
        names = args.names or [os.path.splitext(os.path.basename(c))[0] for c in args.ckpt]
        if len(names) != len(args.ckpt):
            raise SystemExit('--names needs one entry per --ckpt')
        sets = load_all(root, mode=args.lists)
        desc = describe(sets)
        print('HydraFake test lists:\n' + desc)
        empty = [k for k, r in sets.items() if not r['items']]
        if empty:
            raise SystemExit(f'empty subsets {empty}; the table would be incomplete')
        keys, items, si, sources = [], [], [], []
        rng = np.random.default_rng(0)
        for k, (key, r) in enumerate(sets.items()):
            it = r['items']
            if args.limit_per_subset:
                sel = []
                for cls in (0, 1):
                    c = [x for x in it if x[1] == cls]
                    if len(c) > args.limit_per_subset // 2:
                        c = [c[i] for i in sorted(rng.choice(len(c), args.limit_per_subset // 2, replace=False))]
                    sel += c
                it = sel
            keys.append(key)
            sources.append(r['source'])
            items += it
            si += [k] * len(it)
        print(f'{len(items)} images in {len(keys)} subsets'
              + (' (SMOKE TEST)' if args.limit_per_subset else ''))
        import torch
        device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
        rows, ok, params = score_items(items, args.ckpt, names, args, device)
        data = dict(keys=keys, sources=sources, subset_index=np.asarray(si, dtype=np.int64),
                    labels=np.array([y for _, y in items], dtype=np.int64), ok=ok,
                    paths=[os.path.relpath(p, root).replace('\\', '/') for p, _ in items], describe=desc)
        if (~ok).any():
            print(f'  {int((~ok).sum())} unreadable images left out of every metric')
        if args.save_scores:
            save_scores(args.save_scores, rows, data, params)
            print(f'  scores saved: {args.save_scores}')

    res = compute_all(rows, data)
    ref = args.ref if args.ref in rows else None
    boot = bootstrap(rows, data, B=args.bootstrap, ref=ref) if args.bootstrap > 0 else None
    src_flags = dict(zip(data['keys'], data['sources']))
    md = ['# HydraFake results', '',
          f'{time.strftime("%Y-%m-%d %H:%M")}. Protocol of Tan et al. (ICLR 2026): direct resize to '
          '224 x 224, no TTA, decision = argmax (p(fake) >= 0.5), accuracy in %. '
          'Avg = (In-Domain mean + 18 OOD subsets) / 19.', '']
    if any(v != 'json' for v in src_flags.values()):
        approx = 'folder+approx' in src_flags.values()
        md += ['> Some subsets were read from folders instead of the official lists'
               + (' (* = facevid2vid with substituted reals)' if approx else '') + '.', '']
    md += ['## Accuracy per subset (Table 1 format)', '', table1(res, src_flags, args.with_paper_rows), '',
           '## Per level', '', level_table(res), '']
    if boot:
        md += [f'## Bootstrap ({args.bootstrap} paired resamples, test-set uncertainty only)', '',
               boot_table(boot, ref), '']
    md += ['## Accuracy / precision / recall / AUC per subset (fake = positive)', '', per_subset_table(res), '',
           '## Data', '', '```', data['describe'], '```', '']
    print('\n' + level_table(res))
    if args.out:
        os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
        with open(args.out + '.md', 'w', encoding='utf-8') as f:
            f.write('\n'.join(md))
        with open(args.out + '.json', 'w', encoding='utf-8') as f:
            json.dump(dict(params_M=params, sources={'/'.join(k): s for k, s in src_flags.items()},
                           rows={n: dict(acc=r['acc'], auc=r['auc'],
                                         subsets={'/'.join(k): m for k, m in r['subsets'].items()})
                                 for n, r in res.items()}, bootstrap=boot),
                      f, indent=2, default=lambda o: None)
        print(f'\nwritten: {args.out}.md, {args.out}.json')


if __name__ == '__main__':
    main()
