#!/usr/bin/env python
"""Fully decode every image of one or more folders and list (or set aside) unreadable files.

The HydraFake release contains at least one empty file
(``train/fake/FR/AniPortrait/05490--aU4rjIl5160_0--AniPortraitVideo/086.png``); a truncated file
would only fail halfway through training. With ``--move``, unreadable files are moved to
``<root>/_unreadable/<same relative path>`` (nothing is deleted) and listed in
``_unreadable/list.txt``.

    python tools/check_images.py --root data/HydraFake --split train val --move
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import time
from multiprocessing import Pool

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')


def check(path):
    try:
        if os.path.getsize(path) == 0:
            return path, 'empty file'
        from PIL import Image
        with Image.open(path) as im:
            im.load()                                  # full decode: catches truncated files
            if im.width < 8 or im.height < 8:
                return path, f'too small ({im.width}x{im.height})'
        return path, None
    except Exception as e:                             # noqa: BLE001
        return path, f'{type(e).__name__}: {e}'


def main():
    ap = argparse.ArgumentParser(description='Find unreadable images')
    ap.add_argument('--root', required=True)
    ap.add_argument('--split', nargs='+', default=['.'], help='sub-folders of --root to check')
    ap.add_argument('--move', action='store_true', help='move unreadable files to <root>/_unreadable/')
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    root = os.path.abspath(args.root)
    files = []
    for sp in args.split:
        d = os.path.join(root, sp)
        if not os.path.isdir(d):
            print(f'  [warn] missing: {d}')
            continue
        for dp, _, names in os.walk(d):
            if '_unreadable' in dp:
                continue
            files += [os.path.join(dp, n) for n in names if n.lower().endswith(IMG_EXT)]
    print(f'{len(files)} images, decoding with {args.workers} processes', flush=True)
    t0, bad = time.time(), []
    with Pool(args.workers) as pool:
        for i, (p, err) in enumerate(pool.imap_unordered(check, files, chunksize=64), 1):
            if err:
                bad.append((p, err))
                print(f'  UNREADABLE  {os.path.relpath(p, root)}: {err}', flush=True)
            if i % 5000 == 0:
                print(f'    {i}/{len(files)} ({(time.time() - t0) / 60:.1f} min)', flush=True)
    print(f'done in {(time.time() - t0) / 60:.1f} min: {len(bad)} unreadable of {len(files)}')
    if not bad:
        return 0
    if not args.move:
        print('run again with --move to set them aside')
        return 2
    dst_root = os.path.join(root, '_unreadable')
    os.makedirs(dst_root, exist_ok=True)
    with open(os.path.join(dst_root, 'list.txt'), 'a', encoding='utf-8') as f:
        for p, err in bad:
            rel = os.path.relpath(p, root)
            dst = os.path.join(dst_root, rel)
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.move(p, dst)
            f.write(f'{time.strftime("%Y-%m-%d %H:%M")}\t{rel}\t{err}\n')
            print(f'  moved: {rel}')
    return 0


if __name__ == '__main__':
    sys.exit(main())
