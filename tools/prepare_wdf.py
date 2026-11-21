#!/usr/bin/env python
"""Download the WildDeepfake test split (Hugging Face dataset ``xingjunm/WildDeepfake``) and lay
it out as frame folders:

    <out_dir>/test/real/<archive>-<sequence>_frame_<n>.png
    <out_dir>/test/fake/<archive>-<sequence>_frame_<n>.png

Each archive of the release holds one folder per face sequence; that sequence is the "video" of
the video-level AUC. The paper keeps 32 frames per sequence (806 sequences, 25,792 frames).

    python tools/prepare_wdf.py --out_dir data/WildDeepfake --frames_per_seq 32
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import tarfile
from collections import defaultdict

REPO_ID = 'xingjunm/WildDeepfake'
BASE = 'deepfake_in_the_wild'
CLASSES = {'real': f'{BASE}/real_test', 'fake': f'{BASE}/fake_test'}
IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp')


def download(cache_dir, token=None):
    from huggingface_hub import snapshot_download
    patterns = [f'{p}/*' for p in CLASSES.values()]
    print('downloading only the test archives:', ', '.join(patterns))
    return snapshot_download(repo_id=REPO_ID, repo_type='dataset', allow_patterns=patterns,
                             local_dir=cache_dir, token=token)


def archives(root, rel_dir):
    d = os.path.join(root, *rel_dir.split('/'))
    return sorted(os.path.join(d, f) for f in os.listdir(d) if f.endswith('.tar.gz')) if os.path.isdir(d) else []


def open_archive(path):
    with open(path, 'rb') as f:
        head = f.read(512)
    if head.startswith(b'\x1f\x8b'):
        mode = 'r|gz'
    elif head.startswith(b'BZh'):
        mode = 'r|bz2'
    elif head.startswith(b'\xfd7zXZ\x00'):
        mode = 'r|xz'
    elif len(head) >= 262 and head[257:262] == b'ustar':
        mode = 'r|'
    elif head.startswith(b'version https://git-lfs'):
        raise ValueError(f'{os.path.basename(path)} is a Git-LFS pointer: the download did not complete')
    else:
        raise ValueError(f'{os.path.basename(path)} is not a readable archive')
    return tarfile.open(path, mode)


def extract(tar_path, out_dir, prefix, limit_per_seq=0):
    """Stream one archive into <out_dir>/<prefix>-<sequence>_frame_<n>.<ext>"""
    os.makedirs(out_dir, exist_ok=True)
    per_seq, written = defaultdict(int), 0
    with open_archive(tar_path) as tf:
        for m in tf:
            if not m.isfile():
                continue
            parts = m.name.replace('\\', '/').strip('/').split('/')
            if len(parts) < 2 or not parts[-1].lower().endswith(IMG_EXT):
                continue                                  # no sequence folder: grouping would be lost
            key = f'{prefix}-{parts[-2].replace("_frame_", "-")}'
            if limit_per_seq and per_seq[key] >= limit_per_seq:
                continue
            src = tf.extractfile(m)
            if src is None:
                continue
            stem, ext = os.path.splitext(parts[-1])
            with open(os.path.join(out_dir, f'{key}_frame_{stem}{ext.lower()}'), 'wb') as w:
                shutil.copyfileobj(src, w)
            per_seq[key] += 1
            written += 1
    return written, len(per_seq)


def main():
    ap = argparse.ArgumentParser(description='Prepare the WildDeepfake test split')
    ap.add_argument('--out_dir', required=True, help='dataset root; test/real and test/fake are created')
    ap.add_argument('--cache_dir', default='./wdf_hf_cache', help='where the archives are downloaded')
    ap.add_argument('--frames_per_seq', type=int, default=32, help='frames kept per sequence (0 = all)')
    ap.add_argument('--skip_download', action='store_true', help='archives are already in --cache_dir')
    ap.add_argument('--token', default=None, help='Hugging Face token, if the dataset is gated')
    args = ap.parse_args()

    root = args.cache_dir if args.skip_download else download(args.cache_dir, args.token)
    total_f = total_s = 0
    for cls, rel in CLASSES.items():
        arcs = archives(root, rel)
        if not arcs:
            sys.exit(f'no archives for "{cls}" under {os.path.join(root, rel)}')
        dest = os.path.join(args.out_dir, 'test', cls)
        print(f'{cls}: {len(arcs)} archives -> {dest}')
        for i, tp in enumerate(arcs, 1):
            prefix = cls + os.path.basename(tp).replace('.tar.gz', '')
            n, s = extract(tp, dest, prefix, args.frames_per_seq)
            total_f += n
            total_s += s
            print(f'  [{i}/{len(arcs)}] {os.path.basename(tp)}: {n} frames, {s} sequences', flush=True)
    print(f'done: {total_f} frames in {total_s} sequences under {os.path.join(args.out_dir, "test")}')


if __name__ == '__main__':
    main()
