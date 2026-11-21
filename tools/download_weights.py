#!/usr/bin/env python
"""Download the FS-VFM weights needed for training (Hugging Face repo ``Wolowolo/fsfm-3c``,
released by the FS-VFM authors under CC BY-NC 4.0):

    pretrained/fsvfm_vit_small/FS-VFM_ViT-S_VF2_600e.pth (+ pretrain_ds_mean_std.txt)
        FS-VFM ViT-S/16 pre-trained on VGGFace2: the face prior of the student
    pretrained/fsvfm_vit_large_ffpp_c23.pth
        FS-VFM ViT-L/16 fine-tuned on FF++ c23: the online teacher (weights only, no optimiser)

PE-Core weights are fetched by timm on first use; GenD weights by tools/cache_gend.py.
Released UNMASK checkpoints contain every weight and need none of these files.

    python tools/download_weights.py [--out pretrained]
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys

REPO = 'Wolowolo/fsfm-3c'
PRIOR = 'pretrained_models/FS-VFM_ViT-S_VF2_600e/'
TEACHER = ('finetuned_models/FS-VFM_extensions/cross_dataset_DFD_and_DiFF/ViT-L_VF2_600e/'
           'FT_on_FF++_c23_32frames/checkpoint-min_val_loss.pth')


def list_files():
    from huggingface_hub import list_repo_files
    try:
        return list_repo_files(REPO)
    except Exception as e:
        print(f'  [warn] cannot list {REPO} ({type(e).__name__}: {e}); trying the known file names')
        return None


def pick(files, candidates):
    if files is None:
        return candidates[0]
    return next((c for c in candidates if c in files), None)


def fetch(filename):
    from huggingface_hub import hf_hub_download
    print(f'  downloading {REPO}/{filename}', flush=True)
    return hf_hub_download(REPO, filename)


def keep_model_only(src, dst):
    """Copy only the model weights of a released checkpoint (the ViT-L file carries its optimiser)."""
    import torch
    ck = None
    for kw in (dict(weights_only=False, mmap=True), dict(weights_only=False), {}):
        try:
            ck = torch.load(src, map_location='cpu', **kw)
            break
        except (TypeError, RuntimeError, ValueError):
            continue
    if ck is None:
        raise RuntimeError(f'cannot read {src}')
    sd = ck.get('model', ck) if isinstance(ck, dict) else ck
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    torch.save({'model': sd, 'epoch': ck.get('epoch') if isinstance(ck, dict) else None, 'source': REPO}, dst)
    print(f'    -> {dst} ({os.path.getsize(dst) / 1e6:.0f} MB)')


def main():
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--out', default='pretrained')
    ap.add_argument('--skip_teacher', action='store_true', help='only the student prior')
    args = ap.parse_args()
    try:
        import huggingface_hub  # noqa: F401
    except ImportError:
        sys.exit('pip install -U huggingface_hub')
    files = list_files()

    dst_dir = os.path.join(args.out, 'fsvfm_vit_small')
    os.makedirs(dst_dir, exist_ok=True)
    ck = pick(files, [PRIOR + 'checkpoint-599.pth', PRIOR + 'checkpoint-600.pth'])
    if ck is None:
        sys.exit(f'FS-VFM ViT-S checkpoint not found under {PRIOR}')
    for f, name in ((ck, 'FS-VFM_ViT-S_VF2_600e.pth'),
                    (pick(files, [PRIOR + 'pretrain_ds_mean_std.txt']), 'pretrain_ds_mean_std.txt')):
        if f is None:
            continue
        dst = os.path.join(dst_dir, name)
        if os.path.isfile(dst):
            print(f'  already present: {dst}')
        else:
            shutil.copyfile(fetch(f), dst)
            print(f'    -> {dst}')

    if not args.skip_teacher:
        dst = os.path.join(args.out, 'fsvfm_vit_large_ffpp_c23.pth')
        if os.path.isfile(dst):
            print(f'  already present: {dst}')
        else:
            src = fetch(TEACHER)
            keep_model_only(src, dst)
            print(f'    the full download (with optimiser state) stays in the Hugging Face cache: {src}')
    print('done')


if __name__ == '__main__':
    main()
