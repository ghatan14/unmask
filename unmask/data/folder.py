"""Frame folders: scanning, labels, video ids and FF++ naming helpers.

Every split is a folder of class sub-folders. A folder whose name contains one of the
``real_hint`` substrings (default ``"real"``) is real, every other folder is fake:

    FF++ (FSFM preprocessing)   <split>/real_youtube, <split>/fake_four
    Celeb-DF-v2, WildDeepfake   test/real, test/fake
    HydraFake                   train/real, train/fake/<type>/<subset>/...

Frames of one video share the file-name stem before ``_frame_`` (``video_sep``); the video id
is ``<class folder>/<stem>``. Labels are ``1 = fake``, ``0 = real``.
"""

from __future__ import annotations

import os
import re
from typing import List, Tuple

import torch
from PIL import Image
from torch.utils.data import Dataset

IMG_EXT = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')

FFPP_MANIPULATIONS = {
    'Deepfakes': ('deepfakes', 'df'),
    'Face2Face': ('face2face', 'f2f'),
    'FaceSwap': ('faceswap', 'fs'),
    'NeuralTextures': ('neuraltextures', 'nt'),
}


def _subdirs(root):
    """Sub-folders, ignoring hidden ones and those starting with '_' (e.g. ``_unreadable``)."""
    return sorted(d for d in os.listdir(root)
                  if os.path.isdir(os.path.join(root, d)) and not d.startswith(('.', '_')))


def resolve_class_root(root, max_depth=3, verbose=True):
    """Descend through single-child folders until the class folders appear
    (e.g. ``Celeb-DF-v2/`` -> ``Celeb-DF-v2/test/``)."""
    root = os.path.abspath(root)
    for _ in range(max_depth):
        names = os.listdir(root)
        subs = _subdirs(root)
        has_img = any(n.lower().endswith(IMG_EXT) for n in names)
        if len(subs) >= 2 or has_img or len(subs) == 0:
            break
        if verbose:
            print(f'    descending into "{subs[0]}/"')
        root = os.path.join(root, subs[0])
    return root


def split_real_fake(classes, real_hint='real'):
    """Class folders -> (real folders, fake folders) by substring match on ``real_hint``."""
    hints = [h.strip().lower() for h in real_hint.split(',') if h.strip()]
    real = [c for c in classes if any(h in c.lower() for h in hints)]
    fake = [c for c in classes if c not in real]
    return real, fake


def scan_frames(root, real_hint='real', verbose=True) -> Tuple[List[Tuple[str, int]], str]:
    """All images below the class folders of ``root``.

    Returns ``(items, class_root)`` with ``items = [(absolute path, is_fake), ...]`` sorted by
    path (a stable order, so seeded samplers see the same index on every machine).
    """
    croot = resolve_class_root(root, verbose=verbose)
    classes = _subdirs(croot)
    if len(classes) < 2:
        raise ValueError(f'{croot} holds {len(classes)} class folder(s) {classes}; '
                         f'both a real and a fake folder are needed')
    real, fake = split_real_fake(classes, real_hint)
    if not real or not fake:
        raise ValueError(f'cannot tell real from fake in {classes} with real_hint "{real_hint}"')
    items = []
    for c in classes:
        y = 0 if c in real else 1
        for dp, _, names in os.walk(os.path.join(croot, c)):
            for n in sorted(names):
                if n.lower().endswith(IMG_EXT):
                    items.append((os.path.join(dp, n), y))
    if not items:
        raise ValueError(f'no images under {croot}')
    items.sort(key=lambda t: t[0])
    return items, croot


def video_stem(path, sep='_frame_'):
    name = os.path.basename(str(path).replace('\\', '/'))
    return name.split(sep)[0] if sep in name else os.path.splitext(name)[0]


def video_id(path, sep='_frame_'):
    """``<parent folder>/<file stem before sep>``: the unit of video-level AUC."""
    p = str(path).replace('\\', '/')
    return f'{os.path.basename(os.path.dirname(p))}/{video_stem(p, sep)}'


def identity_of(stem):
    """The FF++ video identity of a frame: its first purely numeric token.

    ``DeepFakes_001_870_frame_0.png`` (target 001) and ``youtube_001_frame_216.png`` share
    identity ``001``: the forgery and the real video it was made from.
    """
    for tok in stem.split('_'):
        if tok.isdigit():
            return tok
    return stem.split('_')[0]


def manipulation_of(rel_path):
    """Which FF++ manipulation a path belongs to (from its folder / file name), or None."""
    low = str(rel_path).lower().replace('\\', '/')
    for canon, keys in FFPP_MANIPULATIONS.items():
        for k in keys:
            if re.search(rf'(^|[/_\-]){re.escape(k)}([/_\-]|$)', low):
                return canon
    return None


class FrameDataset(Dataset):
    """``(image, label, index)`` over a list of ``(path, label)``.

    Datasets live in this module (not inside a function) so that DataLoader workers can unpickle
    them under the ``spawn`` start method, the default on Windows and macOS.
    """

    def __init__(self, items, transform, sep='_frame_'):
        self.items = list(items)
        self.transform = transform
        self.sep = sep

    def __len__(self):
        return len(self.items)

    def video_id(self, i):
        return video_id(self.items[i][0], self.sep)

    def __getitem__(self, i):
        p, y = self.items[i]
        with Image.open(p) as im:
            img = im.convert('RGB')
        return self.transform(img), int(y), i


class ImageList(Dataset):
    """``(image, label, index, ok)`` over a list of ``(path, label)``.

    An unreadable file gives a blank image and ``ok = 0`` instead of stopping the run, so the caller
    can keep it out of every metric.
    """

    def __init__(self, items, transform, size=224):
        self.items = list(items)
        self.transform = transform
        self.size = int(size)

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        try:
            with Image.open(p) as im:
                return self.transform(im.convert('RGB')), int(y), i, 1
        except Exception as e:                                         # noqa: BLE001
            print(f'  [warn] cannot read {p}: {e}', flush=True)
            return torch.zeros(3, self.size, self.size), int(y), i, 0
