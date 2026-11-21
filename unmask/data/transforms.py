"""Image transforms: training augmentation with codec / sensor degradation, and the
deterministic evaluation transform of the paper protocol (direct resize to 224, bicubic)."""

from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageFilter
from torchvision import transforms
from torchvision.transforms import functional as TF

from ..utils import IMAGENET_MEAN, IMAGENET_STD

__all__ = ['Degrade', 'TrainTransform', 'FixedDegrade', 'eval_transform']


class Degrade:
    """Random codec and sensor damage on a PIL image: rescaling, blur, noise and JPEG
    (each with probability 0.55, in random order), plus occasional grayscale.

    With ``seed=None`` the global ``random`` module is used on purpose: PyTorch re-seeds it in
    every DataLoader worker, whereas a private generator would be copied to every worker in the
    same state (on Windows workers are spawned and would replay one identical sequence).
    """

    def __init__(self, p=0.9, jpeg=(30, 95), scale=(0.35, 1.0), blur=(0.0, 1.4),
                 noise=(0.0, 7.0), gray_p=0.08, seed=None):
        self.p = p
        self.jpeg = jpeg
        self.scale = scale
        self.blur = blur
        self.noise = noise
        self.gray_p = gray_p
        self._rng = random.Random(seed) if seed is not None else None

    @property
    def rng(self):
        return self._rng if self._rng is not None else random

    def _jpeg(self, img):
        q = self.rng.randint(*self.jpeg)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).convert('RGB')

    def _rescale(self, img):
        w, h = img.size
        s = self.rng.uniform(*self.scale)
        small = img.resize((max(int(w * s), 16), max(int(h * s), 16)), Image.BILINEAR)
        return small.resize((w, h), self.rng.choice([Image.BILINEAR, Image.BICUBIC, Image.NEAREST]))

    def _blur(self, img):
        r = self.rng.uniform(*self.blur)
        return img.filter(ImageFilter.GaussianBlur(radius=r)) if r > 0.05 else img

    def _noise(self, img):
        s = self.rng.uniform(*self.noise)
        if s < 0.3:
            return img
        a = np.asarray(img, dtype=np.float32)
        g = np.random.default_rng(self.rng.getrandbits(63))      # per-worker Python RNG
        a = a + g.normal(0, s, a.shape).astype(np.float32)
        return Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))

    def __call__(self, img):
        if self.rng.random() > self.p:
            return img
        ops = [self._rescale, self._blur, self._noise, self._jpeg]
        self.rng.shuffle(ops)
        for op in ops:
            if self.rng.random() < 0.55:
                img = op(img)
        if self.rng.random() < self.gray_p:
            img = img.convert('L').convert('RGB')
        return img


class TrainTransform:
    """PIL -> normalised tensor for training.

    Mild geometry (random resized crop with scale >= ``scale_min``, horizontal flip), colour
    jitter, degradation and random erasing.
    """

    def __init__(self, size=224, scale_min=0.75, hflip=0.5, jitter=0.25, erase_p=0.25,
                 degrade=True, mean=IMAGENET_MEAN, std=IMAGENET_STD):
        self.size = size
        self.scale = (scale_min, 1.0)
        self.hflip = hflip
        self.degrade = Degrade() if degrade else None
        self.jitter = transforms.ColorJitter(jitter, jitter, jitter, jitter / 4) if jitter > 0 else None
        self.to_tensor = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean, std)])
        self.erase = transforms.RandomErasing(p=erase_p, scale=(0.02, 0.12), ratio=(0.4, 2.5),
                                              value='random') if erase_p > 0 else None

    def __call__(self, img):
        i, j, h, w = transforms.RandomResizedCrop.get_params(img, list(self.scale), [3. / 4., 4. / 3.])
        img = TF.resized_crop(img, i, j, h, w, [self.size, self.size], TF.InterpolationMode.BICUBIC)
        if random.random() < self.hflip:
            img = TF.hflip(img)
        if self.jitter is not None and random.random() < 0.8:
            img = self.jitter(img)
        if self.degrade is not None:
            img = self.degrade(img)
        x = self.to_tensor(img)
        if self.erase is not None:
            x = self.erase(x)
        return x


class FixedDegrade:
    """Deterministic damage for the 'degraded' validation split: half resolution and back,
    then JPEG quality 40. (Module-level so it can be pickled into spawned workers.)"""

    def __call__(self, img):
        w, h = img.size
        img = img.resize((max(w // 2, 16), max(h // 2, 16)), Image.BILINEAR)
        img = img.resize((w, h), Image.BILINEAR)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=40)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


def eval_transform(size=224, mean=IMAGENET_MEAN, std=IMAGENET_STD, degrade=False,
                   resize='direct', with_normalize=True):
    """Evaluation transform. ``resize='direct'`` (resize the whole crop to size x size,
    bicubic, no centre crop) is the protocol of every number in the paper."""
    if resize == 'centercrop':
        pre = [transforms.Resize(int(size / (224 / 256)),
                                 interpolation=transforms.InterpolationMode.BICUBIC),
               transforms.CenterCrop(size)]
    else:
        pre = [transforms.Resize((size, size), interpolation=transforms.InterpolationMode.BICUBIC)]
    if degrade:
        pre.append(FixedDegrade())
    post = [transforms.ToTensor()] + ([transforms.Normalize(mean, std)] if with_normalize else [])
    return transforms.Compose(pre + post)
