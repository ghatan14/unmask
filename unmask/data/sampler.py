"""Paired real/fake sampling for FF++.

Every batch is half real, half fake, and each fake frame comes from a video identity whose
real frames are in the same batch (an FF++ forgery ``<target>_<source>`` is paired with real
video ``<target>``: same scene, background and framing). Training on such pairs keeps the
detector from separating the classes by identity, background or lighting
(Yermakov et al., WACV 2026).
"""

from __future__ import annotations

import random
from collections import defaultdict

from torch.utils.data import Sampler

from .folder import identity_of, video_stem

__all__ = ['PairedRealFakeSampler']


class PairedRealFakeSampler(Sampler):
    """Yields dataset positions; one epoch = ``epoch_mult`` passes over the real frames.

    ``items`` = ``[(path, is_fake), ...]`` of the training dataset (positions = indices).
    Falls back, loudly, to class-balanced random sampling when identities cannot be paired.
    """

    def __init__(self, items, batch_size, seed=0, epoch_mult=1.0, sep='_frame_', verbose=True):
        self.batch_size = batch_size - (batch_size % 2)
        self.half = self.batch_size // 2
        self.rng = random.Random(seed)
        self.epoch_mult = epoch_mult

        real_by_id, fake_by_id = defaultdict(list), defaultdict(list)
        for k, (p, y) in enumerate(items):
            ident = identity_of(video_stem(p, sep))
            (fake_by_id if y == 1 else real_by_id)[ident].append(k)
        self.shared = sorted(set(real_by_id) & set(fake_by_id))
        self.real_by_id = real_by_id
        self.fake_by_id = fake_by_id
        self.all_real = [k for ident in real_by_id for k in real_by_id[ident]]
        self.all_fake = [k for ident in fake_by_id for k in fake_by_id[ident]]

        covered = sum(len(fake_by_id[i]) for i in self.shared)
        self.coverage = covered / max(len(self.all_fake), 1)
        self.paired = self.coverage >= 0.5 and len(self.shared) >= self.half
        self.n_batches = max(int(len(self.all_real) / self.half * self.epoch_mult), 1)
        if verbose:
            print(f'  [sampler] identities: {len(real_by_id)} real {sorted(real_by_id)[:5]}, '
                  f'{len(fake_by_id)} fake {sorted(fake_by_id)[:5]}')
            if self.paired:
                print(f'  [sampler] paired sampling ON: {len(self.shared)} shared identities '
                      f'covering {self.coverage * 100:.0f}% of the fakes')
            else:
                print(f'  [sampler] PAIRING OFF: only {len(self.shared)} shared identities '
                      f'({self.coverage * 100:.0f}% of the fakes) -> class-balanced random '
                      f'sampling. Check that file names carry the FF++ video ids.')

    def __iter__(self):
        for _ in range(self.n_batches):
            batch = []
            if self.paired:
                ids = self.rng.sample(self.shared, min(self.half, len(self.shared)))
                while len(ids) < self.half:
                    ids.append(self.rng.choice(self.shared))
                for ident in ids:
                    batch.append(self.rng.choice(self.real_by_id[ident]))
                    batch.append(self.rng.choice(self.fake_by_id[ident]))
            else:
                batch += [self.rng.choice(self.all_real) for _ in range(self.half)]
                batch += [self.rng.choice(self.all_fake) for _ in range(self.half)]
            self.rng.shuffle(batch)
            yield from batch

    def __len__(self):
        return self.n_batches * self.batch_size
