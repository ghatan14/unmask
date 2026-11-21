"""DataLoader workers under the ``spawn`` start method, the default on Windows and macOS.

Linux uses ``fork``, where any dataset works; with ``spawn`` the dataset and its transform are
pickled into fresh processes, so they must be importable module-level objects.
"""
import multiprocessing as mp

import numpy as np
import pytest
from PIL import Image
from torch.utils.data import DataLoader

from unmask.data import FrameDataset, ImageList, TrainTransform, eval_transform


def _frames(tmp_path, n=4):
    items = []
    for i in range(n):
        p = tmp_path / f'vid{i // 2}_frame_{i}.png'
        Image.fromarray(np.full((40, 40, 3), 40 * i, np.uint8)).save(p)
        items.append((str(p), i % 2))
    return items


@pytest.mark.parametrize('kind', ['eval', 'train', 'image_list'])
def test_spawn_workers(tmp_path, kind):
    items = _frames(tmp_path)
    if kind == 'image_list':
        empty = tmp_path / 'empty.png'
        empty.write_bytes(b'')                       # the HydraFake release has such a file
        ds = ImageList(items + [(str(empty), 1)], eval_transform(size=32), size=32)
    else:
        ds = FrameDataset(items, eval_transform(size=32) if kind == 'eval' else TrainTransform(size=32))
    dl = DataLoader(ds, batch_size=2, num_workers=1, multiprocessing_context=mp.get_context('spawn'))
    out = list(dl)
    assert sum(len(b[0]) for b in out) == len(ds)
    assert tuple(out[0][0].shape[1:]) == (3, 32, 32)
    if kind == 'image_list':
        ok = np.concatenate([b[3].numpy() for b in out])
        assert ok.tolist() == [1, 1, 1, 1, 0]
