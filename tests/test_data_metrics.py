import itertools

import numpy as np
import torch
from PIL import Image

from unmask.data import PairedRealFakeSampler, identity_of, manipulation_of, scan_frames, video_id
from unmask.engine import kd_kl, soft_ce
from unmask.metrics import accuracy_metrics, roc_auc, video_scores


def brute_auc(y, s):
    pos = [v for v, t in zip(s, y) if t == 1]
    neg = [v for v, t in zip(s, y) if t == 0]
    wins = sum((p > n) + 0.5 * (p == n) for p, n in itertools.product(pos, neg))
    return 100.0 * wins / (len(pos) * len(neg))


def test_auc_with_ties():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = np.round(rng.random(200), 1)                  # many ties
    assert abs(roc_auc(y, s) - brute_auc(y, s)) < 1e-9
    assert np.isnan(roc_auc(np.ones(5), np.arange(5)))


def test_video_scores_and_accuracy():
    vy, vs, ids = video_scores([0.1, 0.3, 0.9, 0.7], [0, 0, 1, 1], ['a', 'a', 'b', 'b'])
    assert ids == ['a', 'b'] and np.allclose(vs, [0.2, 0.8]) and list(vy) == [0, 1]
    m = accuracy_metrics([0.2, 0.6, 0.5, 0.4], [0, 0, 1, 1])
    assert m['acc'] == 50.0 and m['rec'] == 50.0 and m['tnr'] == 50.0


def test_ffpp_naming():
    assert identity_of('DeepFakes_001_870') == '001'
    assert identity_of('youtube_001') == '001'
    assert manipulation_of('fake_four/NeuralTextures_033_097_frame_3.png') == 'NeuralTextures'
    assert manipulation_of('real_youtube/youtube_033_frame_3.png') is None
    assert video_id('/x/test/fake/id0_id1_0003_frame_12.png') == 'fake/id0_id1_0003'


def test_scan_and_paired_sampler(tmp_path):
    for c in ('real_youtube', 'fake_four', '_unreadable'):
        (tmp_path / c).mkdir()
    for i in range(1, 5):
        for f in range(2):
            Image.new('RGB', (8, 8)).save(tmp_path / 'real_youtube' / f'youtube_{i:03d}_frame_{f}.png')
            Image.new('RGB', (8, 8)).save(tmp_path / 'fake_four' / f'Deepfakes_{i:03d}_9{i:02d}_frame_{f}.png')
    items, root = scan_frames(str(tmp_path), verbose=False)
    assert len(items) == 16 and sum(y for _, y in items) == 8
    sampler = PairedRealFakeSampler(items, batch_size=4, seed=0, epoch_mult=1.0, verbose=False)
    assert sampler.paired and len(sampler) == 16
    idx = list(sampler)
    for b in range(0, len(idx), 4):
        batch = [items[k] for k in idx[b:b + 4]]
        assert sum(y for _, y in batch) == 2           # half real, half fake


def test_losses():
    logits = torch.randn(4, 2)
    y = torch.tensor([0, 1, 1, 0])
    ce = torch.nn.functional.cross_entropy(logits, y)
    assert torch.allclose(soft_ce(logits, y, y, 1.0, 2, 0.0), ce)
    p = torch.softmax(logits / 3, 1)
    assert kd_kl(logits, p, 3.0).abs() < 1e-6
