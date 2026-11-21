import copy

import pytest
import torch

from unmask import FAKE_INDEX, build_unmask, load_unmask, save_unmask, unmask_config

# (inference params M, trainable params M, GMACs of conv + linear layers) as in the paper
EXPECTED = {'lite': (22.02, 1.56, 4.30), 'mid': (28.18, 2.35, 5.42), 'base': (45.83, 3.13, 8.66)}


@pytest.mark.parametrize('variant', ['lite', 'mid', 'base'])
def test_size_matches_paper(variant):
    model = build_unmask(unmask_config(variant)).eval()
    r = model.param_report()
    params, trainable, macs = EXPECTED[variant]
    assert round(r['inference'] / 1e6, 2) == params
    assert round(r['trainable'] / 1e6, 2) == trainable
    from torch.utils.flop_counter import FlopCounterMode
    model.fold_lora()
    with torch.no_grad(), FlopCounterMode(display=False) as fc:
        model(torch.randn(1, 3, 224, 224))
    counts = fc.get_flop_counts()['Global']
    linear_conv = sum(c for op, c in counts.items() if 'scaled_dot_product' not in str(op))
    assert abs(linear_conv / 2e9 - macs) < 0.01


def test_forward_readouts_and_folding(tmp_path):
    torch.manual_seed(0)
    cfg = unmask_config('mid')
    model = build_unmask(cfg).eval()
    for p in model.parameters():                      # make LoRA non-trivial before folding
        if p.ndim == 2 and p.shape[0] < p.shape[1] and p.shape[0] == 16:
            p.data.normal_(0, 0.02)
    with torch.no_grad():
        for m in model.modules():
            if hasattr(m, 'B') and isinstance(m.B, torch.nn.Parameter):
                m.B.normal_(0, 0.02)
    x = torch.randn(2, 3, 224, 224)
    out = model(x, return_all=True, want_tokens=('face',))
    assert out['fused'].shape == (2, 2) and out['gates'].shape == (2, 8)
    assert out['tok_face'].shape == (2, 196, 384)
    ro = model.readouts(x)
    assert set(ro) == {'fused', 'face', 'clip', 'mean'}
    p = model.predict_proba(x)
    assert torch.allclose(p, torch.softmax(ro['fused'], 1)[:, FAKE_INDEX])

    path = tmp_path / 'm.pth'
    save_unmask(path, model, cfg)
    folded, cfg2 = load_unmask(str(path), fold=True, verbose=False)
    assert cfg2['lora_folded']
    assert torch.allclose(folded.predict_proba(x), p, atol=1e-5)
    assert abs(folded.param_report()['total'] - model.param_report()['inference']) == 0
    save_unmask(tmp_path / 'f.pth', folded, cfg2)
    again, _ = load_unmask(str(tmp_path / 'f.pth'), verbose=False)
    assert torch.allclose(again.predict_proba(x), p, atol=1e-5)


def test_se_starts_as_identity():
    model = build_unmask(unmask_config('lite')).eval()
    with torch.no_grad():
        g = model(torch.randn(1, 3, 224, 224), return_all=True)['gates']
    assert torch.allclose(g, torch.ones_like(g))


def test_init_from_folded_checkpoint(tmp_path):
    from unmask.models import load_weights
    cfg = unmask_config('lite')
    src = build_unmask(copy.deepcopy(cfg)).eval()
    src.fold_lora()
    save_unmask(tmp_path / 's.pth', src, cfg)
    dst = build_unmask(copy.deepcopy(cfg)).eval()
    load_weights(dst, torch.load(tmp_path / 's.pth', weights_only=False)['model'])
    x = torch.randn(1, 3, 224, 224)
    assert torch.allclose(dst.predict_proba(x), src.predict_proba(x), atol=1e-6)
