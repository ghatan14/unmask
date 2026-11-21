#!/usr/bin/env python
"""HiResCAM evidence maps for UNMASK (Draelos & Carin, 2020), as in the paper's Figure 7.

What is explained: the logit margin ``y = log p(fake) - log p(real)`` of the fused head, for real
and fake images alike. HiResCAM multiplies gradient and activation per token and sums over
channels. Maps are taken at the inputs of every read-out path (the LayerNorm output of the taps
after blocks 4 / 8 / 12, and the summary of the last block) and summed over paths and branches
into one 14 x 14 map, upsampled bilinearly.

Conservative gradients (default): the fused head reads all levels through a LayerNorm, which is
scale invariant, so with plain gradients the sum of gradient x activation over the levels is
always zero (Euler's theorem) and plain maps are small residuals of cancelling terms. As in the
conservative propagation for transformers (Ali et al., ICML 2022), the standard deviation of every
LayerNorm on the read-out path, the attention weights of the pooling layers and the SE gates are
treated as constants in the backward pass. The forward pass (and the prediction) is unchanged, and
the map sums to the logit margin minus the bias contributions. ``--plain_gradients`` restores
ordinary gradients for comparison.

Red = evidence for "fake", blue = evidence for "real", transparent = neutral. One colour scale is
shared by all images of a call (99.5th percentile of |map|), so magnitudes are comparable.

    python tools/visualize_hirescam.py --ckpt checkpoints/unmask_base_ffpp.pth \
        --images path/to/face_crops/ --out hirescam.png
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import sys
import types

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from unmask.data import IMG_EXT, eval_transform  # noqa: E402
from unmask.models import FAKE_INDEX, load_unmask  # noqa: E402

FAKE_RGB = np.array([0xe3, 0x49, 0x48], np.float32)
REAL_RGB = np.array([0x2a, 0x78, 0xd6], np.float32)


# ==========================================================================
# conservative gradients: same forward, constants in the backward pass
# ==========================================================================
def _layernorm_const_sigma(self, x):
    mu = x.mean(-1, keepdim=True)
    var = ((x - mu) ** 2).mean(-1, keepdim=True)
    xh = (x - mu) / torch.sqrt(var + self.eps).detach()
    return xh * self.weight + self.bias if self.elementwise_affine else xh


def _pool_const_weights(self, tokens):
    a = self.w(torch.tanh(self.V(tokens)) * torch.sigmoid(self.U(tokens)))
    a = torch.softmax(a, dim=1)
    return (a.detach() * tokens).sum(1), a.squeeze(-1)


def _se_const_gates(self, levels):
    s = torch.cat([F.gelu(m(v)) for m, v in zip(self.squeeze, levels)], dim=1)
    g = 2.0 * torch.sigmoid(self.excite(s))
    return [v * g[:, k:k + 1].detach() for k, v in enumerate(levels)], g


def _attnpool_const_weights(self, x, attn_mask=None):
    """timm AttentionPoolLatent (PE-Core) with the attention weights held constant."""
    B, N, C = x.shape
    if self.pos_embed is not None:
        x = x + self.pos_embed.unsqueeze(0).to(x.dtype)
    q = self.q(self.latent.expand(B, -1, -1)).reshape(B, self.latent_len, self.num_heads, self.head_dim).transpose(1, 2)
    kv = self.kv(x).reshape(B, N, 2, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
    k, v = kv.unbind(0)
    q, k = self.q_norm(q), self.k_norm(k)
    attn = (q * self.scale) @ k.transpose(-2, -1)
    if attn_mask is not None:
        attn = attn + attn_mask
    attn = attn.softmax(dim=-1).detach()
    x = (attn @ v).transpose(1, 2).reshape(B, self.latent_len, C)
    x = self.proj_drop(self.proj(x))
    if self.mlp is not None:
        x = x + self.mlp(self.norm(x))
    if self.pool == 'token':
        x = x[:, 0]
    elif self.pool == 'avg':
        x = x.mean(1)
    return x


@contextlib.contextmanager
def conservative_gradients(model, active=True):
    patched = []

    def patch(mod, fn):
        patched.append((mod, mod.__dict__.get('forward', None)))
        mod.forward = types.MethodType(fn, mod)

    if active:
        patch(model.fc_norm, _layernorm_const_sigma)
        patch(model.se, _se_const_gates)
        for b in model.branch_names:
            br = model.branches[b]
            for p in br.tap_pools:
                patch(p, _pool_const_weights)
            if br.final_norm is not None:
                patch(br.final_norm, _layernorm_const_sigma)
            ap = getattr(br.vit, 'attn_pool', None)
            if ap is not None:
                patch(ap, _attnpool_const_weights)
                if isinstance(getattr(ap, 'norm', None), torch.nn.LayerNorm):
                    patch(ap.norm, _layernorm_const_sigma)
    try:
        yield
    finally:
        for mod, fw in reversed(patched):
            if fw is None:
                del mod.forward
            else:
                mod.forward = fw


def hirescam(model, x, conservative=True):
    """-> (map (S, S) in logit units per pixel, positive evidence per branch / path,
           sum of the map, logit margin y)"""
    store, hooks = {}, []

    def keep(key):
        def f(mod, inp, out):
            o = out[0] if isinstance(out, (tuple, list)) else out
            o.retain_grad()
            store[key] = o
        return f

    for b in model.branch_names:
        br = model.branches[b]
        for j, tn in enumerate(br.tap_norms):
            hooks.append(tn.register_forward_hook(keep((b, f'block{br.taps[j] + 1}'))))
        if br.family == 'fsvfm':
            hooks.append(br.vit.blocks[-1].register_forward_hook(keep((b, 'last'))))
            hooks.append(br.final_norm.register_forward_hook(keep((b, 'summary'))))
        elif br.is_eva and getattr(br.vit, 'attn_pool', None) is not None:
            hooks.append(br.vit.norm.register_forward_hook(keep((b, 'norm'))))
        else:
            raise NotImplementedError(f'branch {b}: CLS-token summaries are not supported')
    try:
        with conservative_gradients(model, conservative):
            x = x.clone().requires_grad_(True)
            out = model(x, return_all=True)
            lp = F.log_softmax(out['fused'].float(), 1)
            y = lp[0, FAKE_INDEX] - lp[0, 1 - FAKE_INDEX]
            model.zero_grad(set_to_none=True)
            y.backward()
    finally:
        for h in hooks:
            h.remove()

    S = x.shape[-1]
    total = torch.zeros(S, S)
    share, map_sum = {}, 0.0
    for b in model.branch_names:
        br = model.branches[b]
        npf, g = br.n_prefix, br.grid
        paths = {}
        for t in br.taps:
            A = store[(b, f'block{t + 1}')]
            paths[f'block{t + 1}'] = (A.grad * A).sum(-1)[0, npf:].detach().cpu().view(g, g)
        if br.family == 'fsvfm':
            # LN(m), m = mean of the patch tokens, sigma held constant -> contribution of token i:
            #   s_i = <grad * gamma, h_i - mean_c(h_i)> / (N * sigma(m))
            h = store[(b, 'last')][0, npf:].detach().float().cpu()
            gr = store[(b, 'summary')].grad[0].detach().float().cpu()
            m = h.mean(0)
            sigma = torch.sqrt(m.var(unbiased=False) + br.final_norm.eps)
            hc = h - h.mean(-1, keepdim=True)
            paths['summary'] = ((hc @ (gr * br.final_norm.weight.detach().float().cpu()))
                                / (h.shape[0] * sigma)).view(g, g)
        else:
            A = store[(b, 'norm')]
            paths['summary'] = (A.grad * A).sum(-1)[0, npf:].detach().cpu().view(g, g)
        area = (S / g) ** 2
        share[b] = {}
        for name, mp in paths.items():
            share[b][name] = float(F.relu(mp).sum())
            map_sum += float(mp.sum())
            total += F.interpolate(mp.float()[None, None] / area, size=(S, S), mode='bilinear',
                                   align_corners=False)[0, 0]
    return total, share, map_sum, float(y.detach())


# ==========================================================================
def overlay(img, cam, scale, alpha_max=0.85, size=256):
    """Image (aspect kept) on a white square canvas; the map is stretched to the image box (the
    inverse of the direct resize). Red = fake evidence, blue = real evidence."""
    w, h = img.size
    s = size / max(w, h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    x0, y0 = (size - nw) // 2, (size - nh) // 2
    base = np.full((size, size, 3), 255, np.float32)
    base[y0:y0 + nh, x0:x0 + nw] = np.asarray(img.resize((nw, nh), Image.BICUBIC), np.float32)
    v = np.zeros((size, size), np.float32)
    cam_img = Image.fromarray(cam.astype(np.float32) / scale)
    v[y0:y0 + nh, x0:x0 + nw] = np.asarray(cam_img.resize((nw, nh), Image.BILINEAR))
    lum = base @ np.array([0.299, 0.587, 0.114], np.float32)
    gray = np.repeat((0.25 * 255 + 0.75 * lum)[..., None], 3, -1)
    v = np.clip(v, -1, 1)[..., None]
    col = np.where(v >= 0, FAKE_RGB, REAL_RGB)
    a = alpha_max * np.abs(v)
    return base.astype(np.uint8), (gray * (1 - a) + col * a).clip(0, 255).astype(np.uint8)


def collect(paths, limit):
    files = []
    for p in paths:
        if os.path.isdir(p):
            for dp, _, names in os.walk(p):
                files += [os.path.join(dp, n) for n in sorted(names) if n.lower().endswith(IMG_EXT)]
        elif os.path.isfile(p):
            files.append(p)
    return sorted(files)[:limit] if limit else sorted(files)


def main():
    ap = argparse.ArgumentParser(description='HiResCAM evidence maps of UNMASK')
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--images', nargs='+', required=True, help='image files and/or folders (face crops)')
    ap.add_argument('--out', default='hirescam.png')
    ap.add_argument('--max_images', type=int, default=16)
    ap.add_argument('--cols', type=int, default=8)
    ap.add_argument('--plain_gradients', action='store_true')
    ap.add_argument('--device', default='cuda')
    args = ap.parse_args()

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    model, cfg = load_unmask(args.ckpt, device=device)
    for p in model.parameters():
        p.requires_grad_(False)
    tf = eval_transform(size=cfg['input_size'], mean=cfg['input_mean'], std=cfg['input_std'])
    files = collect(args.images, args.max_images)
    if not files:
        raise SystemExit('no images found')
    results = []
    for f in files:
        with Image.open(f) as im:
            img = im.convert('RGB')
        x = tf(img).unsqueeze(0).to(device)
        cam, share, map_sum, margin = hirescam(model, x, not args.plain_gradients)
        p_fake = float(torch.sigmoid(torch.tensor(margin)))
        pos = {b: sum(v.values()) for b, v in share.items()}
        tot = sum(pos.values()) or 1.0
        results.append(dict(path=f, img=img, cam=cam.numpy(), p_fake=round(p_fake, 4),
                            logit_margin=round(margin, 4), map_sum=round(map_sum, 4),
                            positive_share_per_branch={b: round(v / tot, 3) for b, v in pos.items()}))
        print(f'  {os.path.basename(f)}: p(fake) {p_fake:.3f} | margin {margin:+.3f} | map sum {map_sum:+.3f} | '
              + ', '.join(f'{b} {v / tot:.0%}' for b, v in pos.items()))
    scale = max(float(np.quantile(np.abs(r['cam']), 0.995)) for r in results) or 1.0
    cols = min(args.cols, len(results))
    rows = int(np.ceil(len(results) / cols))
    fig, axes = plt.subplots(2 * rows, cols, figsize=(1.6 * cols, 3.3 * rows), squeeze=False)
    for ax in axes.ravel():
        ax.set_axis_off()
    for i, r in enumerate(results):
        base, ov = overlay(r['img'], r['cam'], scale)
        rr, cc = 2 * (i // cols), i % cols
        axes[rr, cc].imshow(base)
        axes[rr, cc].set_title(os.path.basename(r['path'])[:22], fontsize=6)
        axes[rr + 1, cc].imshow(ov)
        axes[rr + 1, cc].set_title(f'p(fake) {r["p_fake"]:.2f}', fontsize=7)
    fig.suptitle('HiResCAM' + (' (plain gradients)' if args.plain_gradients else ' (conservative gradients)')
                 + ': red = evidence for fake, blue = evidence for real', fontsize=8)
    fig.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or '.', exist_ok=True)
    fig.savefig(args.out, dpi=200)
    with open(os.path.splitext(args.out)[0] + '.json', 'w', encoding='utf-8') as fh:
        json.dump([{k: v for k, v in r.items() if k not in ('img', 'cam')} for r in results], fh, indent=1)
    print(f'written: {args.out}')


if __name__ == '__main__':
    main()
