"""Building blocks of UNMASK: LoRA, gated attention pooling and multi-level SE fusion."""

from __future__ import annotations

import math
from typing import List, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

__all__ = ['LoRALinear', 'apply_lora', 'fold_lora', 'GatedAttentionPool', 'MultiLevelSE']

# Linear layers of a ViT block that receive a LoRA adapter.
LORA_TARGETS = {
    'attn': (('attn', 'qkv'), ('attn', 'proj')),
    'attn+mlp': (('attn', 'qkv'), ('attn', 'proj'), ('mlp', 'fc1'), ('mlp', 'fc2')),
}


class LoRALinear(nn.Module):
    """``W x + (alpha / r) * B A x`` with ``W`` frozen and only ``A`` and ``B`` trained.

    ``B`` is zero-initialised, so at step 0 the wrapped layer computes exactly the
    pretrained function (Hu et al., ICLR 2022).
    """

    def __init__(self, base: nn.Linear, r: int = 16, alpha: float = 32, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.r = r
        self.scale = alpha / r
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.base(x) + F.linear(F.linear(self.drop(x), self.A), self.B) * self.scale

    @torch.no_grad()
    def merged_linear(self) -> nn.Linear:
        """A plain ``nn.Linear`` with the low-rank update folded into its weight."""
        lin = nn.Linear(self.base.in_features, self.base.out_features,
                        bias=self.base.bias is not None)
        lin.weight.copy_(self.base.weight + (self.B @ self.A) * self.scale)
        lin.weight.requires_grad_(self.base.weight.requires_grad)
        if self.base.bias is not None:
            lin.bias.copy_(self.base.bias)
            lin.bias.requires_grad_(self.base.bias.requires_grad)
        return lin.to(device=self.base.weight.device, dtype=self.base.weight.dtype)


def apply_lora(vit: nn.Module, r: int = 16, alpha: float = 32, dropout: float = 0.0,
               targets: str = 'attn+mlp') -> int:
    """Wrap the target ``nn.Linear`` layers of every block of ``vit`` with LoRA."""
    n = 0
    for blk in vit.blocks:
        for parent_name, name in LORA_TARGETS[targets]:
            parent = getattr(blk, parent_name, None)
            mod = getattr(parent, name, None) if parent is not None else None
            if isinstance(mod, nn.Linear):
                setattr(parent, name, LoRALinear(mod, r, alpha, dropout))
                n += 1
    return n


@torch.no_grad()
def fold_lora(module: nn.Module) -> int:
    """Replace every ``LoRALinear`` below ``module`` by its merged ``nn.Linear``."""
    n = 0
    for name, child in list(module.named_children()):
        if isinstance(child, LoRALinear):
            setattr(module, name, child.merged_linear())
            n += 1
        else:
            n += fold_lora(child)
    return n


class GatedAttentionPool(nn.Module):
    """Gated attention pooling over tokens (Ilse et al., ICML 2018).

    The output layer is zero-initialised, so pooling starts as a plain mean and only
    learns to concentrate on a few tokens if that helps.
    """

    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.V = nn.Linear(dim, hidden)
        self.U = nn.Linear(dim, hidden)
        self.w = nn.Linear(hidden, 1)
        nn.init.zeros_(self.w.weight)
        nn.init.zeros_(self.w.bias)

    def forward(self, tokens):                                   # (B, N, D)
        a = self.w(torch.tanh(self.V(tokens)) * torch.sigmoid(self.U(tokens)))
        a = torch.softmax(a, dim=1)                              # (B, N, 1)
        return (a * tokens).sum(1), a.squeeze(-1)


class MultiLevelSE(nn.Module):
    """Squeeze-and-excitation across pooled levels.

    Every level vector is squeezed to a small descriptor; one excitation layer sees all
    descriptors and emits one gate per level, ``gate = 2 * sigmoid(z)``. ``z`` is
    zero-initialised, so the fusion starts as a plain concatenation (all gates = 1).
    """

    def __init__(self, dims: Sequence[int], hidden: int = 32):
        super().__init__()
        self.squeeze = nn.ModuleList([nn.Sequential(nn.LayerNorm(d), nn.Linear(d, hidden))
                                      for d in dims])
        self.excite = nn.Linear(hidden * len(dims), len(dims))
        nn.init.zeros_(self.excite.weight)
        nn.init.zeros_(self.excite.bias)

    def forward(self, levels: List[torch.Tensor]):
        s = torch.cat([F.gelu(m(v)) for m, v in zip(self.squeeze, levels)], dim=1)
        g = 2.0 * torch.sigmoid(self.excite(s))                  # (B, L)
        return [v * g[:, k:k + 1] for k, v in enumerate(levels)], g
