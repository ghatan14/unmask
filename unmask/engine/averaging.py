"""Weight averaging: EMA over the trainable tensors and SWA with LoRA folded."""

from __future__ import annotations

import copy

import torch

from ..models.layers import LoRALinear

__all__ = ['EMA', 'MergedSWA']


class EMA:
    """Exponential moving average of the trainable parameters only (the frozen backbone never
    changes, so averaging it would be pure overhead)."""

    def __init__(self, model, decay):
        self.decay = decay
        self.shadow = copy.deepcopy(model).eval()
        for p in self.shadow.parameters():
            p.requires_grad_(False)
        sp = dict(self.shadow.named_parameters())
        self.pairs = [(sp[n], p) for n, p in model.named_parameters() if p.requires_grad]

    @torch.no_grad()
    def update(self):
        for s, p in self.pairs:
            s.mul_(self.decay).add_(p.detach(), alpha=1 - self.decay)

    def state_dict(self):
        return self.shadow.state_dict()


class MergedSWA:
    """Running average of the weights visited in the second half of training (SWA/SWAD).

    LoRA factors are not averaged directly (mean(B) @ mean(A) is not the mean update): the
    running mean is kept of the folded update ``scale * B @ A``, and the result is a plain model
    with LoRA folded."""

    def __init__(self, model):
        self.n = 0
        self.lora = {}
        lora_params = set()
        for name, mod in model.named_modules():
            if isinstance(mod, LoRALinear) and mod.A.requires_grad:
                self.lora[name] = (mod, torch.zeros_like(mod.base.weight, dtype=torch.float32))
                lora_params.update({id(mod.A), id(mod.B)})
        self.other = {n: (p, torch.zeros_like(p, dtype=torch.float32))
                      for n, p in model.named_parameters()
                      if p.requires_grad and id(p) not in lora_params}

    @torch.no_grad()
    def update(self):
        for mod, acc in self.lora.values():
            acc.add_((mod.B.float() @ mod.A.float()) * mod.scale)
        for p, acc in self.other.values():
            acc.add_(p.detach().float())
        self.n += 1

    @torch.no_grad()
    def merged_state_dict(self, model):
        sd = {k: v.detach().clone() for k, v in model.state_dict().items()}
        for pre, (mod, acc) in self.lora.items():
            w = sd.pop(pre + '.base.weight')
            sd[pre + '.weight'] = (w.float() + acc / self.n).to(w.dtype)
            if pre + '.base.bias' in sd:
                sd[pre + '.bias'] = sd.pop(pre + '.base.bias')
            sd.pop(pre + '.A', None)
            sd.pop(pre + '.B', None)
        for n, (p, acc) in self.other.items():
            sd[n] = (acc / self.n).to(p.dtype)
        for k in [k for k in sd if k.endswith('.A')]:       # untrained LoRA factors, if any
            pre = k[:-2]
            A, B = sd.pop(pre + '.A').float(), sd.pop(pre + '.B').float()
            w = sd.pop(pre + '.base.weight')
            mod = model.get_submodule(pre)
            sd[pre + '.weight'] = (w.float() + (B @ A) * mod.scale).to(w.dtype)
            if pre + '.base.bias' in sd:
                sd[pre + '.bias'] = sd.pop(pre + '.base.bias')
        return {k: v.cpu() for k, v in sd.items()}

    def state(self):
        return dict(n=self.n, lora={k: a.cpu() for k, (_, a) in self.lora.items()},
                    other={k: a.cpu() for k, (_, a) in self.other.items()})

    def load_state(self, st):
        self.n = st['n']
        for k, (_, acc) in self.lora.items():
            acc.copy_(st['lora'][k].to(acc.device))
        for k, (_, acc) in self.other.items():
            acc.copy_(st['other'][k].to(acc.device))
