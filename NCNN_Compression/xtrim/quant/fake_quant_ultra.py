from __future__ import annotations

from typing import Any, List

import torch
import torch.nn.functional as F


def _ste_round(x: torch.Tensor) -> torch.Tensor:
    return x + (torch.round(x) - x).detach()


def _fake_quant_per_tensor_symmetric(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    scale = x.detach().abs().amax().clamp(min=1e-8) / qmax
    y = _ste_round(x / scale).clamp(-qmax, qmax) * scale
    return y


def _fake_quant_per_channel_symmetric_w(w: torch.Tensor, bits: int = 8) -> torch.Tensor:
    qmax = (1 << (bits - 1)) - 1
    amax = w.detach().abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8)
    scale = amax / qmax
    y = _ste_round(w / scale).clamp(-qmax, qmax) * scale
    return y


def _is_ultralytics_conv(m: Any) -> bool:
    return hasattr(m, "conv") and isinstance(getattr(m, "conv"), torch.nn.Conv2d) and hasattr(m, "bn")


def patch_ultralytics_convs_for_fake_quant(model: torch.nn.Module) -> List[torch.nn.Module]:
    patched: List[torch.nn.Module] = []

    for _, m in model.named_modules():
        if not _is_ultralytics_conv(m):
            continue
        if hasattr(m, "_xtrim_fq_patched") and getattr(m, "_xtrim_fq_patched"):
            patched.append(m)
            continue

        m._xtrim_fq_patched = True
        m._xtrim_fq_enabled = False
        m._xtrim_fq_bits_w = 8
        m._xtrim_fq_bits_a = 8

        orig_forward = m.forward
        m._xtrim_orig_forward = orig_forward

        def _forward(self, x):
            if not getattr(self, "_xtrim_fq_enabled", False):
                return self._xtrim_orig_forward(x)

            conv = self.conv
            bn = self.bn
            act = self.act

            xa = _fake_quant_per_tensor_symmetric(x, bits=int(self._xtrim_fq_bits_a))
            wq = _fake_quant_per_channel_symmetric_w(conv.weight, bits=int(self._xtrim_fq_bits_w))
            y = F.conv2d(
                xa,
                wq,
                conv.bias,
                stride=conv.stride,
                padding=conv.padding,
                dilation=conv.dilation,
                groups=conv.groups,
            )
            y = bn(y)
            y = act(y)
            return y

        m.forward = _forward.__get__(m, m.__class__)
        patched.append(m)

    return patched


def set_fake_quant_enabled(model: torch.nn.Module, enabled: bool) -> None:
    for _, m in model.named_modules():
        if hasattr(m, "_xtrim_fq_patched") and getattr(m, "_xtrim_fq_patched"):
            m._xtrim_fq_enabled = bool(enabled)


def set_fake_quant_bits(model: torch.nn.Module, bits_w: int = 8, bits_a: int = 8) -> None:
    for _, m in model.named_modules():
        if hasattr(m, "_xtrim_fq_patched") and getattr(m, "_xtrim_fq_patched"):
            m._xtrim_fq_bits_w = int(bits_w)
            m._xtrim_fq_bits_a = int(bits_a)
