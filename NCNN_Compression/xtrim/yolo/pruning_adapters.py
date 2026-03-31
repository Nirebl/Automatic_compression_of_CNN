from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


try:
    from ultralytics.nn.modules.block import C2f, Bottleneck
    from ultralytics.nn.modules.conv import Conv
except Exception:
    # fallback для некоторых версий ultralytics
    from ultralytics.nn.modules import C2f, Bottleneck, Conv  # type: ignore


class C2fPrunable(nn.Module):
    """
    Pruning-friendly replacement for Ultralytics C2f.

    Оригинальный C2f делает:
        cv1(x) -> chunk(2, dim=1) -> ... -> cat(...)
    Это неудобно для dependency-graph pruning.

    Здесь split делается двумя явными ветками:
        cv0(x), cv1(x)

    ВАЖНО:
    - forward написан явным циклом, без generator expression,
      чтобы трассировка у torch-pruning была стабильнее;
    - верхний parent.cv2 (агрегационный conv после concat) не должен
      использоваться как root prune target.
    """

    def __init__(
        self,
        c1: int,
        c2: int,
        n: int = 1,
        shortcut: bool = False,
        g: int = 1,
        e: float = 0.5,
    ):
        super().__init__()
        self.c = int(c2 * e)

        self.cv0 = Conv(c1, self.c, 1, 1)
        self.cv1 = Conv(c1, self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1)

        self.m = nn.ModuleList(
            Bottleneck(
                self.c,
                self.c,
                shortcut,
                g,
                k=((3, 3), (3, 3)),
                e=1.0,
            )
            for _ in range(n)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y0 = self.cv0(x)
        y1 = self.cv1(x)

        ys = [y0, y1]
        z = y1
        for block in self.m:
            z = block(z)
            ys.append(z)

        return self.cv2(torch.cat(ys, 1))

    def forward_split(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def _copy_conv_bn_(dst: Any, src: Any, out_slice: slice | None = None) -> None:
    """
    Копирует Ultralytics Conv() + BN, optionally slicing output channels.
    """
    with torch.no_grad():
        src_w = src.conv.weight.data
        if out_slice is None:
            dst.conv.weight.data.copy_(src_w)
        else:
            dst.conv.weight.data.copy_(src_w[out_slice].clone())

        if getattr(src.conv, "bias", None) is not None and getattr(dst.conv, "bias", None) is not None:
            if out_slice is None:
                dst.conv.bias.data.copy_(src.conv.bias.data)
            else:
                dst.conv.bias.data.copy_(src.conv.bias.data[out_slice].clone())

        for attr in ("weight", "bias", "running_mean", "running_var"):
            s = getattr(src.bn, attr)
            d = getattr(dst.bn, attr)
            if out_slice is None:
                d.data.copy_(s.data)
            else:
                d.data.copy_(s.data[out_slice].clone())

        if hasattr(src.bn, "num_batches_tracked") and hasattr(dst.bn, "num_batches_tracked"):
            dst.bn.num_batches_tracked.data.copy_(src.bn.num_batches_tracked.data)

        dst.act = src.act


def _copy_ultralytics_meta(dst: nn.Module, src: nn.Module) -> None:
    """
    Ultralytics BaseModel._predict_once использует у модулей поля:
    - i
    - f
    - type
    - np

    Их надо перенести при замене блока.
    """
    for attr in ("i", "f", "type", "np"):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def convert_c2f_to_prunable(c2f: Any) -> C2fPrunable:
    if not isinstance(c2f, C2f):
        raise TypeError(f"Expected ultralytics C2f, got {type(c2f).__name__}")

    c1 = int(c2f.cv1.conv.in_channels)
    c2 = int(c2f.cv2.conv.out_channels)
    n = len(c2f.m)
    g = int(getattr(c2f.m[0].cv2.conv, "groups", 1)) if n > 0 else 1
    shortcut = bool(getattr(c2f.m[0], "add", False)) if n > 0 else False
    e = float(c2f.c) / float(c2) if c2 > 0 else 0.5

    repl = C2fPrunable(c1=c1, c2=c2, n=n, shortcut=shortcut, g=g, e=e)
    repl.to(device=c2f.cv1.conv.weight.device, dtype=c2f.cv1.conv.weight.dtype)
    repl.train(c2f.training)

    _copy_ultralytics_meta(repl, c2f)

    # split original cv1 -> (cv0, cv1)
    c = int(c2f.c)
    _copy_conv_bn_(repl.cv0, c2f.cv1, slice(0, c))
    _copy_conv_bn_(repl.cv1, c2f.cv1, slice(c, 2 * c))

    # copy cv2 and bottlenecks as-is
    repl.cv2.load_state_dict(c2f.cv2.state_dict())
    repl.m.load_state_dict(c2f.m.state_dict())

    return repl


def replace_c2f_with_prunable(module: nn.Module, verbose: bool = True) -> int:
    """
    Рекурсивно заменяет все Ultralytics C2f внутри модели на C2fPrunable.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        if isinstance(child, C2f):
            setattr(module, name, convert_c2f_to_prunable(child))
            replaced += 1
            if verbose:
                print(f"[adapt] replaced C2f -> C2fPrunable at {name}")
        else:
            replaced += replace_c2f_with_prunable(child, verbose=verbose)
    return replaced