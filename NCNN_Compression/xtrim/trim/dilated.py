from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch.nn as nn


def _find_head_module(torch_model: Any) -> Optional[Any]:
    if hasattr(torch_model, "model"):
        seq = getattr(torch_model, "model")
        try:
            if hasattr(seq, "__len__") and len(seq) > 0:
                return seq[-1]
        except Exception:
            pass
    return None


def _is_stem_layer(name: str) -> bool:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "model" and i + 1 < len(parts) and parts[i + 1] == "0":
            return True
    if parts[0] == "0":
        return True
    return False


def find_conv_blocks_for_dilation(
    torch_model: Any,
    *,
    exclude_head: bool = True,
    target_layers: Tuple[str, ...] = (),
    target_n_blocks: int = 4,
) -> List[Tuple[str, nn.Conv2d]]:
    head_mod = _find_head_module(torch_model) if exclude_head else None
    head_ids: set = set()
    if head_mod is not None:
        for _, hm in head_mod.named_modules():
            head_ids.add(id(hm))

    candidates: List[Tuple[str, nn.Conv2d]] = []

    for name, m in torch_model.named_modules():
        if head_ids and id(m) in head_ids:
            continue
        if _is_stem_layer(name):
            continue

        if not (hasattr(m, "conv") and isinstance(getattr(m, "conv"), nn.Conv2d)):
            continue

        conv: nn.Conv2d = getattr(m, "conv")
        k = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
        if k != 3:
            continue
        if conv.groups != 1:
            continue

        candidates.append((name, conv))

    if target_layers:
        target_set = set(target_layers)
        candidates = [(n, c) for n, c in candidates if n in target_set]
    else:
        candidates = candidates[-target_n_blocks:] if target_n_blocks > 0 else candidates

    return candidates


def apply_dilation(
    torch_model: Any,
    *,
    rates: Tuple[int, ...] = (1, 2),
    exclude_head: bool = True,
    target_layers: Tuple[str, ...] = (),
    target_n_blocks: int = 4,
    verbose: bool = True,
) -> Dict[str, Any]:
    if not rates:
        rates = (1,)

    candidates = find_conv_blocks_for_dilation(
        torch_model,
        exclude_head=exclude_head,
        target_layers=target_layers,
        target_n_blocks=target_n_blocks,
    )

    if not candidates:
        if verbose:
            print("[dilated] No eligible 3x3 Conv layers found")
        return {"layers_modified": 0, "layers": []}

    layers_modified = 0
    layer_report: List[Dict[str, Any]] = []

    for idx, (parent_name, conv) in enumerate(candidates):
        rate = int(rates[idx % len(rates)])

        k = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
        new_padding = rate * (k - 1) // 2

        old_dilation = conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation
        old_padding = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding

        if rate == old_dilation:
            if verbose:
                print(f"[dilated] {parent_name}.conv: already dilation={rate}, skip")
            continue

        conv.dilation = (rate, rate)
        conv.padding = (new_padding, new_padding)

        layers_modified += 1
        info = {
            "name": f"{parent_name}.conv",
            "dilation": rate,
            "padding_before": old_padding,
            "padding_after": new_padding,
        }
        layer_report.append(info)

        if verbose:
            print(
                f"[dilated] {parent_name}.conv: "
                f"dilation {old_dilation}->{rate}, padding {old_padding}->{new_padding}"
            )

    return {"layers_modified": layers_modified, "rates": list(rates), "layers": layer_report}
