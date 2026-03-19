from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.utils.prune as prune


def find_1x1_convs(
    model: nn.Module,
    *,
    exclude_head: bool = True,
    exclude_name_regex: Optional[str] = None,
    min_channels: int = 16,
) -> List[Tuple[str, nn.Conv2d]]:
    exclude_re = re.compile(exclude_name_regex) if exclude_name_regex else None

    head_ids: set = set()
    if exclude_head and hasattr(model, "model"):
        seq = getattr(model, "model")
        try:
            if hasattr(seq, "__len__") and len(seq) > 0:
                head = seq[-1]
                for _, hm in head.named_modules():
                    head_ids.add(id(hm))
        except Exception:
            pass

    result: List[Tuple[str, nn.Conv2d]] = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if m.kernel_size != (1, 1):
            continue
        if m.groups != 1:
            continue
        if m.out_channels < min_channels or m.in_channels < min_channels:
            continue
        if exclude_re and exclude_re.search(name):
            continue
        if head_ids and id(m) in head_ids:
            continue
        result.append((name, m))

    return result


def apply_1x1_weight_sparsity(
    model: nn.Module,
    sparsity: float,
    *,
    method: str = "l1",
    exclude_head: bool = True,
    exclude_name_regex: Optional[str] = None,
    min_channels: int = 16,
    verbose: bool = True,
) -> Dict[str, Any]:
    if sparsity <= 0.0:
        return {"layers_sparsified": 0, "total_params_masked": 0, "avg_sparsity": 0.0}

    sparsity = min(sparsity, 0.99)

    convs = find_1x1_convs(
        model,
        exclude_head=exclude_head,
        exclude_name_regex=exclude_name_regex,
        min_channels=min_channels,
    )

    if not convs:
        if verbose:
            print("[sparse_1x1] No eligible 1x1 Conv2d layers found.")
        return {"layers_sparsified": 0, "total_params_masked": 0, "avg_sparsity": 0.0}

    pruning_method = prune.L1Unstructured if method == "l1" else prune.RandomUnstructured

    layers_done = 0
    total_masked = 0
    total_params = 0

    for name, conv in convs:
        n_params = conv.weight.numel()
        prune.global_unstructured(
            [(conv, "weight")],
            pruning_method=pruning_method,
            amount=sparsity,
        )

        zeros = int((conv.weight == 0).sum().item())
        actual_sparsity = zeros / n_params if n_params > 0 else 0.0

        if verbose:
            print(
                f"[sparse_1x1] {name}: "
                f"{conv.in_channels}x{conv.out_channels} "
                f"sparsity={actual_sparsity:.1%} "
                f"({zeros}/{n_params} zeros)"
            )

        total_masked += zeros
        total_params += n_params
        layers_done += 1

    avg_sparsity = total_masked / total_params if total_params > 0 else 0.0

    if verbose:
        print(
            f"[sparse_1x1] Done: {layers_done} layers, "
            f"avg sparsity={avg_sparsity:.1%}, "
            f"{total_masked}/{total_params} weights zeroed"
        )

    return {
        "layers_sparsified": layers_done,
        "total_params_masked": total_masked,
        "total_params_1x1": total_params,
        "avg_sparsity": avg_sparsity,
    }


def remove_pruning_reparam(model: nn.Module, verbose: bool = False) -> int:
    count = 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue
        if not hasattr(m, "weight_orig"):
            continue
        try:
            prune.remove(m, "weight")
            count += 1
            if verbose:
                print(f"[sparse_1x1] remove reparam: {name}")
        except Exception:
            pass
    return count
