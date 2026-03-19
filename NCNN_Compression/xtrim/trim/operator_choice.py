from __future__ import annotations

import re
from typing import Any, Dict, List, Literal, Optional, Tuple

import torch.nn as nn

from .sparse_1x1 import apply_1x1_weight_sparsity, find_1x1_convs
from .lowrank import LowRankConv2d

OpMode = Literal["dense", "sparse", "lowrank"]

VALID_MODES = {"dense", "sparse", "lowrank"}


def build_auto_plan(
    model: nn.Module,
    *,
    default: OpMode = "dense",
    exclude_head: bool = True,
    exclude_name_regex: Optional[str] = None,
    min_channels: int = 16,
) -> Dict[str, OpMode]:
    convs = find_1x1_convs(
        model,
        exclude_head=exclude_head,
        exclude_name_regex=exclude_name_regex,
        min_channels=min_channels,
    )
    return {name: default for name, _ in convs}


def validate_plan(plan: Dict[str, Any]) -> Dict[str, OpMode]:
    out: Dict[str, OpMode] = {}
    for name, mode in plan.items():
        mode = str(mode).strip().lower()
        if mode not in VALID_MODES:
            raise ValueError(
                f"Invalid operator mode '{mode}' for layer '{name}'. "
                f"Must be one of: {sorted(VALID_MODES)}"
            )
        out[name] = mode  # type: ignore[assignment]
    return out


def _get_parent_and_attr(model: nn.Module, full_name: str) -> Tuple[nn.Module, str]:
    parts = full_name.split(".")
    parent = model
    for part in parts[:-1]:
        if part.isdigit():
            parent = parent[int(part)]  # type: ignore[index]
        else:
            parent = getattr(parent, part)
    return parent, parts[-1]


def apply_operator_plan(
    model: nn.Module,
    plan: Dict[str, OpMode],
    *,
    sparse_sparsity: float = 0.5,
    sparse_method: str = "l1",
    lowrank_rank: int = 8,
    verbose: bool = True,
) -> Dict[str, Any]:
    plan = validate_plan(plan)

    stats: Dict[str, Any] = {
        "dense": 0,
        "sparse": 0,
        "lowrank": 0,
        "skipped": 0,
        "errors": [],
        "layer_results": {},
    }

    conv_index: Dict[str, nn.Conv2d] = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            conv_index[name] = m

    for layer_name, mode in plan.items():
        if mode == "dense":
            stats["dense"] += 1
            stats["layer_results"][layer_name] = "dense (unchanged)"
            if verbose:
                print(f"[op_choice] {layer_name}: dense (skip)")
            continue

        conv = conv_index.get(layer_name)
        if conv is None:
            msg = f"Layer '{layer_name}' not found or is not Conv2d"
            stats["skipped"] += 1
            stats["errors"].append(msg)
            if verbose:
                print(f"[op_choice] WARNING: {msg}")
            continue

        if conv.kernel_size not in ((1, 1), 1):
            msg = f"Layer '{layer_name}' is not 1x1 (kernel={conv.kernel_size}), skipping"
            stats["skipped"] += 1
            stats["errors"].append(msg)
            if verbose:
                print(f"[op_choice] WARNING: {msg}")
            continue

        try:
            if mode == "sparse":
                import torch.nn.utils.prune as prune_utils
                from .sparse_1x1 import remove_pruning_reparam

                pruning_method = (
                    prune_utils.L1Unstructured
                    if sparse_method == "l1"
                    else prune_utils.RandomUnstructured
                )
                prune_utils.global_unstructured(
                    [(conv, "weight")],
                    pruning_method=pruning_method,
                    amount=float(sparse_sparsity),
                )
                remove_pruning_reparam(conv)

                n_zeros = int((conv.weight == 0).sum().item())
                n_total = conv.weight.numel()
                actual_sparsity = n_zeros / n_total if n_total > 0 else 0.0

                stats["sparse"] += 1
                stats["layer_results"][layer_name] = {
                    "mode": "sparse",
                    "sparsity": actual_sparsity,
                    "zeros": n_zeros,
                    "total": n_total,
                }
                if verbose:
                    print(
                        f"[op_choice] {layer_name}: sparse "
                        f"sparsity={actual_sparsity:.1%} ({n_zeros}/{n_total})"
                    )

            elif mode == "lowrank":
                effective_rank = max(1, int(lowrank_rank))
                max_rank = min(conv.out_channels, conv.in_channels)
                effective_rank = min(effective_rank, max_rank)

                orig_params = conv.out_channels * conv.in_channels
                new_params = effective_rank * (conv.in_channels + conv.out_channels)

                if new_params >= orig_params:
                    if verbose:
                        print(
                            f"[op_choice] {layer_name}: lowrank rank={effective_rank} "
                            f"doesn't save params ({new_params} >= {orig_params}), "
                            f"falling back to dense"
                        )
                    stats["dense"] += 1
                    stats["layer_results"][layer_name] = "lowrank->dense (no savings)"
                    continue

                lowrank_conv = LowRankConv2d.from_conv2d(conv, effective_rank)

                parent, attr = _get_parent_and_attr(model, layer_name)
                setattr(parent, attr, lowrank_conv)

                stats["lowrank"] += 1
                stats["layer_results"][layer_name] = {
                    "mode": "lowrank",
                    "rank": effective_rank,
                    "params_before": orig_params,
                    "params_after": new_params,
                    "compression": orig_params / new_params,
                }
                if verbose:
                    print(
                        f"[op_choice] {layer_name}: lowrank rank={effective_rank} "
                        f"params {orig_params} -> {new_params} "
                        f"({100*new_params/orig_params:.0f}%)"
                    )

        except Exception as e:
            msg = f"Failed to apply '{mode}' to '{layer_name}': {e}"
            stats["errors"].append(msg)
            stats["skipped"] += 1
            if verbose:
                print(f"[op_choice] ERROR: {msg}")

    if verbose:
        print(
            f"[op_choice] Done: dense={stats['dense']}, "
            f"sparse={stats['sparse']}, lowrank={stats['lowrank']}, "
            f"skipped={stats['skipped']}"
        )

    return stats


def plan_from_config(
    model: nn.Module,
    cfg_plan: Optional[Dict[str, str]],
    *,
    default: OpMode = "dense",
    exclude_head: bool = True,
    exclude_name_regex: Optional[str] = None,
    min_channels: int = 16,
) -> Dict[str, OpMode]:
    auto_default = default
    explicit: Dict[str, str] = {}

    if cfg_plan:
        for k, v in cfg_plan.items():
            if k.strip().lower() == "auto":
                auto_default = str(v).strip().lower()  # type: ignore[assignment]
            else:
                explicit[k] = v

    base_plan = build_auto_plan(
        model,
        default=auto_default,  # type: ignore[arg-type]
        exclude_head=exclude_head,
        exclude_name_regex=exclude_name_regex,
        min_channels=min_channels,
    )
    base_plan.update(explicit)  # type: ignore[arg-type]
    return validate_plan(base_plan)
