from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple
import torch

import numpy as np


def _is_ultralytics_conv(m: Any) -> bool:
    try:
        import torch
        return hasattr(m, "conv") and isinstance(getattr(m, "conv"), torch.nn.Conv2d) and hasattr(m, "bn")
    except Exception:
        return False


def _find_head_module(torch_model: Any) -> Optional[Any]:
    head = None
    if hasattr(torch_model, "model"):
        seq = getattr(torch_model, "model")
        try:
            if hasattr(seq, "__len__") and len(seq) > 0:
                head = seq[-1]
        except Exception:
            head = None
    return head


def _get_module_by_qualname(root: Any, qualname: str) -> Any:
    cur = root
    if not qualname:
        return cur
    for part in qualname.split("."):
        if part.isdigit():
            cur = cur[int(part)]
        else:
            cur = getattr(cur, part)
    return cur


def _should_skip_prune(
    torch_model: Any,
    name: str,
    *,
    skip_inner_m: bool,
    skip_cv1_if_parent_has_m: bool,
) -> bool:
    if skip_inner_m and re.search(r"\.m\.\d+\.", name):
        return True

    if skip_cv1_if_parent_has_m and name.endswith(".cv1"):
        parent_name = ".".join(name.split(".")[:-1])
        try:
            parent = _get_module_by_qualname(torch_model, parent_name)
            if hasattr(parent, "m"):
                return True
        except Exception:
            pass

    return False


def _collect_prunable_convs(
    torch_model: Any,
    *,
    exclude_head: bool,
    exclude_name_regex: Optional[str],
    skip_inner_m: bool = True,
    skip_cv1_if_parent_has_m: bool = True,
) -> List[Tuple[str, Any]]:
    exclude_re = re.compile(exclude_name_regex) if exclude_name_regex else None
    head_mod = _find_head_module(torch_model) if exclude_head else None

    head_ids = set()
    if head_mod is not None:
        for _, hm in head_mod.named_modules():
            head_ids.add(id(hm))

    prunable: List[Tuple[str, Any]] = []
    for name, m in torch_model.named_modules():
        if exclude_re and exclude_re.search(name):
            continue
        if not _is_ultralytics_conv(m):
            continue
        if head_ids and id(m) in head_ids:
            continue
        if _should_skip_prune(
            torch_model,
            name,
            skip_inner_m=skip_inner_m,
            skip_cv1_if_parent_has_m=skip_cv1_if_parent_has_m,
        ):
            continue
        prunable.append((name, m))
    return prunable


def _global_threshold_from_bn_gammas(prunable: List[Tuple[str, Any]], prune_ratio: float) -> float:
    all_g = []
    for _, m in prunable:
        bn = getattr(m, "bn", None)
        if bn is None or getattr(bn, "weight", None) is None:
            continue
        g = bn.weight.detach().abs().cpu().numpy()
        all_g.append(g)
    if not all_g:
        return 0.0
    vec = np.concatenate(all_g, axis=0)
    if vec.size == 0:
        return 0.0
    q = float(max(0.0, min(0.95, prune_ratio)))
    return float(np.quantile(vec, q))


def _select_prune_idxs_by_gamma(bn, threshold: float, channel_round: int, min_channels: int) -> List[int]:
    import torch

    g = bn.weight.detach().abs()
    C = int(g.numel())
    if C <= min_channels:
        return []

    keep = (g > threshold).nonzero(as_tuple=False).view(-1).tolist()
    if len(keep) < min_channels:
        keep = torch.topk(g, k=min_channels, largest=True).indices.tolist()

    if channel_round > 1:
        target_keep = int(np.ceil(len(keep) / channel_round) * channel_round)
        target_keep = max(min_channels, min(C, target_keep))
        if target_keep != len(keep):
            keep = torch.topk(g, k=target_keep, largest=True).indices.tolist()

    keep_set = set(keep)
    prune = [i for i in range(C) if i not in keep_set]

    if len(prune) >= C:
        prune = list(range(C - min_channels))
    return prune


def _select_prune_idxs_layerwise(
    bn,
    prune_ratio: float,
    channel_round: int,
    min_channels: int,
    max_prune_ratio: Optional[float] = None,
    importance_mode: str = "bn_gamma",
    conv: Optional[Any] = None,
) -> List[int]:
    import torch

    if importance_mode == "uniform" and conv is not None:
        weight = conv.weight.detach()
        C = weight.shape[0]
        g = weight.view(C, -1).norm(dim=1)
    else:
        g = bn.weight.detach().abs()
        C = int(g.numel())

    if C <= min_channels:
        return []

    target_keep = int(np.ceil((1.0 - float(prune_ratio)) * C))
    target_keep = max(min_channels, min(C, target_keep))

    if max_prune_ratio is not None:
        cap = float(max_prune_ratio)
        cap = max(0.0, min(0.95, cap))
        max_prune = int(np.floor(cap * C))
        max_prune = max(0, min(C - min_channels, max_prune))
        min_keep_cap = C - max_prune
        target_keep = max(target_keep, min_keep_cap)

    if channel_round > 1:
        target_keep = int(np.ceil(target_keep / channel_round) * channel_round)
        target_keep = max(min_channels, min(C, target_keep))

    if target_keep >= C:
        return []

    keep = torch.topk(g, k=target_keep, largest=True).indices.tolist()
    keep_set = set(keep)
    prune = [i for i in range(C) if i not in keep_set]

    if len(prune) > C - min_channels:
        order = torch.argsort(g, descending=False).tolist()
        prune = order[: max(0, C - min_channels)]
    return prune


def bn_sparsity_regularizer(
    torch_model: Any,
    l1_weight: float = 1e-4,
    *,
    exclude_head: bool = True,
    exclude_name_regex: Optional[str] = None,
):
    import torch
    reg = torch.zeros((), device=next(torch_model.parameters()).device)
    prunable = _collect_prunable_convs(
        torch_model,
        exclude_head=exclude_head,
        exclude_name_regex=exclude_name_regex,
        skip_inner_m=True,
        skip_cv1_if_parent_has_m=True,
    )
    for _, m in prunable:
        bn = getattr(m, "bn", None)
        if bn is not None and getattr(bn, "weight", None) is not None:
            reg = reg + bn.weight.abs().sum()
    return reg * float(l1_weight)


def _as_tensor_list(x: Any) -> List["torch.Tensor"]:
    out: List[torch.Tensor] = []
    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for t in x:
            out.extend(_as_tensor_list(t))
    elif isinstance(x, dict):
        for v in x.values():
            out.extend(_as_tensor_list(v))
    return out


def _make_example_inputs_for_tp(example_input):
    import torch

    if torch.is_tensor(example_input):
        x = example_input.detach()
        x.requires_grad_(True)
        return x

    if isinstance(example_input, (list, tuple)) and len(example_input) == 1 and torch.is_tensor(example_input[0]):
        x = example_input[0].detach()
        x.requires_grad_(True)
        return x

    return example_input


def _build_dependency_graph(tp, torch_model, example_input):
    import torch

    dg = tp.DependencyGraph()
    ex_inputs = _make_example_inputs_for_tp(example_input)

    def _out_transform(out):
        ts = _as_tensor_list(out)
        if not ts:
            return out
        return max(ts, key=lambda t: t.numel())

    with torch.enable_grad():
        try:
            dg.build_dependency(torch_model, example_inputs=ex_inputs, output_transform=_out_transform)
        except TypeError:
            dg.build_dependency(torch_model, example_inputs=ex_inputs)

    return dg


def _forward_check(torch_model, example_input):
    import torch

    x = example_input
    if isinstance(x, (list, tuple)) and len(x) == 1:
        x = x[0]
    if torch.is_tensor(x):
        x = x.detach()

    with torch.no_grad():
        _ = torch_model(x)


def structured_trim_yolo(
    torch_model: Any,
    *,
    example_input,
    prune_ratio: float,
    channel_round: int = 8,
    min_channels: int = 8,
    exclude_head: bool = True,
    exclude_name_regex: Optional[str] = None,
    strategy: str = "layerwise",
    max_prune_per_layer: Optional[float] = None,
    protect_last_n: int = 0,
    verbose: bool = True,
    skip_inner_m: bool = True,
    skip_cv1_if_parent_has_m: bool = True,
    importance_mode: str = "bn_gamma",
) -> Dict[str, int]:
    if not (0.0 < prune_ratio < 0.95):
        raise ValueError("prune_ratio must be in (0, 0.95)")

    try:
        import torch_pruning as tp
    except Exception:
        raise RuntimeError("torch-pruning is required: pip install torch-pruning")

    import torch

    torch_model.eval()

    prunable = _collect_prunable_convs(
        torch_model,
        exclude_head=exclude_head,
        exclude_name_regex=exclude_name_regex,
        skip_inner_m=skip_inner_m,
        skip_cv1_if_parent_has_m=skip_cv1_if_parent_has_m,
    )
    if not prunable:
        if verbose:
            print("[trim] No prunable layers after filtering. Nothing to prune.")
        return {"layers_pruned": 0, "channels_pruned": 0}

    if protect_last_n > 0 and len(prunable) > int(protect_last_n):
        prunable = prunable[:-int(protect_last_n)]

    strategy = str(strategy).lower().strip()
    if strategy not in ("layerwise", "global"):
        strategy = "layerwise"

    thr = 0.0
    if strategy == "global":
        thr = _global_threshold_from_bn_gammas(prunable, prune_ratio)

    prune_bn_fn = getattr(tp, "prune_batchnorm_out_channels", None)
    if prune_bn_fn is None:
        prune_bn_fn = getattr(tp, "prune_bn_out_channels", None)
    if prune_bn_fn is None:
        raise RuntimeError("torch-pruning: cannot find BN pruning fn (prune_batchnorm_out_channels/prune_bn_out_channels).")

    dg = _build_dependency_graph(tp, torch_model, example_input)

    layers_pruned = 0
    channels_pruned = 0

    for name, m in prunable:
        conv: torch.nn.Conv2d = getattr(m, "conv")
        bn: torch.nn.BatchNorm2d = getattr(m, "bn")

        if conv.out_channels <= min_channels:
            continue

        if conv.groups not in (1, conv.in_channels, conv.out_channels):
            continue

        if strategy == "global":
            idxs = _select_prune_idxs_by_gamma(bn, thr, channel_round, min_channels)

            cap = max_prune_per_layer
            if cap is None:
                cap = prune_ratio
            try:
                cap = float(cap)
            except Exception:
                cap = float(prune_ratio)
            cap = max(0.0, min(0.95, cap))

            if cap > 0.0:
                max_prune = int(np.floor(cap * conv.out_channels))
                max_prune = max(0, min(conv.out_channels - min_channels, max_prune))
                if len(idxs) > max_prune:
                    g = bn.weight.detach().abs()
                    order = torch.argsort(g, descending=False).tolist()
                    idxs = order[:max_prune]
        else:
            idxs = _select_prune_idxs_layerwise(
                bn,
                prune_ratio=float(prune_ratio),
                channel_round=channel_round,
                min_channels=min_channels,
                max_prune_ratio=max_prune_per_layer,
                importance_mode=importance_mode,
                conv=conv,
            )

        if not idxs:
            continue
        if len(idxs) > conv.out_channels - min_channels:
            idxs = idxs[: max(0, conv.out_channels - min_channels)]
        if not idxs:
            continue

        if verbose:
            print(f"[trim] {name}: prune {len(idxs)}/{conv.out_channels} out-ch")

        try:
            if hasattr(dg, "get_pruning_group"):
                group = dg.get_pruning_group(bn, prune_bn_fn, idxs=idxs)
                ok = True
                if hasattr(dg, "check_pruning_group"):
                    ok = bool(dg.check_pruning_group(group))
                if not ok:
                    continue
                group.prune()
            else:
                plan = dg.get_pruning_plan(bn, prune_bn_fn, idxs=idxs)
                if plan is None:
                    continue
                plan.exec()

            _forward_check(torch_model, example_input)

            layers_pruned += 1
            channels_pruned += len(idxs)

            dg = _build_dependency_graph(tp, torch_model, example_input)

        except Exception as e:
            raise RuntimeError(f"Pruning failed at {name}: {e}") from e

    return {"layers_pruned": layers_pruned, "channels_pruned": channels_pruned}
