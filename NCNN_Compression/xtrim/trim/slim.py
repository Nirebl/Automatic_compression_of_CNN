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


def _is_xtrim_prunable_c2f_like(m: Any) -> bool:
    return bool(getattr(m, "_xtrim_prunable_c2f_like", False))


def _looks_like_c2f_like_parent(m: Any) -> bool:
    return (
        hasattr(m, "m")
        and hasattr(m, "cv1")
        and hasattr(m, "cv2")
        and hasattr(m, "c")
    )


def _is_original_c2f_like_parent(m: Any) -> bool:
    return _looks_like_c2f_like_parent(m) and not _is_xtrim_prunable_c2f_like(m)

PSA_FAMILY_CLASSNAMES = {"Attention", "PSABlock", "PSA", "C2PSA", "C2fPSA"}
DETECT_FAMILY_CLASSNAMES = {"Detect"}


def _has_ancestor_class(root: Any, name: str, class_names: set[str]) -> Optional[str]:
    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        qn = ".".join(parts[:i])
        try:
            m = _get_module_by_qualname(root, qn)
        except Exception:
            continue
        if m.__class__.__name__ in class_names:
            return m.__class__.__name__
    return None

def _skip_reason(
    torch_model: Any,
    name: str,
    *,
    skip_inner_m: bool,
    skip_cv1_if_parent_has_m: bool,
    include_inner_m_regex: Optional[str] = None,
) -> Optional[str]:
    detect_cls = _has_ancestor_class(torch_model, name, DETECT_FAMILY_CLASSNAMES)
    if detect_cls is not None:
        return f"handled_by_detect_head_pruner:{detect_cls}"

    include_inner_re = re.compile(include_inner_m_regex) if include_inner_m_regex else None

    psa_cls = _has_ancestor_class(torch_model, name, PSA_FAMILY_CLASSNAMES)
    if psa_cls is not None:
        return f"handled_by_composite_psa_pruner:{psa_cls}"

    is_inner_m = re.search(r"\.m\.\d+\.", name) is not None
    if skip_inner_m and is_inner_m:
        if include_inner_re is None or not include_inner_re.search(name):
            return "skip_inner_m"

    # NEW:
    # If ANY ancestor belongs to an adapted carrier that contains PSA-family blocks,
    # do not prune this conv with the generic channel-pruning path.
    parts = name.split(".")
    for i in range(len(parts) - 1, 0, -1):
        qn = ".".join(parts[:i])
        try:
            anc = _get_module_by_qualname(torch_model, qn)
        except Exception:
            continue
        if bool(getattr(anc, "_xtrim_contains_psa", False)):
            if include_inner_re is None or not include_inner_re.search(name):
                return "handled_by_composite_psa_carrier_subtree"

    parent_name = ".".join(name.split(".")[:-1]) if "." in name else ""
    parent = None
    if parent_name:
        try:
            parent = _get_module_by_qualname(torch_model, parent_name)
        except Exception:
            parent = None

    if name.endswith(".cv2"):
        try:
            if parent is not None and hasattr(parent, "m"):
                return "skip_parent_cv2_aggregator"
        except Exception:
            pass

    if name.endswith(".cv1"):
        try:
            if parent is not None and _is_original_c2f_like_parent(parent):
                if include_inner_re is None or not include_inner_re.search(name):
                    return "unsafe_original_c2f_like_cv1"

            if parent is not None and skip_cv1_if_parent_has_m and hasattr(parent, "m"):
                if include_inner_re is None or not include_inner_re.search(name):
                    return "skip_cv1_if_parent_has_m"
        except Exception:
            pass

    return None


def _should_skip_prune(
    torch_model: Any,
    name: str,
    *,
    skip_inner_m: bool,
    skip_cv1_if_parent_has_m: bool,
    include_inner_m_regex: Optional[str] = None,
) -> bool:
    return _skip_reason(
        torch_model,
        name,
        skip_inner_m=skip_inner_m,
        skip_cv1_if_parent_has_m=skip_cv1_if_parent_has_m,
        include_inner_m_regex=include_inner_m_regex,
    ) is not None


def _collect_prunable_convs(
    torch_model: Any,
    *,
    exclude_head: bool,
    exclude_name_regex: Optional[str],
    skip_inner_m: bool = True,
    skip_cv1_if_parent_has_m: bool = True,
    include_inner_m_regex: Optional[str] = None,
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
            include_inner_m_regex=include_inner_m_regex,
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
        return (x,)

    if isinstance(example_input, (list, tuple)):
        prepared = []
        for item in example_input:
            if torch.is_tensor(item):
                t = item.detach()
                t.requires_grad_(True)
                prepared.append(t)
            else:
                prepared.append(item)
        return tuple(prepared)

    return (example_input,)


import torch


def _build_dependency_graph(tp, torch_model, example_input):
    import torch

    dg = tp.DependencyGraph()
    ex_inputs = _make_example_inputs_for_tp(example_input)

    def _out_transform(out):
        ts = _as_tensor_list(out)
        if not ts:
            return out

        # ВАЖНО:
        # dependency graph должен зависеть от ВСЕХ выходных веток модели,
        # а не только от самого большого тензора.
        acc = None
        for t in ts:
            if not torch.is_floating_point(t):
                t = t.float()
            s = t.reshape(-1).sum()
            acc = s if acc is None else acc + s
        return acc

    with torch.enable_grad():
        try:
            dg.build_dependency(
                torch_model,
                example_inputs=ex_inputs,
                output_transform=_out_transform,
            )
        except TypeError:
            dg.build_dependency(torch_model, example_inputs=ex_inputs)

    return dg


def _pick_pruning_target(tp, module):
    import torch

    conv: torch.nn.Conv2d = getattr(module, "conv")

    if conv.groups == 1:
        return conv, tp.prune_conv_out_channels

    prune_depthwise_fn = getattr(tp, "prune_depthwise_conv_out_channels", None)
    if conv.groups == conv.in_channels == conv.out_channels and prune_depthwise_fn is not None:
        return conv, prune_depthwise_fn

    return None, None

def _forward_check(torch_model, example_input):
    import torch

    if isinstance(example_input, (list, tuple)):
        args = []
        for item in example_input:
            if torch.is_tensor(item):
                args.append(item.detach())
            else:
                args.append(item)
        args = tuple(args)
    else:
        args = (example_input.detach(),) if torch.is_tensor(example_input) else (example_input,)

    with torch.no_grad():
        _ = torch_model(*args)


def _validate_c2f_like_invariants(torch_model: Any) -> None:
    """
    У оригинальных C2f/C3k2-like блоков инвариант жёсткий:
        cv1.out_channels == 2 * self.c
    Если нарушить его pruning'ом, следующий forward упадёт на split/chunk.

    Для адаптированного C2fPrunable этот инвариант не обязателен,
    поэтому проверяем только неадаптированные блоки.
    """
    import torch

    for name, m in torch_model.named_modules():
        if not _is_original_c2f_like_parent(m):
            continue

        conv = getattr(getattr(m, "cv1", None), "conv", None)
        if not isinstance(conv, torch.nn.Conv2d):
            continue

        c = int(getattr(m, "c"))
        out_ch = int(conv.out_channels)
        expected = 2 * c
        if out_ch != expected:
            raise RuntimeError(
                f"{name}: invalid C2f-like invariant: cv1.out_channels={out_ch}, expected={expected} (=2*self.c)"
            )

def _validate_psa_like_invariants(torch_model: Any) -> None:
    import torch

    for name, m in torch_model.named_modules():
        cls = m.__class__.__name__
        if cls not in {"PSA", "C2PSA", "C2fPSA"}:
            continue

        cv1 = getattr(getattr(m, "cv1", None), "conv", None)
        cv2 = getattr(getattr(m, "cv2", None), "conv", None)
        c = int(getattr(m, "c", 0))
        if isinstance(cv1, torch.nn.Conv2d) and int(cv1.out_channels) != 2 * c:
            raise RuntimeError(
                f"{name}: invalid {cls} invariant: cv1.out_channels={int(cv1.out_channels)}, expected={2*c}"
            )
        if isinstance(cv2, torch.nn.Conv2d) and int(cv2.in_channels) != 2 * c and cls in {"PSA", "C2PSA"}:
            raise RuntimeError(
                f"{name}: invalid {cls} invariant: cv2.in_channels={int(cv2.in_channels)}, expected={2*c}"
            )


def _module_param_count(m: Any) -> int:
    try:
        return sum(p.numel() for p in m.parameters(recurse=False))
    except Exception:
        return 0


def _conv_bn_param_count(m: Any) -> int:
    total = 0
    try:
        conv = getattr(m, "conv", None)
        bn = getattr(m, "bn", None)
        if conv is not None:
            total += sum(p.numel() for p in conv.parameters(recurse=False))
        if bn is not None:
            total += sum(p.numel() for p in bn.parameters(recurse=False))
    except Exception:
        pass
    return int(total)


def _coverage_report(
    torch_model: Any,
    *,
    exclude_head: bool,
    exclude_name_regex: Optional[str],
    skip_inner_m: bool,
    skip_cv1_if_parent_has_m: bool,
    include_inner_m_regex: Optional[str],
    protect_last_n: int,
    topk: int = 20,
):
    exclude_re = re.compile(exclude_name_regex) if exclude_name_regex else None
    include_inner_re = re.compile(include_inner_m_regex) if include_inner_m_regex else None
    head_mod = _find_head_module(torch_model) if exclude_head else None

    head_ids = set()
    if head_mod is not None:
        for _, hm in head_mod.named_modules():
            head_ids.add(id(hm))

    all_conv = []
    prunable = []
    skipped = []

    for name, m in torch_model.named_modules():
        if not _is_ultralytics_conv(m):
            continue

        size = _conv_bn_param_count(m)
        all_conv.append((name, size))

        reason = None
        if exclude_re and exclude_re.search(name):
            reason = "exclude_name_regex"
        elif head_ids and id(m) in head_ids:
            reason = "exclude_head"
        else:
            reason = _skip_reason(
                torch_model,
                name,
                skip_inner_m=skip_inner_m,
                skip_cv1_if_parent_has_m=skip_cv1_if_parent_has_m,
                include_inner_m_regex=include_inner_m_regex,
            )

        if reason is None:
            prunable.append((name, size))
        else:
            skipped.append((name, size, reason))

    prunable_before_tail = list(prunable)
    if protect_last_n > 0 and len(prunable) > int(protect_last_n):
        protected_tail = prunable[-int(protect_last_n):]
        prunable = prunable[:-int(protect_last_n)]
        skipped.extend([(n, s, "protect_last_n") for (n, s) in protected_tail])

    prunable_sorted = sorted(prunable, key=lambda x: x[1], reverse=True)
    skipped_sorted = sorted(skipped, key=lambda x: x[1], reverse=True)

    return {
        "total_ultra_convs": len(all_conv),
        "prunable_before_tail": len(prunable_before_tail),
        "prunable_after_tail": len(prunable),
        "total_prunable_params": int(sum(s for _, s in prunable)),
        "top_prunable": prunable_sorted[:topk],
        "top_skipped": skipped_sorted[:topk],
    }

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
    include_inner_m_regex: Optional[str] = None,
    importance_mode: str = "bn_gamma",
) -> dict[str, int] | dict[str, int | list[str]]:
    if not (0.0 < prune_ratio < 0.95):
        raise ValueError("prune_ratio must be in (0, 0.95)")

    try:
        import torch_pruning as tp
    except Exception:
        raise RuntimeError("torch-pruning is required: pip install torch-pruning")

    import torch

    torch_model.eval()

    if verbose:
        rep = _coverage_report(
            torch_model,
            exclude_head=exclude_head,
            exclude_name_regex=exclude_name_regex,
            skip_inner_m=skip_inner_m,
            skip_cv1_if_parent_has_m=skip_cv1_if_parent_has_m,
            include_inner_m_regex=include_inner_m_regex,
            protect_last_n=protect_last_n,
            topk=20,
        )
        print(
            f"[trim] coverage: total_ultra_convs={rep['total_ultra_convs']}, "
            f"prunable_before_tail={rep['prunable_before_tail']}, "
            f"prunable_after_tail={rep['prunable_after_tail']}, "
            f"prunable_params={rep['total_prunable_params']:,}"
        )
        if rep["top_prunable"]:
            print("[trim] top prunable modules:")
            for i, (n, s) in enumerate(rep["top_prunable"], 1):
                print(f"  [{i:02d}] {n:<40} {s:,}")
        if rep["top_skipped"]:
            print("[trim] top skipped modules:")
            for i, (n, s, r) in enumerate(rep["top_skipped"], 1):
                print(f"  [{i:02d}] {n:<40} {s:,}  reason={r}")

    # Снимок на старте нужен только для:
    # 1) списка имён
    # 2) глобального порога по BN-gamma
    initial_prunable = _collect_prunable_convs(
        torch_model,
        exclude_head=exclude_head,
        exclude_name_regex=exclude_name_regex,
        skip_inner_m=skip_inner_m,
        skip_cv1_if_parent_has_m=skip_cv1_if_parent_has_m,
        include_inner_m_regex=include_inner_m_regex,
    )
    if not initial_prunable:
        if verbose:
            print("[trim] No prunable layers after filtering. Nothing to prune.")
        return {
            "layers_pruned": 0,
            "channels_pruned": 0,
            "prunable_count": 0,
        }

    prunable_names = [name for name, _ in initial_prunable]

    if protect_last_n > 0 and len(prunable_names) > int(protect_last_n):
        if verbose:
            print(f"[trim] protect_last_n={protect_last_n}: keeping last {protect_last_n} prunable layers untouched")
        prunable_names = prunable_names[:-int(protect_last_n)]

    strategy = str(strategy).lower().strip()
    if strategy not in ("layerwise", "global"):
        strategy = "layerwise"

    thr = 0.0
    if strategy == "global":
        thr = _global_threshold_from_bn_gammas(initial_prunable, prune_ratio)

    dg = _build_dependency_graph(tp, torch_model, example_input)

    layers_pruned = 0
    channels_pruned = 0
    pruned_layer_names: List[str] = []

    for name in prunable_names:
        # Всегда берём АКТУАЛЬНЫЙ модуль из текущей модели, а не старую ссылку
        try:
            m = _get_module_by_qualname(torch_model, name)
        except Exception:
            if verbose:
                print(f"[trim] skip {name}: module no longer exists")
            continue

        if not _is_ultralytics_conv(m):
            if verbose:
                print(f"[trim] skip {name}: no longer an ultralytics conv")
            continue

        if _should_skip_prune(
            torch_model,
            name,
            skip_inner_m=skip_inner_m,
            skip_cv1_if_parent_has_m=skip_cv1_if_parent_has_m,
            include_inner_m_regex=include_inner_m_regex,
        ):
            if verbose:
                print(f"[trim] skip {name}: filtered by current model state")
            continue

        conv: torch.nn.Conv2d = getattr(m, "conv")
        bn: torch.nn.BatchNorm2d = getattr(m, "bn")

        if conv.out_channels <= min_channels:
            continue

        root_module, prune_fn = _pick_pruning_target(tp, m)
        if root_module is None or prune_fn is None:
            if verbose:
                print(f"[trim] skip {name}: unsupported grouped conv (groups={conv.groups})")
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

        # Проверяем наличие root в актуальном dependency graph
        if hasattr(dg, "module2node") and root_module not in dg.module2node:
            if verbose:
                print(f"[trim] {name}: root conv is not in dependency graph, rebuilding DG")
            dg = _build_dependency_graph(tp, torch_model, example_input)

            # После rebuild нужно заново взять актуальный модуль и root
            try:
                m = _get_module_by_qualname(torch_model, name)
            except Exception:
                if verbose:
                    print(f"[trim] skip {name}: module disappeared after rebuild")
                continue

            if not _is_ultralytics_conv(m):
                if verbose:
                    print(f"[trim] skip {name}: no longer an ultralytics conv after rebuild")
                continue

            if _should_skip_prune(
                torch_model,
                name,
                skip_inner_m=skip_inner_m,
                skip_cv1_if_parent_has_m=skip_cv1_if_parent_has_m,
                include_inner_m_regex=include_inner_m_regex,
            ):
                if verbose:
                    print(f"[trim] skip {name}: filtered after rebuild")
                continue

            conv = getattr(m, "conv")
            bn = getattr(m, "bn")

            if conv.out_channels <= min_channels:
                if verbose:
                    print(f"[trim] skip {name}: too few channels after rebuild")
                continue

            root_module, prune_fn = _pick_pruning_target(tp, m)
            if root_module is None or prune_fn is None:
                if verbose:
                    print(f"[trim] skip {name}: unsupported conv after rebuild")
                continue

            # Пересчитываем idxs на АКТУАЛЬНОМ bn/conv
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
                if verbose:
                    print(f"[trim] skip {name}: no prune idxs after rebuild")
                continue
            if len(idxs) > conv.out_channels - min_channels:
                idxs = idxs[: max(0, conv.out_channels - min_channels)]
            if not idxs:
                if verbose:
                    print(f"[trim] skip {name}: no valid prune idxs after rebuild")
                continue

            if hasattr(dg, "module2node") and root_module not in dg.module2node:
                if verbose:
                    print(f"[trim] skip {name}: root conv still not in dependency graph")
                continue

        try:
            if hasattr(dg, "get_pruning_group"):
                group = dg.get_pruning_group(root_module, prune_fn, idxs=idxs)
                ok = True
                if hasattr(dg, "check_pruning_group"):
                    ok = bool(dg.check_pruning_group(group))
                if not ok:
                    if verbose:
                        print(f"[trim] skip {name}: pruning group check failed")
                    continue
                group.prune()
            else:
                plan = dg.get_pruning_plan(root_module, prune_fn, idxs=idxs)
                if plan is None:
                    continue
                plan.exec()

            _forward_check(torch_model, example_input)

            layers_pruned += 1
            channels_pruned += len(idxs)
            pruned_layer_names.append(name)

            dg = _build_dependency_graph(tp, torch_model, example_input)

        except Exception as e:
            raise RuntimeError(f"Pruning failed at {name}: {e}") from e

    return {
        "layers_pruned": layers_pruned,
        "channels_pruned": channels_pruned,
        "prunable_count": len(prunable_names),
        "pruned_layer_names": pruned_layer_names,
    }