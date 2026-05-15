from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy

import torch
import torch.nn as nn

try:
    from ultralytics.nn.modules.head import Detect
    from ultralytics.nn.modules.conv import Conv, DWConv
except Exception:
    from ultralytics.nn.modules import Detect, Conv, DWConv  # type: ignore

# reuse helpers from pruning_adapters.py
from .pruning_adapters import (
    _copy_conv_bn_,
    _channel_importance_from_bn,
    _target_keep,
    _topk_indices,
)


def _copy_plain_conv2d_(
    dst: nn.Conv2d,
    src: nn.Conv2d,
    *,
    out_idx: Sequence[int] | None = None,
    in_idx: Sequence[int] | None = None,
) -> None:
    with torch.no_grad():
        w = src.weight.data
        if out_idx is not None:
            idx = torch.as_tensor(list(out_idx), device=w.device, dtype=torch.long)
            w = w.index_select(0, idx)
        if in_idx is not None:
            idx = torch.as_tensor(list(in_idx), device=w.device, dtype=torch.long)
            w = w.index_select(1, idx)
        dst.weight.data.copy_(w.contiguous())

        if src.bias is not None and dst.bias is not None:
            b = src.bias.data
            if out_idx is not None:
                idx = torch.as_tensor(list(out_idx), device=b.device, dtype=torch.long)
                b = b.index_select(0, idx)
            dst.bias.data.copy_(b.contiguous())


def _make_box_branch(in_ch: int, hidden: int, reg_max: int) -> nn.Sequential:
    # same structure as Ultralytics Detect.cv2[i]:
    # Conv(x, c2, 3) -> Conv(c2, c2, 3) -> Conv2d(c2, 4*reg_max, 1)
    return nn.Sequential(
        Conv(in_ch, hidden, 3),
        Conv(hidden, hidden, 3),
        nn.Conv2d(hidden, 4 * reg_max, 1),
    )

def _cls_branch_layout(old_branch: nn.Sequential) -> str:
    """
    Detect actual cls-branch layout from modules, not from head.legacy flag.

    Returns:
        - "legacy" for Conv -> Conv -> Conv2d
        - "dw"     for Sequential(DWConv, Conv) -> Sequential(DWConv, Conv) -> Conv2d
    """
    first = old_branch[0]

    if hasattr(first, "conv"):
        return "legacy"

    if isinstance(first, nn.Sequential):
        return "dw"

    raise TypeError(f"Unsupported cls branch layout: first module type = {type(first).__name__}")

def _make_cls_branch(in_ch: int, hidden: int, nc: int, layout: str) -> nn.Sequential:
    if layout == "legacy":
        return nn.Sequential(
            Conv(in_ch, hidden, 3),
            Conv(hidden, hidden, 3),
            nn.Conv2d(hidden, nc, 1),
        )

    if layout == "dw":
        return nn.Sequential(
            nn.Sequential(DWConv(in_ch, in_ch, 3), Conv(in_ch, hidden, 1)),
            nn.Sequential(DWConv(hidden, hidden, 3), Conv(hidden, hidden, 1)),
            nn.Conv2d(hidden, nc, 1),
        )

    raise ValueError(f"Unsupported cls branch layout: {layout}")


def _pick_box_keep(old_branch: nn.Sequential, keep: int) -> List[int]:
    # use first two BN gammas together
    imp = _channel_importance_from_bn(old_branch[0]) + _channel_importance_from_bn(old_branch[1])
    return _topk_indices(imp, keep)


def _pick_cls_keep(old_branch: nn.Sequential, keep: int) -> List[int]:
    layout = _cls_branch_layout(old_branch)

    if layout == "legacy":
        imp = _channel_importance_from_bn(old_branch[0]) + _channel_importance_from_bn(old_branch[1])
        return _topk_indices(imp, keep)

    if layout == "dw":
        imp = _channel_importance_from_bn(old_branch[0][1]) + _channel_importance_from_bn(old_branch[1][1])
        return _topk_indices(imp, keep)

    raise ValueError(f"Unsupported cls branch layout: {layout}")


def _shrink_box_branch(
    old_branch: nn.Sequential,
    *,
    reg_max: int,
    prune_ratio: float,
    round_to: int,
    min_channels: int,
    target_hidden: Optional[int] = None,
) -> Tuple[nn.Sequential, Dict[str, int]]:
    in_ch = int(old_branch[0].conv.in_channels)
    old_hidden = int(old_branch[0].conv.out_channels)

    keep = (
        max(int(min_channels), min(old_hidden, int(target_hidden)))
        if target_hidden is not None
        else _target_keep(old_hidden, prune_ratio, round_to=round_to, min_channels=min_channels)
    )
    if keep >= old_hidden:
        return copy.deepcopy(old_branch), {"hidden_old": old_hidden, "hidden_new": old_hidden, "shrunk": 0}

    keep_idx = _pick_box_keep(old_branch, keep)

    new_branch = _make_box_branch(in_ch, keep, reg_max)
    new_branch.to(device=old_branch[0].conv.weight.device, dtype=old_branch[0].conv.weight.dtype)
    new_branch.train(old_branch.training)

    _copy_conv_bn_(new_branch[0], old_branch[0], out_idx=keep_idx)
    _copy_conv_bn_(new_branch[1], old_branch[1], out_idx=keep_idx, in_idx=keep_idx)
    _copy_plain_conv2d_(new_branch[2], old_branch[2], in_idx=keep_idx)

    return new_branch, {"hidden_old": old_hidden, "hidden_new": keep, "shrunk": old_hidden - keep}


def _shrink_cls_branch(
    old_branch: nn.Sequential,
    *,
    nc: int,
    prune_ratio: float,
    round_to: int,
    min_channels: int,
    target_hidden: Optional[int] = None,
) -> Tuple[nn.Sequential, Dict[str, int]]:
    layout = _cls_branch_layout(old_branch)

    if layout == "legacy":
        in_ch = int(old_branch[0].conv.in_channels)
        old_hidden = int(old_branch[0].conv.out_channels)
        dev = old_branch[0].conv.weight.device
        dt = old_branch[0].conv.weight.dtype
    elif layout == "dw":
        # non-legacy:
        # [0][0] = DWConv(in_ch, in_ch, 3)
        # [0][1] = Conv(in_ch, hidden, 1)
        # [1][0] = DWConv(hidden, hidden, 3)
        # [1][1] = Conv(hidden, hidden, 1)
        in_ch = int(old_branch[0][0].conv.in_channels)
        old_hidden = int(old_branch[0][1].conv.out_channels)
        dev = old_branch[0][1].conv.weight.device
        dt = old_branch[0][1].conv.weight.dtype
    else:
        raise ValueError(f"Unsupported cls branch layout: {layout}")

    keep = (
        max(int(min_channels), min(old_hidden, int(target_hidden)))
        if target_hidden is not None
        else _target_keep(old_hidden, prune_ratio, round_to=round_to, min_channels=min_channels)
    )
    if keep >= old_hidden:
        return copy.deepcopy(old_branch), {"hidden_old": old_hidden, "hidden_new": old_hidden, "shrunk": 0}

    keep_idx = _pick_cls_keep(old_branch, keep)

    new_branch = _make_cls_branch(in_ch, keep, nc, layout=layout)
    new_branch.to(device=dev, dtype=dt)
    new_branch.train(old_branch.training)

    if layout == "legacy":
        _copy_conv_bn_(new_branch[0], old_branch[0], out_idx=keep_idx)
        _copy_conv_bn_(new_branch[1], old_branch[1], out_idx=keep_idx, in_idx=keep_idx)
        _copy_plain_conv2d_(new_branch[2], old_branch[2], in_idx=keep_idx)

    elif layout == "dw":
        # first DWConv keeps input width unchanged: x -> x
        new_branch[0][0].load_state_dict(old_branch[0][0].state_dict())

        # first pointwise conv: in_ch -> hidden
        _copy_conv_bn_(new_branch[0][1], old_branch[0][1], out_idx=keep_idx)

        # second DWConv is depthwise on hidden channels
        _copy_conv_bn_(new_branch[1][0], old_branch[1][0], out_idx=keep_idx)

        # second pointwise conv: hidden -> hidden
        _copy_conv_bn_(new_branch[1][1], old_branch[1][1], out_idx=keep_idx, in_idx=keep_idx)

        # final classifier conv: hidden -> nc
        _copy_plain_conv2d_(new_branch[2], old_branch[2], in_idx=keep_idx)

    return new_branch, {"hidden_old": old_hidden, "hidden_new": keep, "shrunk": old_hidden - keep}


def _shrink_detect_head_impl(
    head: Detect,
    *,
    prune_ratio: float,
    round_to: int,
    min_channels: int,
    verbose: bool,
) -> Tuple[Detect, Dict[str, Any]]:
    new_head = copy.deepcopy(head)
    new_head.train(head.training)

    items: List[Dict[str, Any]] = []
    changed = 0

    legacy = bool(getattr(head, "legacy", False))

    # one2many head
    for i in range(head.nl):
        box_new, box_info = _shrink_box_branch(
            head.cv2[i],
            reg_max=int(head.reg_max),
            prune_ratio=prune_ratio,
            round_to=round_to,
            min_channels=min_channels,
        )
        cls_new, cls_info = _shrink_cls_branch(
            head.cv3[i],
            nc=int(head.nc),
            prune_ratio=prune_ratio,
            round_to=round_to,
            min_channels=min_channels,
        )

        new_head.cv2[i] = box_new
        new_head.cv3[i] = cls_new

        if box_info["shrunk"] > 0:
            changed += 1
            items.append({"branch": f"cv2[{i}]", **box_info})
            if verbose:
                print(f"[detect] cv2[{i}]: hidden {box_info['hidden_old']} -> {box_info['hidden_new']}")

        if cls_info["shrunk"] > 0:
            changed += 1
            items.append({"branch": f"cv3[{i}]", **cls_info})
            if verbose:
                print(f"[detect] cv3[{i}]: hidden {cls_info['hidden_old']} -> {cls_info['hidden_new']}")

    # end2end one2one branches, if present
    if hasattr(head, "one2one_cv2") and hasattr(head, "one2one_cv3"):
        for i in range(head.nl):
            box_new, _ = _shrink_box_branch(
                head.one2one_cv2[i],
                reg_max=int(head.reg_max),
                prune_ratio=prune_ratio,
                round_to=round_to,
                min_channels=min_channels,
            )
            cls_new, _ = _shrink_cls_branch(
                head.one2one_cv3[i],
                nc=int(head.nc),
                prune_ratio=prune_ratio,
                round_to=round_to,
                min_channels=min_channels,
            )
            new_head.one2one_cv2[i] = box_new
            new_head.one2one_cv3[i] = cls_new

    return new_head, {"branches_shrunk": changed, "items": items}


def shrink_detect_heads(
    module: nn.Module,
    *,
    prune_ratio: float,
    round_to: int = 8,
    min_channels: int = 8,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Recursively find Ultralytics Detect heads and shrink their internal hidden channels
    while preserving final output widths:
      - box head last conv out = 4 * reg_max
      - cls head last conv out = nc
    """
    items: List[Dict[str, Any]] = []
    replaced = 0

    for name, child in list(module.named_children()):
        if isinstance(child, Detect):
            new_child, stats = _shrink_detect_head_impl(
                child,
                prune_ratio=prune_ratio,
                round_to=round_to,
                min_channels=min_channels,
                verbose=verbose,
            )
            setattr(module, name, new_child)
            replaced += 1
            items.append({"name": name, **stats})
            if verbose:
                print(f"[detect] Detect head at {name}: {stats['branches_shrunk']} branches shrunk")
        else:
            sub = shrink_detect_heads(
                child,
                prune_ratio=prune_ratio,
                round_to=round_to,
                min_channels=min_channels,
                verbose=verbose,
            )
            replaced += int(sub.get("heads_shrunk", 0))
            items.extend(list(sub.get("items", [])))

    return {"heads_shrunk": replaced, "items": items}


def _box_hidden(branch: nn.Sequential) -> int:
    return int(branch[0].conv.out_channels)


def _cls_hidden(branch: nn.Sequential) -> int:
    layout = _cls_branch_layout(branch)
    if layout == "legacy":
        return int(branch[0].conv.out_channels)
    if layout == "dw":
        return int(branch[0][1].conv.out_channels)
    raise ValueError(f"Unsupported cls branch layout: {layout}")


def collect_detect_head_hidden_channels(module: nn.Module) -> Dict[str, int]:
    """Collect hidden widths for every Detect box/classification branch."""
    widths: Dict[str, int] = {}
    for name, child in module.named_modules():
        if not isinstance(child, Detect):
            continue
        prefix = name if name else "<root>"
        for i in range(child.nl):
            widths[f"{prefix}.cv2[{i}]"] = _box_hidden(child.cv2[i])
            widths[f"{prefix}.cv3[{i}]"] = _cls_hidden(child.cv3[i])
        if hasattr(child, "one2one_cv2") and hasattr(child, "one2one_cv3"):
            for i in range(child.nl):
                widths[f"{prefix}.one2one_cv2[{i}]"] = _box_hidden(child.one2one_cv2[i])
                widths[f"{prefix}.one2one_cv3[{i}]"] = _cls_hidden(child.one2one_cv3[i])
    return widths


def _shrink_detect_head_to_targets_impl(
    head: Detect,
    *,
    prefix: str,
    target_hidden_channels: Dict[str, int],
    round_to: int,
    min_channels: int,
    verbose: bool,
) -> Tuple[Detect, Dict[str, Any]]:
    new_head = copy.deepcopy(head)
    new_head.train(head.training)

    items: List[Dict[str, Any]] = []
    changed = 0

    for i in range(head.nl):
        box_key = f"{prefix}.cv2[{i}]"
        cls_key = f"{prefix}.cv3[{i}]"
        box_new, box_info = _shrink_box_branch(
            head.cv2[i],
            reg_max=int(head.reg_max),
            prune_ratio=0.0,
            round_to=round_to,
            min_channels=min_channels,
            target_hidden=target_hidden_channels.get(box_key),
        )
        cls_new, cls_info = _shrink_cls_branch(
            head.cv3[i],
            nc=int(head.nc),
            prune_ratio=0.0,
            round_to=round_to,
            min_channels=min_channels,
            target_hidden=target_hidden_channels.get(cls_key),
        )
        new_head.cv2[i] = box_new
        new_head.cv3[i] = cls_new

        if box_info["shrunk"] > 0:
            changed += 1
            items.append({"branch": box_key, **box_info})
            if verbose:
                print(f"[detect-target] {box_key}: hidden {box_info['hidden_old']} -> {box_info['hidden_new']}")
        if cls_info["shrunk"] > 0:
            changed += 1
            items.append({"branch": cls_key, **cls_info})
            if verbose:
                print(f"[detect-target] {cls_key}: hidden {cls_info['hidden_old']} -> {cls_info['hidden_new']}")

    if hasattr(head, "one2one_cv2") and hasattr(head, "one2one_cv3"):
        for i in range(head.nl):
            box_key = f"{prefix}.one2one_cv2[{i}]"
            cls_key = f"{prefix}.one2one_cv3[{i}]"
            box_new, box_info = _shrink_box_branch(
                head.one2one_cv2[i],
                reg_max=int(head.reg_max),
                prune_ratio=0.0,
                round_to=round_to,
                min_channels=min_channels,
                target_hidden=target_hidden_channels.get(box_key),
            )
            cls_new, cls_info = _shrink_cls_branch(
                head.one2one_cv3[i],
                nc=int(head.nc),
                prune_ratio=0.0,
                round_to=round_to,
                min_channels=min_channels,
                target_hidden=target_hidden_channels.get(cls_key),
            )
            new_head.one2one_cv2[i] = box_new
            new_head.one2one_cv3[i] = cls_new
            if box_info["shrunk"] > 0:
                changed += 1
                items.append({"branch": box_key, **box_info})
            if cls_info["shrunk"] > 0:
                changed += 1
                items.append({"branch": cls_key, **cls_info})

    return new_head, {"branches_shrunk": changed, "items": items}


def shrink_detect_heads_to_targets(
    module: nn.Module,
    *,
    target_hidden_channels: Dict[str, int],
    round_to: int = 8,
    min_channels: int = 8,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Shrink Detect hidden branches to exact target widths where possible."""
    items: List[Dict[str, Any]] = []
    replaced = 0

    def _recurse(parent: nn.Module, prefix: str = "") -> None:
        nonlocal replaced
        for name, child in list(parent.named_children()):
            qn = f"{prefix}.{name}" if prefix else name
            if isinstance(child, Detect):
                new_child, stats = _shrink_detect_head_to_targets_impl(
                    child,
                    prefix=qn,
                    target_hidden_channels=target_hidden_channels,
                    round_to=round_to,
                    min_channels=min_channels,
                    verbose=verbose,
                )
                setattr(parent, name, new_child)
                replaced += 1
                items.append({"name": qn, **stats})
                if verbose:
                    print(f"[detect-target] Detect head at {qn}: {stats['branches_shrunk']} branches shrunk")
            else:
                _recurse(child, qn)

    _recurse(module)
    return {"heads_shrunk": replaced, "items": items, "targeted": True}
