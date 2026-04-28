from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple
import copy

import torch
import torch.nn as nn


try:
    from ultralytics.nn.modules.block import (
        C2f,
        Bottleneck,
        C3k2,
        Attention,
        PSA,
        C2PSA,
        C2fPSA,
    )
    from ultralytics.nn.modules.conv import Conv
except Exception:
    from ultralytics.nn.modules import C2f, Bottleneck, Conv  # type: ignore
    C3k2 = None  # type: ignore
    Attention = None  # type: ignore
    PSA = None  # type: ignore
    C2PSA = None  # type: ignore
    C2fPSA = None  # type: ignore


PSA_FAMILY_CLASSNAMES = {"Attention", "PSABlock", "PSA", "C2PSA", "C2fPSA"}


def _class_name(m: Any) -> str:
    return type(m).__name__


def _is_exact_c2f(m: nn.Module) -> bool:
    return type(m) is C2f


def _is_c3k2(m: nn.Module) -> bool:
    if C3k2 is not None and isinstance(m, C3k2):
        return True
    return m.__class__.__name__ == "C3k2"


def _is_c2fpsa(m: nn.Module) -> bool:
    if C2fPSA is not None and isinstance(m, C2fPSA):
        return True
    return m.__class__.__name__ == "C2fPSA"


def _c2f_like_kind(m: nn.Module) -> str:
    if _is_c3k2(m):
        return "C3k2"
    if _is_c2fpsa(m):
        return "C2fPSA"
    if _is_exact_c2f(m):
        return "C2f"
    if isinstance(m, C2f):
        return type(m).__name__
    return type(m).__name__


class C2fPrunable(nn.Module):
    """
    Pruning-friendly replacement for Ultralytics C2f/C3k2-like blocks.

    Вместо:
        cv1(x) -> split((c, c)) -> ...
    делаем две явные ветки:
        cv0(x), cv1(x)
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

        self.src_block_type = "unknown"
        self._xtrim_prunable_c2f_like = True

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


# -----------------------------
# PSA-family prunable classes
# -----------------------------

def _safe_num_heads(dim: int, preferred_heads: Optional[int] = None, min_head_dim: int = 8) -> int:
    """
    Safely choose a valid number of heads for Attention(dim, num_heads=...).

    Правила:
    - num_heads >= 1
    - dim % num_heads == 0
    - стараемся сохранить число голов не больше старого preferred_heads
    - стараемся не делать слишком маленький head_dim
    """
    dim = int(dim)
    if dim <= 0:
        return 1

    if preferred_heads is None:
        preferred_heads = max(1, dim // 64)

    preferred_heads = max(1, min(int(preferred_heads), dim))

    # Сначала ищем делитель, где head_dim не слишком маленький
    for h in range(preferred_heads, 0, -1):
        if dim % h == 0 and (dim // h) >= int(min_head_dim):
            return h

    # Потом любой делитель
    for h in range(preferred_heads, 0, -1):
        if dim % h == 0:
            return h

    return 1


class PSABlockPrunable(nn.Module):
    """
    Pruning-friendly analogue of Ultralytics PSABlock.
    """
    def __init__(
        self,
        c: int,
        attn_ratio: float = 0.5,
        num_heads: int = 1,
        shortcut: bool = True,
    ):
        super().__init__()
        if Attention is None:
            raise RuntimeError("Ultralytics Attention class is unavailable")

        safe_heads = _safe_num_heads(c, preferred_heads=num_heads)
        self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=safe_heads)
        self.ffn = nn.Sequential(
            Conv(c, 2 * c, 1),
            Conv(2 * c, c, 1, act=False),
        )
        self.add = bool(shortcut)
        self._xtrim_prunable_psa_block = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(x) if self.add else self.attn(x)
        x = x + self.ffn(x) if self.add else self.ffn(x)
        return x


class PSAPrunable(nn.Module):
    """
    Pruning-friendly analogue of Ultralytics PSA.
    """
    def __init__(self, c1: int, c2: int, e: float = 0.5):
        super().__init__()
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1, 1)

        safe_heads = _safe_num_heads(self.c)
        self.attn = Attention(self.c, attn_ratio=0.5, num_heads=safe_heads)
        self.ffn = nn.Sequential(
            Conv(self.c, 2 * self.c, 1),
            Conv(2 * self.c, self.c, 1, act=False),
        )
        self._xtrim_prunable_psa_like = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = b + self.attn(b)
        b = b + self.ffn(b)
        return self.cv2(torch.cat((a, b), 1))


class C2PSAPrunable(nn.Module):
    """
    Pruning-friendly analogue of Ultralytics C2PSA.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        assert c1 == c2
        self.c = int(c1 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv(2 * self.c, c1, 1, 1)

        safe_heads = _safe_num_heads(self.c)
        self.m = nn.Sequential(
            *(PSABlockPrunable(self.c, attn_ratio=0.5, num_heads=safe_heads, shortcut=True) for _ in range(n))
        )
        self._xtrim_prunable_psa_like = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        a, b = self.cv1(x).split((self.c, self.c), dim=1)
        b = self.m(b)
        return self.cv2(torch.cat((a, b), 1))


class C2fPSAPrunable(nn.Module):
    """
    Pruning-friendly analogue of Ultralytics C2fPSA.
    """
    def __init__(self, c1: int, c2: int, n: int = 1, e: float = 0.5):
        super().__init__()
        self.c = int(c2 * e)
        self.cv1 = Conv(c1, 2 * self.c, 1, 1)
        self.cv2 = Conv((2 + n) * self.c, c2, 1, 1)

        safe_heads = _safe_num_heads(self.c)
        self.m = nn.ModuleList(
            PSABlockPrunable(self.c, attn_ratio=0.5, num_heads=safe_heads, shortcut=True) for _ in range(n)
        )
        self._xtrim_prunable_psa_like = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = list(self.cv1(x).split((self.c, self.c), dim=1))
        for m in self.m:
            y.append(m(y[-1]))
        return self.cv2(torch.cat(y, 1))


# -----------------------------
# Copy helpers
# -----------------------------

def _copy_conv_bn_(
    dst: Any,
    src: Any,
    out_slice: slice | None = None,
    in_slice: slice | None = None,
    out_idx: Optional[Sequence[int]] = None,
    in_idx: Optional[Sequence[int]] = None,
) -> None:
    """
    Copy Ultralytics Conv() + BN with optional output/input slicing.
    Supports:
      - standard conv
      - depthwise conv (groups == in_channels == out_channels, weight shape [C, 1, k, k])

    Important:
      - for depthwise conv we NEVER index input-channel dim, because it is always size 1
      - grouped non-depthwise conv is not safely supported by arbitrary in_idx pruning here
    """
    with torch.no_grad():
        src_conv = src.conv
        dst_conv = dst.conv

        if not isinstance(src_conv, torch.nn.Conv2d) or not isinstance(dst_conv, torch.nn.Conv2d):
            raise TypeError("Expected src.conv and dst.conv to be torch.nn.Conv2d")

        w = src_conv.weight.data

        is_depthwise = (
            src_conv.groups == src_conv.in_channels == src_conv.out_channels
            and w.ndim == 4
            and w.shape[1] == 1
        )
        is_grouped = src_conv.groups > 1 and not is_depthwise

        # output channel selection is always valid
        if out_slice is not None:
            w = w[out_slice].clone()
        if out_idx is not None:
            idx = torch.as_tensor(list(out_idx), device=w.device, dtype=torch.long)
            w = w.index_select(0, idx).clone()

        # input channel selection:
        # - standard conv: allowed
        # - depthwise conv: NEVER index dim=1 (it is 1)
        # - grouped non-depthwise: reject for now, unless no input pruning requested
        if in_slice is not None or in_idx is not None:
            if is_depthwise:
                # For depthwise conv, selecting output channels already selects the corresponding channel groups.
                # Input dim stays 1 and must not be indexed.
                pass
            elif is_grouped:
                raise NotImplementedError(
                    "Grouped non-depthwise Conv2d with arbitrary input-channel remapping is not supported "
                    "by _copy_conv_bn_. Add a dedicated grouped-conv copy path."
                )
            else:
                if in_slice is not None:
                    w = w[:, in_slice].clone()
                if in_idx is not None:
                    idx = torch.as_tensor(list(in_idx), device=w.device, dtype=torch.long)
                    w = w.index_select(1, idx).clone()

        if tuple(dst_conv.weight.shape) != tuple(w.shape):
            raise RuntimeError(
                f"_copy_conv_bn_: shape mismatch for conv weight: dst={tuple(dst_conv.weight.shape)} vs src={tuple(w.shape)}"
            )

        dst_conv.weight.data.copy_(w.contiguous())

        if getattr(src_conv, "bias", None) is not None and getattr(dst_conv, "bias", None) is not None:
            b = src_conv.bias.data
            if out_slice is not None:
                b = b[out_slice].clone()
            if out_idx is not None:
                idx = torch.as_tensor(list(out_idx), device=b.device, dtype=torch.long)
                b = b.index_select(0, idx).clone()
            if tuple(dst_conv.bias.shape) != tuple(b.shape):
                raise RuntimeError(
                    f"_copy_conv_bn_: shape mismatch for conv bias: dst={tuple(dst_conv.bias.shape)} vs src={tuple(b.shape)}"
                )
            dst_conv.bias.data.copy_(b.contiguous())

        for attr in ("weight", "bias", "running_mean", "running_var"):
            s = getattr(src.bn, attr)
            d = getattr(dst.bn, attr)

            data = s.data
            if out_slice is not None:
                data = data[out_slice].clone()
            if out_idx is not None:
                idx = torch.as_tensor(list(out_idx), device=data.device, dtype=torch.long)
                data = data.index_select(0, idx).clone()

            if tuple(d.shape) != tuple(data.shape):
                raise RuntimeError(
                    f"_copy_conv_bn_: shape mismatch for BN.{attr}: dst={tuple(d.shape)} vs src={tuple(data.shape)}"
                )

            d.data.copy_(data.contiguous())

        if hasattr(src.bn, "num_batches_tracked") and hasattr(dst.bn, "num_batches_tracked"):
            dst.bn.num_batches_tracked.data.copy_(src.bn.num_batches_tracked.data)

        dst.act = src.act


def _copy_ultralytics_meta(dst: nn.Module, src: nn.Module) -> None:
    for attr in ("i", "f", "type", "np"):
        if hasattr(src, attr):
            setattr(dst, attr, getattr(src, attr))


def _contains_psa_family(module: nn.Module) -> bool:
    for sm in module.modules():
        if _class_name(sm) in PSA_FAMILY_CLASSNAMES:
            return True
    return False


# -----------------------------
# C2f/C3k2 adapter
# -----------------------------

def convert_c2f_to_prunable(c2f: Any, *, verbose: bool = False) -> C2fPrunable:
    if not isinstance(c2f, C2f):
        raise TypeError(f"Expected ultralytics C2f-like block, got {type(c2f).__name__}")

    src_kind = _c2f_like_kind(c2f)

    c1 = int(c2f.cv1.conv.in_channels)
    c2 = int(c2f.cv2.conv.out_channels)
    n = len(c2f.m)
    e = float(c2f.c) / float(c2) if c2 > 0 else 0.5

    shortcut = False
    g = 1
    if n > 0:
        try:
            shortcut = bool(getattr(c2f.m[0], "add", False))
        except Exception:
            shortcut = False
        try:
            g = int(getattr(getattr(getattr(c2f.m[0], "cv2", None), "conv", None), "groups", 1))
        except Exception:
            g = 1

    repl = C2fPrunable(c1=c1, c2=c2, n=n, shortcut=shortcut, g=g, e=e)
    repl.to(device=c2f.cv1.conv.weight.device, dtype=c2f.cv1.conv.weight.dtype)
    repl.train(c2f.training)
    repl.src_block_type = src_kind
    repl._xtrim_contains_psa = _contains_psa_family(c2f)

    _copy_ultralytics_meta(repl, c2f)

    c = int(c2f.c)
    _copy_conv_bn_(repl.cv0, c2f.cv1, out_slice=slice(0, c))
    _copy_conv_bn_(repl.cv1, c2f.cv1, out_slice=slice(c, 2 * c))
    repl.cv2.load_state_dict(c2f.cv2.state_dict())

    repl.m = copy.deepcopy(c2f.m)
    repl.m.to(device=c2f.cv1.conv.weight.device, dtype=c2f.cv1.conv.weight.dtype)

    # Nested C2f/C3k2 are adapted too. PSA-family is handled separately.
    replace_c2f_with_prunable(repl.m, verbose=verbose)

    return repl


def replace_c2f_with_prunable(module: nn.Module, verbose: bool = True) -> int:
    """
    Replace only real C2f/C3k2 blocks with C2fPrunable.
    C2fPSA is handled separately by the composite PSA shrink path.

    Important:
    - If a C2f/C3k2 carrier contains PSA-family submodules, we still adapt it to
      C2fPrunable, but later the generic channel-pruning path must skip its carrier
      convs (cv0/cv1/cv2) and let the composite PSA pruner handle the inner PSA
      blocks atomically.
    """
    replaced = 0
    for name, child in list(module.named_children()):
        kind = _c2f_like_kind(child) if isinstance(child, C2f) else _class_name(child)
        if isinstance(child, C2f) and kind in {"C2f", "C3k2"}:
            if verbose:
                print(f"[adapt] found {kind} at {name}")
            setattr(module, name, convert_c2f_to_prunable(child, verbose=verbose))
            replaced += 1
            if verbose:
                print(f"[adapt] replaced {kind} -> C2fPrunable at {name}")
        else:
            replaced += replace_c2f_with_prunable(child, verbose=verbose)
    return replaced


# -----------------------------
# Composite PSA-family shrinking
# -----------------------------

def _topk_indices(vec: torch.Tensor, k: int) -> List[int]:
    k = int(max(0, min(k, int(vec.numel()))))
    if k <= 0:
        return []
    idx = torch.topk(vec, k=k, largest=True).indices
    idx = torch.sort(idx).values
    return idx.tolist()


def _channel_importance_from_bn(conv: Any) -> torch.Tensor:
    bn = getattr(conv, "bn", None)
    if bn is None or getattr(bn, "weight", None) is None:
        w = conv.conv.weight.detach()
        return w.view(w.shape[0], -1).abs().sum(dim=1)
    return bn.weight.detach().abs()


def _channel_importance_from_weight(conv: Any, in_idx: Optional[Sequence[int]] = None) -> torch.Tensor:
    w = conv.conv.weight.detach()
    if in_idx is not None:
        idx = torch.as_tensor(list(in_idx), device=w.device, dtype=torch.long)
        w = w.index_select(1, idx)
    return w.view(w.shape[0], -1).abs().sum(dim=1)


def _target_keep(c: int, prune_ratio: float, round_to: int, min_channels: int) -> int:
    keep = int(round((1.0 - float(prune_ratio)) * c))
    keep = max(int(min_channels), min(int(c), keep))
    if round_to > 1:
        keep = int((keep + round_to - 1) // round_to * round_to)
        keep = max(int(min_channels), min(int(c), keep))
    return keep


def _attention_attn_ratio(attn: Any) -> float:
    try:
        head_dim = int(getattr(attn, "head_dim"))
        key_dim = int(getattr(attn, "key_dim"))
        if head_dim > 0:
            return float(key_dim) / float(head_dim)
    except Exception:
        pass
    return 0.5


def _attention_old_heads(attn: Any) -> int:
    try:
        return max(1, int(getattr(attn, "num_heads")))
    except Exception:
        return 1


def _attention_qkv_layout(attn: Any) -> Tuple[int, int, int]:
    qk = int(getattr(attn, "key_dim")) * int(getattr(attn, "num_heads"))
    v = int(getattr(attn, "head_dim")) * int(getattr(attn, "num_heads"))
    return qk, qk, v


def _shrink_attention(attn: Any, keep_hidden: Sequence[int]) -> Any:
    if Attention is None:
        raise RuntimeError("Ultralytics Attention class is unavailable")

    old_dim = int(attn.proj.conv.out_channels)
    new_dim = len(keep_hidden)
    if new_dim <= 0 or new_dim >= old_dim:
        return copy.deepcopy(attn)

    attn_ratio = _attention_attn_ratio(attn)
    old_heads = _attention_old_heads(attn)
    new_heads = _safe_num_heads(new_dim, preferred_heads=min(old_heads, new_dim))
    new_attn = Attention(new_dim, num_heads=new_heads, attn_ratio=attn_ratio)
    new_attn.to(device=attn.qkv.conv.weight.device, dtype=attn.qkv.conv.weight.dtype)
    new_attn.train(attn.training)

    old_q, old_k, old_v = _attention_qkv_layout(attn)
    new_q, new_k, new_v = _attention_qkv_layout(new_attn)

    qkv_imp = _channel_importance_from_weight(attn.qkv, in_idx=keep_hidden)
    q_imp = qkv_imp[:old_q]
    k_imp = qkv_imp[old_q:old_q + old_k]

    q_keep = _topk_indices(q_imp, new_q)
    k_keep = _topk_indices(k_imp, new_k)

    # Для V и proj сохраняем каналы, согласованные с keep_hidden
    v_keep = list(keep_hidden[:new_v])

    qkv_out_idx = q_keep + [old_q + i for i in k_keep] + [old_q + old_k + i for i in v_keep]

    _copy_conv_bn_(new_attn.qkv, attn.qkv, out_idx=qkv_out_idx, in_idx=keep_hidden)
    _copy_conv_bn_(new_attn.proj, attn.proj, out_idx=keep_hidden, in_idx=keep_hidden)

    _copy_conv_bn_(new_attn.pe, attn.pe, out_idx=keep_hidden)
    return new_attn


def _shrink_ffn(ffn: Any, keep_hidden: Sequence[int]) -> Any:
    c = len(keep_hidden)
    old_mid = int(ffn[0].conv.out_channels)
    new_mid = min(old_mid, max(2, 2 * c))

    mid_imp = _channel_importance_from_bn(ffn[0])
    keep_mid = _topk_indices(mid_imp, new_mid)

    new_ffn = nn.Sequential(
        Conv(c, new_mid, 1),
        Conv(new_mid, c, 1, act=False),
    )
    new_ffn.to(device=ffn[0].conv.weight.device, dtype=ffn[0].conv.weight.dtype)
    new_ffn.train(ffn.training)

    _copy_conv_bn_(new_ffn[0], ffn[0], out_idx=keep_mid, in_idx=keep_hidden)
    _copy_conv_bn_(new_ffn[1], ffn[1], out_idx=keep_hidden, in_idx=keep_mid)
    return new_ffn


def _psablock_keep_hidden(block: Any, target_keep: int) -> List[int]:
    proj_imp = _channel_importance_from_bn(block.attn.proj)
    ffn_imp = _channel_importance_from_bn(block.ffn[1])
    imp = proj_imp + ffn_imp
    return _topk_indices(imp, target_keep)


def _shrink_psablock(
    block: Any,
    prune_ratio: float,
    round_to: int,
    min_channels: int,
    keep_hidden: Optional[Sequence[int]] = None,
) -> Tuple[Any, Dict[str, int]]:
    c = int(block.attn.proj.conv.out_channels)
    if keep_hidden is None:
        keep = _target_keep(c, prune_ratio, round_to=round_to, min_channels=min_channels)
        if keep >= c:
            return copy.deepcopy(block), {"hidden_old": c, "hidden_new": c, "shrunk": 0}
        keep_hidden = _psablock_keep_hidden(block, keep)
    else:
        keep_hidden = list(keep_hidden)
        keep = len(keep_hidden)
        if keep >= c:
            return copy.deepcopy(block), {"hidden_old": c, "hidden_new": c, "shrunk": 0}

    attn_ratio = _attention_attn_ratio(block.attn)
    old_heads = _attention_old_heads(block.attn)
    new_heads = _safe_num_heads(keep, preferred_heads=min(old_heads, keep))

    new_block = PSABlockPrunable(
        c=keep,
        attn_ratio=attn_ratio,
        num_heads=new_heads,
        shortcut=bool(getattr(block, "add", True)),
    )
    new_block.to(device=block.attn.qkv.conv.weight.device, dtype=block.attn.qkv.conv.weight.dtype)
    new_block.train(block.training)

    new_block.attn = _shrink_attention(block.attn, keep_hidden)
    new_block.ffn = _shrink_ffn(block.ffn, keep_hidden)
    return new_block, {"hidden_old": c, "hidden_new": keep, "shrunk": c - keep}


def _split_half_keep_from_cv1(cv1: Any, c: int, keep: int) -> Tuple[List[int], List[int]]:
    imp = _channel_importance_from_bn(cv1)
    keep_a = _topk_indices(imp[:c], keep)
    keep_b = _topk_indices(imp[c:2 * c], keep)
    return keep_a, keep_b


def _copy_psa_inner(dst: Any, src: Any, keep_hidden: Sequence[int]) -> None:
    dst.attn = _shrink_attention(src.attn, keep_hidden)
    dst.ffn = _shrink_ffn(src.ffn, keep_hidden)


def _copy_c2psa_inner(
    dst_seq: Any,
    src_seq: Any,
    keep_hidden: Sequence[int],
    prune_ratio: float,
    round_to: int,
    min_channels: int,
) -> None:
    for i, src_block in enumerate(src_seq):
        dst_seq[i], _ = _shrink_psablock(
            src_block,
            prune_ratio=prune_ratio,
            round_to=round_to,
            min_channels=min_channels,
            keep_hidden=keep_hidden,
        )


def _copy_c2fpsa_inner(
    dst_list: Any,
    src_list: Any,
    keep_hidden: Sequence[int],
    prune_ratio: float,
    round_to: int,
    min_channels: int,
) -> None:
    for i, src_block in enumerate(src_list):
        dst_list[i], _ = _shrink_psablock(
            src_block,
            prune_ratio=prune_ratio,
            round_to=round_to,
            min_channels=min_channels,
            keep_hidden=keep_hidden,
        )


def _copy_conv_bn_with_concat_inputs(dst: Any, src: Any, chunk_indices: Sequence[Sequence[int]]) -> None:
    in_idx: List[int] = []
    old_chunk = src.conv.in_channels // len(chunk_indices)
    for chunk_id, keep in enumerate(chunk_indices):
        base = chunk_id * old_chunk
        in_idx.extend([base + int(i) for i in keep])
    _copy_conv_bn_(dst, src, in_idx=in_idx)


def _shrink_psa(block: Any, prune_ratio: float, round_to: int, min_channels: int) -> Tuple[Any, Dict[str, int]]:
    c1 = int(block.cv1.conv.in_channels)
    c2 = int(block.cv2.conv.out_channels)
    old_c = int(block.c)
    keep = _target_keep(old_c, prune_ratio, round_to=round_to, min_channels=min_channels)
    if keep >= old_c:
        return copy.deepcopy(block), {"hidden_old": old_c, "hidden_new": old_c, "shrunk": 0}

    new_e = float(keep) / float(c1)
    new_block = PSAPrunable(c1, c2, e=new_e)
    new_block.to(device=block.cv1.conv.weight.device, dtype=block.cv1.conv.weight.dtype)
    new_block.train(block.training)
    _copy_ultralytics_meta(new_block, block)

    keep_a, keep_b = _split_half_keep_from_cv1(block.cv1, old_c, keep)
    cv1_out_idx = keep_a + [old_c + i for i in keep_b]
    _copy_conv_bn_(new_block.cv1, block.cv1, out_idx=cv1_out_idx)
    _copy_psa_inner(new_block, block, keep_b)
    cv2_in_idx = keep_a + [old_c + i for i in keep_b]
    _copy_conv_bn_(new_block.cv2, block.cv2, in_idx=cv2_in_idx)
    return new_block, {"hidden_old": old_c, "hidden_new": keep, "shrunk": old_c - keep}


def _shrink_c2psa(block: Any, prune_ratio: float, round_to: int, min_channels: int) -> Tuple[Any, Dict[str, int]]:
    c1 = int(block.cv1.conv.in_channels)
    c2 = int(block.cv2.conv.out_channels)
    old_c = int(block.c)
    keep = _target_keep(old_c, prune_ratio, round_to=round_to, min_channels=min_channels)
    if keep >= old_c:
        return copy.deepcopy(block), {"hidden_old": old_c, "hidden_new": old_c, "shrunk": 0}

    new_e = float(keep) / float(c1)
    new_block = C2PSAPrunable(c1, c2, n=len(block.m), e=new_e)
    new_block.to(device=block.cv1.conv.weight.device, dtype=block.cv1.conv.weight.dtype)
    new_block.train(block.training)
    _copy_ultralytics_meta(new_block, block)

    keep_a, keep_b = _split_half_keep_from_cv1(block.cv1, old_c, keep)
    cv1_out_idx = keep_a + [old_c + i for i in keep_b]
    _copy_conv_bn_(new_block.cv1, block.cv1, out_idx=cv1_out_idx)
    _copy_c2psa_inner(
        new_block.m,
        block.m,
        keep_b,
        prune_ratio=prune_ratio,
        round_to=round_to,
        min_channels=min_channels,
    )
    cv2_in_idx = keep_a + [old_c + i for i in keep_b]
    _copy_conv_bn_(new_block.cv2, block.cv2, in_idx=cv2_in_idx)
    return new_block, {"hidden_old": old_c, "hidden_new": keep, "shrunk": old_c - keep}


def _shrink_c2fpsa(block: Any, prune_ratio: float, round_to: int, min_channels: int) -> Tuple[Any, Dict[str, int]]:
    c1 = int(block.cv1.conv.in_channels)
    c2 = int(block.cv2.conv.out_channels)
    old_c = int(block.c)
    keep = _target_keep(old_c, prune_ratio, round_to=round_to, min_channels=min_channels)
    if keep >= old_c:
        return copy.deepcopy(block), {"hidden_old": old_c, "hidden_new": old_c, "shrunk": 0}

    new_e = float(keep) / float(c2)
    new_block = C2fPSAPrunable(c1, c2, n=len(block.m), e=new_e)
    new_block.to(device=block.cv1.conv.weight.device, dtype=block.cv1.conv.weight.dtype)
    new_block.train(block.training)
    _copy_ultralytics_meta(new_block, block)

    keep0, keep1 = _split_half_keep_from_cv1(block.cv1, old_c, keep)
    cv1_out_idx = keep0 + [old_c + i for i in keep1]
    _copy_conv_bn_(new_block.cv1, block.cv1, out_idx=cv1_out_idx)
    _copy_c2fpsa_inner(
        new_block.m,
        block.m,
        keep1,
        prune_ratio=prune_ratio,
        round_to=round_to,
        min_channels=min_channels,
    )

    chunk_indices: List[Sequence[int]] = [keep0, keep1]
    for _ in range(len(block.m)):
        chunk_indices.append(keep1)
    _copy_conv_bn_with_concat_inputs(new_block.cv2, block.cv2, chunk_indices)
    return new_block, {"hidden_old": old_c, "hidden_new": keep, "shrunk": old_c - keep}


def _shrink_composite_module(
    module: Any,
    prune_ratio: float,
    round_to: int,
    min_channels: int,
) -> Tuple[Any, Optional[Dict[str, int]]]:
    name = _class_name(module)

    # IMPORTANT:
    # Do NOT shrink bare PSABlock here.
    # A standalone PSABlock may live inside a larger carrier/container
    # whose surrounding convs are still unchanged. Replacing only the block
    # breaks input channel compatibility (e.g. block expects 56, parent still feeds 256).
    if name == "PSA":
        return _shrink_psa(module, prune_ratio, round_to, min_channels)
    if name == "C2PSA":
        return _shrink_c2psa(module, prune_ratio, round_to, min_channels)
    if name == "C2fPSA":
        return _shrink_c2fpsa(module, prune_ratio, round_to, min_channels)

    return module, None


def shrink_psa_family_blocks(
    module: nn.Module,
    *,
    prune_ratio: float,
    round_to: int = 8,
    min_channels: int = 8,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Prune PSA-family blocks atomically instead of pruning their internal convs independently.
    This removes the need for regex exclusions like ".attn."/".ffn.".
    """
    items: List[Dict[str, Any]] = []
    replaced = 0

    for name, child in list(module.named_children()):
        cls = _class_name(child)
        if cls in {"PSA", "C2PSA", "C2fPSA"}:
            new_child, info = _shrink_composite_module(
                child,
                prune_ratio=prune_ratio,
                round_to=round_to,
                min_channels=min_channels,
            )
            if info is not None and int(info.get("shrunk", 0)) > 0:
                setattr(module, name, new_child)
                replaced += 1
                item = {"name": name, "type": cls, **info}
                items.append(item)
                if verbose:
                    print(f"[composite] {cls} at {name}: hidden {info['hidden_old']} -> {info['hidden_new']}")
            else:
                sub = shrink_psa_family_blocks(
                    child,
                    prune_ratio=prune_ratio,
                    round_to=round_to,
                    min_channels=min_channels,
                    verbose=verbose,
                )
                replaced += int(sub.get("blocks_shrunk", 0))
                items.extend(list(sub.get("items", [])))
        else:
            sub = shrink_psa_family_blocks(
                child,
                prune_ratio=prune_ratio,
                round_to=round_to,
                min_channels=min_channels,
                verbose=verbose,
            )
            replaced += int(sub.get("blocks_shrunk", 0))
            items.extend(list(sub.get("items", [])))

    return {"blocks_shrunk": replaced, "items": items}