from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class LowRankConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        rank: int,
        stride: int = 1,
        padding: int = 0,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size, kernel_size) if isinstance(kernel_size, int) else kernel_size
        self.stride = (stride, stride) if isinstance(stride, int) else stride
        self.padding = (padding, padding) if isinstance(padding, int) else padding
        self.dilation = (dilation, dilation) if isinstance(dilation, int) else dilation
        self.groups = groups
        self.rank = rank
        self.padding_mode = "zeros"
        self.transposed = False
        self.output_padding = (0, 0)

        k = self.kernel_size[0]

        self.weight_u = nn.Parameter(torch.empty(rank, in_channels, k, k, device=device, dtype=dtype))
        self.weight_v = nn.Parameter(torch.empty(out_channels, rank, 1, 1, device=device, dtype=dtype))

        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, device=device, dtype=dtype))
        else:
            self.register_parameter("bias", None)

        nn.init.kaiming_uniform_(self.weight_u)
        nn.init.kaiming_uniform_(self.weight_v)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    @property
    def weight(self) -> torch.Tensor:
        rank = self.weight_u.shape[0]
        in_ch = self.weight_u.shape[1]
        k = self.weight_u.shape[2]
        out_ch = self.weight_v.shape[0]
        U = self.weight_u.view(rank, -1)
        V = self.weight_v.view(out_ch, rank)
        W = torch.mm(V, U)
        return W.view(out_ch, in_ch, k, k)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.conv2d(
            x, self.weight_u,
            bias=None,
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
            groups=self.groups,
        )
        x = F.conv2d(x, self.weight_v, bias=self.bias)
        return x

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}, {self.out_channels}, kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding}, rank={self.rank}"
        )

    @classmethod
    def from_conv2d(cls, conv: nn.Conv2d, rank: int) -> "LowRankConv2d":
        out_ch = conv.out_channels
        in_ch = conv.in_channels
        k = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
        stride = conv.stride[0] if isinstance(conv.stride, tuple) else conv.stride
        padding = conv.padding[0] if isinstance(conv.padding, tuple) else conv.padding
        dilation = conv.dilation[0] if isinstance(conv.dilation, tuple) else conv.dilation

        W = conv.weight.detach().view(out_ch, -1)

        max_rank = min(out_ch, W.shape[1])
        effective_rank = min(rank, max_rank)
        effective_rank = max(1, effective_rank)

        try:
            U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        except Exception:
            U, S, V = torch.svd(W)
            Vh = V.t()

        U_r = U[:, :effective_rank]
        S_r = S[:effective_rank]
        Vh_r = Vh[:effective_rank, :]

        S_sqrt = torch.sqrt(S_r + 1e-8)
        U_scaled = U_r * S_sqrt.unsqueeze(0)
        Vh_scaled = Vh_r * S_sqrt.unsqueeze(1)

        lowrank = cls(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=k,
            rank=effective_rank,
            stride=stride,
            padding=padding,
            dilation=dilation,
            groups=conv.groups,
            bias=conv.bias is not None,
            device=conv.weight.device,
            dtype=conv.weight.dtype,
        )

        lowrank.weight_u.data = Vh_scaled.view(effective_rank, in_ch, k, k).contiguous()
        lowrank.weight_v.data = U_scaled.view(out_ch, effective_rank, 1, 1).contiguous()

        if conv.bias is not None:
            lowrank.bias.data = conv.bias.data.clone()

        return lowrank


def select_rank_by_energy(W_2d: torch.Tensor, threshold: float) -> Tuple[int, float]:
    threshold = float(threshold)
    try:
        _, S, _ = torch.linalg.svd(W_2d, full_matrices=False)
    except Exception:
        _, S, _ = torch.svd(W_2d)

    energy = S ** 2
    total = energy.sum()
    if total < 1e-12:
        return 1, 0.0

    cumulative = torch.cumsum(energy, dim=0) / total
    above = (cumulative >= threshold).nonzero(as_tuple=False)
    if above.numel() == 0:
        rank = len(S)
    else:
        rank = int(above[0].item()) + 1

    actual = float(cumulative[rank - 1].item())
    return rank, actual


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


def _is_stem_layer(name: str) -> bool:
    parts = name.split(".")
    for i, p in enumerate(parts):
        if p == "model" and i + 1 < len(parts) and parts[i + 1] == "0":
            return True
    if parts[0] == "0":
        return True
    return False


def _collect_convs_for_lowrank(
    torch_model: Any,
    *,
    min_channels: int = 32,
    exclude_head: bool = True,
    exclude_stem: bool = True,
    exclude_name_regex: Optional[str] = None,
    max_layers: int = 0,
    include_1x1: bool = False,
) -> List[Tuple[str, str, nn.Conv2d]]:
    exclude_re = re.compile(exclude_name_regex) if exclude_name_regex else None
    head_mod = _find_head_module(torch_model) if exclude_head else None

    head_ids: set = set()
    if head_mod is not None:
        for _, hm in head_mod.named_modules():
            head_ids.add(id(hm))

    candidates: List[Tuple[str, str, nn.Conv2d]] = []

    for name, m in torch_model.named_modules():
        if exclude_re and exclude_re.search(name):
            continue
        if head_ids and id(m) in head_ids:
            continue
        if exclude_stem and _is_stem_layer(name):
            continue

        if hasattr(m, "conv") and isinstance(getattr(m, "conv"), nn.Conv2d):
            conv: nn.Conv2d = getattr(m, "conv")

            if conv.out_channels < min_channels or conv.in_channels < min_channels:
                continue

            k = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size
            if k == 1 and not include_1x1:
                continue

            if conv.groups != 1:
                continue

            candidates.append((name, "conv", conv))

    if max_layers > 0:
        candidates = candidates[:max_layers]

    return candidates


def apply_lowrank_decomposition(
    torch_model: Any,
    *,
    rank: int = 0,
    energy_threshold: float = 0.0,
    exclude_head: bool = True,
    exclude_stem: bool = True,
    exclude_name_regex: Optional[str] = None,
    min_channels: int = 32,
    max_layers: int = 0,
    include_1x1: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    use_energy = float(energy_threshold) > 0.0
    use_fixed = not use_energy and int(rank) > 0

    if not use_energy and not use_fixed:
        if verbose:
            print("[lowrank] Neither rank nor energy_threshold set — skipping")
        return {"layers_decomposed": 0, "params_before": 0, "params_after": 0}

    torch_model.eval()

    candidates = _collect_convs_for_lowrank(
        torch_model,
        min_channels=min_channels,
        exclude_head=exclude_head,
        exclude_stem=exclude_stem,
        exclude_name_regex=exclude_name_regex,
        max_layers=max_layers,
        include_1x1=include_1x1,
    )

    if not candidates:
        if verbose:
            print("[lowrank] No eligible Conv layers found")
        return {"layers_decomposed": 0, "params_before": 0, "params_after": 0}

    layers_decomposed = 0
    params_before_total = 0
    params_after_total = 0
    layer_report: List[Dict[str, Any]] = []

    for parent_name, attr_name, conv in candidates:
        out_ch = conv.out_channels
        in_ch = conv.in_channels
        k = conv.kernel_size[0] if isinstance(conv.kernel_size, tuple) else conv.kernel_size

        W_2d = conv.weight.detach().view(out_ch, -1)
        max_rank_layer = min(out_ch, W_2d.shape[1])

        if use_energy:
            target_rank, energy_frac = select_rank_by_energy(W_2d, energy_threshold)
            target_rank = min(target_rank, max_rank_layer)
            target_rank = max(1, target_rank)
        else:
            target_rank = min(int(rank), max_rank_layer)
            target_rank = max(1, target_rank)
            _, energy_frac = select_rank_by_energy(W_2d, 0.0)
            try:
                _, S, _ = torch.linalg.svd(W_2d, full_matrices=False)
            except Exception:
                _, S, _ = torch.svd(W_2d)
            energy_total = (S ** 2).sum()
            energy_kept = (S[:target_rank] ** 2).sum()
            energy_frac = float(energy_kept / energy_total) if energy_total > 1e-12 else 0.0

        orig_params = out_ch * in_ch * k * k + (out_ch if conv.bias is not None else 0)
        new_params = (
            target_rank * in_ch * k * k
            + out_ch * target_rank
            + (out_ch if conv.bias is not None else 0)
        )

        layer_info: Dict[str, Any] = {
            "name": f"{parent_name}.{attr_name}",
            "in_ch": in_ch,
            "out_ch": out_ch,
            "kernel": k,
            "rank": target_rank,
            "max_rank": max_rank_layer,
            "energy_frac": round(energy_frac, 4),
            "params_before": orig_params,
            "params_after": new_params,
        }

        if new_params >= orig_params:
            layer_info["skipped"] = "no_param_saving"
            layer_report.append(layer_info)
            if verbose:
                print(f"[lowrank] {parent_name}.{attr_name}: skip (rank {target_rank} saves no params)")
            continue

        try:
            lowrank_conv = LowRankConv2d.from_conv2d(conv, target_rank)

            parent = torch_model
            for part in parent_name.split("."):
                if part.isdigit():
                    parent = parent[int(part)]
                else:
                    parent = getattr(parent, part)

            setattr(parent, attr_name, lowrank_conv)

            layers_decomposed += 1
            params_before_total += orig_params
            params_after_total += new_params
            layer_info["applied"] = True
            layer_report.append(layer_info)

            if verbose:
                ratio = new_params / orig_params
                print(
                    f"[lowrank] {parent_name}.{attr_name}: "
                    f"rank={target_rank}/{max_rank_layer}, "
                    f"energy={energy_frac:.1%}, "
                    f"params {orig_params}->{new_params} ({ratio:.1%})"
                )

        except Exception as e:
            layer_info["error"] = str(e)
            layer_report.append(layer_info)
            if verbose:
                print(f"[lowrank] Warning: failed to decompose {parent_name}.{attr_name}: {e}")
            continue

    compression_ratio = (
        params_before_total / params_after_total if params_after_total > 0 else 1.0
    )

    return {
        "layers_decomposed": layers_decomposed,
        "params_before": params_before_total,
        "params_after": params_after_total,
        "compression_ratio": compression_ratio,
        "rank": rank,
        "energy_threshold": energy_threshold,
        "layers": layer_report,
    }


def recalibrate_bn(
    torch_model: Any,
    data_loader,
    n_batches: int = 20,
    device: Optional[torch.device] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    if n_batches <= 0:
        return {"batches_processed": 0, "skipped": True}

    if device is None:
        try:
            device = next(torch_model.parameters()).device
        except StopIteration:
            device = torch.device("cpu")

    bn_count = sum(1 for m in torch_model.modules() if isinstance(m, (nn.BatchNorm2d, nn.SyncBatchNorm)))
    if bn_count == 0:
        if verbose:
            print("[lowrank/bn_recalib] No BatchNorm layers found, skipping")
        return {"batches_processed": 0, "bn_layers": 0, "skipped": True}

    if verbose:
        print(f"[lowrank/bn_recalib] Recalibrating {bn_count} BN layers over {n_batches} batches...")

    torch_model.train()

    for p in torch_model.parameters():
        p.requires_grad_(False)

    batches_done = 0
    try:
        for batch in data_loader:
            if batches_done >= n_batches:
                break
            try:
                if isinstance(batch, (list, tuple)):
                    imgs = batch[0]
                else:
                    imgs = batch

                if not isinstance(imgs, torch.Tensor):
                    continue

                imgs = imgs.to(device, non_blocking=True)
                if imgs.dtype != torch.float32:
                    imgs = imgs.float() / 255.0

                with torch.no_grad():
                    torch_model(imgs)

                batches_done += 1
                if verbose and batches_done % 5 == 0:
                    print(f"[lowrank/bn_recalib]   {batches_done}/{n_batches} batches done")

            except Exception as e:
                if verbose:
                    print(f"[lowrank/bn_recalib] Warning: batch {batches_done} failed: {e}")
                continue
    finally:
        torch_model.eval()
        for p in torch_model.parameters():
            p.requires_grad_(True)

    if verbose:
        print(f"[lowrank/bn_recalib] Done. Processed {batches_done} batches.")

    return {"batches_processed": batches_done, "bn_layers": bn_count}
