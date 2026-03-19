from __future__ import annotations

import copy
import re
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lowrank import LowRankConv2d
from .sparse_1x1 import find_1x1_convs


class MixedOp1x1(nn.Module):
    BRANCH_NAMES = ("dense", "sparse", "lowrank")

    def __init__(
        self,
        conv: nn.Conv2d,
        *,
        lowrank_rank: int = 8,
        sparsity: float = 0.5,
        hard: bool = False,
    ):
        super().__init__()

        assert conv.kernel_size in ((1, 1), 1), "MixedOp1x1 only works for 1x1 Conv2d"

        self.in_channels = conv.in_channels
        self.out_channels = conv.out_channels
        self.hard = hard

        self.dense = copy.deepcopy(conv)

        self.sparse = copy.deepcopy(conv)
        with torch.no_grad():
            w = self.sparse.weight.data
            n_zero = int(sparsity * w.numel())
            if n_zero > 0:
                threshold = w.abs().flatten().kthvalue(n_zero).values
                mask = (w.abs() > threshold).float()
                self.sparse.weight.data = w * mask

        effective_rank = max(1, min(lowrank_rank, min(conv.out_channels, conv.in_channels)))
        orig_params = conv.out_channels * conv.in_channels
        new_params = effective_rank * (conv.in_channels + conv.out_channels)
        if new_params < orig_params:
            self.lowrank: nn.Module = LowRankConv2d.from_conv2d(conv, effective_rank)
        else:
            self.lowrank = copy.deepcopy(conv)

        self.logits = nn.Parameter(torch.randn(3) * 0.01)
        self.tau: float = 1.0
        self._rank = effective_rank
        self._sparsity = sparsity

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w = F.gumbel_softmax(self.logits, tau=self.tau, hard=self.hard, dim=0)
        out = (
            w[0] * self.dense(x)
            + w[1] * self.sparse(x)
            + w[2] * self.lowrank(x)
        )
        return out

    @torch.no_grad()
    def chosen_branch_idx(self) -> int:
        return int(self.logits.argmax().item())

    @property
    def chosen_branch_name(self) -> str:
        return self.BRANCH_NAMES[self.chosen_branch_idx()]

    def freeze(self) -> nn.Module:
        idx = self.chosen_branch_idx()
        branch = [self.dense, self.sparse, self.lowrank][idx]
        return branch

    def extra_repr(self) -> str:
        return (
            f"{self.in_channels}->{self.out_channels}, "
            f"rank={self._rank}, sparsity={self._sparsity}, "
            f"tau={self.tau:.2f}, chosen={self.chosen_branch_name}"
        )


def insert_mixed_ops(
    model: nn.Module,
    *,
    lowrank_rank: int = 8,
    sparsity: float = 0.5,
    hard: bool = False,
    exclude_head: bool = True,
    exclude_name_regex: Optional[str] = None,
    min_channels: int = 16,
    verbose: bool = True,
) -> Dict[str, Any]:
    convs = find_1x1_convs(
        model,
        exclude_head=exclude_head,
        exclude_name_regex=exclude_name_regex,
        min_channels=min_channels,
    )

    if not convs:
        if verbose:
            print("[gumbel] No eligible 1x1 Conv2d layers found.")
        return {"layers_replaced": 0, "layer_names": []}

    replaced = 0
    layer_names = []

    for full_name, conv in convs:
        try:
            mixed = MixedOp1x1(
                conv,
                lowrank_rank=lowrank_rank,
                sparsity=sparsity,
                hard=hard,
            )

            parts = full_name.split(".")
            parent = model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]  # type: ignore[index]
                else:
                    parent = getattr(parent, part)
            setattr(parent, parts[-1], mixed)

            replaced += 1
            layer_names.append(full_name)

            if verbose:
                print(
                    f"[gumbel] {full_name}: "
                    f"{conv.in_channels}->{conv.out_channels} "
                    f"rank={mixed._rank} sparsity={sparsity}"
                )
        except Exception as e:
            if verbose:
                print(f"[gumbel] Warning: failed to replace '{full_name}': {e}")

    if verbose:
        print(f"[gumbel] Inserted MixedOp1x1 into {replaced} layers")

    return {"layers_replaced": replaced, "layer_names": layer_names}


def set_gumbel_temperature(model: nn.Module, tau: float) -> int:
    count = 0
    for m in model.modules():
        if isinstance(m, MixedOp1x1):
            m.tau = tau
            count += 1
    return count


def freeze_mixed_ops(
    model: nn.Module,
    verbose: bool = True,
) -> Dict[str, Any]:
    to_replace: List[Tuple[str, MixedOp1x1]] = []
    for name, m in model.named_modules():
        if isinstance(m, MixedOp1x1):
            to_replace.append((name, m))

    branch_counts: Dict[str, int] = {"dense": 0, "sparse": 0, "lowrank": 0}
    frozen = 0

    for full_name, mixed in to_replace:
        try:
            winner = mixed.freeze()
            branch_name = mixed.chosen_branch_name

            parts = full_name.split(".")
            parent = model
            for part in parts[:-1]:
                if part.isdigit():
                    parent = parent[int(part)]  # type: ignore[index]
                else:
                    parent = getattr(parent, part)
            setattr(parent, parts[-1], winner)

            branch_counts[branch_name] = branch_counts.get(branch_name, 0) + 1
            frozen += 1

            if verbose:
                logits_str = ", ".join(
                    f"{MixedOp1x1.BRANCH_NAMES[i]}={mixed.logits[i].item():.3f}"
                    for i in range(3)
                )
                print(
                    f"[gumbel] {full_name}: chose '{branch_name}' "
                    f"[{logits_str}]"
                )
        except Exception as e:
            if verbose:
                print(f"[gumbel] Warning: failed to freeze '{full_name}': {e}")

    if verbose:
        print(
            f"[gumbel] Frozen {frozen} layers: "
            + ", ".join(f"{k}={v}" for k, v in branch_counts.items() if v > 0)
        )

    return {"layers_frozen": frozen, "branch_counts": branch_counts}


def count_mixed_ops(model: nn.Module) -> int:
    return sum(1 for m in model.modules() if isinstance(m, MixedOp1x1))


def tau_linear(epoch: int, total_epochs: int, tau_start: float, tau_end: float) -> float:
    if total_epochs <= 1:
        return tau_end
    frac = epoch / (total_epochs - 1)
    return tau_start + frac * (tau_end - tau_start)


def tau_exponential(epoch: int, total_epochs: int, tau_start: float, tau_end: float) -> float:
    import math
    if total_epochs <= 1:
        return tau_end
    frac = epoch / (total_epochs - 1)
    log_start = math.log(max(tau_start, 1e-8))
    log_end = math.log(max(tau_end, 1e-8))
    return math.exp(log_start + frac * (log_end - log_start))
