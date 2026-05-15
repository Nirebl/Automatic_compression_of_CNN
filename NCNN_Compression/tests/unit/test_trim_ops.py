from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from xtrim.trim.dilated import apply_dilation
from xtrim.trim.gumbel_choice import (
    MixedOp1x1,
    count_mixed_ops,
    freeze_mixed_ops,
    insert_mixed_ops,
    set_gumbel_temperature,
    tau_exponential,
    tau_linear,
)
from xtrim.trim.lowrank import LowRankConv2d, select_rank_by_energy
from xtrim.trim.operator_choice import apply_operator_plan, plan_from_config, validate_plan
from xtrim.trim.sparse_1x1 import apply_1x1_weight_sparsity, find_1x1_convs, remove_pruning_reparam


pytestmark = pytest.mark.unit


def test_lowrank_full_rank_reconstructs_conv_output():
    torch.manual_seed(0)
    conv = nn.Conv2d(3, 4, 3, padding=1, bias=True)
    lowrank = LowRankConv2d.from_conv2d(conv, rank=4)
    x = torch.randn(2, 3, 8, 8)

    assert torch.allclose(lowrank(x), conv(x), atol=1e-4, rtol=1e-4)


def test_select_rank_by_energy_handles_zero_and_nonzero_weights():
    rank_zero, actual_zero = select_rank_by_energy(torch.zeros(4, 4), 0.9)
    rank, actual = select_rank_by_energy(torch.diag(torch.tensor([3.0, 1.0, 0.1])), 0.9)

    assert (rank_zero, actual_zero) == (1, 0.0)
    assert rank == 2
    assert actual >= 0.9


def test_sparse_and_operator_choice_pipeline(tiny_yolo_like):
    names = [name for name, _ in find_1x1_convs(tiny_yolo_like, exclude_head=False, min_channels=1)]
    assert "model.0.conv" in names

    sparse_stats = apply_1x1_weight_sparsity(tiny_yolo_like, 0.5, exclude_head=False, min_channels=1, verbose=False)
    removed = remove_pruning_reparam(tiny_yolo_like)

    assert sparse_stats["layers_sparsified"] >= 1
    assert sparse_stats["avg_sparsity"] > 0
    assert removed >= 1

    plan = plan_from_config(tiny_yolo_like, {"auto": "dense", "model.0.conv": "lowrank"}, exclude_head=False, min_channels=1)
    stats = apply_operator_plan(tiny_yolo_like, plan, lowrank_rank=4, verbose=False)

    assert stats["lowrank"] == 1
    assert isinstance(tiny_yolo_like.model[0].conv, LowRankConv2d)


def test_validate_plan_rejects_unknown_mode():
    with pytest.raises(ValueError, match="Invalid operator mode"):
        validate_plan({"x": "magic"})


def test_gumbel_insert_temperature_and_freeze(tiny_yolo_like):
    stats = insert_mixed_ops(tiny_yolo_like, exclude_head=False, min_channels=1, lowrank_rank=4, verbose=False)
    assert stats["layers_replaced"] >= 1
    assert count_mixed_ops(tiny_yolo_like) >= 1

    changed = set_gumbel_temperature(tiny_yolo_like, 0.7)
    assert changed == count_mixed_ops(tiny_yolo_like)

    for m in tiny_yolo_like.modules():
        if isinstance(m, MixedOp1x1):
            with torch.no_grad():
                m.logits[:] = torch.tensor([10.0, 0.0, -1.0])

    frozen = freeze_mixed_ops(tiny_yolo_like, verbose=False)
    assert frozen["layers_frozen"] >= 1
    assert count_mixed_ops(tiny_yolo_like) == 0


def test_tau_schedules_hit_endpoints():
    assert tau_linear(0, 5, 5.0, 0.5) == 5.0
    assert tau_linear(4, 5, 5.0, 0.5) == pytest.approx(0.5)
    assert tau_exponential(0, 5, 5.0, 0.5) == pytest.approx(5.0)
    assert tau_exponential(4, 5, 5.0, 0.5) == pytest.approx(0.5)


def test_apply_dilation_changes_last_eligible_blocks(tiny_yolo_like):
    stats = apply_dilation(tiny_yolo_like, exclude_head=False, target_n_blocks=1, rates=(2,), verbose=False)

    assert stats["layers_modified"] == 1
    assert stats["layers"][0]["dilation"] == 2
