from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from xtrim.trim.dilated import apply_dilation, find_conv_blocks_for_dilation
from xtrim.trim.lowrank import LowRankConv2d, apply_lowrank_decomposition, recalibrate_bn, select_rank_by_energy
from xtrim.trim.operator_choice import apply_operator_plan, plan_from_config, validate_plan
from xtrim.trim.sparse_1x1 import apply_1x1_weight_sparsity, find_1x1_convs, remove_pruning_reparam

pytestmark = pytest.mark.unit


class Block(nn.Module):
    def __init__(self, cin, cout, k):
        super().__init__()
        self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(cout)
        self.act = nn.Identity()
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class TinyCompressModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList([
            Block(16, 16, 3),
            Block(16, 16, 1),
            Block(16, 16, 3),
            nn.Sequential(nn.Conv2d(16, 16, 1)),
        ])
    def forward(self, x):
        for m in self.model:
            x = m(x)
        return x


def test_sparse_and_operator_choice_paths():
    model = TinyCompressModel()
    convs = find_1x1_convs(model, exclude_head=True, min_channels=8)
    assert [n for n, _ in convs] == ["model.1.conv"]
    assert apply_1x1_weight_sparsity(model, 0.0)["layers_sparsified"] == 0
    stats = apply_1x1_weight_sparsity(model, 0.5, min_channels=8, verbose=False)
    assert stats["layers_sparsified"] == 1
    assert remove_pruning_reparam(model) == 1

    with pytest.raises(ValueError):
        validate_plan({"x": "bad"})

    plan = plan_from_config(model, {"auto": "dense", "model.1.conv": "sparse"}, min_channels=8)
    assert plan["model.1.conv"] == "sparse"
    stats2 = apply_operator_plan(model, {"model.1.conv": "dense", "missing": "sparse", "model.0.conv": "sparse"}, verbose=False)
    assert stats2["dense"] == 1 and stats2["skipped"] == 2

    model2 = TinyCompressModel()
    stats3 = apply_operator_plan(model2, {"model.1.conv": "sparse"}, sparse_sparsity=0.5, verbose=False)
    assert stats3["sparse"] == 1

    model3 = TinyCompressModel()
    stats4 = apply_operator_plan(model3, {"model.1.conv": "lowrank"}, lowrank_rank=2, verbose=False)
    assert stats4["lowrank"] == 1
    assert isinstance(model3.model[1].conv, LowRankConv2d)

    model4 = TinyCompressModel()
    stats5 = apply_operator_plan(model4, {"model.1.conv": "lowrank"}, lowrank_rank=16, verbose=False)
    assert stats5["dense"] == 1


def test_dilation_paths():
    model = TinyCompressModel()
    cands = find_conv_blocks_for_dilation(model, target_n_blocks=2)
    assert [n for n, _ in cands] == ["model.2"]
    out = apply_dilation(model, rates=(2,), target_n_blocks=2, verbose=False)
    assert out["layers_modified"] == 1
    assert model.model[2].conv.dilation == (2, 2)
    assert apply_dilation(nn.Sequential(nn.Conv2d(1, 1, 1)), verbose=False)["layers_modified"] == 0
    assert apply_dilation(model, rates=(), target_layers=("model.2",), verbose=False)["rates"] == [1]


def test_lowrank_select_apply_and_recalibrate():
    conv = nn.Conv2d(16, 16, 1, bias=True)
    lr = LowRankConv2d.from_conv2d(conv, rank=4)
    x = torch.randn(2, 16, 8, 8)
    assert lr(x).shape == conv(x).shape
    assert lr.weight.shape == conv.weight.shape
    assert "rank=4" in lr.extra_repr()

    w = torch.diag(torch.tensor([4.0, 2.0, 1.0]))
    assert select_rank_by_energy(w, 0.0)[0] == 1
    assert select_rank_by_energy(w, 1.1)[0] == 3
    assert select_rank_by_energy(torch.zeros(2, 2), 0.8) == (1, 0.0)

    model = TinyCompressModel()
    stats = apply_lowrank_decomposition(model, rank=2, min_channels=8, exclude_head=True, exclude_stem=False, include_1x1=True, verbose=False)
    assert stats["layers_decomposed"] >= 1

    loader = [torch.randn(2, 16, 8, 8) for _ in range(2)]
    out = recalibrate_bn(model, loader, device=torch.device("cpu"), n_batches=1, verbose=False)
    assert out["batches_processed"] == 1
