from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn

from xtrim.trim.slim import (
    _as_tensor_list,
    _collect_prunable_convs,
    _coverage_report,
    _get_module_by_qualname,
    _global_threshold_from_bn_gammas,
    _make_example_inputs_for_tp,
    _pick_pruning_target,
    _select_prune_idxs_by_gamma,
    _select_prune_idxs_layerwise,
    _skip_reason,
    _validate_c2f_like_invariants,
    _validate_psa_like_invariants,
    bn_sparsity_regularizer,
)


pytestmark = pytest.mark.unit


class UltraConv(nn.Module):
    def __init__(self, c1: int, c2: int, k: int = 1, groups: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, padding=k // 2, groups=groups, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


class C2fLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = 4
        self.cv1 = UltraConv(8, 8)
        self.cv2 = UltraConv(12, 8)
        self.m = nn.ModuleList([nn.Sequential(UltraConv(4, 4))])


class PSA(nn.Module):
    def __init__(self):
        super().__init__()
        self.c = 4
        self.cv1 = UltraConv(8, 8)
        self.cv2 = UltraConv(8, 8)


class Detect(nn.Module):
    def __init__(self):
        super().__init__()
        self.cv = UltraConv(8, 8)


class SlimToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem = UltraConv(8, 8)
        self.block = C2fLike()
        self.psa = PSA()
        self.detect = Detect()
        self.free = UltraConv(8, 8)
        self.model = nn.ModuleList([self.stem, self.detect])

    def forward(self, x):
        return self.free(self.stem(x))


def test_get_module_by_qualname_and_skip_reasons():
    model = SlimToyModel()

    assert _get_module_by_qualname(model, "block.m.0.0") is model.block.m[0][0]
    assert _skip_reason(model, "block.m.0.0", skip_inner_m=True, skip_cv1_if_parent_has_m=True) == "skip_inner_m"
    assert _skip_reason(model, "block.cv2", skip_inner_m=False, skip_cv1_if_parent_has_m=True) == "skip_parent_cv2_aggregator"
    assert _skip_reason(model, "block.cv1", skip_inner_m=False, skip_cv1_if_parent_has_m=True) == "unsafe_original_c2f_like_cv1"
    assert _skip_reason(model, "psa.cv1", skip_inner_m=False, skip_cv1_if_parent_has_m=False) == "handled_by_composite_psa_pruner:PSA"
    assert _skip_reason(model, "detect.cv", skip_inner_m=False, skip_cv1_if_parent_has_m=False) == "handled_by_detect_head_pruner:Detect"


def test_collect_prunable_convs_and_coverage_report():
    model = SlimToyModel()
    prunable = _collect_prunable_convs(
        model,
        exclude_head=False,
        exclude_name_regex=None,
        skip_inner_m=True,
        skip_cv1_if_parent_has_m=True,
    )
    names = [name for name, _ in prunable]
    report = _coverage_report(
        model,
        exclude_head=False,
        exclude_name_regex=None,
        skip_inner_m=True,
        skip_cv1_if_parent_has_m=True,
        include_inner_m_regex=None,
        protect_last_n=1,
    )

    assert "free" in names
    assert "stem" in names
    assert "block.cv1" not in names
    assert report["total_ultra_convs"] >= 1
    assert report["prunable_before_tail"] >= report["prunable_after_tail"]


def test_bn_gamma_threshold_and_index_selection():
    bn = nn.BatchNorm2d(8)
    with torch.no_grad():
        bn.weight[:] = torch.arange(1, 9, dtype=torch.float32)
    conv = nn.Conv2d(8, 8, 1, bias=False)
    with torch.no_grad():
        conv.weight.copy_(torch.arange(conv.weight.numel(), dtype=torch.float32).view_as(conv.weight))

    threshold = _global_threshold_from_bn_gammas([("x", SimpleNamespace(bn=bn))], 0.5)
    prune_by_gamma = _select_prune_idxs_by_gamma(bn, threshold, channel_round=2, min_channels=2)
    prune_layerwise = _select_prune_idxs_layerwise(bn, 0.5, channel_round=2, min_channels=2)
    prune_uniform = _select_prune_idxs_layerwise(
        bn,
        0.5,
        channel_round=2,
        min_channels=2,
        importance_mode="uniform",
        conv=conv,
    )

    assert threshold == pytest.approx(4.5)
    assert set(prune_by_gamma) == {0, 1, 2, 3}
    assert len(prune_layerwise) == 4
    assert len(prune_uniform) == 4


def test_bn_regularizer_tensor_helpers_and_pruning_target():
    model = SlimToyModel()
    reg = bn_sparsity_regularizer(model, l1_weight=1e-4, exclude_head=False)
    nested = _as_tensor_list({"a": [torch.ones(1), (torch.zeros(1),)]})
    inp = torch.randn(1, 3, 4, 4)
    prepared = _make_example_inputs_for_tp(inp)

    class TP:
        prune_conv_out_channels = object()
        prune_depthwise_conv_out_channels = object()

    regular_target = _pick_pruning_target(TP, model.free)
    depthwise = UltraConv(4, 4, groups=4)
    depthwise_target = _pick_pruning_target(TP, depthwise)

    assert reg.item() > 0
    assert len(nested) == 2
    assert prepared[0].requires_grad is True
    assert regular_target[0] is model.free.conv
    assert regular_target[1] is TP.prune_conv_out_channels
    assert depthwise_target[0] is depthwise.conv
    assert depthwise_target[1] is TP.prune_depthwise_conv_out_channels


def test_invariant_validators_detect_broken_shapes():
    model = SlimToyModel()
    _validate_c2f_like_invariants(model)
    _validate_psa_like_invariants(model)

    model.block.cv1.conv = nn.Conv2d(8, 7, 1)
    with pytest.raises(RuntimeError, match="invalid C2f-like invariant"):
        _validate_c2f_like_invariants(model)

    model = SlimToyModel()
    model.psa.cv2.conv = nn.Conv2d(7, 8, 1)
    with pytest.raises(RuntimeError, match="invalid PSA invariant"):
        _validate_psa_like_invariants(model)
