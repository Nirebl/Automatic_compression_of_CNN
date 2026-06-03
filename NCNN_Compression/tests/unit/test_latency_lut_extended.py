from __future__ import annotations

import json

import pytest
import torch
import torch.nn as nn

from xtrim.latency_lut import LatencyLUT, _get_op_type, build_lut_from_model, estimate_model_latency

pytestmark = pytest.mark.unit


def test_get_op_type_and_lookup_zero_macs_branch(tmp_path):
    p = tmp_path / "lut.json"
    p.write_text(json.dumps({"entries": [{"op": "conv1x1", "cin": 0, "cout": 0, "k": 1, "stride": 1, "h": 0, "w": 0, "groups": 1, "latency_ms": 3.0}]}), encoding="utf-8")
    lut = LatencyLUT(p, verbose=False)
    assert _get_op_type(nn.Conv2d(4, 4, 1)) == "conv1x1"
    assert _get_op_type(nn.Conv2d(4, 4, 3, groups=4)) == "conv2d_dw"
    assert _get_op_type(nn.Conv2d(4, 4, 5)) == "conv5x5"
    assert lut.lookup("conv1x1", 1, 1, 1, 1, 1, 1, 1) == 3.0


def test_estimate_latency_fallback_and_verbose_when_forward_fails(tmp_path, capsys):
    p = tmp_path / "lut.json"
    p.write_text(json.dumps({"entries": []}), encoding="utf-8")
    lut = LatencyLUT(p, verbose=False)

    class BadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 4, 3, padding=1)
        def forward(self, x):
            raise RuntimeError("boom")

    report = estimate_model_latency(BadModel(), lut, input_shape=(1, 3, 8, 8), verbose=True, macs_per_ms=10.0)
    out = capsys.readouterr().out
    assert report["lut_misses"] == 1
    assert report["per_layer"][0]["h"] == 8
    assert "Total" in out


def test_build_lut_from_model_profiles_layers():
    model = nn.Sequential(nn.Conv2d(3, 4, 1), nn.Conv2d(4, 4, 3, padding=1, groups=4))
    data = build_lut_from_model(model, input_shape=(1, 3, 8, 8), warmup=0, repeats=1, verbose=False)
    assert data["device"] == "local_cpu"
    assert [e["op"] for e in data["entries"]] == ["conv1x1", "conv2d_dw"]
