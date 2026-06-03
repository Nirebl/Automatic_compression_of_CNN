from __future__ import annotations

import json

import pytest
import torch.nn as nn

from xtrim.latency_lut import LatencyLUT, estimate_model_latency, latency_penalty


pytestmark = pytest.mark.unit


def _lut_file(tmp_path):
    p = tmp_path / "lut.json"
    p.write_text(
        json.dumps(
            {
                "device": "cpu",
                "unit": "ms",
                "entries": [
                    {"op": "conv3x3", "cin": 3, "cout": 8, "k": 3, "stride": 1, "h": 16, "w": 16, "groups": 1, "latency_ms": 1.0},
                    {"op": "conv1x1", "cin": 8, "cout": 8, "k": 1, "stride": 1, "h": 16, "w": 16, "groups": 1, "latency_ms": 0.25},
                ],
            }
        ),
        encoding="utf-8",
    )
    return p


def test_lut_exact_nearest_scaled_and_fallback(tmp_path):
    lut = LatencyLUT(_lut_file(tmp_path), verbose=False)

    assert lut.lookup("conv3x3", 3, 8, 3, 1, 16, 16, 1) == 1.0
    assert lut.lookup("conv3x3", 3, 16, 3, 1, 16, 16, 1) == pytest.approx(2.0)
    assert lut.lookup("unknown", 1, 1, 1, 1, 1, 1, 1) is None
    assert lut.lookup_with_fallback("unknown", 1, 1, 1, 1, 10, 10, 1, macs_per_ms=100.0) == pytest.approx(2.0)


def test_estimate_model_latency_reports_hits(tmp_path):
    lut = LatencyLUT(_lut_file(tmp_path), verbose=False)
    model = nn.Sequential(
        nn.Conv2d(3, 8, 3, padding=1),
        nn.Conv2d(8, 8, 1),
    )

    report = estimate_model_latency(model, lut, input_shape=(1, 3, 16, 16))

    assert report["layers_estimated"] == 2
    assert report["lut_hits"] == 2
    assert report["lut_misses"] == 0
    assert report["latency_est_ms"] == pytest.approx(1.25)


def test_latency_penalty_only_applies_to_excess():
    assert latency_penalty(8.0, budget_ms=10.0, lambda_lat=0.5) == 0.0
    assert latency_penalty(12.0, budget_ms=10.0, lambda_lat=0.5) == 1.0
