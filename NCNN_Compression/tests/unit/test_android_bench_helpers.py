from __future__ import annotations

import pytest

from xtrim.android_app_bench import AndroidAppBench
from xtrim.android_ort_bench import AndroidOrtBench
from xtrim.types import AndroidAppBenchConfig, OrtAndroidBenchConfig, ToolsConfig


pytestmark = pytest.mark.unit


def test_android_app_extract_last_json_returns_last_valid_object():
    bench = AndroidAppBench(ToolsConfig(), AndroidAppBenchConfig())
    text = 'noise {"avg_ms": 1.0} more {bad} tail {"avg_ms": 2.5, "run_id": "x"}'

    assert bench._extract_last_json(text) == {"avg_ms": 2.5, "run_id": "x"}


def test_android_ort_extract_last_json_returns_none_without_json():
    bench = AndroidOrtBench(ToolsConfig(), OrtAndroidBenchConfig())

    assert bench._extract_last_json("plain log") is None
