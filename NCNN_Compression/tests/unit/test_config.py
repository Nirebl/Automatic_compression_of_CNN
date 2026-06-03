from __future__ import annotations

import pytest

from xtrim.config import parse_config
from xtrim.types import DilatedConfig, StagedPruningConfig, TrimConfig


pytestmark = pytest.mark.unit


def test_parse_config_uses_defaults_and_extracts_operator_plan():
    parsed = parse_config({"operator_choice": {"enabled": True, "plan": {"auto": "lowrank", "model.1": "dense"}}})

    trim_cfg = parsed[7]
    op_choice_cfg = parsed[17]
    op_choice_plan = parsed[18]

    assert trim_cfg == TrimConfig()
    assert op_choice_cfg.enabled is True
    assert op_choice_plan == {"auto": "lowrank", "model.1": "dense"}


def test_parse_config_converts_dilated_lists_to_tuples():
    parsed = parse_config({"dilated": {"enabled": True, "rates": [1, 2, 3], "target_layers": ["a", "b"]}})
    dilated_cfg = parsed[-1]

    assert dilated_cfg == DilatedConfig(enabled=True, rates=(1, 2, 3), target_layers=("a", "b"))


def test_parse_config_converts_staged_pruning_milestones_to_tuple():
    parsed = parse_config({"staged_pruning": {"milestones": [0.6, 0.7], "intermediate_epochs": 12}})
    staged_cfg = parsed[-2]

    assert staged_cfg == StagedPruningConfig(milestones=(0.6, 0.7), intermediate_epochs=12)


def test_parse_config_reads_benchmark_profiles():
    parsed = parse_config({
        "benchmark_profiles": [
            {"name": "cpu_ort", "backend": "ort_android", "provider": "xnnpack", "devices": ["phone1"], "threads": 4},
            {"name": "npu_nnapi", "backend": "ort_android", "provider": "nnapi", "device": "phone2", "required": False},
        ]
    })
    profiles = parsed[-3]

    assert len(profiles) == 2
    assert profiles[0].name == "cpu_ort"
    assert profiles[0].backend == "ort_android"
    assert profiles[0].provider == "xnnpack"
    assert profiles[0].device_names == ("phone1",)
    assert profiles[0].threads == 4
    assert profiles[1].provider == "nnapi"
    assert profiles[1].device_names == ("phone2",)
    assert profiles[1].required is False


def test_parse_config_accepts_device_runtime_fields():
    parsed = parse_config({
        "devices": [
            {"name": "phone_cpu", "serial": "s1", "device": -1},
            {"name": "phone_gpu", "serial": "s1", "runtime": "ncnn_vulkan"},
        ],
        "benchmark_profiles": [
            {"name": "ncnn_gpu", "backend": "android_app", "devices": ["phone_gpu"], "runtime": "ncnn_vulkan"},
        ],
    })

    devices = parsed[1]
    profiles = parsed[-3]

    assert devices[0].device == -1
    assert devices[0].runtime is None
    assert devices[1].runtime == "ncnn_vulkan"
    assert profiles[0].runtime == "ncnn_vulkan"
    assert profiles[0].device_names == ("phone_gpu",)
