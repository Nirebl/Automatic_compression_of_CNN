from __future__ import annotations

import pytest

from xtrim.runtime_backend import (
    effective_ncnn_gpu_device,
    normalize_ncnn_runtime,
    resolve_ncnn_runtime,
)
from xtrim.types import DeviceConfig

pytestmark = pytest.mark.unit


def test_runtime_backend_explicit_runtime_has_priority():
    dev = DeviceConfig(name="p", serial="s", runtime="ncnn_cpu", gpu_device=0)
    assert resolve_ncnn_runtime(dev) == "ncnn_cpu"
    assert effective_ncnn_gpu_device(dev) == -1


def test_runtime_backend_legacy_device_and_gpu_device_mapping():
    assert resolve_ncnn_runtime(DeviceConfig(name="cpu", serial="s", device=-1, gpu_device=0)) == "ncnn_cpu"
    assert resolve_ncnn_runtime(DeviceConfig(name="gpu", serial="s", device=0, gpu_device=-1)) == "ncnn_vulkan"
    assert resolve_ncnn_runtime(DeviceConfig(name="old", serial="s", gpu_device=0)) == "ncnn_vulkan"


def test_runtime_backend_rejects_unknown_runtime():
    with pytest.raises(ValueError, match="Unknown NCNN runtime"):
        normalize_ncnn_runtime("npu")
