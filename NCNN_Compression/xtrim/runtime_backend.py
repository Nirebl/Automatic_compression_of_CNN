from __future__ import annotations

from typing import Any, Optional


NCNN_CPU = "ncnn_cpu"
NCNN_VULKAN = "ncnn_vulkan"

_CPU_ALIASES = {
    "cpu",
    "ncnn_cpu",
    "ncnn-cpu",
    "ncnn:cpu",
    "phone_cpu",
}

_GPU_ALIASES = {
    "gpu",
    "vulkan",
    "ncnn_gpu",
    "ncnn-gpu",
    "ncnn_vulkan",
    "ncnn-vulkan",
    "ncnn:vulkan",
    "phone_gpu",
}


def normalize_ncnn_runtime(value: Optional[Any]) -> Optional[str]:
    """Normalize user-facing runtime names for NCNN Android benchmarking.

    Returns None when no runtime was specified. Raises ValueError for unknown
    non-empty values, because silently falling back to CPU/GPU would make
    benchmark tables hard to trust.
    """

    if value is None:
        return None

    text = str(value).strip().lower()
    if not text:
        return None

    if text in _CPU_ALIASES:
        return NCNN_CPU
    if text in _GPU_ALIASES:
        return NCNN_VULKAN

    raise ValueError(
        f"Unknown NCNN runtime {value!r}. Use 'ncnn_cpu' or 'ncnn_vulkan' "
        "(aliases: cpu/gpu/vulkan are also accepted)."
    )


def ncnn_runtime_from_legacy_device_id(value: Optional[Any]) -> Optional[str]:
    """Map legacy device/gpu_device numbers to explicit runtime names.

    This keeps old configs working:
      - -1 means NCNN CPU
      -  0 or any non-negative value means NCNN Vulkan GPU
    """

    if value is None:
        return None

    try:
        device_id = int(value)
    except (TypeError, ValueError):
        return normalize_ncnn_runtime(value)

    return NCNN_CPU if device_id < 0 else NCNN_VULKAN


def resolve_ncnn_runtime(device: Any, default: str = NCNN_CPU) -> str:
    """Resolve the effective NCNN runtime from a DeviceConfig-like object.

    Priority:
      1. devices[].runtime
      2. devices[].device      # new readable legacy alias requested by user
      3. devices[].gpu_device  # old benchncnn-style field
      4. default
    """

    explicit = normalize_ncnn_runtime(getattr(device, "runtime", None))
    if explicit is not None:
        return explicit

    from_device = ncnn_runtime_from_legacy_device_id(getattr(device, "device", None))
    if from_device is not None:
        return from_device

    from_gpu = ncnn_runtime_from_legacy_device_id(getattr(device, "gpu_device", None))
    if from_gpu is not None:
        return from_gpu

    normalized_default = normalize_ncnn_runtime(default)
    return normalized_default or NCNN_CPU


def ncnn_gpu_device_for_runtime(runtime: Any) -> int:
    runtime_norm = normalize_ncnn_runtime(runtime) or NCNN_CPU
    return 0 if runtime_norm == NCNN_VULKAN else -1


def effective_ncnn_gpu_device(device: Any) -> int:
    return ncnn_gpu_device_for_runtime(resolve_ncnn_runtime(device))


def ncnn_runtime_label(device: Any) -> str:
    """Small helper for cache keys, logs and table/debug text."""

    return resolve_ncnn_runtime(device)
