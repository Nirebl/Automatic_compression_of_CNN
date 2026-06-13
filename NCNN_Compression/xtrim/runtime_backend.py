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
    """Приводит название NCNN runtime к единому виду."""

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
    """Преобразует старые значения device/gpu_device в явный NCNN runtime."""

    if value is None:
        return None

    try:
        device_id = int(value)
    except (TypeError, ValueError):
        return normalize_ncnn_runtime(value)

    return NCNN_CPU if device_id < 0 else NCNN_VULKAN


def resolve_ncnn_runtime(device: Any, default: str = NCNN_CPU) -> str:
    """Определяет, какой NCNN runtime нужно использовать для устройства."""

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
    """Возвращает короткую подпись runtime для логов, таблиц и кэша."""

    return resolve_ncnn_runtime(device)
