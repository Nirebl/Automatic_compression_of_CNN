from .calib import make_calib_imagelist
from .fake_quant_ultra import (
    patch_ultralytics_convs_for_fake_quant,
    set_fake_quant_enabled,
    set_fake_quant_bits,
    unpatch_ultralytics_convs_for_fake_quant,
)


def ort_static_quantize_yolo(*args, **kwargs):
    """Загружает квантование ONNX Runtime только при реальном вызове."""
    from .ort_ptq import ort_static_quantize_yolo as _ort_static_quantize_yolo

    return _ort_static_quantize_yolo(*args, **kwargs)


__all__ = [
    "make_calib_imagelist",
    "ort_static_quantize_yolo",
    "patch_ultralytics_convs_for_fake_quant",
    "set_fake_quant_enabled",
    "set_fake_quant_bits",
    "unpatch_ultralytics_convs_for_fake_quant",
]
