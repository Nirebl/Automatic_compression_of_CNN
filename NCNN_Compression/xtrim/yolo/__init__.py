"""Вспомогательные функции для интеграции YOLO в конвейер xtrim."""
__all__ = [
    "build_ultralytics_candidate",
    "eval_ultralytics_map",
    "eval_ultralytics_metrics",
    "eval_exported_onnx_map",
    "eval_exported_onnx_metrics",
    "make_ultralytics_export_onnx_fn",
    "save_student_torchscript",
    "warmstart_noop",
    "finetune_noop",
    "finetune_kd",
    "finetune_qat_recover",
]


def __getattr__(name: str):
    if name in __all__:
        from . import ultralytics_io

        return getattr(ultralytics_io, name)
    raise AttributeError(name)
