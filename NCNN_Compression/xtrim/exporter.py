from __future__ import annotations

from pathlib import Path
from typing import Callable

from .utils import ensure_dir


class Exporter:
    def __init__(self, export_fn: Callable[[Path], None]):
        self._export_fn = export_fn

    def export_onnx(self, onnx_path: Path) -> Path:
        ensure_dir(onnx_path.parent)
        self._export_fn(onnx_path)
        if not onnx_path.exists():
            raise RuntimeError(f"ONNX export did not create file: {onnx_path}")
        return onnx_path