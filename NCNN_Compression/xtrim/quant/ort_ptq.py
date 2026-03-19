from __future__ import annotations

from pathlib import Path
from typing import Iterator, List, Optional, Dict

import numpy as np

import onnxruntime as ort
from onnxruntime.quantization import CalibrationDataReader, quantize_static, QuantFormat, QuantType, CalibrationMethod


def _letterbox(img: np.ndarray, new_shape: int = 640, color=(114, 114, 114)) -> np.ndarray:
    h, w = img.shape[:2]
    r = min(new_shape / h, new_shape / w)
    nh, nw = int(round(h * r)), int(round(w * r))

    import cv2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)

    pad_w, pad_h = new_shape - nw, new_shape - nh
    top = pad_h // 2
    bottom = pad_h - top
    left = pad_w // 2
    right = pad_w - left
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img


class YoloCalibReader(CalibrationDataReader):
    def __init__(self, image_list_txt: Path, input_name: str, imgsz: int):
        self.input_name = input_name
        self.imgsz = int(imgsz)
        self.paths: List[str] = [p.strip() for p in image_list_txt.read_text(encoding="utf-8").splitlines() if p.strip()]
        self._iter: Optional[Iterator[Dict[str, np.ndarray]]] = None

    def get_next(self) -> Optional[Dict[str, np.ndarray]]:
        if self._iter is None:
            self._iter = iter(self._gen())
        return next(self._iter, None)

    def rewind(self) -> None:
        self._iter = None

    def _gen(self) -> Iterator[Dict[str, np.ndarray]]:
        import cv2
        for p in self.paths:
            img = cv2.imread(p)  # BGR uint8
            if img is None:
                continue
            img = _letterbox(img, new_shape=self.imgsz)

            img = img[:, :, ::-1]

            x = img.astype(np.float32) / 255.0

            x = np.transpose(x, (2, 0, 1))

            x = np.expand_dims(x, axis=0)

            yield {self.input_name: x}


def ort_static_quantize_yolo(
    *,
    onnx_fp32: Path,
    onnx_int8: Path,
    image_list_txt: Path,
    imgsz: int,
    per_channel: bool,
    quant_format: str,
    activation_type: str,
    weight_type: str,
    calibrate_method: str,
    input_name: str = "images",
) -> Path:
    qformat = QuantFormat.QDQ if str(quant_format).upper() == "QDQ" else QuantFormat.QOperator

    atype = QuantType.QUInt8 if str(activation_type).upper() in ("QUINT8", "UINT8") else QuantType.QInt8
    wtype = QuantType.QInt8 if str(weight_type).upper() in ("QINT8", "INT8") else QuantType.QUInt8

    cm = CalibrationMethod.Entropy if str(calibrate_method).lower() in ("entropy", "kl") else CalibrationMethod.MinMax

    reader = YoloCalibReader(Path(image_list_txt), input_name=input_name, imgsz=int(imgsz))

    quantize_static(
        model_input=str(onnx_fp32),
        model_output=str(onnx_int8),
        calibration_data_reader=reader,
        quant_format=qformat,
        activation_type=atype,
        weight_type=wtype,
        per_channel=bool(per_channel),
        calibrate_method=cm,
        op_types_to_quantize=["Conv"],
        extra_options={
            "ActivationSymmetric": False,
            "WeightSymmetric": True,
        },
    )

    return Path(onnx_int8)