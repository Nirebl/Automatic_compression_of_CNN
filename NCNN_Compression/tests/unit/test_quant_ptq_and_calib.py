from __future__ import annotations

import importlib
import sys
import types
from pathlib import Path

import numpy as np
import pytest

from xtrim.quant.calib import make_calib_imagelist

pytestmark = pytest.mark.unit


def _install_fake_ort(monkeypatch):
    calls = {}
    ort = types.ModuleType("onnxruntime")
    quant = types.ModuleType("onnxruntime.quantization")
    class Reader: pass
    class QuantFormat:
        QDQ = "QDQ"
        QOperator = "QOperator"
    class QuantType:
        QUInt8 = "QUInt8"
        QInt8 = "QInt8"
    class CalibrationMethod:
        Entropy = "Entropy"
        MinMax = "MinMax"
    def quantize_static(**kwargs):
        calls.update(kwargs)
    quant.CalibrationDataReader = Reader
    quant.QuantFormat = QuantFormat
    quant.QuantType = QuantType
    quant.CalibrationMethod = CalibrationMethod
    quant.quantize_static = quantize_static
    ort.quantization = quant
    monkeypatch.setitem(sys.modules, "onnxruntime", ort)
    monkeypatch.setitem(sys.modules, "onnxruntime.quantization", quant)
    sys.modules.pop("xtrim.quant.ort_ptq", None)
    return importlib.import_module("xtrim.quant.ort_ptq"), calls


def test_make_calib_imagelist_success_and_errors(tmp_path, monkeypatch):
    imgdir = tmp_path / "imgs"
    imgdir.mkdir()
    for i in range(3):
        (imgdir / f"{i}.jpg").write_bytes(b"x")

    ultra = types.ModuleType("ultralytics")
    data = types.ModuleType("ultralytics.data")
    utils = types.ModuleType("ultralytics.data.utils")
    utils.check_det_dataset = lambda _p: {"train": str(imgdir)}
    monkeypatch.setitem(sys.modules, "ultralytics", ultra)
    monkeypatch.setitem(sys.modules, "ultralytics.data", data)
    monkeypatch.setitem(sys.modules, "ultralytics.data.utils", utils)

    out = make_calib_imagelist(data_yaml="data.yaml", split="weird", max_images=2, out_txt=tmp_path / "calib.txt", seed=0)
    assert out.exists()
    assert len(out.read_text(encoding="utf-8").splitlines()) == 2

    utils.check_det_dataset = lambda _p: {}
    with pytest.raises(RuntimeError, match="does not provide split"):
        make_calib_imagelist(data_yaml="data.yaml", split="val", max_images=2, out_txt=tmp_path / "bad.txt")

    empty = tmp_path / "empty"
    empty.mkdir()
    utils.check_det_dataset = lambda _p: {"train": str(empty)}
    with pytest.raises(RuntimeError, match="Could not find images"):
        make_calib_imagelist(data_yaml="data.yaml", split="train", max_images=2, out_txt=tmp_path / "bad2.txt")


def test_ort_ptq_reader_and_quantize(monkeypatch, tmp_path):
    mod, calls = _install_fake_ort(monkeypatch)
    monkeypatch.setitem(sys.modules, "cv2", types.SimpleNamespace(
        INTER_LINEAR=1,
        BORDER_CONSTANT=0,
        resize=lambda img, size, interpolation: np.zeros((size[1], size[0], 3), dtype=np.uint8),
        copyMakeBorder=lambda img, top, bottom, left, right, borderType, value: np.pad(img, ((top, bottom), (left, right), (0, 0))),
        imread=lambda p: None if p.endswith("bad.jpg") else np.zeros((2, 4, 3), dtype=np.uint8),
    ))

    txt = tmp_path / "imgs.txt"
    txt.write_text("bad.jpg\ngood.jpg\n", encoding="utf-8")
    reader = mod.YoloCalibReader(txt, input_name="images", imgsz=4)
    batch = reader.get_next()
    assert batch is not None and batch["images"].shape == (1, 3, 4, 4)
    assert reader.get_next() is None
    reader.rewind()
    assert reader.get_next() is not None

    img = np.zeros((2, 4, 3), dtype=np.uint8)
    assert mod._letterbox(img, new_shape=4).shape == (4, 4, 3)

    out = mod.ort_static_quantize_yolo(
        onnx_fp32=tmp_path / "m.onnx",
        onnx_int8=tmp_path / "q.onnx",
        image_list_txt=txt,
        imgsz=4,
        per_channel=True,
        quant_format="qdq",
        activation_type="uint8",
        weight_type="int8",
        calibrate_method="kl",
    )
    assert out.name == "q.onnx"
    assert calls["quant_format"] == "QDQ"
    assert calls["activation_type"] == "QUInt8"
    assert calls["weight_type"] == "QInt8"
    assert calls["calibrate_method"] == "Entropy"

    mod.ort_static_quantize_yolo(
        onnx_fp32=tmp_path / "m.onnx",
        onnx_int8=tmp_path / "q2.onnx",
        image_list_txt=txt,
        imgsz=4,
        per_channel=False,
        quant_format="operator",
        activation_type="int8",
        weight_type="uint8",
        calibrate_method="minmax",
    )
    assert calls["quant_format"] == "QOperator"
    assert calls["activation_type"] == "QInt8"
    assert calls["weight_type"] == "QUInt8"
    assert calls["calibrate_method"] == "MinMax"
