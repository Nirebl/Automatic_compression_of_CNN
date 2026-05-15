from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from xtrim.ncnn import AdbBench, NcnnConverter, _validate_ncnn_param
from xtrim.types import DeviceConfig, NcnnModelPaths, PTQConfig, ToolsConfig

pytestmark = pytest.mark.unit


def _write_valid_param(path: Path, n_layers: int = 2) -> None:
    body = [
        "Input input 0 1 data",
        "Convolution conv 1 1 data out 0=1",
    ][:n_layers]
    path.write_text("7767517\n" + f"{n_layers} 3\n" + "\n".join(body) + "\n", encoding="utf-8")


def test_validate_ncnn_param_more_invalid_cases(tmp_path):
    missing = tmp_path / "missing.param"
    with pytest.raises(RuntimeError, match="not found"):
        _validate_ncnn_param(missing)

    empty = tmp_path / "empty.param"
    empty.write_text("", encoding="utf-8")
    with pytest.raises(RuntimeError, match="empty"):
        _validate_ncnn_param(empty)

    no_header = tmp_path / "no_header.param"
    no_header.write_text("7767517\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="missing layer"):
        _validate_ncnn_param(no_header)

    bad_header = tmp_path / "bad_header.param"
    bad_header.write_text("7767517\na b\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="bad header"):
        _validate_ncnn_param(bad_header)

    bad_last = tmp_path / "bad_last.param"
    bad_last.write_text("7767517\n1 1\nBad only three\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="last layer"):
        _validate_ncnn_param(bad_last)


def test_ncnn_converter_onnx_ptq_success_and_fallbacks(tmp_path, monkeypatch):
    calls = []

    def fake_sh(cmd):
        calls.append(cmd)
        if "ncnn2int8" in cmd[0]:
            _write_valid_param(Path(cmd[3]))
            Path(cmd[4]).write_bytes(b"x" * 1_100_000)
        return ""

    monkeypatch.setattr("xtrim.ncnn.sh", fake_sh)
    conv = NcnnConverter(ToolsConfig())
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"onnx")
    ncnn = conv.onnx_to_ncnn(onnx, tmp_path / "float")
    assert ncnn.param.name == "model.param"

    imagelist = tmp_path / "imgs.txt"
    imagelist.write_text("a.jpg\n", encoding="utf-8")
    src_param = tmp_path / "src.param"
    src_bin = tmp_path / "src.bin"
    _write_valid_param(src_param)
    src_bin.write_bytes(b"bin")
    out = conv.ptq_int8(NcnnModelPaths(src_param, src_bin), tmp_path / "int8", PTQConfig(imagelist=str(imagelist)))
    assert out.param.exists() and out.bin.exists()
    assert any("shape=[640,640,3]" in part for cmd in calls for part in cmd)

    with pytest.raises(RuntimeError, match="imagelist not found"):
        conv.ptq_int8(NcnnModelPaths(src_param, src_bin), tmp_path / "int8b", PTQConfig(imagelist=str(tmp_path / "none")))


def test_ncnn_converter_ptq_recovers_from_tool_error_with_valid_outputs(tmp_path, monkeypatch):
    imagelist = tmp_path / "imgs.txt"
    imagelist.write_text("a.jpg\n", encoding="utf-8")
    src_param = tmp_path / "src.param"
    src_bin = tmp_path / "src.bin"
    _write_valid_param(src_param)
    src_bin.write_bytes(b"bin")

    def fake_sh(cmd):
        if "ncnn2int8" in cmd[0]:
            _write_valid_param(Path(cmd[3]))
            Path(cmd[4]).write_bytes(b"x" * 1_100_000)
            raise RuntimeError("tool failed")
        return ""

    monkeypatch.setattr("xtrim.ncnn.sh", fake_sh)
    out = NcnnConverter(ToolsConfig()).ptq_int8(NcnnModelPaths(src_param, src_bin), tmp_path / "int8", PTQConfig(imagelist=str(imagelist)))
    assert out.param.exists()


def test_ncnn_converter_ptq_raises_when_failed_outputs_invalid(tmp_path, monkeypatch):
    imagelist = tmp_path / "imgs.txt"
    imagelist.write_text("a.jpg\n", encoding="utf-8")
    src_param = tmp_path / "src.param"
    src_bin = tmp_path / "src.bin"
    _write_valid_param(src_param)
    src_bin.write_bytes(b"bin")

    def fake_sh(cmd):
        if "ncnn2int8" in cmd[0]:
            raise RuntimeError("tool failed")
        return ""

    monkeypatch.setattr("xtrim.ncnn.sh", fake_sh)
    with pytest.raises(RuntimeError, match="output files are invalid"):
        NcnnConverter(ToolsConfig()).ptq_int8(NcnnModelPaths(src_param, src_bin), tmp_path / "int8", PTQConfig(imagelist=str(imagelist)))


def test_pnnx_convert_handles_export_and_postprocess_error(tmp_path, monkeypatch):
    class Detect(nn.Module):
        def __init__(self):
            super().__init__()
            self.export = False
        def forward(self, x):
            return x

    model = nn.Sequential(nn.Conv2d(3, 3, 1), Detect())
    fake_pnnx = types.SimpleNamespace()

    def fake_export(_model, _pt_path, *, inputs, ncnnparam, ncnnbin, **_kwargs):
        Path(ncnnparam).write_text("param data" * 20, encoding="utf-8")
        Path(ncnnbin).write_bytes(b"x" * 2000)
        raise RuntimeError("post")

    fake_pnnx.export = fake_export
    monkeypatch.setitem(sys.modules, "pnnx", fake_pnnx)
    out = NcnnConverter(ToolsConfig()).pnnx_convert(model, tmp_path, imgsz=8)
    assert out is not None
    assert model[1].export is False


def test_adb_bench_ensure_binary_and_failure_paths(tmp_path, monkeypatch):
    local = tmp_path / "benchncnn"
    local.write_text("bin", encoding="utf-8")
    bench = AdbBench(ToolsConfig(benchncnn_local=str(local)))
    dev = DeviceConfig(name="phone", serial="s")
    calls = []

    def fake_adb(_serial, *args):
        calls.append(args)
        if args == ("get-state",):
            return "device"
        if args[:3] == ("shell", "test -x /data/local/tmp/benchncnn && echo OK"):
            return "OK"
        return ""

    monkeypatch.setattr(bench, "adb", fake_adb)
    bench.ensure_benchncnn(dev)
    bench.ensure_benchncnn(dev, force_push=True)
    assert any(args[:1] == ("push",) for args in calls)

    monkeypatch.setattr(bench, "is_device_ready", lambda _d: False)
    with pytest.raises(RuntimeError, match="not ready"):
        bench.ensure_benchncnn(dev)

    bench2 = AdbBench(ToolsConfig(benchncnn_local=str(tmp_path / "missing")))
    monkeypatch.setattr(bench2, "is_device_ready", lambda _d: True)
    monkeypatch.setattr(bench2, "adb", lambda *_a: "")
    with pytest.raises(RuntimeError, match="not found locally"):
        bench2.ensure_benchncnn(dev, force_push=True)


def test_adb_bench_parse_failure(monkeypatch):
    bench = AdbBench(ToolsConfig())
    dev = DeviceConfig(name="phone", serial="s")
    model = NcnnModelPaths(Path("m.param"), Path("m.bin"))
    monkeypatch.setattr(bench, "is_device_ready", lambda _d: True)
    monkeypatch.setattr(bench, "adb", lambda *_a: "bad log")
    with pytest.raises(RuntimeError, match="Could not parse"):
        bench.bench(dev, model, "[1,3,8,8]")
