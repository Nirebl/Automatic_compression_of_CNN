from __future__ import annotations

from pathlib import Path

import pytest

from xtrim.ncnn import AdbBench, NcnnConverter, _normalize_shape_arg, _validate_ncnn_param
from xtrim.types import DeviceConfig, NcnnModelPaths, ToolsConfig


pytestmark = pytest.mark.unit


def _write_valid_param(path: Path, n_layers: int = 2) -> None:
    body = [
        "Input input 0 1 data",
        "Convolution conv 1 1 data out 0=1",
    ][:n_layers]
    path.write_text("7767517\n" + f"{n_layers} 3\n" + "\n".join(body) + "\n", encoding="utf-8")


def test_normalize_shape_arg_accepts_bracketed_and_plain():
    assert _normalize_shape_arg("640,640,3") == "[640,640,3]"
    assert _normalize_shape_arg("[640,640,3]") == "[640,640,3]"


def test_validate_ncnn_param_accepts_structurally_valid_file(tmp_path):
    p = tmp_path / "model.param"
    _write_valid_param(p)

    _validate_ncnn_param(p)


def test_validate_ncnn_param_rejects_wrong_magic_and_truncation(tmp_path):
    bad_magic = tmp_path / "bad_magic.param"
    bad_magic.write_text("123\n1 1\nInput input 0 1 data\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="bad magic"):
        _validate_ncnn_param(bad_magic)

    truncated = tmp_path / "truncated.param"
    truncated.write_text("7767517\n2 3\nInput input 0 1 data\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="truncated"):
        _validate_ncnn_param(truncated)


def test_optimize_falls_back_to_copy_when_tool_fails(tmp_path, monkeypatch):
    src_param = tmp_path / "src.param"
    src_bin = tmp_path / "src.bin"
    src_param.write_text("param", encoding="utf-8")
    src_bin.write_bytes(b"bin")

    monkeypatch.setattr("xtrim.ncnn.sh", lambda _cmd: (_ for _ in ()).throw(RuntimeError("boom")))
    out = NcnnConverter(ToolsConfig()).optimize(NcnnModelPaths(src_param, src_bin), tmp_path / "out")

    assert out.param.read_text(encoding="utf-8") == "param"
    assert out.bin.read_bytes() == b"bin"
    assert (tmp_path / "out" / "ncnnoptimize_failed.txt").exists()


def test_adb_bench_parses_average_latency(monkeypatch):
    bench = AdbBench(ToolsConfig())
    device = DeviceConfig(name="phone", serial="123")
    model = NcnnModelPaths(Path("m.param"), Path("m.bin"))

    def fake_adb(_serial: str, *args: str) -> str:
        if args == ("get-state",):
            return "device\n"
        if args[0] == "push":
            return ""
        return "min = 1.0  max = 3.0  avg = 2.5"

    monkeypatch.setattr(bench, "adb", fake_adb)

    avg, raw = bench.bench(device, model, "[1,3,640,640]")

    assert avg == 2.5
    assert "avg = 2.5" in raw
