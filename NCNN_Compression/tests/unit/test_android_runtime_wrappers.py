from __future__ import annotations

import types
from pathlib import Path

import pytest
from PIL import Image

from xtrim.android.adb_demo import AdbYoloDemo, save_ppm_rgb
from xtrim.android_app_bench import AndroidAppBench
from xtrim.android_ort_bench import AndroidOrtBench
from xtrim.types import AndroidAppBenchConfig, AndroidDemoConfig, DeviceConfig, OrtAndroidBenchConfig, ToolsConfig

pytestmark = pytest.mark.unit


def test_android_app_bench_run_once_success_and_helpers(tmp_path, monkeypatch):
    cfg = AndroidAppBenchConfig(enabled=True, clear_logcat=True, poll_interval_sec=0.0)
    bench = AndroidAppBench(ToolsConfig(), cfg)
    dev = DeviceConfig(name="phone", serial="s", cooling_down=0)
    calls = []

    monkeypatch.setattr("xtrim.android_app_bench.uuid.uuid4", lambda: types.SimpleNamespace(hex="abc1234567dead"))
    monkeypatch.setattr("xtrim.android_app_bench.time.sleep", lambda *_: None)

    def fake_adb(_serial: str, *args: str) -> str:
        calls.append(args)
        if args == ("get-state",):
            return "device\n"
        if args[:1] == ("logcat",) and "-d" in args:
            return '{"avg_ms": 1.0, "run_id": "old"}\n{"avg_ms": 2.5, "run_id": "abc1234567"}'
        return ""

    monkeypatch.setattr(bench, "adb", fake_adb)
    out = bench.run_once(device=dev, local_param=tmp_path / "m.param", local_bin=tmp_path / "m.bin")

    assert out["avg_ms"] == 2.5
    assert any(args[:2] == ("push", str(tmp_path / "m.param")) for args in calls)
    assert any(args[:2] == ("logcat", "-c") for args in calls)


def test_android_app_bench_disabled_and_device_not_ready(tmp_path, monkeypatch):
    dev = DeviceConfig(name="phone", serial="s", cooling_down=0)
    disabled = AndroidAppBench(ToolsConfig(), AndroidAppBenchConfig(enabled=False))
    with pytest.raises(RuntimeError, match="enabled=False"):
        disabled.run_once(device=dev, local_param=tmp_path / "m.param", local_bin=tmp_path / "m.bin")

    enabled = AndroidAppBench(ToolsConfig(), AndroidAppBenchConfig(enabled=True))
    monkeypatch.setattr(enabled, "is_device_ready", lambda _d: False)
    with pytest.raises(RuntimeError, match="Device not ready"):
        enabled.run_once(device=dev, local_param=tmp_path / "m.param", local_bin=tmp_path / "m.bin")


def test_android_ort_run_once_success_and_timeout(tmp_path, monkeypatch):
    dev = DeviceConfig(name="phone", serial="s", cooling_down=0)
    cfg = OrtAndroidBenchConfig(enabled=True, clear_logcat=True, poll_interval_sec=0.0, timeout_sec=1)
    bench = AndroidOrtBench(ToolsConfig(), cfg)
    monkeypatch.setattr("xtrim.android_ort_bench.uuid.uuid4", lambda: types.SimpleNamespace(hex="run1234567dead"))
    monkeypatch.setattr("xtrim.android_ort_bench.time.sleep", lambda *_: None)
    monkeypatch.setattr(bench, "is_device_ready", lambda _d: True)

    responses = iter([
        '{"avg_ms": 1.0, "run_id": "run1234567"}',
    ])

    def fake_adb(_serial: str, *args: str) -> str:
        if args[:1] == ("logcat",) and "-d" in args:
            return next(responses)
        return ""

    monkeypatch.setattr(bench, "adb", fake_adb)
    out = bench.run_once(device=dev, local_onnx=tmp_path / "m.onnx")
    assert out == {"avg_ms": 1.0, "run_id": "run1234567"}

    timeout_bench = AndroidOrtBench(ToolsConfig(), OrtAndroidBenchConfig(enabled=True, clear_logcat=True, poll_interval_sec=0.0, timeout_sec=0))
    monkeypatch.setattr(timeout_bench, "is_device_ready", lambda _d: True)
    monkeypatch.setattr(timeout_bench, "adb", lambda *_a: "")
    times = iter([100.0, 101.0])
    monkeypatch.setattr("xtrim.android_ort_bench.time.time", lambda: next(times))
    with pytest.raises(RuntimeError, match="Timeout waiting"):
        timeout_bench.run_once(device=dev, local_onnx=tmp_path / "m.onnx")


def test_adb_demo_saves_ppm_and_runs_once(tmp_path, monkeypatch):
    img = tmp_path / "demo.png"
    Image.new("RGB", (2, 1), color=(10, 20, 30)).save(img)
    ppm = tmp_path / "out" / "demo.ppm"
    assert save_ppm_rgb(img, ppm) == (2, 1)
    assert ppm.read_bytes().startswith(b"P6\n2 1\n255\n")

    binary = tmp_path / "xtrim_yolo_detect"
    binary.write_text("bin", encoding="utf-8")
    demo = AdbYoloDemo(ToolsConfig(yolo_detect_local=str(binary)))
    dev = DeviceConfig(name="phone", serial="s")
    calls = []

    def fake_adb(_serial: str, *args: str) -> str:
        calls.append(args)
        if args == ("get-state",):
            return "device"
        if args[:1] == ("shell",) and len(args) == 2 and args[1].startswith("cd /data/local/tmp"):
            return "detections"
        return ""

    monkeypatch.setattr(demo, "adb", fake_adb)
    out = demo.run_once(
        device=dev,
        demo_cfg=AndroidDemoConfig(enabled=True, sample_image=str(img), imgsz=320),
        ncnn_param=tmp_path / "m.param",
        ncnn_bin=tmp_path / "m.bin",
        run_dir=tmp_path / "run",
    )

    assert out == "detections"
    log = (tmp_path / "run" / "android_demo" / "phone.log").read_text(encoding="utf-8")
    assert "ppm_wh=2x1" in log
    assert any(args[:1] == ("push",) for args in calls)


def test_adb_demo_failure_branches(tmp_path, monkeypatch):
    dev = DeviceConfig(name="phone", serial="s")
    demo = AdbYoloDemo(ToolsConfig(yolo_detect_local=str(tmp_path / "missing")))
    monkeypatch.setattr(demo, "is_ready", lambda _d: False)
    with pytest.raises(RuntimeError, match="Device not ready"):
        demo.run_once(device=dev, demo_cfg=AndroidDemoConfig(), ncnn_param=tmp_path / "p", ncnn_bin=tmp_path / "b", run_dir=tmp_path)

    with pytest.raises(RuntimeError, match="binary not found"):
        demo.ensure_binary(dev)
