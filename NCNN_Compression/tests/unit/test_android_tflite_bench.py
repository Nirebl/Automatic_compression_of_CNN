from __future__ import annotations

import types
from pathlib import Path

import pytest

from xtrim.android_tflite_bench import AndroidTfliteBench
from xtrim.types import DeviceConfig, OrtAndroidBenchConfig, ToolsConfig

pytestmark = pytest.mark.unit


def test_tflite_extract_last_json_and_device_ready(monkeypatch):
    bench = AndroidTfliteBench(ToolsConfig(), OrtAndroidBenchConfig(), delegate="GPU")
    assert bench.delegate == "gpu"
    assert bench._extract_last_json('noise {bad} {"a": 1} tail {"a": 2}') == {"a": 2}
    assert bench._extract_last_json("plain") is None

    dev = DeviceConfig(name="p", serial="s")
    monkeypatch.setattr(bench, "adb", lambda _s, *args: "device\n" if args == ("get-state",) else "")
    assert bench.is_device_ready(dev) is True
    monkeypatch.setattr(bench, "adb", lambda *_a: (_ for _ in ()).throw(RuntimeError("adb")))
    assert bench.is_device_ready(dev) is False


def test_tflite_run_once_success_with_dataset_push(tmp_path, monkeypatch):
    cfg = OrtAndroidBenchConfig(
        enabled=True,
        clear_logcat=True,
        push_dataset_images=True,
        poll_interval_sec=0.0,
        timeout_sec=5,
        remote_dir="/data/local/tmp/xtrim",
        provider="cpu",
    )
    bench = AndroidTfliteBench(ToolsConfig(), cfg, delegate="gpu")
    dev = DeviceConfig(name="phone", serial="serial", cooling_down=0)
    model = tmp_path / "model_fp16.tflite"
    model.write_bytes(b"model")
    images = tmp_path / "images"
    images.mkdir()
    (images / "b.jpg").write_bytes(b"b")
    (images / "a.png").write_bytes(b"a")
    image_list = tmp_path / "list.txt"
    image_list.write_text("a\nb\n", encoding="utf-8")
    calls = []

    monkeypatch.setattr("xtrim.android_tflite_bench.uuid.uuid4", lambda: types.SimpleNamespace(hex="run1234567ffff"))
    monkeypatch.setattr("xtrim.android_tflite_bench.time.sleep", lambda *_: None)

    dumps = iter([
        '{"avg_ms": 100, "run_id": "old"}',
        '{"avg_ms": 12.5, "run_id": "run1234567"}',
    ])

    def fake_adb(_serial: str, *args: str) -> str:
        calls.append(args)
        if args == ("get-state",):
            return "device\n"
        if args[:1] == ("logcat",) and "-d" in args:
            return next(dumps)
        return ""

    monkeypatch.setattr(bench, "adb", fake_adb)
    out = bench.run_once(
        device=dev,
        local_tflite=model,
        local_images_dir=images,
        local_image_list=image_list,
        remote_images_dir="/remote/images",
        remote_image_list="/remote/list.txt",
        dataset_image_count=2,
    )

    assert out["avg_ms"] == 12.5
    assert out["backend"] == "tflite"
    assert out["delegate"] == "gpu"
    assert out["runtime"] == "tflite_gpu"
    assert out["device_name"] == "phone"
    start_cmd = next(c for c in calls if c[:4] == ("shell", "am", "start", "-W"))
    assert "--es" in start_cmd and "delegate" in start_cmd and "gpu" in start_cmd
    assert "use_pushed_images" in start_cmd
    assert any(c[:2] == ("push", str(model)) for c in calls)
    assert any(c[:2] == ("push", str(images / "a.png")) for c in calls)
    assert any(c[:2] == ("logcat", "-c") for c in calls)


def test_tflite_failure_branches(tmp_path, monkeypatch):
    dev = DeviceConfig(name="phone", serial="s", cooling_down=0)
    disabled = AndroidTfliteBench(ToolsConfig(), OrtAndroidBenchConfig(enabled=False))
    with pytest.raises(RuntimeError, match="enabled=False"):
        disabled.run_once(device=dev, local_tflite=tmp_path / "m.tflite")

    bench = AndroidTfliteBench(ToolsConfig(), OrtAndroidBenchConfig(enabled=True, poll_interval_sec=0.0))
    monkeypatch.setattr(bench, "is_device_ready", lambda _d: False)
    with pytest.raises(RuntimeError, match="Device not ready"):
        bench.run_once(device=dev, local_tflite=tmp_path / "m.tflite")

    cfg = OrtAndroidBenchConfig(enabled=True, push_dataset_images=True, poll_interval_sec=0.0)
    bench = AndroidTfliteBench(ToolsConfig(), cfg)
    monkeypatch.setattr(bench, "is_device_ready", lambda _d: True)
    monkeypatch.setattr(bench, "push_model", lambda *_a, **_k: "/remote/model.tflite")
    monkeypatch.setattr(bench, "force_stop", lambda *_a, **_k: None)
    with pytest.raises(RuntimeError, match="dataset subset paths"):
        bench.run_once(device=dev, local_tflite=tmp_path / "m.tflite")


def test_tflite_run_once_timeout(tmp_path, monkeypatch):
    dev = DeviceConfig(name="phone", serial="s", cooling_down=0)
    cfg = OrtAndroidBenchConfig(enabled=True, clear_logcat=False, poll_interval_sec=0.0, timeout_sec=0)
    bench = AndroidTfliteBench(ToolsConfig(), cfg)
    monkeypatch.setattr(bench, "is_device_ready", lambda _d: True)
    monkeypatch.setattr(bench, "push_model", lambda *_a, **_k: "/remote/model.tflite")
    monkeypatch.setattr(bench, "force_stop", lambda *_a, **_k: None)
    monkeypatch.setattr(bench, "adb", lambda *_a: "")
    times = iter([0.0, 10.0])
    monkeypatch.setattr("xtrim.android_tflite_bench.time.time", lambda: next(times))
    monkeypatch.setattr("xtrim.android_tflite_bench.time.sleep", lambda *_: None)

    with pytest.raises(RuntimeError, match="Timeout waiting"):
        bench.run_once(device=dev, local_tflite=tmp_path / "m.tflite")


def test_tflite_push_model_and_dataset_helpers_ignore_chmod_errors(tmp_path, monkeypatch):
    cfg = OrtAndroidBenchConfig(remote_dir="/remote")
    bench = AndroidTfliteBench(ToolsConfig(), cfg)
    dev = DeviceConfig(name="phone", serial="s")
    model = tmp_path / "m"
    model.write_bytes(b"x")
    images = tmp_path / "imgs"
    images.mkdir()
    (images / "one.jpg").write_bytes(b"1")
    (images / "note.txt").write_text("also pushed by helper", encoding="utf-8")
    image_list = tmp_path / "list.txt"
    image_list.write_text("x", encoding="utf-8")
    calls = []

    def fake_adb(_serial, *args):
        calls.append(args)
        if args[:2] == ("shell", "chmod") or (args[:1] == ("shell",) and "chmod" in args[-1]):
            raise RuntimeError("chmod denied")
        return ""

    monkeypatch.setattr(bench, "adb", fake_adb)
    assert bench.push_model(dev, model) == "/remote/model.tflite"
    rdir, rlist = bench.push_dataset_subset(
        dev,
        local_images_dir=images,
        local_image_list=image_list,
        remote_images_dir="/r/imgs",
        remote_image_list="/r/list.txt",
    )
    assert (rdir, rlist) == ("/r/imgs", "/r/list.txt")
    assert any(c[:2] == ("push", str(images / "one.jpg")) for c in calls)
    assert any(c[:2] == ("push", str(image_list)) for c in calls)
