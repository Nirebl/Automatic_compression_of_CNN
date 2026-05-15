from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch.nn as nn

from xtrim.types import AndroidDemoConfig, DeviceConfig, LatencyConfig, NcnnModelPaths, OnnxPTQConfig, PTQConfig

pytestmark = pytest.mark.integration


def _write_ncnn_pair(root: Path, stem: str = "m") -> NcnnModelPaths:
    p = root / f"{stem}.param"
    b = root / f"{stem}.bin"
    p.write_text("param", encoding="utf-8")
    b.write_bytes(b"bin")
    return NcnnModelPaths(p, b)


def test_orchestrator_backend_helpers_and_error_paths(tmp_path, make_orchestrator, monkeypatch):
    dev = DeviceConfig(name="phone", serial="s")
    orch = make_orchestrator(devices=[dev], latency_cfg=LatencyConfig(backend="ort_android", use_cache=True))
    onnx = tmp_path / "m.onnx"
    onnx.write_bytes(b"onnx")

    assert orch._latency_aggregate({}) == float("inf")
    orch.latency_cfg = LatencyConfig(backend="ort_android", aggregate="max")
    assert orch._latency_aggregate({"a": 1.0, "b": 2.0}) == 2.0
    assert len(orch._hash_file_model(onnx)) == 32

    calls = []
    monkeypatch.setattr(orch.android_ort, "run_once", lambda **_kw: calls.append(1) or {"mean_ms": 3.5})
    first = orch._bench_devices_with_android_ort(onnx, tmp_path / "run1")
    second = orch._bench_devices_with_android_ort(onnx, tmp_path / "run2")
    assert first == {"phone": 3.5}
    assert second == {"phone": 3.5}
    assert len(calls) == 1

    orch_bad = make_orchestrator(devices=[dev], latency_cfg=LatencyConfig(backend="ort_android", use_cache=False))
    monkeypatch.setattr(orch_bad.android_ort, "run_once", lambda **_kw: {"x": 1})
    with pytest.raises(RuntimeError, match="no avg latency"):
        orch_bad._bench_devices_with_android_ort(onnx, tmp_path / "bad")

    with pytest.raises(RuntimeError, match="onnx_model is required"):
        orch._bench_latency(None, None, tmp_path / "x")
    orch.latency_cfg = LatencyConfig(backend="android_app")
    with pytest.raises(RuntimeError, match="NCNN model is required"):
        orch._bench_latency(None, None, tmp_path / "x")
    orch.latency_cfg = LatencyConfig(backend="benchncnn")
    with pytest.raises(RuntimeError, match="this backend"):
        orch._bench_latency(None, None, tmp_path / "x")


def test_orchestrator_android_app_bench_cache_and_missing_avg(tmp_path, make_orchestrator, monkeypatch):
    dev = DeviceConfig(name="phone", serial="s")
    orch = make_orchestrator(devices=[dev], latency_cfg=LatencyConfig(backend="android_app", use_cache=True))
    pair = _write_ncnn_pair(tmp_path)
    calls = []
    monkeypatch.setattr(orch.android_app, "run_once", lambda **_kw: calls.append(1) or {"avg_ms": 4.0})

    assert orch._bench_devices_with_android_app(pair.param, pair.bin, tmp_path / "a") == {"phone": 4.0}
    assert orch._bench_devices_with_android_app(pair.param, pair.bin, tmp_path / "b") == {"phone": 4.0}
    assert len(calls) == 1

    orch_bad = make_orchestrator(devices=[dev], latency_cfg=LatencyConfig(backend="android_app", use_cache=False))
    monkeypatch.setattr(orch_bad.android_app, "run_once", lambda **_kw: {})
    with pytest.raises(RuntimeError, match="missing avg_ms"):
        orch_bad._bench_devices_with_android_app(pair.param, pair.bin, tmp_path / "c")


def test_reference_and_candidate_ncnn_paths_with_demo_and_ptq(tmp_path, make_orchestrator, monkeypatch):
    dev_ok = DeviceConfig(name="ok", serial="1")
    dev_fail = DeviceConfig(name="fail", serial="2")
    orch = make_orchestrator(
        devices=[dev_ok, dev_fail],
        latency_cfg=LatencyConfig(backend="benchncnn", use_cache=False),
    )
    orch.android_demo_cfg = AndroidDemoConfig(enabled=True)
    orch.ptq_cfg = PTQConfig(enabled=True)
    orch.build_candidate_fn = lambda cand: SimpleNamespace(candidate=cand, torch_model=nn.Conv2d(3, 3, 1))
    pair = _write_ncnn_pair(tmp_path, "n")
    monkeypatch.setattr(orch.converter, "pnnx_convert", lambda *_a, **_kw: pair)
    monkeypatch.setattr(orch.converter, "optimize", lambda *_a, **_kw: pair)
    monkeypatch.setattr(orch.converter, "ptq_int8", lambda *_a, **_kw: pair)
    monkeypatch.setattr(orch, "_bench_latency", lambda *_a, **_kw: {"ok": 1.0})

    def fake_demo(*, device, **_kw):
        if device.name == "fail":
            raise RuntimeError("demo bad")
        return "ok"
    monkeypatch.setattr(orch.demo, "run_once", fake_demo)

    baseline = orch._process_reference_baseline(tmp_path / "baseline")
    assert baseline.extra["ncnn_source"] == "pnnx"
    assert baseline.extra["deploy_backend"] == "ncnn"

    item = orch._process_candidate(orch.policy.next_candidate([baseline]), tmp_path / "cand")
    assert item.extra["ncnn_source"] == "ncnn_int8"
    assert item.extra["android_demo_ok"] == ["ok"]
    assert "fail" in item.extra["android_demo_fail"]


def test_run_filters_unready_devices_and_records_failed_candidate(make_orchestrator, monkeypatch):
    d1 = DeviceConfig(name="ready", serial="1")
    d2 = DeviceConfig(name="bad", serial="2")
    orch = make_orchestrator(devices=[d1, d2], latency_cfg=LatencyConfig(backend="benchncnn", use_cache=False))
    monkeypatch.setattr(orch.bench, "is_device_ready", lambda d: d.name == "ready")
    monkeypatch.setattr(orch.bench, "ensure_benchncnn", lambda *_a, **_kw: None)
    monkeypatch.setattr(orch, "_process_reference_baseline", lambda run_dir: __import__("xtrim.types", fromlist=["HistoryItem", "CandidateConfig", "Metrics"]).HistoryItem(
        candidate=__import__("xtrim.types", fromlist=["CandidateConfig"]).CandidateConfig(tag="baseline_raw"),
        metrics=__import__("xtrim.types", fromlist=["Metrics"]).Metrics(acc=0.8, size_bytes=1, latency_ms={}),
        artifacts_dir=str(run_dir),
        extra={"is_reference_baseline": True},
    ))
    monkeypatch.setattr(orch, "_process_candidate", lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("boom")))
    history = orch.run(max_candidates=1)
    assert [d.name for d in orch.devices] == ["ready"]
    assert history[-1].extra["failed"] is True
    assert (Path(history[-1].artifacts_dir) / "error.txt").exists()
