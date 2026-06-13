from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from xtrim.rebench_existing import (
    _append_history,
    _copy_source_history,
    _extract_candidate_name_from_dir,
    _find_existing_ncnn_pair,
    _find_existing_ncnn_source_onnx,
    _find_existing_onnx,
    _find_existing_tflite_models,
    _history_from_dirs,
    _is_failed_or_non_deploy,
    _json_safe,
    _load_history_jsonl,
    _path_from_extra,
    _score_path,
    rebench_existing,
)
from xtrim.types import CandidateConfig, HistoryItem, Metrics, NcnnModelPaths

pytestmark = pytest.mark.unit


def _hist(tag: str, artifacts_dir: Path, *, failed: bool = False, extra: dict | None = None) -> HistoryItem:
    data = dict(extra or {})
    if failed:
        data["failed"] = True
    return HistoryItem(
        candidate=CandidateConfig(width_mult=0.5, prune_ratio=0.8, lowrank_rank=0, sparse_1x1=0.0, tag=tag),
        metrics=Metrics(acc=0.7, size_bytes=1234, latency_ms={"old": 999.0}, precision=0.6, recall=0.5),
        artifacts_dir=str(artifacts_dir),
        extra=data,
    )


def test_rebench_history_jsonl_and_json_safe_roundtrip(tmp_path):
    history = tmp_path / "history.jsonl"
    item = _hist("w1.0_p0.0_r0_s0", tmp_path / "cand", extra={"path": tmp_path / "m.onnx"})

    assert _load_history_jsonl(history) == []
    _append_history(history, item)
    loaded = _load_history_jsonl(history)

    assert loaded[0].candidate.tag == item.candidate.tag
    assert loaded[0].metrics.precision == pytest.approx(0.6)
    assert _json_safe({"p": tmp_path, "items": [CandidateConfig(tag="x")]})["items"][0]["tag"] == "x"


def test_rebench_candidate_name_path_lookup_and_sorting(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    rel = source / "rel.onnx"
    rel.write_bytes(b"onnx")
    absolute = tmp_path / "absolute.onnx"
    absolute.write_bytes(b"onnx")

    assert _extract_candidate_name_from_dir(Path("001_baseline_raw")) == "baseline_raw"
    assert _extract_candidate_name_from_dir(Path("002_w0.5_p0.8_r0_s0")) == "w0.5_p0.8_r0_s0"
    assert _extract_candidate_name_from_dir(Path("misc")) is None
    assert _path_from_extra({"onnx_path": str(rel)}, "onnx_path", source) == rel.resolve()
    assert _path_from_extra({"onnx_path": str(absolute)}, "onnx_path", source) == absolute.resolve()
    assert _path_from_extra({}, "missing", source) is None

    poorer = tmp_path / "a" / "model.onnx"
    better = tmp_path / "deep" / "qat_int8" / "model.onnx"
    assert _score_path(better, ("qat_int8", "model")) < _score_path(poorer, ("qat_int8", "model"))


def test_rebench_finds_existing_artifacts_by_extra_and_priority(tmp_path):
    source = tmp_path / "source"
    cand = source / "001_w0.5_p0.8_r0_s0"
    (cand / "export").mkdir(parents=True)
    (cand / "deploy").mkdir()

    param = cand / "deploy" / "model_int8.param"
    binf = cand / "deploy" / "model_int8.bin"
    param.write_text("p", encoding="utf-8")
    binf.write_bytes(b"b")
    onnx = cand / "export" / "model.onnx"
    qdq = cand / "export" / "model_qat_int8.onnx"
    onnx.write_bytes(b"fp32")
    qdq.write_bytes(b"int8")
    int8 = cand / "model_int8.tflite"
    fp16 = cand / "model_fp16.tflite"
    int8.write_bytes(b"i")
    fp16.write_bytes(b"f")

    ncnn_param, ncnn_bin = _find_existing_ncnn_pair(cand, {}, source)
    assert ncnn_param == param.resolve()
    assert ncnn_bin == binf.resolve()
    assert _find_existing_onnx(cand, {}, source) == qdq.resolve()
    assert _find_existing_ncnn_source_onnx(cand, {}, source) == onnx.resolve()

    models = _find_existing_tflite_models(cand, {}, source)
    assert models["int8"] == int8.resolve()
    assert models["tflite_int8"] == int8.resolve()
    assert models["fp16"] == fp16.resolve()
    assert models["tflite_fp16"] == fp16.resolve()

    by_extra = _find_existing_tflite_models(cand, {"tflite_artifacts": {"custom": str(fp16)}}, source)
    assert by_extra["custom"] == fp16.resolve()


def test_rebench_history_from_dirs_filters_and_archive(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    (source / "001_baseline_raw").mkdir()
    (source / "002_w0.25_p0.75_r0_s0").mkdir()
    (source / "ignored").mkdir()

    items = _history_from_dirs(source)
    tags = [x.candidate.tag for x in items]
    assert tags == ["baseline_raw", "w0.25_p0.75_r0_s0"]
    assert items[0].extra["is_reference_baseline"] is True
    assert _is_failed_or_non_deploy(HistoryItem(CandidateConfig(tag="x"), Metrics(0, 0, {}), "", {"failed": True})) is True
    assert _is_failed_or_non_deploy(HistoryItem(CandidateConfig(tag="x"), Metrics(0, 0, {}), "", {"search_excluded": True})) is True
    assert _is_failed_or_non_deploy(items[0]) is False

    history = source / "history.jsonl"
    history.write_text("{}\n", encoding="utf-8")
    monkeypatch.setattr("xtrim.rebench_existing.time.time", lambda: 123.0)
    _copy_source_history(history, tmp_path / "archive")
    archived = tmp_path / "archive" / "source_history_123.jsonl"
    assert archived.read_text(encoding="utf-8") == "{}\n"


class _Cache:
    def __init__(self):
        self.saved = False

    def save(self):
        self.saved = True


class _FakeOrchestrator:
    def __init__(self, out_root: Path, *, fail_tags: set[str] | None = None, needs_ncnn: bool = False):
        self.out_root = out_root
        self.latency_cfg = SimpleNamespace(backend="tflite_android", use_cache=True, force_rebench=False)
        self.benchmark_profiles = [SimpleNamespace(name="p")]
        self.cache = _Cache()
        self.fail_tags = fail_tags or set()
        self.needs_ncnn = needs_ncnn
        self.prepared = False
        self.bench_calls = []

    def prepare_benchmark_devices(self):
        self.prepared = True

    def _needs_ncnn_artifact(self):
        return self.needs_ncnn

    def prepare_ncnn_from_existing_onnx(self, *, source_onnx: Path, run_dir: Path, source_label: str):
        param = run_dir / "generated.param"
        binf = run_dir / "generated.bin"
        param.write_text("p", encoding="utf-8")
        binf.write_bytes(b"b")
        return NcnnModelPaths(param=param, bin=binf), {"generated_from": source_label, "source_onnx": str(source_onnx)}

    def _benchmark_existing_deploy_model(self, *, ncnn_param, ncnn_bin, onnx_model, tflite_models, run_dir):
        tag = run_dir.name.split("_", 1)[1]
        self.bench_calls.append((ncnn_param, ncnn_bin, onnx_model, tflite_models, run_dir))
        if tag in self.fail_tags:
            raise RuntimeError("bench failed")
        assert onnx_model is not None
        return {"phone": 10.0, "gpu": 5.0}, {"bench_extra": True}, "tflite_fp16"

    def _latency_aggregate(self, latency_ms):
        return sum(latency_ms.values()) / len(latency_ms)

    def _scalarize(self, acc, latency, size):
        return acc - latency / 1000 - size / 1_000_000


def test_rebench_existing_success_failure_and_outputs(tmp_path, monkeypatch):
    source = tmp_path / "source"
    out = tmp_path / "out"
    source.mkdir()
    ok_dir = source / "ok_artifacts"
    fail_dir = source / "fail_artifacts"
    ok_dir.mkdir()
    fail_dir.mkdir()
    (ok_dir / "model.onnx").write_bytes(b"ok")
    (fail_dir / "model.onnx").write_bytes(b"fail")
    (ok_dir / "model_fp16.tflite").write_bytes(b"fp16")

    _append_history(source / "history.jsonl", _hist("ok", ok_dir))
    _append_history(source / "history.jsonl", _hist("bad name", fail_dir))
    old_history = out / "history.jsonl"
    out.mkdir()
    old_history.write_text("old\n", encoding="utf-8")

    monkeypatch.setattr("xtrim.rebench_existing.time.time", lambda: 1000.0)
    orch = _FakeOrchestrator(out, fail_tags={"bad_name"}, needs_ncnn=True)
    result = rebench_existing(orch, source_run_dir=source, out_root=out, latency_cfg=orch.latency_cfg)

    assert orch.prepared is True
    assert orch.latency_cfg.use_cache is False
    assert orch.latency_cfg.force_rebench is True
    assert len(result) == 2
    assert result[0].metrics.latency_ms == {"phone": 10.0, "gpu": 5.0}
    assert result[0].extra["rebench"] is True
    assert result[0].extra["latency_agg_ms"] == pytest.approx(7.5)
    assert result[0].extra["rebench_tflite_models"]["fp16"].endswith("model_fp16.tflite")
    assert result[1].extra["failed"] is True
    assert "bench failed" in result[1].extra["error"]
    assert (out / "history" / "previous_rebench_1000.jsonl").read_text(encoding="utf-8") == "old\n"
    assert (out / "rebench_results.jsonl").exists()
    assert (out / "pareto.json").exists()
    assert orch.cache.saved is True


def test_rebench_existing_uses_dirs_when_no_history_and_handles_empty(tmp_path):
    missing = tmp_path / "missing"
    with pytest.raises(FileNotFoundError):
        rebench_existing(_FakeOrchestrator(tmp_path / "out"), source_run_dir=missing, out_root=tmp_path / "out")

    empty = tmp_path / "empty"
    empty.mkdir()
    assert rebench_existing(_FakeOrchestrator(tmp_path / "out2"), source_run_dir=empty, out_root=tmp_path / "out2") == []

    source = tmp_path / "source"
    cand = source / "001_w0.5_p0.8_r0_s0"
    cand.mkdir(parents=True)
    (cand / "model.onnx").write_bytes(b"onnx")
    orch = _FakeOrchestrator(tmp_path / "out3")
    result = rebench_existing(orch, source_run_dir=source, out_root=tmp_path / "out3")
    assert result[0].candidate.tag == "w0.5_p0.8_r0_s0"
