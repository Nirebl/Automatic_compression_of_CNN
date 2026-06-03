from __future__ import annotations

import json
from pathlib import Path

import pytest

from xtrim.results_table import load_history_jsonl
from xtrim.types import OnnxPTQConfig, QATConfig


pytestmark = pytest.mark.integration


def test_orchestrator_run_creates_baseline_candidate_history_and_pareto(make_orchestrator):
    orch = make_orchestrator()

    history = orch.run(max_candidates=1)

    assert len(history) == 2
    assert history[0].candidate.tag == "baseline_raw"
    assert history[0].extra["is_reference_baseline"] is True
    assert history[1].extra["finetune_logs"] == {"epochs": 0}
    assert orch.history_path.exists()
    assert (orch.out_root / "pareto.json").exists()
    assert list(orch.history_archive_dir.glob("*.jsonl"))

    reloaded = load_history_jsonl(orch.history_path)
    assert [h.candidate.tag for h in reloaded] == [h.candidate.tag for h in history]


def test_orchestrator_qat_path_uses_int8_after_qat(make_orchestrator):
    def eval_onnx(path: Path) -> float:
        name = path.name
        if "int8_before_qat" in name:
            return 0.60
        if "int8_after_qat" in name:
            return 0.72
        if "model_qat" in name:
            return 0.74
        return 0.75

    def quantize(src: Path, dst: Path, _run_dir: Path) -> Path:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_bytes(src.read_bytes() + b"-int8")
        return dst

    orch = make_orchestrator(
        onnx_ptq_cfg=OnnxPTQConfig(enabled=True),
        qat_cfg=QATConfig(enabled=True, acc_drop_threshold=0.01),
        eval_exported_onnx_fn=eval_onnx,
        quantize_onnx_fn=quantize,
        finetune_qat_fn=lambda _student, _cfg: {"qat": "ok"},
    )

    history = orch.run(max_candidates=1)
    candidate = history[1]

    assert candidate.extra["qat_triggered"] is True
    assert candidate.extra["acc_onnx_int8"] == 0.60
    assert candidate.extra["acc_onnx_int8_after_qat"] == 0.72
    assert candidate.extra["qat_logs"] == {"qat": "ok"}
    assert candidate.extra["deploy_onnx_kind"] == "int8_after_qat"


def test_orchestrator_resume_does_not_duplicate_reference_baseline(make_orchestrator):
    orch1 = make_orchestrator()
    first = orch1.run(max_candidates=0)
    assert len(first) == 1

    orch2 = make_orchestrator(out_root=orch1.out_root)
    second = orch2.run(max_candidates=0)

    assert len(second) == 1
    assert second[0].candidate.tag == "baseline_raw"


def test_orchestrator_exports_optional_tflite_int8_artifact(make_orchestrator):
    from xtrim.types import ExportConfig

    def tflite_factory(_student, _cfg):
        def _export(path: Path) -> Path:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-tflite-int8")
            return path
        return _export

    orch = make_orchestrator(
        export_tflite_int8_fn_factory=tflite_factory,
    )
    orch.export_cfg = ExportConfig(tflite=True, tflite_int8=True)

    history = orch.run(max_candidates=1)
    baseline = history[0]
    candidate = history[1]

    assert baseline.extra["tflite_int8_export"] == "ok"
    assert candidate.extra["tflite_int8_export"] == "ok"
    assert Path(candidate.extra["tflite_int8_path"]).exists()
    assert candidate.extra["deploy_size_bytes_by_backend"]["tflite_int8"] == len(b"fake-tflite-int8")
