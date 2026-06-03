from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Callable

import pytest
import torch
import torch.nn as nn

from xtrim.orchestrator import XTrimOrchestrator
from xtrim.types import (
    AndroidAppBenchConfig,
    AndroidDemoConfig,
    CandidateConfig,
    DeviceConfig,
    ExportConfig,
    LatencyConfig,
    Metrics,
    OnnxPTQConfig,
    OrtAndroidBenchConfig,
    PTQConfig,
    QATConfig,
    SearchConfig,
    StagedPruningConfig,
    ToolsConfig,
    TrainConfig,
    TrimConfig,
    HistoryItem,
)


@pytest.fixture
def make_history_item() -> Callable[..., HistoryItem]:
    def _make(
        *,
        tag: str = "c",
        acc: float = 0.5,
        size: int = 100,
        latency: dict[str, float] | None = None,
        failed: bool = False,
        baseline: bool = False,
        width: float = 1.0,
        prune: float = 0.0,
        rank: int = 0,
        sparse: float = 0.0,
        extra: dict | None = None,
    ) -> HistoryItem:
        payload = dict(extra or {})
        if failed:
            payload["failed"] = True
        if baseline:
            payload["is_reference_baseline"] = True
        return HistoryItem(
            candidate=CandidateConfig(
                width_mult=width,
                prune_ratio=prune,
                lowrank_rank=rank,
                sparse_1x1=sparse,
                tag=tag,
            ),
            metrics=Metrics(
                acc=acc,
                size_bytes=size,
                latency_ms=latency if latency is not None else {"phone": 10.0},
            ),
            artifacts_dir="artifacts",
            extra=payload,
        )

    return _make


class TinyConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, k: int = 1):
        super().__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, k, padding=k // 2, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.bn(self.conv(x)))


class TinyYoloLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.ModuleList(
            [
                TinyConvBlock(16, 16, 1),
                TinyConvBlock(16, 16, 3),
                nn.Sequential(nn.Conv2d(16, 16, 1), nn.Conv2d(16, 16, 3, padding=1)),
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.model:
            x = layer(x)
        return x


@pytest.fixture
def tiny_yolo_like() -> TinyYoloLike:
    return TinyYoloLike()


@pytest.fixture
def fake_student_factory():
    class Student:
        def __init__(self, cand: CandidateConfig):
            self.candidate = cand
            self.torch_model = None

    def _build(cand: CandidateConfig) -> Student:
        return Student(cand)

    return _build


@pytest.fixture
def fake_export_factory():
    def _factory(_student, _cfg):
        def _export(path: Path) -> None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"fake-onnx")
        return _export

    return _factory


@pytest.fixture
def make_orchestrator(tmp_path, fake_student_factory, fake_export_factory):
    def _make(
        *,
        out_root: Path | None = None,
        devices: list[DeviceConfig] | None = None,
        latency_cfg: LatencyConfig | None = None,
        onnx_ptq_cfg: OnnxPTQConfig | None = None,
        qat_cfg: QATConfig | None = None,
        search_cfg: SearchConfig | None = None,
        search_space: dict | None = None,
        eval_acc_fn=None,
        eval_metrics_fn=None,
        eval_exported_onnx_fn=None,
        eval_exported_onnx_metrics_fn=None,
        quantize_onnx_fn=None,
        export_tflite_int8_fn_factory=None,
        finetune_qat_fn=None,
        build_candidate_fn=None,
        apply_pruning_stage_fn=None,
        extract_pruning_architecture_fn=None,
        finetune_fn=None,
        trim_cfg: TrimConfig | None = None,
        staged_pruning_cfg: StagedPruningConfig | None = None,
    ) -> XTrimOrchestrator:
        return XTrimOrchestrator(
            out_root=out_root or tmp_path / "out",
            tools=ToolsConfig(),
            devices=devices or [],
            train_cfg=TrainConfig(),
            export_cfg=ExportConfig(),
            ptq_cfg=PTQConfig(),
            latency_cfg=latency_cfg or LatencyConfig(backend="ort_android", use_cache=False),
            onnx_ptq_cfg=onnx_ptq_cfg or OnnxPTQConfig(),
            qat_cfg=qat_cfg or QATConfig(),
            android_demo_cfg=AndroidDemoConfig(),
            search_cfg=search_cfg or SearchConfig(method="grid", init_random=0),
            search_space=search_space or {"width_mult": [1.0], "prune_ratio": [0.2], "lowrank_rank": [0], "sparse_1x1": [0.0]},
            trim_cfg=trim_cfg,
            staged_pruning_cfg=staged_pruning_cfg,
            build_candidate_fn=build_candidate_fn or fake_student_factory,
            apply_pruning_stage_fn=apply_pruning_stage_fn,
            extract_pruning_architecture_fn=extract_pruning_architecture_fn,
            warmstart_fn=lambda _student: None,
            finetune_fn=finetune_fn or (lambda _student, _cfg: {"epochs": 0}),
            finetune_qat_fn=finetune_qat_fn,
            eval_acc_fn=eval_acc_fn or (lambda student: 0.8 if student.candidate.tag == "baseline_raw" else 0.7),
            eval_metrics_fn=eval_metrics_fn,
            export_onnx_fn_factory=fake_export_factory,
            export_tflite_int8_fn_factory=export_tflite_int8_fn_factory,
            eval_exported_onnx_fn=eval_exported_onnx_fn,
            eval_exported_onnx_metrics_fn=eval_exported_onnx_metrics_fn,
            quantize_onnx_fn=quantize_onnx_fn,
            save_student_pt_fn=None,
            android_app_bench_cfg=AndroidAppBenchConfig(),
            ort_android_bench_cfg=OrtAndroidBenchConfig(),
        )

    return _make
