from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import yaml

from .types import (
    ToolsConfig,
    DeviceConfig,
    TrainConfig,
    ExportConfig,
    PTQConfig,
    ModelConfig,
    TrimConfig,
    StagedPruningConfig,
    LatencyConfig,
    EvalConfig,
    KDConfig,
    OnnxPTQConfig,
    QATConfig,
    AndroidDemoConfig,
    SearchConfig,
    AndroidAppBenchConfig,
    OperatorChoiceConfig,
    LatencyLUTConfig,
    GumbelChoiceConfig,
    LowRankConfig,
    DilatedConfig,
    OrtAndroidBenchConfig,
    BenchmarkProfileConfig,
)


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def parse_config(cfg: Dict[str, Any]) -> Tuple[
    ToolsConfig,
    List[DeviceConfig],
    TrainConfig,
    ExportConfig,
    PTQConfig,
    Dict[str, list],
    ModelConfig,
    TrimConfig,
    LatencyConfig,
    EvalConfig,
    KDConfig,
    OnnxPTQConfig,
    QATConfig,
    AndroidDemoConfig,
    SearchConfig,
    AndroidAppBenchConfig,
    OrtAndroidBenchConfig,
    OperatorChoiceConfig,
    Dict[str, str],
    LatencyLUTConfig,
    GumbelChoiceConfig,
    LowRankConfig,
    List[BenchmarkProfileConfig],
    StagedPruningConfig,
    DilatedConfig,
]:
    tools = ToolsConfig(**cfg.get("tools", {}))
    devices = [DeviceConfig(**d) for d in cfg.get("devices", [])]
    train_cfg = TrainConfig(**cfg.get("train", {}))
    export_cfg = ExportConfig(**cfg.get("export", {}))
    ptq_cfg = PTQConfig(**cfg.get("ptq", {}))
    search_space = cfg.get("search_space", {})
    model_cfg = ModelConfig(**cfg.get("model", {}))
    trim_cfg = TrimConfig(**cfg.get("trim", {}))
    latency_cfg = LatencyConfig(**cfg.get("latency", {}))
    eval_cfg = EvalConfig(**cfg.get("eval", {}))
    kd_cfg = KDConfig(**cfg.get("kd", {}))
    onnx_ptq_cfg = OnnxPTQConfig(**cfg.get("onnx_ptq", {}))
    qat_cfg = QATConfig(**cfg.get("qat", {}))
    android_demo_cfg = AndroidDemoConfig(**cfg.get("android_demo", {}))
    search_cfg = SearchConfig(**cfg.get("search", {}))
    android_app_bench_cfg = AndroidAppBenchConfig(**cfg.get("android_app_bench", {}))

    op_choice_raw = dict(cfg.get("operator_choice", {}))
    op_choice_plan: Dict[str, str] = op_choice_raw.pop("plan", {}) or {}
    op_choice_cfg = OperatorChoiceConfig(**op_choice_raw)

    lut_cfg = LatencyLUTConfig(**cfg.get("latency_lut", {}))
    gumbel_cfg = GumbelChoiceConfig(**cfg.get("gumbel_choice", {}))
    lowrank_cfg = LowRankConfig(**cfg.get("lowrank", {}))

    staged_raw = dict(cfg.get("staged_pruning", {}) or {})
    if "milestones" in staged_raw:
        staged_raw["milestones"] = tuple(staged_raw["milestones"] or ())
    staged_pruning_cfg = StagedPruningConfig(**staged_raw)

    ort_android_bench_cfg = OrtAndroidBenchConfig(**cfg.get("ort_android_bench", {}))

    benchmark_profiles: List[BenchmarkProfileConfig] = []
    for i, profile_raw in enumerate(cfg.get("benchmark_profiles", []) or []):
        raw = dict(profile_raw or {})
        # Historically benchmark_profiles used either devices: [...] or
        # device: "phone1" to select Android devices by name/serial.
        # New NCNN runtime selection should be done with runtime: ncnn_cpu|ncnn_vulkan
        # or gpu_device. If device is numeric, keep it as an override field.
        if "devices" in raw:
            devices_value = raw.pop("devices")
        elif "device_names" in raw:
            devices_value = raw.pop("device_names")
        elif "device" in raw and not isinstance(raw.get("device"), (int, float)):
            devices_value = raw.pop("device")
        else:
            devices_value = ()
        if devices_value is None or devices_value == "":
            device_names = ()
        elif isinstance(devices_value, str):
            device_names = (devices_value,)
        else:
            device_names = tuple(devices_value)
        raw["device_names"] = device_names
        if not raw.get("name"):
            raw["name"] = str(raw.get("backend") or f"profile_{i + 1}")
        benchmark_profiles.append(BenchmarkProfileConfig(**raw))

    dilated_raw = dict(cfg.get("dilated", {}))
    if "rates" in dilated_raw:
        dilated_raw["rates"] = tuple(dilated_raw["rates"])
    if "target_layers" in dilated_raw:
        dilated_raw["target_layers"] = tuple(dilated_raw["target_layers"])
    dilated_cfg = DilatedConfig(**dilated_raw)

    return (
        tools,
        devices,
        train_cfg,
        export_cfg,
        ptq_cfg,
        search_space,
        model_cfg,
        trim_cfg,
        latency_cfg,
        eval_cfg,
        kd_cfg,
        onnx_ptq_cfg,
        qat_cfg,
        android_demo_cfg,
        search_cfg,
        android_app_bench_cfg,
        ort_android_bench_cfg,
        op_choice_cfg,
        op_choice_plan,
        lut_cfg,
        gumbel_cfg,
        lowrank_cfg,
        benchmark_profiles,
        staged_pruning_cfg,
        dilated_cfg,
    )
