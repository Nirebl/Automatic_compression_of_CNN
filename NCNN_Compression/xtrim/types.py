from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class DeviceConfig:
    name: str
    serial: str
    threads: int = 4
    loops: int = 50
    powersave: int = 2
    gpu_device: int = -1
    cooling_down: int = 1


@dataclass(frozen=True)
class TrainConfig:
    short_epochs: int = 0
    lr: float = 1e-4
    kd_enabled: bool = False
    fake_quant: bool = False
    seed: int = 42


@dataclass(frozen=True)
class ExportConfig:
    opset: int = 12
    dynamo: bool = False
    bench_shape: str = "[640,640,3]"


@dataclass(frozen=True)
class PTQConfig:
    enabled: bool = False
    imagelist: str = "calib.txt"
    mean: str = "0,0,0"
    norm: str = "1,1,1"
    shape: str = "640,640,3"
    pixel: str = "BGR"
    thread: int = 8
    method: str = "kl"


@dataclass(frozen=True)
class ToolsConfig:
    adb: str = "adb"
    onnx2ncnn: str = "onnx2ncnn"
    pnnx: str = "pnnx"
    ncnnoptimize: str = "ncnnoptimize"
    ncnn2table: str = "ncnn2table"
    ncnn2int8: str = "ncnn2int8"
    benchncnn_local: str = "benchncnn"
    yolo_detect_local: str = "android_native/build-android/bin/xtrim_yolo_detect"


@dataclass(frozen=True)
class LowRankConfig:
    enabled: bool = False
    energy_threshold: float = 0.0
    min_channels: int = 32
    exclude_head: bool = True
    exclude_stem: bool = True
    max_layers: int = 0
    bn_recalib_batches: int = 20
    finetune_epochs: int = 0
    lowrank_1x1: bool = False


@dataclass(frozen=True)
class Sparse1x1Config:
    enabled: bool = False
    sparsity: float = 0.5
    method: str = "l1"
    min_channels: int = 16


@dataclass(frozen=True)
class GumbelChoiceConfig:
    enabled: bool = False
    tau_start: float = 5.0
    tau_end: float = 0.5
    tau_schedule: str = "exp"
    hard: bool = False
    lowrank_rank: int = 8
    sparse_sparsity: float = 0.5
    min_channels: int = 16


@dataclass(frozen=True)
class LatencyLUTConfig:
    enabled: bool = False
    lut_path: str = "assets/lut_sample.json"
    budget_ms: float = 15.0
    lambda_lat: float = 0.05
    macs_per_ms: float = 100_000_000.0
    log_every_n_batches: int = 50


@dataclass(frozen=True)
class OperatorChoiceConfig:
    enabled: bool = False
    default: str = "dense"
    allow_sparse: bool = True
    allow_lowrank: bool = True
    lowrank_rank: int = 8
    sparse_sparsity: float = 0.5
    sparse_method: str = "l1"
    min_channels: int = 16


@dataclass(frozen=True)
class CandidateConfig:
    width_mult: float = 1.0
    prune_ratio: float = 0.0
    lowrank_rank: int = 0
    sparse_1x1: float = 0.0
    tag: str = ""


@dataclass
class Metrics:
    acc: float
    size_bytes: int
    latency_ms: Dict[str, float]


@dataclass
class HistoryItem:
    candidate: CandidateConfig
    metrics: Metrics
    artifacts_dir: str
    extra: Dict[str, Any] = dataclasses.field(default_factory=dict)


@dataclass(frozen=True)
class NcnnModelPaths:
    param: Path
    bin: Path


@dataclass(frozen=True)
class ModelConfig:
    weights: str = "yolov8n.pt"
    data: str = "coco128.yaml"
    imgsz: int = 640
    device: str = "cpu"
    task: str = "detect"


@dataclass(frozen=True)
class TrimConfig:
    channel_round: int = 8
    min_channels: int = 8
    exclude_head: bool = True
    exclude_name_regex: Optional[str] = None
    strategy: str = "layerwise"
    max_prune_per_layer: Optional[float] = None
    protect_last_n: int = 0


@dataclass(frozen=True)
class LatencyConfig:
    use_cache: bool = True
    force_rebench: bool = False
    cache_file: str = "bench_cache.json"
    aggregate: str = "avg"
    backend: str = "benchncnn"
    repeats: int = 1
    scalar_alpha: float = 0.0
    scalar_beta: float = 0.0


@dataclass(frozen=True)
class EvalConfig:
    conf: float = 0.001
    iou: float = 0.6
    max_det: int = 300
    half: bool = False
    dnn: bool = False
    rect: bool = False
    split: str = "val"
    batch: int = 16
    workers: int = 8
    verbose: bool = False
    plots: bool = False
    augment: bool = False
    agnostic_nms: bool = False


@dataclass(frozen=True)
class KDConfig:
    enabled: bool = False
    teacher: Optional[str] = None
    num_feature_layers: int = 3
    lambda_feat: float = 1.0
    lambda_head: float = 0.2
    lambda_bn: float = 1e-5
    batch: int = 8
    workers: int = 4
    max_train_batches: int = 0


@dataclass(frozen=True)
class OnnxPTQConfig:
    enabled: bool = False
    calib_split: str = "train"
    calib_max_images: int = 256
    per_channel: bool = True
    quant_format: str = "qdq"
    activation_type: str = "uint8"
    weight_type: str = "int8"
    calibrate_method: str = "minmax"


@dataclass(frozen=True)
class QATConfig:
    enabled: bool = False
    acc_drop_threshold: float = 0.01
    epochs: int = 1
    lr: float = 2e-5
    max_train_batches: int = 50
    bits_w: int = 8
    bits_a: int = 8


@dataclass(frozen=True)
class DilatedConfig:
    enabled: bool = False
    rates: tuple = (1, 2)
    target_layers: tuple = ()
    target_n_blocks: int = 4
    exclude_head: bool = True


@dataclass(frozen=True)
class AndroidDemoConfig:
    enabled: bool = False
    sample_image: str = "assets/demo.jpg"
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.6
    max_det: int = 100


@dataclass(frozen=True)
class SearchConfig:
    method: str = "grid"
    seed: int = 42
    init_random: int = 6
    population: int = 24
    offspring: int = 12
    mutation_prob: float = 0.35
    crossover_prob: float = 0.9
    tournament_k: int = 2


@dataclass(frozen=True)
class AndroidAppBenchConfig:
    enabled: bool = False
    package: str = "com.example.testyolo"
    activity: str = ".CliBenchActivity"
    dataset: str = "coco"
    imgsz: int = 640
    loops: int = 50
    warmup: int = 10
    threads: int = 4
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 100
    optimized: bool = True
    result_tag: str = "XTRIM_RESULT"
    timeout_sec: int = 180
    poll_interval_sec: float = 0.6
    clear_logcat: bool = True
    remote_dir: str = "/data/local/tmp"
