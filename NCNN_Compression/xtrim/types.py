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
    # Явный runtime для Android/NCNN: ncnn_cpu или ncnn_vulkan.
    # Если не задан, используются старые поля device/gpu_device.
    runtime: Optional[str] = None
    # Старый короткий вариант для devices[]: -1 — CPU, 0 и выше — Vulkan GPU.
    # Не путать с model.device, который относится к PyTorch на рабочей станции.
    device: Optional[int] = None
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

    # Дополнительно сохранять NCNN-файлы рядом с ONNX, не меняя выбранную deploy-модель.
    ncnn: bool = False
    # Какой FP32 ONNX использовать для NCNN. По умолчанию не берем INT8/QDQ-графы.
    # fp32 — model.onnx; qat_fp32 — model_qat.onnx при наличии; deploy_fp32 — FP32-версия выбранной deploy-модели.
    ncnn_source: str = "qat_fp32"
    ncnn_optimize: bool = True
    # None сохраняет старое поведение для NCNN-only запусков. Для ONNX+NCNN обычно лучше явно ставить false.
    ncnn_int8: Optional[bool] = None
    # Если false, ошибка NCNN-экспорта пишется в историю, но не останавливает ORT-only эксперимент.
    ncnn_required: bool = False

    # Дополнительный экспорт в TFLite для Android-бенчмарков CPU/GPU. Выбранная ONNX deploy-модель при этом не меняется.
    tflite: bool = False
    tflite_int8: bool = False
    tflite_int8_required: bool = False
    tflite_int8_name: str = "model_int8.tflite"
    # FP32 TFLite для исходной модели, без случайного перевода baseline в INT8/FP16.
    tflite_fp32_name: str = "model_fp32.tflite"
    tflite_fp32_required: bool = False
    # FP16 TFLite сохраняется под стабильным именем для GPU-бенчмарка на Android.
    tflite_fp16: bool = False
    tflite_fp16_required: bool = False
    tflite_fp16_name: str = "model_fp16.tflite"
    # TFLite экспорт по умолчанию идет на CPU, чтобы не зависеть от GPU-настроек обучения.
    tflite_int8_device: str = "cpu"
    # Упрощение ONNX отключено, чтобы не тянуть лишние зависимости при TFLite-экспорте.
    tflite_int8_simplify: bool = False


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
    # Legacy NCNN demo. The current Android path uses the app through ADB.
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
    """Legacy LUT regularization. It is kept only for old experiments."""

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
    # acc оставлен для совместимости; для детекции это mAP50-95. Остальные метрики сохраняются при наличии.
    acc: float
    size_bytes: int
    latency_ms: Dict[str, float]
    precision: Optional[float] = None
    recall: Optional[float] = None
    iou: Optional[float] = None
    map50: Optional[float] = None


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
    # Как достигается итоговый prune_ratio: сразу одним шагом или через стадии.
    prune_mode: str = "one_shot"
    channel_round: int = 8
    min_channels: int = 8
    exclude_head: bool = True
    exclude_name_regex: Optional[str] = None
    strategy: str = "layerwise"
    max_prune_per_layer: Optional[float] = None
    protect_last_n: int = 0

    skip_inner_m: bool = True
    skip_cv1_if_parent_has_m: bool = True
    include_inner_m_regex: Optional[str] = None
    adapt_c2f_for_pruning: bool = False


@dataclass(frozen=True)
class StagedPruningConfig:
    # Промежуточные цели прунинга относительно исходной модели. Итоговый prune_ratio кандидата добавляется автоматически.
    milestones: tuple = ()

    # Режим стадийного прунинга: локальные шаги или подгонка к итоговой one-shot архитектуре.
    target_mode: str = "ratio_schedule"

    # Настройки восстановления качества для промежуточных стадий. Финальная стадия использует обычный train config, если не задано другое.
    intermediate_epochs: int = 30
    intermediate_lr: Optional[float] = None
    final_epochs: Optional[int] = None
    final_lr: Optional[float] = None

    # Если включено, каждая стадия отдельно попадает в history.jsonl.
    eval_after_each_stage: bool = True


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
class BenchmarkProfileConfig:
    # Дополнительный профиль бенчмарка. Если список пуст, используется старый latency.backend.
    name: str = ""
    backend: str = ""  # поддерживаемые профили: benchncnn, android_app, ort_android, tflite_android
    enabled: bool = True
    required: bool = True
    device_names: tuple = ()  # имена или serial устройств; пусто = все устройства

    # Переопределения для устройства и бенчмарка. None означает значение из основного конфига.
    threads: Optional[int] = None
    loops: Optional[int] = None
    powersave: Optional[int] = None
    runtime: Optional[str] = None
    device: Optional[int] = None
    gpu_device: Optional[int] = None
    cooling_down: Optional[int] = None
    shape: Optional[str] = None

    # Переопределения для Android app / ORT / TFLite бенчмарков.
    package: Optional[str] = None
    activity: Optional[str] = None
    dataset: Optional[str] = None
    push_dataset_images: Optional[bool] = None
    dataset_split: Optional[str] = None
    dataset_max_images: Optional[int] = None
    dataset_seed: Optional[int] = None
    dataset_remote_subdir: Optional[str] = None
    imgsz: Optional[int] = None
    warmup: Optional[int] = None
    conf: Optional[float] = None
    iou: Optional[float] = None
    max_det: Optional[int] = None
    provider: Optional[str] = None  # ORT: xnnpack, nnapi или cpu; для TFLite может быть fallback delegate
    delegate: Optional[str] = None  # TFLite: xnnpack, cpu или gpu
    artifact: Optional[str] = None  # TFLite-артефакт: tflite_int8, tflite_fp16 или явный путь
    optimized: Optional[bool] = None
    result_tag: Optional[str] = None
    timeout_sec: Optional[int] = None
    poll_interval_sec: Optional[float] = None
    clear_logcat: Optional[bool] = None
    remote_dir: Optional[str] = None


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
    """Legacy NCNN demo with a standalone native binary."""

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
    # Если включено, на устройство отправляется фиксированный набор изображений для бенчмарка.
    push_dataset_images: bool = False
    dataset_split: str = "val"
    dataset_max_images: int = 32
    dataset_seed: int = 42
    dataset_remote_subdir: str = "xtrim_bench_images"
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

@dataclass(frozen=True)
class OrtAndroidBenchConfig:
    enabled: bool = False
    package: str = "com.example.testyolo"
    activity: str = ".CliBenchActivity"
    dataset: str = "coco"
    # То же поведение, что и в AndroidAppBenchConfig: Python отправляет изображения, приложение их читает.
    push_dataset_images: bool = False
    dataset_split: str = "val"
    dataset_max_images: int = 32
    dataset_seed: int = 42
    dataset_remote_subdir: str = "xtrim_bench_images"
    imgsz: int = 640
    loops: int = 50
    warmup: int = 10
    threads: int = 4
    conf: float = 0.25
    iou: float = 0.45
    max_det: int = 100
    provider: str = "xnnpack"
    result_tag: str = "XTRIM_RESULT"
    timeout_sec: int = 180
    poll_interval_sec: float = 0.6
    clear_logcat: bool = True
    remote_dir: str = "/data/local/tmp"