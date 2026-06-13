# Automatic_compression_of_CNN

A project for automatic compression of YOLO models and evaluation of their inference performance on mobile devices. The pipeline performs structural pruning, quality recovery, model export, quantization, and latency measurement on Android devices.

The main goal of the project is to generate several compressed model variants and compare them by accuracy, model size, and inference latency. The current version uses ONNX Runtime and TensorFlow Lite. Older NCNN demo and LUT-based modes are kept in the code only for legacy compatibility and are not part of the main workflow.

## Features

- loading Ultralytics YOLO models from `.pt` files;
- structural channel pruning;
- staged pruning, where the model is compressed gradually through several pruning stages;
- quality recovery using fine-tuning and knowledge distillation;
- QAT-based recovery for INT8 deployment;
- export to ONNX;
- ONNX INT8 post-training quantization using ONNX Runtime;
- export to TensorFlow Lite INT8 and TensorFlow Lite FP16;
- latency measurement on Android through ADB and an Android benchmark application;
- comparison of candidates by mAP50-95, Precision, Recall, model size, and latency;
- repeated benchmarking of already exported models without retraining;
- generation of experiment history files and result tables.

## Current workflow

```text
Workstation
  ├─ main.py
  ├─ xtrim/
  ├─ configs/
  └─ outputs/
        ↓
      ADB
        ↓
Android device
  └─ application com.example.testyolo / .CliBenchActivity
        ↓
benchmark result is returned through logcat with the XTRIM_RESULT tag
```

Main benchmark profiles:

- `ort_xnnpack` — ONNX Runtime CPU;
- `tflite_int8_cpu` — TFLite INT8 CPU;
- `tflite_fp16_gpu` — TFLite FP16 GPU.

## Project structure

```text
.
├─ main.py                         # main pipeline entry point
├─ configs/                        # YAML experiment configs
├─ xtrim/                          # main Python package
│  ├─ orchestrator.py              # pipeline orchestration
│  ├─ config.py                    # YAML config loading
│  ├─ types.py                     # configuration data structures
│  ├─ rebench_existing.py          # rebenchmarking of exported models
│  ├─ results_table.py             # result table output
│  ├─ android_ort_bench.py         # ONNX Runtime benchmark on Android
│  ├─ android_app_bench.py         # Android application benchmark through ADB
│  ├─ android_tflite_bench.py      # TFLite benchmark helpers on Android
│  ├─ quant/                       # PTQ/QAT helper modules
│  ├─ trim/                        # compression methods
│  └─ yolo/                        # YOLO/Ultralytics integration
├─ tools/                          # helper scripts
├─ tests/                          # unit, integration, and smoke tests
├─ requirements.txt                # project dependencies
├─ requirements-dev.txt            # test dependencies
└─ pytest.ini                      # pytest configuration
```

## Requirements

Recommended environment:

- Windows 10/11;
- Python 3.10;
- CUDA-compatible GPU for training and fine-tuning;
- installed ADB;
- Android device with USB debugging enabled;
- installed Android benchmark application: `com.example.testyolo`;
- YOLO model weights in `.pt` format;
- Ultralytics dataset YAML file.

Check that the Android device is connected:

```bash
adb devices
```

The device should be listed with the `device` status.

## Installation

Create a virtual environment:

```bash
python -m venv .venv
```

Activate it on Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the dependencies:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

For running tests, also install development dependencies:

```bash
pip install -r requirements-dev.txt
```

## Preparing the config

Main configs are stored in the `configs/` directory:

```text
configs/config_yolov8m_p08_30x_staged_tflite_android.yaml
configs/config_yolo11m_p09_30x_staged_tflite_android.yaml
configs/config_yolo26m_p09_30x_staged_tflite_android.yaml
```

Before running an experiment, check the following fields in the YAML config.

Model and dataset paths:

```yaml
model:
  weights: path/to/best.pt
  data: path/to/data.yaml
  imgsz: 640
  device: 0
  task: detect
```

Android devices:

```yaml
devices:
- name: Your_device_name
  serial: Device_serial_from_adb
  threads: 4
  loops: 50
  cooling_down: 5
```

Android benchmark application settings:

```yaml
ort_android_bench:
  enabled: true
  package: com.example.testyolo
  activity: .CliBenchActivity
  imgsz: 640
  loops: 50
  warmup: 10
  threads: 4
  provider: xnnpack
  result_tag: XTRIM_RESULT
```

Benchmark profiles are defined in the `benchmark_profiles` section:

```yaml
benchmark_profiles:
- name: ort_xnnpack
  backend: ort_android
  provider: xnnpack
  required: true

- name: tflite_int8_cpu
  backend: tflite_android
  delegate: xnnpack
  artifact: tflite_int8
  required: false

- name: tflite_fp16_gpu
  backend: tflite_android
  delegate: gpu
  artifact: tflite_fp16
  required: false
```

If only ONNX Runtime CPU benchmarking is needed, keep only the `ort_xnnpack` profile.

## Running the full pipeline

Example for YOLOv8m:

```bash
python main.py ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --out outputs/yolov8m_full ^
  --max_candidates 10 ^
  --hide-extra
```

The same command in one line:

```bash
python main.py --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml --out outputs/yolov8m_full --max_candidates 10 --hide-extra
```

Example for YOLO11m:

```bash
python main.py --config configs/config_yolo11m_p09_30x_staged_tflite_android.yaml --out outputs/yolo11m_full --max_candidates 10 --hide-extra
```

Example for YOLO26m:

```bash
python main.py --config configs/config_yolo26m_p09_30x_staged_tflite_android.yaml --out outputs/yolo26m_full --max_candidates 10 --hide-extra
```

The `--max_candidates` argument limits the number of candidates taken from `search_space`. For a quick test, use `1`. For a full experiment, use the required number of candidates from the config.

## What the full pipeline does

During a full run, the system:

1. loads the YAML config;
2. loads the original YOLO model;
3. evaluates the baseline model;
4. selects the next candidate from `search_space`;
5. applies structural pruning;
6. restores model quality;
7. exports the model to ONNX;
8. creates INT8/FP16 variants when the corresponding settings are enabled;
9. sends the model to the Android device through ADB;
10. starts the Android application to measure latency;
11. saves the results to `history.jsonl`.

## Experiment results

After a run, the output directory contains files such as:

```text
history.jsonl              # experiment history by candidates
bench_cache.json           # latency measurement cache, if enabled
*/export/model.onnx        # ONNX FP32
*/export/model_int8.onnx   # ONNX INT8
*/export/model_int8.tflite # TFLite INT8
*/export/model_fp16.tflite # TFLite FP16
```

To print a result table from an existing `history.jsonl` file:

```bash
python tools/show_results.py --history outputs/yolov8m_full/history.jsonl --hide-extra
```

To save a Pareto plot:

```bash
python tools/show_results.py --history outputs/yolov8m_full/history.jsonl --plot-save outputs/yolov8m_full/pareto.png
```

## Rebenchmarking exported models

If the models have already been trained and exported, latency can be measured again without retraining:

```bash
python main.py ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --rebench-existing outputs/yolov8m_full ^
  --out outputs/yolov8m_rebench_huawei ^
  --hide-extra
```

One-line version:

```bash
python main.py --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml --rebench-existing outputs/yolov8m_full --out outputs/yolov8m_rebench_huawei --hide-extra
```

This mode is useful when the same models need to be tested on another device or with different numbers of loops, warmup runs, threads, or backend profiles.

## Android rebenchmark with confidence intervals

For repeated latency measurements of the same model set, use:

```bash
python tools/android_ci_rebench_raw.py ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --models-root outputs/yolov8m_full ^
  --out outputs/yolov8m_rebench_raw ^
  --serial 3GKUN24525G13087 ^
  --repeat 5
```

Run only one profile:

```bash
python tools/android_ci_rebench_raw.py ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --models-root outputs/yolov8m_full ^
  --out outputs/yolov8m_rebench_int8 ^
  --serial 3GKUN24525G13087 ^
  --repeat 5 ^
  --only-profile tflite_int8_cpu
```

The script saves:

```text
history.jsonl
summary.csv
```

The `summary.csv` file contains mean latency, standard deviation, and the 95% confidence interval.

## Validating TFLite INT8 and FP16 accuracy

To validate `.tflite` models on the dataset:

```bash
python tools/validate_tflite_fp16_candidates.py ^
  --outputs outputs/yolov8m_full ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --baseline-map50-95 0.7411
```

Limit validation to selected candidates:

```bash
python tools/validate_tflite_fp16_candidates.py ^
  --outputs outputs/yolov8m_full ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --include-candidates-regex "p=0.86|p=0.88" ^
  --baseline-map50-95 0.7411
```

Results are saved to:

```text
outputs/.../fp16_metrics_report/
```

Main report files:

```text
tflite_int8_fp16_metrics.csv
tflite_int8_fp16_metrics.json
tflite_int8_fp16_metrics.md
tflite_int8_fp16_errors.json
```

## Benchmarking the FP32 TFLite baseline

To measure the FP32 TFLite baseline separately:

```bash
python tools/bench_tflite_fp32_baselines.py --threads 4 --loops 50 --warmup 10 --imgsz 640
```

If needed, specify the Android device explicitly:

```bash
python tools/bench_tflite_fp32_baselines.py --serial 3GKUN24525G13087 --threads 4 --loops 50 --warmup 10 --imgsz 640
```

## Tests

Run all tests:

```bash
pytest
```

Run only unit tests:

```bash
pytest -m unit
```

Run integration tests:

```bash
pytest -m integration
```

Run smoke tests:

```bash
pytest -m smoke
```

Check coverage:

```bash
pytest --cov=xtrim --cov-report=term-missing
```

Tests do not require an Android device, real YOLO weights, ONNX Runtime, or NCNN binaries. External dependencies are replaced with mocks and fake objects where needed.

## Important config sections

### `search_space`

Defines the set of candidates for enumeration:

```yaml
search_space:
  width_mult:
  - 1
  prune_ratio:
  - 0.0
  - 0.8
  - 0.86
  - 0.88
  - 0.9
  lowrank_rank:
  - 0
  sparse_1x1:
  - 0
```

### `trim`

Controls structural pruning:

```yaml
trim:
  channel_round: 8
  min_channels: 8
  prune_mode: staged
  adapt_c2f_for_pruning: true
```

### `staged_pruning`

Defines intermediate pruning stages:

```yaml
staged_pruning:
  target_mode: match_one_shot_architecture
  milestones:
  - 0.3
  - 0.45
  - 0.6
  - 0.7
  - 0.8
  intermediate_epochs: 30
  final_epochs: 150
  eval_after_each_stage: true
```

### `onnx_ptq`

ONNX INT8 PTQ settings:

```yaml
onnx_ptq:
  enabled: true
  calib_split: train
  calib_max_images: 256
  per_channel: true
  quant_format: qdq
  activation_type: quint8
  weight_type: qint8
  calibrate_method: entropy
```

### `qat`

QAT recovery settings:

```yaml
qat:
  enabled: true
  epochs: 75
  lr: 5.0e-05
  bits_w: 8
  bits_a: 8
```

## Legacy parts

Some older parts of the code are kept for compatibility with previous experiments:

- `android_demo`;
- `latency_lut`;
- NCNN-related helpers.

They are not used in the current main pipeline. The current workflow is based on ONNX Runtime, TensorFlow Lite, the Android benchmark application, and ADB.

## Common issues

### ADB does not see the device

Check:

```bash
adb devices
```

If the device is shown as `unauthorized`, confirm the RSA key on the phone or tablet.

### The Android application does not return a result

Check that the package, activity, and result tag are correct in the config:

```yaml
package: com.example.testyolo
activity: .CliBenchActivity
result_tag: XTRIM_RESULT
```

You can also inspect logcat manually:

```bash
adb logcat -d -v raw -s XTRIM_RESULT:I
```

### Model file was not found

Check the model path in the config:

```yaml
model:
  weights: path/to/best.pt
```

Also check the dataset path:

```yaml
model:
  data: path/to/data.yaml
```

### TFLite GPU is slower than CPU

This can happen. Latency depends not only on model size, but also on the device, delegate, model format, and runtime overhead.

## Minimal quick scenario

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Check the Android device:

```bash
adb devices
```

3. Update `model.weights`, `model.data`, and `devices.serial` in the YAML config.

4. Run a small experiment:

```bash
python main.py --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml --out outputs/yolov8m_test --max_candidates 1 --hide-extra
```

5. Print the results:

```bash
python tools/show_results.py --history outputs/yolov8m_test/history.jsonl --hide-extra
```
