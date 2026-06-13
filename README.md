# Automatic_compression_of_CNN

Проект для автоматического сжатия моделей YOLO и оценки скорости их инференса на мобильных устройствах. Конвейер выполняет структурный прунинг, восстановление качества модели, экспорт, квантование и измерение задержки на Android-устройствах.

Главная задача проекта — сформировать несколько сжатых вариантов модели и сравнить их по точности, размеру файла и задержке инференса. В актуальной версии используются ONNX Runtime и TensorFlow Lite. Старые режимы NCNN demo и LUT оставлены в коде только для совместимости со старыми экспериментами и не входят в основной сценарий запуска.

## Авторская роль

Проект выполнен в рамках выпускной квалификационной работы. Моя задача заключалась в разработке программного конвейера сжатия моделей и его интеграции с мобильным бенчмарком.

В рамках проекта мной реализованы:

- модуль управления экспериментом и последовательным запуском этапов конвейера;
- структурный прунинг YOLO-моделей с поддержкой staged pruning;
- восстановление качества модели после сжатия с использованием дообучения, дистилляции знаний и QAT;
- экспорт моделей в ONNX и TensorFlow Lite;
- подготовка INT8- и FP16-вариантов моделей для разных профилей инференса;
- запуск измерений задержки на Android-устройствах через ADB;
- повторный бенчмарк уже экспортированных моделей без повторного обучения и экспорта;
- сбор истории экспериментов, расчет метрик и формирование таблиц результатов;
- набор unit-, integration- и smoke-тестов для основных модулей проекта.

## Использование стороннего кода и библиотек

Проект не является форком готового решения. Он использует существующие библиотеки и фреймворки, включая Ultralytics YOLO, PyTorch, ONNX Runtime и TensorFlow Lite. Эти инструменты применяются для загрузки моделей, обучения, экспорта, квантования и запуска инференса.

Основная авторская часть проекта — это конвейер, который объединяет эти инструменты в единый процесс: выбирает кандидатов сжатия, выполняет прунинг, запускает восстановление качества, экспортирует модели, измеряет задержку на Android-устройствах и сохраняет результаты экспериментов в едином формате.

## Что нетривиального в реализации

Нетривиальная часть реализации состоит не в отдельном запуске готовой модели, а в объединении нескольких этапов в воспроизводимый экспериментальный конвейер. Структурный прунинг изменяет архитектуру YOLO-модели, после чего качество нужно восстановить обучением, затем модель нужно экспортировать в несколько форматов и проверить на реальном Android-устройстве.

Для каждого кандидата система сохраняет точность, размер модели и задержку инференса. Это позволяет сравнивать варианты сжатия не по одной метрике, а сразу по нескольким критериям.

Отдельно реализован режим повторного бенчмарка. Он позволяет брать уже экспортированные модели и заново измерять задержку на другом устройстве или при других настройках запуска, не повторяя обучение и экспорт. Это важно для честного сравнения моделей на разных мобильных устройствах.

## Возможности

- загрузка моделей Ultralytics YOLO из файлов `.pt`;
- структурный прунинг каналов;
- staged pruning, при котором модель сжимается постепенно через несколько этапов;
- восстановление качества с помощью дообучения и дистилляции знаний;
- восстановление качества через QAT для INT8-развертывания;
- экспорт модели в ONNX;
- пост-тренировочное INT8-квантование ONNX средствами ONNX Runtime;
- экспорт в TensorFlow Lite INT8 и TensorFlow Lite FP16;
- измерение задержки на Android через ADB и Android-приложение для бенчмарка;
- сравнение кандидатов по mAP50-95, Precision, Recall, размеру модели и задержке;
- повторное измерение задержки уже экспортированных моделей без повторного обучения;
- сохранение истории экспериментов и формирование таблиц результатов.

## Текущий сценарий работы

```text
Рабочая станция
  ├─ main.py
  ├─ xtrim/
  ├─ configs/
  └─ outputs/
        ↓
      ADB
        ↓
Android-устройство
  └─ приложение com.example.testyolo / .CliBenchActivity
        ↓
результат бенчмарка возвращается через logcat с тегом XTRIM_RESULT
```

Основные профили бенчмарка:

- `ort_xnnpack` — ONNX Runtime CPU;
- `tflite_int8_cpu` — TFLite INT8 CPU;
- `tflite_fp16_gpu` — TFLite FP16 GPU.

## Структура проекта

```text
.
├─ main.py                         # основная точка входа конвейера
├─ configs/                        # YAML-конфиги экспериментов
├─ xtrim/                          # основной Python-пакет
│  ├─ orchestrator.py              # управление этапами конвейера
│  ├─ config.py                    # загрузка YAML-конфигов
│  ├─ types.py                     # структуры данных для конфигурации
│  ├─ rebench_existing.py          # повторный бенчмарк экспортированных моделей
│  ├─ results_table.py             # вывод таблиц результатов
│  ├─ android_ort_bench.py         # бенчмарк ONNX Runtime на Android
│  ├─ android_app_bench.py         # бенчмарк через Android-приложение и ADB
│  ├─ android_tflite_bench.py      # вспомогательные функции для TFLite-бенчмарка
│  ├─ quant/                       # вспомогательные модули PTQ/QAT
│  ├─ trim/                        # методы сжатия
│  └─ yolo/                        # интеграция с YOLO/Ultralytics
├─ tools/                          # вспомогательные скрипты
├─ tests/                          # unit, integration и smoke-тесты
├─ requirements.txt                # зависимости проекта
├─ requirements-dev.txt            # зависимости для тестов
└─ pytest.ini                      # настройки pytest
```

## Требования

Рекомендуемое окружение:

- Windows 10/11;
- Python 3.10;
- CUDA-совместимая видеокарта для обучения и дообучения;
- установленный ADB;
- Android-устройство с включенной отладкой по USB;
- установленное Android-приложение для бенчмарка: `com.example.testyolo`;
- веса YOLO-модели в формате `.pt`;
- YAML-файл датасета для Ultralytics.

Проверьте, что Android-устройство подключено:

```bash
adb devices
```

Устройство должно отображаться со статусом `device`.

## Установка

Создайте виртуальное окружение:

```bash
python -m venv .venv
```

Активируйте его в Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Установите зависимости:

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Для запуска тестов также установите зависимости для разработки:

```bash
pip install -r requirements-dev.txt
```

## Подготовка конфига

Основные конфиги лежат в папке `configs/`:

```text
configs/config_yolov8m_p08_30x_staged_tflite_android.yaml
configs/config_yolo11m_p09_30x_staged_tflite_android.yaml
configs/config_yolo26m_p09_30x_staged_tflite_android.yaml
```

Перед запуском эксперимента проверьте следующие поля в YAML-конфиге.

Пути к модели и датасету:

```yaml
model:
  weights: path/to/best.pt
  data: path/to/data.yaml
  imgsz: 640
  device: 0
  task: detect
```

Android-устройства:

```yaml
devices:
- name: Huawei
  serial: 3GKUN24525G13087
  threads: 4
  loops: 50
  cooling_down: 5
```

Настройки Android-приложения для бенчмарка:

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

Профили бенчмарка задаются в секции `benchmark_profiles`:

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

Если нужен только бенчмарк ONNX Runtime CPU, оставьте только профиль `ort_xnnpack`.

## Запуск полного конвейера

Пример для YOLOv8m:

```bash
python main.py ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --out outputs/yolov8m_full ^
  --max_candidates 10 ^
  --hide-extra
```

Та же команда в одну строку:

```bash
python main.py --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml --out outputs/yolov8m_full --max_candidates 10 --hide-extra
```

Пример для YOLO11m:

```bash
python main.py --config configs/config_yolo11m_p09_30x_staged_tflite_android.yaml --out outputs/yolo11m_full --max_candidates 10 --hide-extra
```

Пример для YOLO26m:

```bash
python main.py --config configs/config_yolo26m_p09_30x_staged_tflite_android.yaml --out outputs/yolo26m_full --max_candidates 10 --hide-extra
```

Аргумент `--max_candidates` ограничивает количество кандидатов из `search_space`. Для быстрой проверки можно поставить `1`. Для полного эксперимента укажите нужное количество кандидатов из конфига.

## Что делает полный конвейер

Во время полного запуска система:

1. загружает YAML-конфиг;
2. загружает исходную YOLO-модель;
3. оценивает базовую модель;
4. выбирает следующего кандидата из `search_space`;
5. выполняет структурный прунинг;
6. восстанавливает качество модели;
7. экспортирует модель в ONNX;
8. формирует INT8/FP16-варианты, если включены соответствующие настройки;
9. передает модель на Android-устройство через ADB;
10. запускает Android-приложение для измерения задержки;
11. сохраняет результаты в `history.jsonl`.

## Результаты эксперимента

После запуска в выходной папке появляются файлы такого вида:

```text
history.jsonl              # история эксперимента по кандидатам
bench_cache.json           # кэш измерений задержки, если включен
*/export/model.onnx        # ONNX FP32
*/export/model_int8.onnx   # ONNX INT8
*/export/model_int8.tflite # TFLite INT8
*/export/model_fp16.tflite # TFLite FP16
```

Чтобы вывести таблицу результатов из существующего файла `history.jsonl`, используйте:

```bash
python tools/show_results.py --history outputs/yolov8m_full/history.jsonl --hide-extra
```

Чтобы сохранить Pareto-график:

```bash
python tools/show_results.py --history outputs/yolov8m_full/history.jsonl --plot-save outputs/yolov8m_full/pareto.png
```

## Повторный бенчмарк экспортированных моделей

Если модели уже обучены и экспортированы, задержку можно измерить повторно без нового обучения:

```bash
python main.py ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --rebench-existing outputs/yolov8m_full ^
  --out outputs/yolov8m_rebench_huawei ^
  --hide-extra
```

Команда в одну строку:

```bash
python main.py --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml --rebench-existing outputs/yolov8m_full --out outputs/yolov8m_rebench_huawei --hide-extra
```

Этот режим полезен, когда одни и те же модели нужно проверить на другом устройстве или с другими значениями количества прогонов, warmup-запусков, потоков или backend-профилей.

## Android rebench с доверительными интервалами

Для повторных измерений задержки одного и того же набора моделей используйте:

```bash
python tools/android_ci_rebench_raw.py ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --models-root outputs/yolov8m_full ^
  --out outputs/yolov8m_rebench_raw ^
  --serial 3GKUN24525G13087 ^
  --repeat 5
```

Запуск только одного профиля:

```bash
python tools/android_ci_rebench_raw.py ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --models-root outputs/yolov8m_full ^
  --out outputs/yolov8m_rebench_int8 ^
  --serial 3GKUN24525G13087 ^
  --repeat 5 ^
  --only-profile tflite_int8_cpu
```

Скрипт сохраняет:

```text
history.jsonl
summary.csv
```

Файл `summary.csv` содержит среднюю задержку, стандартное отклонение и 95% доверительный интервал.

## Проверка точности TFLite INT8 и FP16

Для валидации `.tflite`-моделей на датасете используйте:

```bash
python tools/validate_tflite_fp16_candidates.py ^
  --outputs outputs/yolov8m_full ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --baseline-map50-95 0.7411
```

Ограничение проверки выбранными кандидатами:

```bash
python tools/validate_tflite_fp16_candidates.py ^
  --outputs outputs/yolov8m_full ^
  --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml ^
  --include-candidates-regex "p=0.86|p=0.88" ^
  --baseline-map50-95 0.7411
```

Результаты сохраняются в:

```text
outputs/.../fp16_metrics_report/
```

Основные файлы отчета:

```text
tflite_int8_fp16_metrics.csv
tflite_int8_fp16_metrics.json
tflite_int8_fp16_metrics.md
tflite_int8_fp16_errors.json
```

## Бенчмарк базовой TFLite FP32-модели

Чтобы отдельно измерить задержку базовой TFLite FP32-модели:

```bash
python tools/bench_tflite_fp32_baselines.py --threads 4 --loops 50 --warmup 10 --imgsz 640
```

Если нужно явно указать Android-устройство:

```bash
python tools/bench_tflite_fp32_baselines.py --serial 3GKUN24525G13087 --threads 4 --loops 50 --warmup 10 --imgsz 640
```

## Тесты

Запустить все тесты:

```bash
pytest
```

Запустить только unit-тесты:

```bash
pytest -m unit
```

Запустить integration-тесты:

```bash
pytest -m integration
```

Запустить smoke-тесты:

```bash
pytest -m smoke
```

Проверить покрытие:

```bash
pytest --cov=xtrim --cov-report=term-missing
```

Тесты не требуют Android-устройства, реальных весов YOLO, ONNX Runtime или NCNN-бинарников. Внешние зависимости заменяются mock-объектами и fake-объектами там, где это нужно.

## Важные секции конфига

### `search_space`

Задает набор кандидатов для перебора:

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

Управляет структурным прунингом:

```yaml
trim:
  channel_round: 8
  min_channels: 8
  prune_mode: staged
  adapt_c2f_for_pruning: true
```

### `staged_pruning`

Задает промежуточные этапы прунинга:

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

Настройки ONNX INT8 PTQ:

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

Настройки восстановления качества через QAT:

```yaml
qat:
  enabled: true
  epochs: 75
  lr: 5.0e-05
  bits_w: 8
  bits_a: 8
```

## Legacy-части

Некоторые старые части кода оставлены для совместимости с предыдущими экспериментами:

- `android_demo`;
- `latency_lut`;
- вспомогательные части, связанные с NCNN.

Они не используются в текущем основном конвейере. Актуальный сценарий основан на ONNX Runtime, TensorFlow Lite, Android-приложении для бенчмарка и ADB.

## Частые проблемы

### ADB не видит устройство

Проверьте:

```bash
adb devices
```

Если устройство отображается как `unauthorized`, подтвердите RSA-ключ на телефоне или планшете.

### Android-приложение не возвращает результат

Проверьте, что в конфиге правильно указаны package, activity и result tag:

```yaml
package: com.example.testyolo
activity: .CliBenchActivity
result_tag: XTRIM_RESULT
```

Также можно вручную посмотреть logcat:

```bash
adb logcat -d -v raw -s XTRIM_RESULT:I
```

### Файл модели не найден

Проверьте путь к модели в конфиге:

```yaml
model:
  weights: path/to/best.pt
```

Также проверьте путь к датасету:

```yaml
model:
  data: path/to/data.yaml
```

### TFLite GPU медленнее CPU

Это может быть нормальной ситуацией. Задержка зависит не только от размера модели, но и от устройства, delegate, формата модели и накладных расходов runtime.

## Минимальный быстрый сценарий

1. Установите зависимости:

```bash
pip install -r requirements.txt
```

2. Проверьте Android-устройство:

```bash
adb devices
```

3. Обновите `model.weights`, `model.data` и `devices.serial` в YAML-конфиге.

4. Запустите небольшой эксперимент:

```bash
python main.py --config configs/config_yolov8m_p08_30x_staged_tflite_android.yaml --out outputs/yolov8m_test --max_candidates 1 --hide-extra
```

5. Выведите результаты:

```bash
python tools/show_results.py --history outputs/yolov8m_test/history.jsonl --hide-extra
```
