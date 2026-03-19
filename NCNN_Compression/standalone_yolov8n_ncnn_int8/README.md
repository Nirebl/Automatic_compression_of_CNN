# YOLOv8n -> NCNN -> INT8

Это полностью standalone-папка. Она не импортирует код из текущего проекта и нужна только для отдельного эксперимента с `yolov8n` и `ncnn`.

Что делает скрипт:

1. Скачивает `yolov8n.pt`.
2. Экспортирует модель в NCNN через `ultralytics`.
3. Собирает calibration list.
4. Запускает `ncnn2table` и `ncnn2int8`.
5. Проверяет float NCNN и INT8 NCNN на тестовом изображении.
6. Сохраняет картинки с предсказаниями и JSON-сводки.

## Быстрый старт

Нужен распакованный SDK `ncnn`, где лежат `ncnn2table` и `ncnn2int8`.

PowerShell:

```powershell
.\setup_and_run.ps1 -NcnnToolsDir "C:\path\to\ncnn\x64\bin"
```

Ручной запуск:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe .\quantize_yolov8n_ncnn_int8.py --ncnn-tools-dir "C:\path\to\ncnn\x64\bin"
```

## Свои calibration images

Для нормальной квантизации лучше дать реальные картинки:

```powershell
.\.venv\Scripts\python.exe .\quantize_yolov8n_ncnn_int8.py `
  --ncnn-tools-dir "C:\path\to\ncnn\x64\bin" `
  --calib-dir "D:\datasets\calibration_images"
```

Если `--calib-dir` и `--calib-list` не заданы, скрипт использует demo-набор из `bus.jpg` и `zidane.jpg`, повторяя их несколько раз. Для реальной оценки качества этого мало, но для smoke test достаточно.

## Что сохраняется

По умолчанию всё складывается в `work/`:

- `work/demo_images/` - тестовые изображения
- `work/calibration_list.txt` - список для `ncnn2table`
- `work/exports/..._ncnn_model/` - float NCNN модель
- `work/int8_model/` - INT8 NCNN модель
- `work/verification/` - картинки и JSON со сводкой предсказаний

## Важное замечание про Windows

На этой машине float NCNN экспорт и float-проверка отрабатывают нормально.

Но native Windows `ncnn2table` / `ncnn2int8` из доступных SDK воспроизводимо ломали `*.param` для `yolov8n` при записи, из-за чего INT8-проверка падала уже на загрузке модели. Поэтому в скрипте есть явная диагностика этого случая. Если увидишь ошибку про malformed `model.ncnn.param`, запускай тот же скрипт в Linux/WSL с Linux-версией `ncnn` tools.

Это наблюдение получено локально при реальном прогоне, а не из документации.

## Официальные ссылки

- Ultralytics NCNN integration: https://docs.ultralytics.com/integrations/ncnn/
- NCNN INT8 quantization: https://ncnn.readthedocs.io/en/latest/how-to-use-and-FAQ/quantized-int8-inference.html
- NCNN wiki mirror for INT8: https://github.com/Tencent/ncnn/wiki/quantized-int8-inference
