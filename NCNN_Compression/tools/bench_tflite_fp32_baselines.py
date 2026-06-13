import argparse
import csv
import json
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MODELS = {
    "yolov8m": "outputs/baselines/yolov8m_tflite_fp32/baseline_tflite_fp32.tflite",
    "yolo11m": "outputs/baselines/yolo11m_tflite_fp32/baseline_tflite_fp32.tflite",
    "yolo26m": "outputs/baselines/yolo26m_tflite_fp32/baseline_tflite_fp32.tflite",
}


def run_command(cmd: list[str], cwd: Path | None = None) -> subprocess.CompletedProcess:
    print("\n[cmd]", " ".join(f'"{x}"' if " " in x else x for x in cmd))

    return subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        text=True,
        encoding="utf-8",
        errors="replace",
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )


def get_first_android_device(adb: str = "adb") -> str:
    result = run_command([adb, "devices"])

    if result.returncode != 0:
        raise RuntimeError(
            "Не удалось выполнить adb devices.\n\n"
            f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        )

    devices = []

    for line in result.stdout.splitlines():
        line = line.strip()

        if not line or line.startswith("List of devices"):
            continue

        parts = line.split()

        if len(parts) >= 2 and parts[1] == "device":
            devices.append(parts[0])

    if not devices:
        raise RuntimeError(
            "Не найдено ни одного доступного Android-устройства.\n"
            "Проверь, что телефон подключен, включена отладка по USB и в adb devices статус именно device."
        )

    return devices[0]


def parse_json_from_output(text: str) -> dict[str, Any] | None:
    """Достает последний JSON-объект из вывода Android-бенчмарка."""
    candidates = []

    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            candidates.append(line)

    for candidate in reversed(candidates):
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            continue

    return None


def save_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", errors="replace")


def bench_one(
    project_root: Path,
    python_exe: str,
    model_name: str,
    model_path: Path,
    delegate: str,
    serial: str,
    out_dir: Path,
    threads: int,
    loops: int,
    warmup: int,
    imgsz: int,
) -> dict[str, Any]:
    log_base = out_dir / f"{model_name}_{delegate}"
    stdout_path = log_base.with_suffix(".stdout.txt")
    stderr_path = log_base.with_suffix(".stderr.txt")

    if not model_path.exists():
        row = {
            "model": model_name,
            "delegate": delegate,
            "status": "missing_model",
            "model_path": str(model_path),
            "avg_ms": "",
            "min_ms": "",
            "max_ms": "",
            "std_ms": "",
            "n": "",
            "stdout_log": str(stdout_path),
            "stderr_log": str(stderr_path),
        }
        return row

    cmd = [
        python_exe,
        "-m",
        "xtrim.android_app_bench",
        "--model",
        str(model_path),
        "--serial",
        serial,
        "--delegate",
        delegate,
        "--threads",
        str(threads),
        "--loops",
        str(loops),
        "--warmup",
        str(warmup),
        "--imgsz",
        str(imgsz),
    ]

    result = run_command(cmd, cwd=project_root)

    save_text(stdout_path, result.stdout)
    save_text(stderr_path, result.stderr)

    parsed = parse_json_from_output(result.stdout)

    status = "ok" if result.returncode == 0 else "failed"

    if parsed is not None and parsed.get("ok") is False:
        status = "bench_error"

    row = {
        "model": model_name,
        "delegate": delegate,
        "status": status,
        "model_path": str(model_path),
        "avg_ms": parsed.get("avg_ms", "") if parsed else "",
        "min_ms": parsed.get("min_ms", "") if parsed else "",
        "max_ms": parsed.get("max_ms", "") if parsed else "",
        "std_ms": parsed.get("std_ms", "") if parsed else "",
        "n": parsed.get("n", "") if parsed else "",
        "actual_delegate": parsed.get("actual_delegate", "") if parsed else "",
        "delegate_error": parsed.get("delegate_error", "") if parsed else "",
        "stdout_log": str(stdout_path),
        "stderr_log": str(stderr_path),
    }

    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "model",
        "delegate",
        "status",
        "avg_ms",
        "min_ms",
        "max_ms",
        "std_ms",
        "n",
        "actual_delegate",
        "delegate_error",
        "model_path",
        "stdout_log",
        "stderr_log",
    ]

    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for row in rows:
            writer.writerow(row)


def parse_model_overrides(items: list[str] | None) -> dict[str, str]:
    models = dict(DEFAULT_MODELS)

    if not items:
        return models

    for item in items:
        if "=" not in item:
            raise ValueError(
                f"Неверный формат --model: {item}\n"
                "Нужно так: --model yolov8m=path/to/model.tflite"
            )

        name, path = item.split("=", 1)
        models[name.strip()] = path.strip()

    return models


def main() -> None:
    parser = argparse.ArgumentParser()

    parser.add_argument("--project-root", default=".", help="Корень проекта")
    parser.add_argument("--out", default=None, help="Папка для результатов")
    parser.add_argument("--serial", default=None, help="Serial устройства. Если не указан, берется первое adb device")
    parser.add_argument("--adb", default="adb")
    parser.add_argument("--python", default=sys.executable)

    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--loops", type=int, default=50)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--cooldown", type=float, default=3.0, help="Пауза между замерами, сек")

    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help="Переопределить путь модели: --model yolov8m=path/to/model.tflite",
    )

    args = parser.parse_args()

    project_root = Path(args.project_root).resolve()

    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = Path(args.out).resolve() if args.out else project_root / "outputs" / "baseline_tflite_fp32_bench" / timestamp
    out_dir.mkdir(parents=True, exist_ok=True)

    serial = args.serial or get_first_android_device(args.adb)

    print(f"\n[device] {serial}")
    print(f"[out] {out_dir}")

    models = parse_model_overrides(args.model)

    rows = []

    for model_name, rel_path in models.items():
        model_path = Path(rel_path)
        if not model_path.is_absolute():
            model_path = project_root / model_path

        for delegate in ["cpu", "gpu"]:
            print("\n" + "=" * 80)
            print(f"[bench] model={model_name}, delegate={delegate}")
            print("=" * 80)

            row = bench_one(
                project_root=project_root,
                python_exe=args.python,
                model_name=model_name,
                model_path=model_path,
                delegate=delegate,
                serial=serial,
                out_dir=out_dir,
                threads=args.threads,
                loops=args.loops,
                warmup=args.warmup,
                imgsz=args.imgsz,
            )

            rows.append(row)

            print(
                f"[result] {model_name} {delegate}: "
                f"status={row['status']}, avg_ms={row['avg_ms']}"
            )

            if args.cooldown > 0:
                time.sleep(args.cooldown)

    csv_path = out_dir / "summary.csv"
    write_csv(csv_path, rows)

    json_path = out_dir / "summary.json"
    json_path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\n" + "=" * 80)
    print("Готово")
    print("=" * 80)
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")

    print("\nКраткая таблица:")
    for row in rows:
        print(
            f"- {row['model']:8s} | {row['delegate']:3s} | "
            f"{row['status']:12s} | avg_ms={row['avg_ms']}"
        )


if __name__ == "__main__":
    main()