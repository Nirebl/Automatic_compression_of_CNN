from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

try:
    import yaml
except ImportError as exc:  # pragma: no cover
    raise SystemExit("PyYAML is required. Install it with: pip install pyyaml") from exc

try:
    from ultralytics import YOLO
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Ultralytics is required. Install it with: pip install ultralytics") from exc


@dataclass
class ModelPair:
    candidate: str
    candidate_dir: Path
    p: Optional[float]
    int8_path: Optional[Path]
    fp16_path: Optional[Path]
    all_int8: List[Path]
    all_fp16: List[Path]


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a YAML mapping: {path}")
    return data


def get_from_config(cfg: Dict[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted_key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return default
        cur = cur[part]
    return cur


def parse_p_from_name(name: str) -> Optional[float]:
    s = name.lower()

    patterns = [
        r"(?:^|[_\-])p(?:=|_|-)?(0\.\d+|1\.0|1|\d{1,2})(?:$|[_\-])",
        r"prun(?:e|ing)?(?:=|_|-)?(0\.\d+|1\.0|1|\d{1,2})(?:$|[_\-])",
    ]
    for pattern in patterns:
        m = re.search(pattern, s)
        if not m:
            continue
        raw = m.group(1)
        try:
            if raw.startswith("0.") or raw in {"1", "1.0"}:
                return float(raw)
            if raw.isdigit():
                n = int(raw)
                if n <= 9:
                    return n / 10.0
                if n <= 99:
                    return n / 100.0
            return float(raw)
        except ValueError:
            return None
    return None


def normalize_path_text(path: Path) -> str:
    return str(path).replace("\\", "/").lower()


def classify_tflite(path: Path) -> Optional[str]:
    text = normalize_path_text(path)
    name = path.name.lower()

    fp16_tokens = ("fp16", "float16", "f16")
    int8_tokens = ("int8", "integer", "uint8", "quant", "qat", "ptq")

    has_fp16 = any(t in text for t in fp16_tokens)
    has_int8 = any(t in text for t in int8_tokens)

    name_has_fp16 = any(t in name for t in fp16_tokens)
    name_has_int8 = any(t in name for t in int8_tokens)

    if name_has_fp16 and not name_has_int8:
        return "fp16"
    if name_has_int8 and not name_has_fp16:
        return "int8"
    if has_fp16 and not has_int8:
        return "fp16"
    if has_int8 and not has_fp16:
        return "int8"
    return None


def score_model_path(path: Path, kind: str) -> int:
    text = normalize_path_text(path)
    name = path.name.lower()
    score = 0

    if kind == "fp16":
        for token in ("fp16", "float16"):
            if token in name:
                score += 100
            elif token in text:
                score += 50
    elif kind == "int8":
        for token in ("int8", "qat", "quant"):
            if token in name:
                score += 100
            elif token in text:
                score += 50
        if "qat" in text:
            score += 40
        if "ptq" in text:
            score -= 10

    if any(t in text for t in ("final", "export", "artifact", "deploy")):
        score += 20
    if any(t in text for t in ("tmp", "temp", "backup", "old")):
        score -= 50

    try:
        size_mb = path.stat().st_size / (1024 * 1024)
        if size_mb > 0.5:
            score += 5
    except OSError:
        pass

    return score


def pick_best(paths: List[Path], kind: str) -> Optional[Path]:
    if not paths:
        return None
    return sorted(paths, key=lambda p: (score_model_path(p, kind), p.stat().st_size if p.exists() else 0), reverse=True)[0]


def candidate_dirs(outputs: Path) -> List[Path]:
    dirs: List[Path] = []
    if any(outputs.glob("*.tflite")):
        dirs.append(outputs)

    for child in sorted(outputs.iterdir(), key=lambda p: p.name):
        if child.is_dir() and any(child.rglob("*.tflite")):
            dirs.append(child)
    return dirs


def find_model_pairs(outputs: Path, include_candidates_regex: Optional[str] = None) -> List[ModelPair]:
    regex = re.compile(include_candidates_regex) if include_candidates_regex else None
    pairs: List[ModelPair] = []

    for cdir in candidate_dirs(outputs):
        candidate = cdir.name
        if regex and not regex.search(candidate):
            continue

        int8_files: List[Path] = []
        fp16_files: List[Path] = []
        for tflite in sorted(cdir.rglob("*.tflite")):
            kind = classify_tflite(tflite)
            if kind == "int8":
                int8_files.append(tflite)
            elif kind == "fp16":
                fp16_files.append(tflite)

        if not int8_files and not fp16_files:
            continue

        pairs.append(
            ModelPair(
                candidate=candidate,
                candidate_dir=cdir,
                p=parse_p_from_name(candidate),
                int8_path=pick_best(int8_files, "int8"),
                fp16_path=pick_best(fp16_files, "fp16"),
                all_int8=int8_files,
                all_fp16=fp16_files,
            )
        )

    return pairs


def file_fingerprint(path: Path) -> str:
    st = path.stat()
    raw = f"{path.resolve()}|{st.st_size}|{int(st.st_mtime)}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]


def cache_key(path: Path, data: str, imgsz: int, split: str, task: str, batch: int) -> str:
    raw = f"{file_fingerprint(path)}|data={data}|imgsz={imgsz}|split={split}|task={task}|batch={batch}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def load_cache(cache_path: Path) -> Dict[str, Any]:
    if not cache_path.exists():
        return {}
    try:
        with cache_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def save_cache(cache_path: Path, cache: Dict[str, Any]) -> None:
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with cache_path.open("w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def validate_model(
    model_path: Path,
    *,
    data: str,
    imgsz: int,
    split: str,
    task: str,
    batch: int,
    workers: int,
    project_dir: Path,
    run_name: str,
    verbose: bool,
) -> Dict[str, Any]:
    start = time.time()
    model = YOLO(str(model_path), task=task)
    metrics = model.val(
        data=data,
        imgsz=imgsz,
        split=split,
        batch=batch,
        workers=workers,
        plots=False,
        save=False,
        save_json=False,
        verbose=verbose,
        project=str(project_dir),
        name=run_name,
        exist_ok=True,
    )

    if not hasattr(metrics, "box"):
        raise RuntimeError(f"Unsupported metrics object for task={task}. This script currently expects detection metrics.")

    elapsed = time.time() - start
    return {
        "map50_95": float(metrics.box.map),
        "map50": float(metrics.box.map50),
        "precision": float(metrics.box.mp),
        "recall": float(metrics.box.mr),
        "elapsed_sec": elapsed,
    }


def rounded(value: Optional[float], digits: int = 4) -> Optional[float]:
    if value is None:
        return None
    return round(float(value), digits)


def fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def safe_metric(metrics: Optional[Dict[str, Any]], key: str) -> Optional[float]:
    if not metrics:
        return None
    value = metrics.get(key)
    return None if value is None else float(value)


def build_row(
    pair: ModelPair,
    int8_metrics: Optional[Dict[str, Any]],
    fp16_metrics: Optional[Dict[str, Any]],
    baseline_map50_95: Optional[float],
    max_loss_ratio: float,
) -> Dict[str, Any]:
    int8_map = safe_metric(int8_metrics, "map50_95")
    fp16_map = safe_metric(fp16_metrics, "map50_95")

    delta_fp16_int8 = None
    if int8_map is not None and fp16_map is not None:
        delta_fp16_int8 = fp16_map - int8_map

    fp16_loss_ratio = None
    fp16_pass = None
    if baseline_map50_95 is not None and fp16_map is not None:
        fp16_loss_ratio = (baseline_map50_95 - fp16_map) / baseline_map50_95 if baseline_map50_95 else None
        fp16_pass = fp16_loss_ratio is not None and fp16_loss_ratio <= max_loss_ratio

    int8_size = pair.int8_path.stat().st_size / (1024 * 1024) if pair.int8_path and pair.int8_path.exists() else None
    fp16_size = pair.fp16_path.stat().st_size / (1024 * 1024) if pair.fp16_path and pair.fp16_path.exists() else None

    return {
        "candidate": pair.candidate,
        "p": pair.p,
        "int8_map50_95": rounded(int8_map),
        "int8_map50": rounded(safe_metric(int8_metrics, "map50")),
        "int8_precision": rounded(safe_metric(int8_metrics, "precision")),
        "int8_recall": rounded(safe_metric(int8_metrics, "recall")),
        "fp16_map50_95": rounded(fp16_map),
        "fp16_map50": rounded(safe_metric(fp16_metrics, "map50")),
        "fp16_precision": rounded(safe_metric(fp16_metrics, "precision")),
        "fp16_recall": rounded(safe_metric(fp16_metrics, "recall")),
        "delta_fp16_vs_int8_map50_95": rounded(delta_fp16_int8),
        "fp16_loss_ratio_vs_baseline": rounded(fp16_loss_ratio),
        "fp16_pass_max_loss": fp16_pass,
        "int8_size_mb": rounded(int8_size, 2),
        "fp16_size_mb": rounded(fp16_size, 2),
        "int8_path": str(pair.int8_path) if pair.int8_path else "",
        "fp16_path": str(pair.fp16_path) if pair.fp16_path else "",
        "int8_found_files": len(pair.all_int8),
        "fp16_found_files": len(pair.all_fp16),
    }


def write_csv(path: Path, rows: List[Dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()), delimiter=";")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)


def write_markdown(path: Path, rows: List[Dict[str, Any]]) -> None:
    columns = [
        "candidate",
        "p",
        "int8_map50_95",
        "int8_precision",
        "int8_recall",
        "fp16_map50_95",
        "fp16_precision",
        "fp16_recall",
        "delta_fp16_vs_int8_map50_95",
        "fp16_loss_ratio_vs_baseline",
        "fp16_pass_max_loss",
        "int8_size_mb",
        "fp16_size_mb",
    ]
    headers = [
        "Кандидат",
        "p",
        "INT8 mAP50-95",
        "INT8 Precision",
        "INT8 Recall",
        "FP16 mAP50-95",
        "FP16 Precision",
        "FP16 Recall",
        "FP16−INT8 Δ mAP50-95",
        "Потеря FP16 к baseline",
        "FP16 ≤ порога",
        "INT8, МБ",
        "FP16, МБ",
    ]

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("| " + " | ".join(["---"] * len(headers)) + " |\n")
        for row in rows:
            f.write("| " + " | ".join(fmt(row.get(col)) for col in columns) + " |\n")


def sort_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    def key(row: Dict[str, Any]) -> Tuple[int, float, str]:
        p = row.get("p")
        return (0 if p is not None else 1, float(p) if p is not None else 999.0, str(row.get("candidate", "")))

    return sorted(rows, key=key)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Validate TFLite INT8 and FP16 metrics for all candidates.")
    parser.add_argument("--outputs", required=True, type=Path, help="Path to outputs directory with candidate folders.")
    parser.add_argument("--config", required=True, type=Path, help="YAML config used for the experiment.")
    parser.add_argument("--data", default=None, help="Dataset YAML. Defaults to model.data from config.")
    parser.add_argument("--imgsz", default=None, type=int, help="Image size. Defaults to model.imgsz from config, then 640.")
    parser.add_argument("--task", default=None, help="Ultralytics task. Defaults to model.task from config, then detect.")
    parser.add_argument("--split", default="val", help="Dataset split for validation. Default: val.")
    parser.add_argument("--batch", default=1, type=int, help="Batch size for TFLite validation. Default: 1.")
    parser.add_argument("--workers", default=0, type=int, help="Dataloader workers. Default: 0 for Windows stability.")
    parser.add_argument("--out-dir", default=None, type=Path, help="Directory for reports. Default: <outputs>/fp16_metrics_report.")
    parser.add_argument("--include-candidates-regex", default=None, help="Optional regex to validate only matching candidate folder names.")
    parser.add_argument("--max-candidates", default=None, type=int, help="Limit number of candidate folders for a quick test.")
    parser.add_argument("--baseline-map50-95", default=None, type=float, help="Optional baseline mAP50-95 for 20%% loss check.")
    parser.add_argument("--max-loss-ratio", default=0.20, type=float, help="Allowed loss ratio vs baseline. Default: 0.20.")
    parser.add_argument("--no-cache", action="store_true", help="Disable validation metrics cache.")
    parser.add_argument("--verbose", action="store_true", help="Show verbose Ultralytics validation output.")
    args = parser.parse_args(argv)

    outputs = args.outputs.resolve()
    config = args.config.resolve()
    if not outputs.exists():
        raise SystemExit(f"Outputs directory not found: {outputs}")
    if not config.exists():
        raise SystemExit(f"Config not found: {config}")

    cfg = load_yaml(config)
    data = args.data or get_from_config(cfg, "model.data")
    imgsz = args.imgsz or get_from_config(cfg, "model.imgsz", 640)
    task = args.task or get_from_config(cfg, "model.task", "detect")

    if not data:
        raise SystemExit("Dataset YAML was not provided. Use --data or set model.data in config.")
    imgsz = int(imgsz)

    out_dir = (args.out_dir or (outputs / "fp16_metrics_report")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    val_project_dir = out_dir / "ultralytics_val_runs"
    cache_path = out_dir / "tflite_metrics_cache.json"
    cache = {} if args.no_cache else load_cache(cache_path)

    pairs = find_model_pairs(outputs, include_candidates_regex=args.include_candidates_regex)
    if args.max_candidates is not None:
        pairs = pairs[: args.max_candidates]

    if not pairs:
        raise SystemExit(
            "No candidate folders with recognizable INT8/FP16 .tflite files were found. "
            "Check file names: they should contain tokens like int8/qat/quant or fp16/float16."
        )

    print(f"Found candidates: {len(pairs)}")
    print(f"Dataset: {data}; imgsz={imgsz}; task={task}; split={args.split}")
    print(f"Reports: {out_dir}")

    rows: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for idx, pair in enumerate(pairs, start=1):
        print(f"\n[{idx}/{len(pairs)}] {pair.candidate}")
        print(f"  INT8: {pair.int8_path if pair.int8_path else 'not found'}")
        print(f"  FP16: {pair.fp16_path if pair.fp16_path else 'not found'}")

        metrics_by_kind: Dict[str, Optional[Dict[str, Any]]] = {"int8": None, "fp16": None}

        for kind, path in (("int8", pair.int8_path), ("fp16", pair.fp16_path)):
            if path is None:
                continue
            key = cache_key(path, data=data, imgsz=imgsz, split=args.split, task=task, batch=args.batch)
            if not args.no_cache and key in cache:
                print(f"  {kind.upper()}: using cached metrics")
                metrics_by_kind[kind] = cache[key]
                continue

            try:
                print(f"  {kind.upper()}: validating...")
                run_name = re.sub(r"[^a-zA-Z0-9_.-]+", "_", f"{pair.candidate}_{kind}")
                result = validate_model(
                    path,
                    data=data,
                    imgsz=imgsz,
                    split=args.split,
                    task=task,
                    batch=args.batch,
                    workers=args.workers,
                    project_dir=val_project_dir,
                    run_name=run_name,
                    verbose=args.verbose,
                )
                metrics_by_kind[kind] = result
                cache[key] = result
                if not args.no_cache:
                    save_cache(cache_path, cache)
                print(
                    f"  {kind.upper()}: mAP50-95={result['map50_95']:.4f}, "
                    f"P={result['precision']:.4f}, R={result['recall']:.4f}"
                )
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}"
                print(f"  {kind.upper()}: ERROR: {message}")
                errors.append({"candidate": pair.candidate, "kind": kind, "path": str(path), "error": message})

        row = build_row(
            pair,
            int8_metrics=metrics_by_kind["int8"],
            fp16_metrics=metrics_by_kind["fp16"],
            baseline_map50_95=args.baseline_map50_95,
            max_loss_ratio=args.max_loss_ratio,
        )
        rows.append(row)

    rows = sort_rows(rows)
    csv_path = out_dir / "tflite_int8_fp16_metrics.csv"
    json_path = out_dir / "tflite_int8_fp16_metrics.json"
    md_path = out_dir / "tflite_int8_fp16_metrics.md"
    errors_path = out_dir / "tflite_int8_fp16_errors.json"

    write_csv(csv_path, rows)
    write_json(json_path, rows)
    write_markdown(md_path, rows)
    write_json(errors_path, errors)

    print("\nDone.")
    print(f"CSV:  {csv_path}")
    print(f"JSON: {json_path}")
    print(f"MD:   {md_path}")
    if errors:
        print(f"Errors: {errors_path}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
