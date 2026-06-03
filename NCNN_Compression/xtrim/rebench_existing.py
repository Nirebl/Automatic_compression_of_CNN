from __future__ import annotations

import dataclasses
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any, Iterable, Optional

from .pareto import pareto_front
from .types import CandidateConfig, HistoryItem, Metrics, NcnnModelPaths
from .utils import ensure_dir


_CANDIDATE_RE = re.compile(r"(w[\d.]+_p[\d.]+_r[\d.]+_s[\d.]+)")


def _json_safe(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if dataclasses.is_dataclass(value):
        return _json_safe(dataclasses.asdict(value))
    if hasattr(value, "__dict__"):
        return _json_safe(vars(value))
    return repr(value)


def _load_history_jsonl(path: Path) -> list[HistoryItem]:
    items: list[HistoryItem] = []
    if not path.exists():
        return items

    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        items.append(
            HistoryItem(
                candidate=CandidateConfig(**rec["candidate"]),
                metrics=Metrics(**rec["metrics"]),
                artifacts_dir=str(rec.get("artifacts_dir", "")),
                extra=rec.get("extra", {}) or {},
            )
        )
    return items


def _append_history(path: Path, item: HistoryItem) -> None:
    rec = {
        "candidate": dataclasses.asdict(item.candidate),
        "metrics": dataclasses.asdict(item.metrics),
        "artifacts_dir": item.artifacts_dir,
        "extra": item.extra,
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")


def _extract_candidate_name_from_dir(path: Path) -> Optional[str]:
    if "baseline_raw" in path.name:
        return "baseline_raw"
    match = _CANDIDATE_RE.search(path.name)
    return match.group(1) if match else None


def _path_from_extra(extra: dict[str, Any], key: str, source_root: Path) -> Optional[Path]:
    raw = extra.get(key)
    if not raw:
        return None
    p = Path(str(raw))
    candidates = [p]
    if not p.is_absolute():
        candidates.append(source_root / p)
    for c in candidates:
        if c.exists():
            return c.resolve()
    return None


def _score_path(path: Path, words: Iterable[str]) -> tuple[int, int, str]:
    text = str(path).lower().replace("\\", "/")
    score = 0
    for i, word in enumerate(words):
        if word in text:
            score += 100 - i
    # Prefer deeper/specific artifacts over random root files when scores tie.
    return (-score, -len(path.parts), str(path))


def _find_existing_ncnn_pair(candidate_dir: Path, extra: dict[str, Any], source_root: Path) -> tuple[Optional[Path], Optional[Path]]:
    param = _path_from_extra(extra, "deploy_ncnn_param", source_root)
    binf = _path_from_extra(extra, "deploy_ncnn_bin", source_root)
    if param is not None and binf is not None and param.exists() and binf.exists():
        return param, binf

    param_files = [p for p in candidate_dir.rglob("*.param") if p.with_suffix(".bin").exists()]
    if not param_files:
        return None, None

    priority_words = (
        "int8",
        "qat",
        "deploy",
        "final",
        "opt",
        "model",
        "ncnn",
    )
    param_files.sort(key=lambda p: _score_path(p, priority_words))
    best = param_files[0]
    return best.resolve(), best.with_suffix(".bin").resolve()


def _find_existing_onnx(candidate_dir: Path, extra: dict[str, Any], source_root: Path) -> Optional[Path]:
    for key in (
        "deploy_onnx_path",
        "deploy_onnx",
        "onnx_path",
        "export_onnx",
    ):
        p = _path_from_extra(extra, key, source_root)
        if p is not None:
            return p

    onnx_files = list(candidate_dir.rglob("*.onnx"))
    if not onnx_files:
        return None

    priority_words = (
        "qat_int8",
        "qat-int8",
        "int8_after_qat",
        "after_qat",
        "qat",
        "int8",
        "deploy",
        "final",
        "model",
    )
    onnx_files.sort(key=lambda p: _score_path(p, priority_words))
    return onnx_files[0].resolve()


def _find_existing_tflite_models(candidate_dir: Path, extra: dict[str, Any], source_root: Path) -> dict[str, Path]:
    models: dict[str, Path] = {}

    artifacts = extra.get("tflite_artifacts")
    if isinstance(artifacts, dict):
        for key, raw in artifacts.items():
            p = Path(str(raw))
            candidates = [p] if p.is_absolute() else [source_root / p, candidate_dir / p]
            for c in candidates:
                if c.exists():
                    models[str(key)] = c.resolve()
                    break

    for key, aliases in {
        "int8": ("tflite_int8_path", "model_int8.tflite", "full_integer_quant", "integer_quant", "int8"),
        "fp16": ("tflite_fp16_path", "model_fp16.tflite", "float16", "fp16"),
    }.items():
        if key in models:
            continue
        for extra_key in aliases[:1]:
            p = _path_from_extra(extra, extra_key, source_root)
            if p is not None:
                models[key] = p.resolve()
                break
        if key in models:
            continue
        files = list(candidate_dir.rglob("*.tflite"))
        if not files:
            continue
        # Prefer stable copied filenames, then converter-specific names.
        priority = tuple(str(x) for x in aliases[1:])
        files.sort(key=lambda p: _score_path(p, priority))
        best = files[0]
        if best.exists():
            models[key] = best.resolve()

    if "int8" in models:
        models.setdefault("tflite_int8", models["int8"])
    if "fp16" in models:
        models.setdefault("tflite_fp16", models["fp16"])
    return models



def _find_existing_ncnn_source_onnx(candidate_dir: Path, extra: dict[str, Any], source_root: Path) -> Optional[Path]:
    """Find a FP32/QAT-FP32 ONNX file suitable for onnx2ncnn.

    Deploy ONNX may be INT8/QDQ. That is fine for ORT, but it is not the safest
    source for NCNN conversion. Prefer model_qat.onnx or model.onnx.
    """
    for key in (
        "ncnn_source_onnx_path",
        "deploy_onnx_qat_path",
        "onnx_qat_path",
    ):
        p = _path_from_extra(extra, key, source_root)
        if p is not None:
            return p

    onnx_files = list(candidate_dir.rglob("*.onnx"))
    if not onnx_files:
        return None

    def is_int8(path: Path) -> bool:
        text = path.name.lower()
        return "int8" in text or "quant" in text or "qdq" in text

    fp32_like = [p for p in onnx_files if not is_int8(p)] or onnx_files
    priority_words = (
        "model_qat.onnx",
        "qat",
        "model.onnx",
        "fp32",
        "export",
        "model",
    )
    fp32_like.sort(key=lambda p: _score_path(p, priority_words))
    return fp32_like[0].resolve()


def _history_from_dirs(source_root: Path) -> list[HistoryItem]:
    items: list[HistoryItem] = []
    for path in sorted(source_root.iterdir()):
        if not path.is_dir():
            continue
        name = _extract_candidate_name_from_dir(path)
        if name is None:
            continue
        if name == "baseline_raw":
            cand = CandidateConfig(width_mult=1.0, prune_ratio=0.0, lowrank_rank=0, sparse_1x1=0.0, tag="baseline_raw")
            extra = {"is_reference_baseline": True, "search_excluded": True}
        else:
            m = re.search(r"w(?P<w>[\d.]+)_p(?P<p>[\d.]+)_r(?P<r>[\d.]+)_s(?P<s>[\d.]+)", name)
            cand = CandidateConfig(
                width_mult=float(m.group("w")) if m else 1.0,
                prune_ratio=float(m.group("p")) if m else 0.0,
                lowrank_rank=int(float(m.group("r"))) if m else 0,
                sparse_1x1=float(m.group("s")) if m else 0.0,
                tag=name,
            )
            extra = {}
        items.append(
            HistoryItem(
                candidate=cand,
                metrics=Metrics(acc=0.0, size_bytes=0, latency_ms={}),
                artifacts_dir=str(path),
                extra=extra,
            )
        )
    return items


def _is_failed_or_non_deploy(item: HistoryItem) -> bool:
    if item.extra.get("failed"):
        return True
    # Keep reference baseline and real candidates. Skip other helper/history rows.
    if item.extra.get("search_excluded") and not item.extra.get("is_reference_baseline"):
        return True
    return False


def _copy_source_history(source_history: Path, out_history_archive_dir: Path) -> None:
    if not source_history.exists():
        return
    ensure_dir(out_history_archive_dir)
    dest = out_history_archive_dir / f"source_history_{int(time.time())}.jsonl"
    try:
        shutil.copy2(source_history, dest)
    except Exception:
        pass


def rebench_existing(
    orchestrator: Any,
    run_dir: str | Path | None = None,
    cfg: Any = None,
    latency_cfg: Any = None,
    devices: Any = None,
    android_app_bench_cfg: Any = None,
    ort_android_bench_cfg: Any = None,
    benchmark_profiles: Any = None,
    *,
    source_run_dir: str | Path | None = None,
    out_root: str | Path | None = None,
) -> list[HistoryItem]:
    """Re-runs latency benchmarks for already generated artifacts.

    No pruning, no training, no QAT, no PTQ and no export are performed.
    The returned history keeps the original accuracy/size metrics and replaces
    only latency fields with fresh measurements from the current config.
    """
    source_root = Path(source_run_dir or run_dir or "").resolve()
    if not source_root.exists():
        raise FileNotFoundError(f"Rebench source directory does not exist: {source_root}")

    output_root = Path(out_root or getattr(orchestrator, "out_root", source_root)).resolve()
    ensure_dir(output_root)
    ensure_dir(output_root / "history")

    # Make sure stale latency cache is not reused during rebench.
    try:
        orchestrator.latency_cfg = dataclasses.replace(orchestrator.latency_cfg, use_cache=False, force_rebench=True)
    except Exception:
        pass

    if latency_cfg is not None:
        try:
            latency_cfg.use_cache = False
            latency_cfg.force_rebench = True
        except Exception:
            pass

    try:
        orchestrator.prepare_benchmark_devices()
    except AttributeError:
        # Older orchestrators did not expose the method.  The explicit benchmark
        # method below will still check devices through the concrete backends.
        pass

    source_history_path = source_root / "history.jsonl"
    source_items = _load_history_jsonl(source_history_path)
    if not source_items:
        source_items = _history_from_dirs(source_root)

    source_items = [item for item in source_items if not _is_failed_or_non_deploy(item)]
    if not source_items:
        print(f"[rebench] No deployable candidates found in: {source_root}")
        return []

    # Fresh output history for this rebench run.
    out_history = output_root / "history.jsonl"
    if out_history.exists():
        archive = output_root / "history" / f"previous_rebench_{int(time.time())}.jsonl"
        shutil.copy2(out_history, archive)
        out_history.unlink()
    _copy_source_history(source_history_path, output_root / "history")

    print()
    print("=" * 120)
    print("  REBENCH EXISTING CANDIDATES")
    print("=" * 120)
    print(f"[rebench] source:     {source_root}")
    print(f"[rebench] output:     {output_root}")
    print(f"[rebench] backend:    {getattr(orchestrator.latency_cfg, 'backend', None)}")
    print(f"[rebench] profiles:   {len(getattr(orchestrator, 'benchmark_profiles', []) or [])}")
    print(f"[rebench] candidates: {len(source_items)}")
    print("=" * 120)

    rows: list[dict[str, Any]] = []
    result_history: list[HistoryItem] = []

    for index, item in enumerate(source_items, start=1):
        source_artifacts_dir = Path(item.artifacts_dir).resolve() if item.artifacts_dir else source_root
        if not source_artifacts_dir.exists():
            source_artifacts_dir = source_root

        tag = item.candidate.tag or ("baseline_raw" if item.extra.get("is_reference_baseline") else f"candidate_{index}")
        safe_tag = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in tag) or f"candidate_{index}"
        out_dir = output_root / f"{index:03d}_{safe_tag}"
        ensure_dir(out_dir)

        ncnn_param, ncnn_bin = _find_existing_ncnn_pair(source_artifacts_dir, item.extra, source_root)
        onnx_model = _find_existing_onnx(source_artifacts_dir, item.extra, source_root)
        tflite_models = _find_existing_tflite_models(source_artifacts_dir, item.extra, source_root)

        # If the original run saved only ONNX artifacts, a rebench config can
        # still add NCNN CPU/Vulkan profiles. Generate NCNN from a FP32/QAT-FP32
        # ONNX source in the rebench output directory instead of touching the
        # old run.
        generated_ncnn_extra: dict[str, Any] = {}
        try:
            needs_ncnn = bool(orchestrator._needs_ncnn_artifact())
        except Exception:
            needs_ncnn = False
        if needs_ncnn and (ncnn_param is None or ncnn_bin is None):
            ncnn_source_onnx = _find_existing_ncnn_source_onnx(source_artifacts_dir, item.extra, source_root)
            if ncnn_source_onnx is not None:
                ncnn_paths, generated_ncnn_extra = orchestrator.prepare_ncnn_from_existing_onnx(
                    source_onnx=ncnn_source_onnx,
                    run_dir=out_dir,
                    source_label="rebench_existing_onnx_fp32",
                )
                if ncnn_paths is not None:
                    ncnn_param, ncnn_bin = ncnn_paths.param, ncnn_paths.bin

        print()
        print("-" * 120)
        print(f"[rebench] {index}/{len(source_items)}: {tag}")
        print(f"[rebench] source artifacts: {source_artifacts_dir}")
        print(f"[rebench] ncnn param:       {ncnn_param}")
        print(f"[rebench] ncnn bin:         {ncnn_bin}")
        print(f"[rebench] onnx:             {onnx_model}")
        print(f"[rebench] tflite:           {tflite_models}")
        print("-" * 120)

        try:
            latency_ms, bench_extra, deploy_backend = orchestrator._benchmark_existing_deploy_model(
                ncnn_param=ncnn_param,
                ncnn_bin=ncnn_bin,
                onnx_model=onnx_model,
                tflite_models=tflite_models,
                run_dir=out_dir,
            )

            extra = dict(item.extra or {})
            extra.update(generated_ncnn_extra)
            extra.update(bench_extra)
            extra["rebench"] = True
            extra["rebench_source_artifacts_dir"] = str(source_artifacts_dir)
            extra["rebench_source_root"] = str(source_root)
            extra["deploy_backend"] = deploy_backend
            extra["primary_latency_backend"] = str(getattr(orchestrator.latency_cfg, "backend", ""))
            extra["rebench_ncnn_param"] = str(ncnn_param) if ncnn_param else None
            extra["rebench_ncnn_bin"] = str(ncnn_bin) if ncnn_bin else None
            extra["rebench_onnx_model"] = str(onnx_model) if onnx_model else None
            extra["rebench_tflite_models"] = {k: str(v) for k, v in sorted(tflite_models.items())}

            lat_agg = orchestrator._latency_aggregate(latency_ms)
            extra["latency_agg_ms"] = float(lat_agg)
            try:
                extra["scalar_score"] = orchestrator._scalarize(float(item.metrics.acc), float(lat_agg), int(item.metrics.size_bytes))
            except Exception:
                pass

            metrics = dataclasses.replace(item.metrics, latency_ms=latency_ms)
            new_item = HistoryItem(
                candidate=item.candidate,
                metrics=metrics,
                artifacts_dir=str(out_dir),
                extra=extra,
            )
            result_history.append(new_item)
            _append_history(out_history, new_item)

            rows.append(
                {
                    "candidate": tag,
                    "source_artifacts_dir": str(source_artifacts_dir),
                    "output_dir": str(out_dir),
                    "latency_ms": latency_ms,
                    "status": "ok",
                    "ts": time.time(),
                }
            )
            print(f"[rebench] OK {tag}: {latency_ms}")

        except Exception as exc:
            extra = dict(item.extra or {})
            extra.update(
                {
                    "failed": True,
                    "rebench": True,
                    "rebench_source_artifacts_dir": str(source_artifacts_dir),
                    "error": repr(exc),
                }
            )
            failed_item = HistoryItem(
                candidate=item.candidate,
                metrics=dataclasses.replace(item.metrics, latency_ms={}),
                artifacts_dir=str(out_dir),
                extra=extra,
            )
            result_history.append(failed_item)
            _append_history(out_history, failed_item)
            rows.append(
                {
                    "candidate": tag,
                    "source_artifacts_dir": str(source_artifacts_dir),
                    "output_dir": str(out_dir),
                    "status": "failed",
                    "error": repr(exc),
                    "ts": time.time(),
                }
            )
            print(f"[rebench] FAILED {tag}: {exc}")

    # Save a machine-readable rebench log and Pareto set for the new latency measurements.
    rebench_results = output_root / "rebench_results.jsonl"
    with rebench_results.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(_json_safe(row), ensure_ascii=False, default=str) + "\n")

    ok_items = [h for h in result_history if not h.extra.get("failed") and not h.extra.get("exclude_from_pareto")]
    try:
        front = pareto_front(ok_items)
        (output_root / "pareto.json").write_text(
            json.dumps(
                [
                    {
                        "candidate": dataclasses.asdict(h.candidate),
                        "metrics": dataclasses.asdict(h.metrics),
                        "artifacts_dir": h.artifacts_dir,
                        "extra": h.extra,
                    }
                    for h in front
                ],
                ensure_ascii=False,
                indent=2,
                default=str,
            ),
            encoding="utf-8",
        )
    except Exception:
        pass

    try:
        orchestrator.cache.save()
    except Exception:
        pass

    print()
    print("=" * 120)
    print("  REBENCH DONE")
    print("=" * 120)
    print(f"[rebench] history saved to: {out_history}")
    print(f"[rebench] results saved to: {rebench_results}")
    print(f"[rebench] ok:      {sum(1 for row in rows if row.get('status') == 'ok')}")
    print(f"[rebench] failed:  {sum(1 for row in rows if row.get('status') == 'failed')}")
    print("=" * 120)

    return result_history
