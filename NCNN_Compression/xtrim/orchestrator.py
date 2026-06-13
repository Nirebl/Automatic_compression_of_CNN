from __future__ import annotations

import dataclasses
import json
import math
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional  # noqa: F401

from .android import AdbYoloDemo
from .bench_cache import BenchCache
from .exporter import Exporter
from .ncnn import NcnnConverter, AdbBench
from .pareto import pareto_front, avg_latency
from .search import SearchPolicy
from .types import (
    ToolsConfig,
    DeviceConfig,
    TrainConfig,
    ExportConfig,
    PTQConfig,
    CandidateConfig,
    Metrics,
    HistoryItem,
    LatencyConfig,
    OnnxPTQConfig,
    QATConfig,
    AndroidDemoConfig,
    SearchConfig,
    TrimConfig,
    StagedPruningConfig,
    OrtAndroidBenchConfig,
    BenchmarkProfileConfig,
    NcnnModelPaths,
)
from .android_app_bench import AndroidAppBench, AndroidAppBenchConfig
from .android_ort_bench import AndroidOrtBench
from .android_tflite_bench import AndroidTfliteBench
from .runtime_backend import effective_ncnn_gpu_device, ncnn_runtime_label
from .android_dataset import AndroidDatasetSubset, prepare_android_dataset_subset
from .utils import ensure_dir, sizeof_file, write_json, now_ts, sha256_file


class XTrimOrchestrator:
    def __init__(
        self,
        *,
        out_root: Path,
        tools: ToolsConfig,
        devices: List[DeviceConfig],
        train_cfg: TrainConfig,
        export_cfg: ExportConfig,
        ptq_cfg: PTQConfig,
        latency_cfg: LatencyConfig,
        onnx_ptq_cfg: OnnxPTQConfig,
        qat_cfg: QATConfig,
        android_demo_cfg: AndroidDemoConfig,
        search_cfg: SearchConfig,
        search_space: Dict[str, List[Any]],
        trim_cfg: Optional[TrimConfig] = None,
        staged_pruning_cfg: Optional[StagedPruningConfig] = None,
        build_candidate_fn: Callable[[CandidateConfig], Any],
        apply_pruning_stage_fn: Optional[Callable[..., Dict[str, Any]]] = None,
        extract_pruning_architecture_fn: Optional[Callable[[Any], Dict[str, Any]]] = None,
        warmstart_fn: Callable[[Any], None],
        finetune_fn: Callable[[Any, TrainConfig], Any],
        finetune_qat_fn: Optional[Callable[[Any, TrainConfig], Any]] = None,
        eval_acc_fn: Callable[[Any], float] = lambda _: 0.0,
        eval_metrics_fn: Optional[Callable[[Any], Dict[str, float]]] = None,
        export_onnx_fn_factory: Callable[[Any, ExportConfig], Callable[[Path], None]] = lambda _m, _c: (lambda _p: None),
        export_tflite_int8_fn_factory: Optional[Callable[[Any, ExportConfig], Callable[[Path], Path]]] = None,
        export_tflite_fp32_fn_factory: Optional[Callable[[Any, ExportConfig], Callable[[Path], Path]]] = None,
        eval_exported_onnx_fn: Optional[Callable[[Path], float]] = None,
        eval_exported_onnx_metrics_fn: Optional[Callable[[Path], Dict[str, float]]] = None,
        quantize_onnx_fn: Optional[Callable[[Path, Path, Path], Path]] = None,
        save_student_pt_fn: Optional[Callable[[Any, Path], None]] = None,
        android_app_bench_cfg: AndroidAppBenchConfig,
        ort_android_bench_cfg: OrtAndroidBenchConfig,
        dataset_yaml: str = "coco128.yaml",
        benchmark_profiles: Optional[List[BenchmarkProfileConfig]] = None,
    ):
        self.out_root = out_root
        self.tools = tools
        self.devices = devices
        self.train_cfg = train_cfg
        self.export_cfg = export_cfg
        self.ptq_cfg = ptq_cfg
        self.latency_cfg = latency_cfg
        self.onnx_ptq_cfg = onnx_ptq_cfg
        self.qat_cfg = qat_cfg
        self.android_demo_cfg = android_demo_cfg
        self.search_cfg = search_cfg
        self.search_space = search_space
        self.trim_cfg = trim_cfg or TrimConfig()
        self.staged_pruning_cfg = staged_pruning_cfg or StagedPruningConfig()

        self.build_candidate_fn = build_candidate_fn
        self.apply_pruning_stage_fn = apply_pruning_stage_fn
        self.extract_pruning_architecture_fn = extract_pruning_architecture_fn
        self.warmstart_fn = warmstart_fn
        self.finetune_fn = finetune_fn
        self.finetune_qat_fn = finetune_qat_fn
        self.eval_acc_fn = eval_acc_fn
        self.eval_metrics_fn = eval_metrics_fn
        self.export_onnx_fn_factory = export_onnx_fn_factory
        self.export_tflite_int8_fn_factory = export_tflite_int8_fn_factory
        self.export_tflite_fp32_fn_factory = export_tflite_fp32_fn_factory
        self.eval_exported_onnx_fn = eval_exported_onnx_fn
        self.eval_exported_onnx_metrics_fn = eval_exported_onnx_metrics_fn
        self.quantize_onnx_fn = quantize_onnx_fn
        self.save_student_pt_fn = save_student_pt_fn

        self.converter = NcnnConverter(tools)
        self.bench = AdbBench(tools)
        self.demo = AdbYoloDemo(tools)

        self.android_app_bench_cfg = android_app_bench_cfg
        self.android_app = AndroidAppBench(tools, android_app_bench_cfg)
        self.ort_android_bench_cfg = ort_android_bench_cfg
        self.android_ort = AndroidOrtBench(tools, ort_android_bench_cfg)
        self.dataset_yaml = str(dataset_yaml)
        self.benchmark_profiles = list(benchmark_profiles or [])

        ensure_dir(self.out_root)
        self.history_path = self.out_root / "history.jsonl"
        self.history_archive_dir = self.out_root / "history"
        ensure_dir(self.history_archive_dir)

        cache_path = self.out_root / self.latency_cfg.cache_file
        self.cache = BenchCache(cache_path)

        self.policy = SearchPolicy.create(search_cfg, search_space)


    def prepare_benchmark_devices(self) -> List[DeviceConfig]:
        """Проверяет устройства и оставляет только те, на которых можно запускать выбранные бенчмарки."""
        backends_to_prepare = {str(self.latency_cfg.backend).lower().strip()}
        backends_to_prepare.update(p.backend.lower().strip() for p in self._active_benchmark_profiles())

        ready_devices: List[DeviceConfig] = []
        for d in self.devices:
            if not self.bench.is_device_ready(d):
                print(f"[warn] device {d.name} not ready (adb get-state != device)", file=sys.stderr)
                continue

            if "benchncnn" in backends_to_prepare:
                try:
                    self.bench.ensure_benchncnn(d, force_push=False)
                except Exception as e:
                    print(f"[warn] device {d.name} not usable: {e}", file=sys.stderr)
                    continue

            ready_devices.append(d)

        self.devices = ready_devices
        return ready_devices

    def run(self, max_candidates: Optional[int] = None) -> List[HistoryItem]:
        history: List[HistoryItem] = self._load_history()
        self._ensure_reference_baseline(history)

        self.prepare_benchmark_devices()

        count = 0
        while True:
            if max_candidates is not None and count >= max_candidates:
                break

            cand = self.policy.next_candidate(history)
            if cand is None:
                print("[info] search space exhausted")
                break

            count += 1
            run_dir = self.out_root / f"{now_ts()}_{cand.tag}"
            ensure_dir(run_dir)
            print(f"\n=== Candidate: {cand.tag} -> {run_dir} ===")

            try:
                item = self._process_candidate(cand, run_dir)
                history.append(item)
                self._append_history(item)

                candidate_history = [
                    h for h in history
                    if not self._is_reference_baseline_item(h)
                       and not h.extra.get("failed", False)
                ]
                front = pareto_front(candidate_history)
                self._write_pareto(front)

                lat_agg = item.extra.get("latency_agg_ms", float("inf"))
                print(f"OK: acc={item.metrics.acc:.4f}, size={item.metrics.size_bytes} bytes, lat(agg)={lat_agg:.2f} ms")
                if "acc_onnx_int8" in item.extra:
                    print(f"    acc_onnx_int8={item.extra['acc_onnx_int8']:.4f} drop={item.extra.get('acc_drop_int8', 0.0):+.4f}")
                if item.extra.get("qat_triggered"):
                    print(f"    QAT triggered; int8_after={item.extra.get('acc_onnx_int8_after_qat', None)}")
                n_candidates_ok = len([
                    h for h in history
                    if not self._is_reference_baseline_item(h)
                       and not h.extra.get("failed", False)
                ])
                print(f"Pareto size: {len(front)} / {n_candidates_ok}")

            except Exception as e:
                import traceback
                err_text = traceback.format_exc()
                (run_dir / "error.txt").write_text(err_text, encoding="utf-8")
                print(f"FAILED: {cand.tag}\n{e}", file=sys.stderr)
                failed_item = HistoryItem(
                    candidate=cand,
                    metrics=Metrics(acc=0.0, size_bytes=0, latency_ms={}),
                    artifacts_dir=str(run_dir),
                    extra={"failed": True, "error": str(e)},
                )
                history.append(failed_item)
                self._append_history(failed_item)
                continue

        try:
            self.cache.save()
        except Exception as e:
            print(e)
            pass

        self._archive_history()

        return history

    def _latency_aggregate(self, latency_ms: Dict[str, float]) -> float:
        if not latency_ms:
            return float("inf")
        if str(self.latency_cfg.aggregate).lower() == "max":
            return float(max(latency_ms.values()))
        return float(avg_latency(latency_ms))

    def _scalarize(self, acc: float, latency_agg_ms: float, size_bytes: int) -> float:
        a = float(self.latency_cfg.scalar_alpha)
        b = float(self.latency_cfg.scalar_beta)
        lat_term = math.log1p(max(0.0, latency_agg_ms))
        size_mb = max(0.0, size_bytes) / 1e6
        size_term = math.log1p(size_mb)
        return float(acc - a * lat_term - b * size_term)

    def _active_benchmark_profiles(self) -> List[BenchmarkProfileConfig]:
        return [p for p in self.benchmark_profiles if bool(p.enabled)]

    def _benchmark_profiles_require_backend(self, backend: str) -> bool:
        backend = backend.lower().strip()
        return any(p.backend.lower().strip() == backend for p in self._active_benchmark_profiles())

    def _benchmark_profiles_require_ncnn(self) -> bool:
        return any(
            p.backend.lower().strip() in {"benchncnn", "android_app", "ncnn"}
            for p in self._active_benchmark_profiles()
        )


    def _export_config_requests_ncnn(self) -> bool:
        return bool(getattr(self.export_cfg, "ncnn", False))

    def _needs_ncnn_artifact(self) -> bool:
        backend = str(self.latency_cfg.backend).lower().strip()
        return (
            backend not in {"ort_android", "ort", "onnx"}
            or self._benchmark_profiles_require_ncnn()
            or self._export_config_requests_ncnn()
            # Legacy NCNN demo still needs NCNN files when explicitly enabled.
            or bool(getattr(self.android_demo_cfg, "enabled", False))
        )

    def _ncnn_int8_enabled(self) -> bool:
        explicit = getattr(self.export_cfg, "ncnn_int8", None)
        if explicit is None:
            return bool(self.ptq_cfg.enabled)
        return bool(explicit)

    def _select_ncnn_source_onnx(
        self,
        *,
        onnx_fp32: Path,
        onnx_qat: Optional[Path] = None,
        deploy_onnx: Optional[Path] = None,
        deploy_onnx_kind: str = "",
    ) -> tuple[Path, str]:
        source = str(getattr(self.export_cfg, "ncnn_source", "qat_fp32") or "qat_fp32").lower().strip()

        if source in {"fp32", "raw_fp32", "model", "before_qat"}:
            return onnx_fp32, "onnx_fp32"

        if source in {"qat", "qat_fp32", "after_qat"}:
            if onnx_qat is not None and onnx_qat.exists():
                return onnx_qat, "onnx_qat_fp32"
            return onnx_fp32, "onnx_fp32_fallback_no_qat"

        if source in {"deploy_fp32", "selected_fp32"}:
            if deploy_onnx_kind in {"qat_fp32", "int8_after_qat"} and onnx_qat is not None and onnx_qat.exists():
                return onnx_qat, "onnx_qat_fp32_for_deploy"
            return onnx_fp32, "onnx_fp32_for_deploy"

        if source in {"deploy", "selected"}:
            if deploy_onnx is not None and deploy_onnx.exists():
                return deploy_onnx, f"onnx_selected_{deploy_onnx_kind or 'unknown'}"
            return onnx_fp32, "onnx_fp32_fallback_no_deploy"

        return onnx_qat if (onnx_qat is not None and onnx_qat.exists()) else onnx_fp32, f"onnx_auto_unknown_source_{source}"

    def _prepare_ncnn_from_onnx(
        self,
        *,
        source_onnx: Path,
        ncnn_dir: Path,
        extra: Dict[str, Any],
        source_label: str,
        pnnx_ncnn: Optional[NcnnModelPaths] = None,
        apply_int8: bool = True,
    ) -> Optional[NcnnModelPaths]:
        ensure_dir(ncnn_dir)
        try:
            if pnnx_ncnn is not None:
                ncnn_float = pnnx_ncnn
                source_label = "pnnx"
            else:
                ncnn_float = self.converter.onnx_to_ncnn(source_onnx, ncnn_dir / "float")
            ncnn_final = self.converter.optimize(ncnn_float, ncnn_dir / "opt") if bool(getattr(self.export_cfg, "ncnn_optimize", True)) else ncnn_float

            if apply_int8 and self._ncnn_int8_enabled():
                auto_calib = ncnn_dir.parent / "ptq" / "calib_images.txt"
                ptq_cfg_resolved = self.ptq_cfg
                if auto_calib.exists():
                    ptq_cfg_resolved = dataclasses.replace(self.ptq_cfg, imagelist=str(auto_calib))
                ncnn_final = self.converter.ptq_int8(ncnn_final, ncnn_dir / "int8", ptq_cfg_resolved)
                extra["ncnn_int8"] = True
                source_label = "ncnn_int8"
            else:
                extra["ncnn_int8"] = False

            extra["ncnn_source"] = source_label
            extra["ncnn_source_onnx_path"] = str(source_onnx)
            extra["deploy_ncnn_param"] = str(ncnn_final.param)
            extra["deploy_ncnn_bin"] = str(ncnn_final.bin)
            extra["ncnn_export"] = "ok"
            return ncnn_final
        except Exception as exc:
            extra["ncnn_export"] = "failed"
            extra["ncnn_export_failed"] = str(exc)
            extra["ncnn_source"] = source_label
            extra["ncnn_source_onnx_path"] = str(source_onnx)
            if bool(getattr(self.export_cfg, "ncnn_required", False)):
                raise
            print(f"[warn] NCNN export failed for {source_onnx}: {exc}", file=sys.stderr)
            return None

    def prepare_ncnn_from_existing_onnx(
        self,
        *,
        source_onnx: Path,
        run_dir: Path,
        extra: Optional[Dict[str, Any]] = None,
        source_label: str = "existing_onnx_fp32",
    ) -> tuple[Optional[NcnnModelPaths], Dict[str, Any]]:
        """Готовит NCNN-файлы из уже существующей ONNX-модели для режима rebench.

        Это позволяет добавить NCNN CPU/Vulkan профили без повторной сборки модели.
        """
        extra_out: Dict[str, Any] = dict(extra or {})
        ncnn = self._prepare_ncnn_from_onnx(
            source_onnx=source_onnx,
            ncnn_dir=run_dir / "ncnn",
            extra=extra_out,
            source_label=source_label,
        )
        return ncnn, extra_out

    @staticmethod
    def _profile_backend(profile: BenchmarkProfileConfig) -> str:
        backend = str(profile.backend).lower().strip()
        if backend == "ncnn":
            return "android_app"
        if backend in {"tflite", "tflite_android", "android_tflite"}:
            return "tflite_android"
        return backend

    @staticmethod
    def _profile_display_name(profile: BenchmarkProfileConfig, index: int) -> str:
        name = str(profile.name or profile.backend or f"profile_{index + 1}").strip()
        return name or f"profile_{index + 1}"

    @staticmethod
    def _profile_log_name(profile_name: str) -> str:
        safe = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in profile_name.strip())
        return safe or "profile"

    def _devices_for_profile(self, profile: BenchmarkProfileConfig) -> List[DeviceConfig]:
        wanted = {str(x) for x in (profile.device_names or ())}
        selected = [
            d for d in self.devices
            if not wanted or d.name in wanted or d.serial in wanted
        ]

        patched: List[DeviceConfig] = []
        for d in selected:
            overrides: Dict[str, Any] = {}
            for key in ("threads", "loops", "powersave", "runtime", "device", "gpu_device", "cooling_down"):
                value = getattr(profile, key, None)
                if value is not None:
                    overrides[key] = value
            patched.append(dataclasses.replace(d, **overrides) if overrides else d)
        return patched

    @staticmethod
    def _cfg_overrides_from_profile(profile: BenchmarkProfileConfig, allowed: List[str]) -> Dict[str, Any]:
        overrides: Dict[str, Any] = {}
        for key in allowed:
            value = getattr(profile, key, None)
            if value is not None:
                overrides[key] = value
        return overrides

    def _ort_cfg_for_profile(self, profile: BenchmarkProfileConfig) -> OrtAndroidBenchConfig:
        allowed = [
            "package", "activity", "dataset", "push_dataset_images", "dataset_split",
            "dataset_max_images", "dataset_seed", "dataset_remote_subdir", "imgsz",
            "loops", "warmup", "threads", "conf", "iou", "max_det", "provider",
            "result_tag", "timeout_sec", "poll_interval_sec", "clear_logcat", "remote_dir",
        ]
        return dataclasses.replace(self.ort_android_bench_cfg, **self._cfg_overrides_from_profile(profile, allowed))

    def _android_app_cfg_for_profile(self, profile: BenchmarkProfileConfig) -> AndroidAppBenchConfig:
        allowed = [
            "package", "activity", "dataset", "push_dataset_images", "dataset_split",
            "dataset_max_images", "dataset_seed", "dataset_remote_subdir", "imgsz",
            "loops", "warmup", "threads", "conf", "iou", "max_det", "optimized",
            "result_tag", "timeout_sec", "poll_interval_sec", "clear_logcat", "remote_dir",
        ]
        return dataclasses.replace(self.android_app_bench_cfg, **self._cfg_overrides_from_profile(profile, allowed))

    def _tflite_cfg_for_profile(self, profile: BenchmarkProfileConfig) -> OrtAndroidBenchConfig:
        allowed = [
            "package", "activity", "dataset", "push_dataset_images", "dataset_split",
            "dataset_max_images", "dataset_seed", "dataset_remote_subdir", "imgsz",
            "loops", "warmup", "threads", "conf", "iou", "max_det",
            "result_tag", "timeout_sec", "poll_interval_sec", "clear_logcat", "remote_dir",
        ]
        return dataclasses.replace(self.ort_android_bench_cfg, **self._cfg_overrides_from_profile(profile, allowed))

    @staticmethod
    def _profile_tflite_delegate(profile: BenchmarkProfileConfig) -> str:
        return str(getattr(profile, "delegate", None) or getattr(profile, "provider", None) or "xnnpack").strip().lower()

    @staticmethod
    def _profile_artifact(profile: BenchmarkProfileConfig) -> str:
        return str(getattr(profile, "artifact", None) or "").strip()

    @staticmethod
    def _normalize_eval_metrics(raw: Any, acc_fallback: Optional[float] = None) -> Dict[str, float]:
        """Приводит результат валидации к единому набору метрик конвейера."""
        metrics: Dict[str, float] = {}

        def _put(dst_key: str, value: Any) -> None:
            if value is None:
                return
            try:
                metrics[dst_key] = float(value)
            except (TypeError, ValueError):
                return

        if isinstance(raw, (int, float)):
            _put("map50_95", raw)
        elif isinstance(raw, dict):
            for src_key in ("map50_95", "mAP50-95", "map", "acc", "accuracy"):
                if src_key in raw:
                    _put("map50_95", raw[src_key])
                    break
            for src_key in ("precision", "mp", "metrics/precision(B)", "metrics/precision"):
                if src_key in raw:
                    _put("precision", raw[src_key])
                    break
            for src_key in ("recall", "mr", "metrics/recall(B)", "metrics/recall"):
                if src_key in raw:
                    _put("recall", raw[src_key])
                    break
            for src_key in ("iou", "IoU", "mean_iou", "miou", "mIoU", "metrics/IoU(B)", "metrics/IoU", "metrics/iou(B)", "metrics/iou", "box/iou"):
                if src_key in raw:
                    _put("iou", raw[src_key])
                    break
            for src_key in ("map50", "mAP50", "metrics/mAP50(B)", "metrics/mAP50"):
                if src_key in raw:
                    _put("map50", raw[src_key])
                    break

        if "map50_95" not in metrics and acc_fallback is not None:
            _put("map50_95", acc_fallback)
        return metrics

    def _eval_student_metrics(self, student: Any) -> Dict[str, float]:
        if self.eval_metrics_fn is not None:
            metrics = self._normalize_eval_metrics(self.eval_metrics_fn(student))
        else:
            metrics = self._normalize_eval_metrics(float(self.eval_acc_fn(student)))
        if "map50_95" not in metrics:
            raise RuntimeError(f"student evaluator returned no map50_95/acc metric: {metrics}")
        return metrics

    def _eval_exported_onnx_metrics(self, onnx_path: Path) -> Dict[str, float]:
        if self.eval_exported_onnx_metrics_fn is not None:
            metrics = self._normalize_eval_metrics(self.eval_exported_onnx_metrics_fn(onnx_path))
        elif self.eval_exported_onnx_fn is not None:
            metrics = self._normalize_eval_metrics(float(self.eval_exported_onnx_fn(onnx_path)))
        else:
            return {}
        if "map50_95" not in metrics:
            raise RuntimeError(f"ONNX evaluator returned no map50_95/acc metric for {onnx_path}: {metrics}")
        return metrics

    @staticmethod
    def _metric_stage_from_acc_key(acc_key: str) -> str:
        return acc_key[4:] if acc_key.startswith("acc_") else acc_key

    def _store_eval_metrics(self, extra: Dict[str, Any], acc_key: str, metrics: Dict[str, float]) -> None:
        metrics = self._normalize_eval_metrics(metrics)
        if "map50_95" in metrics:
            extra[acc_key] = float(metrics["map50_95"])

        stage = self._metric_stage_from_acc_key(acc_key)
        compact: Dict[str, float] = {}
        for metric_key in ("map50_95", "precision", "recall", "iou", "map50"):
            if metric_key not in metrics:
                continue
            value = float(metrics[metric_key])
            compact[metric_key] = value
            if metric_key == "precision":
                extra[f"precision_{stage}"] = value
            elif metric_key == "recall":
                extra[f"recall_{stage}"] = value
            elif metric_key == "iou":
                extra[f"iou_{stage}"] = value
            elif metric_key == "map50":
                extra[f"map50_{stage}"] = value

        if compact:
            extra[f"metrics_{stage}"] = compact

    @staticmethod
    def _store_stage_metrics(stage_rec: Dict[str, Any], suffix: str, metrics: Dict[str, float]) -> None:
        compact: Dict[str, float] = {}
        for key in ("map50_95", "precision", "recall", "iou", "map50"):
            if key not in metrics:
                continue
            try:
                value = float(metrics[key])
            except (TypeError, ValueError):
                continue
            compact[key] = value
            if key == "precision":
                stage_rec[f"precision_{suffix}"] = value
            elif key == "recall":
                stage_rec[f"recall_{suffix}"] = value
            elif key == "iou":
                stage_rec[f"iou_{suffix}"] = value
            elif key == "map50":
                stage_rec[f"map50_{suffix}"] = value
        if compact:
            stage_rec[f"metrics_{suffix}"] = compact

    def _metrics_from_extra(self, extra: Dict[str, Any], acc_key: str) -> Dict[str, float]:
        stage = self._metric_stage_from_acc_key(acc_key)
        raw = extra.get(f"metrics_{stage}")
        metrics = self._normalize_eval_metrics(raw if isinstance(raw, dict) else None)
        if "map50_95" not in metrics and acc_key in extra:
            try:
                metrics["map50_95"] = float(extra[acc_key])
            except (TypeError, ValueError):
                pass
        for metric_key, extra_key in (
            ("precision", f"precision_{stage}"),
            ("recall", f"recall_{stage}"),
            ("iou", f"iou_{stage}"),
            ("map50", f"map50_{stage}"),
        ):
            if metric_key not in metrics and extra_key in extra:
                try:
                    metrics[metric_key] = float(extra[extra_key])
                except (TypeError, ValueError):
                    pass
        return metrics

    def _cache_key(self, device: DeviceConfig, model_hash: str, shape: str) -> str:
        runtime = ncnn_runtime_label(device)
        gpu_device = effective_ncnn_gpu_device(device)
        return "|".join([
            "ncnn_android",
            f"serial={device.serial}",
            f"device_name={device.name}",
            f"runtime={runtime}",
            f"loops={device.loops}",
            f"threads={device.threads}",
            f"powersave={device.powersave}",
            f"gpu={gpu_device}",
            f"cool={device.cooling_down}",
            f"shape={shape}",
            f"model={model_hash}",
            f"backend={self.latency_cfg.backend}",
        ])

    def _hash_ncnn_model(self, param: Path, binf: Path) -> str:
        h1 = sha256_file(param)
        h2 = sha256_file(binf)
        return (h1[:16] + h2[:16])

    def _hash_file_model(self, p: Path) -> str:
        return sha256_file(p)[:32]

    @staticmethod
    def _is_reference_baseline_item(item: HistoryItem) -> bool:
        return bool(item.extra.get("is_reference_baseline", False))

    def _find_reference_baseline(self, history: List[HistoryItem]) -> Optional[HistoryItem]:
        for h in history:
            if self._is_reference_baseline_item(h) and not h.extra.get("failed", False):
                return h
        return None

    def _make_reference_baseline_candidate(self) -> CandidateConfig:
        return CandidateConfig(
            width_mult=1.0,
            prune_ratio=0.0,
            lowrank_rank=0,
            sparse_1x1=0.0,
            tag="baseline_raw",
        )

    def _ensure_reference_baseline(self, history: List[HistoryItem]) -> HistoryItem:
        existing = self._find_reference_baseline(history)
        if existing is not None:
            return existing

        run_dir = self.out_root / f"{now_ts()}_baseline_raw"
        ensure_dir(run_dir)

        print(f"\n=== Reference baseline (raw model) -> {run_dir} ===")
        item = self._process_reference_baseline(run_dir)

        history.append(item)
        self._append_history(item)
        return item

    def _process_reference_baseline(self, run_dir: Path) -> HistoryItem:
        cand = self._make_reference_baseline_candidate()
        student = self.build_candidate_fn(cand)

        extra: Dict[str, Any] = {
            "is_reference_baseline": True,
            "search_excluded": True,
            "baseline_kind": "raw_fp32",
            "reference_note": "raw base model before finetune/compression/PTQ/QAT",
            "acc_kind": "mAP50-95",
        }

        recovered_metrics = self._eval_student_metrics(student)
        acc = float(recovered_metrics["map50_95"])
        self._store_eval_metrics(extra, "acc_recovered_torch", recovered_metrics)

        onnx_path = run_dir / "export" / "model.onnx"
        exporter = Exporter(self.export_onnx_fn_factory(student, self.export_cfg))
        exporter.export_onnx(onnx_path)

        self._maybe_eval_onnx(onnx_path, extra, "acc_onnx")

        _pnnx_ncnn: Optional[Any] = None
        torch_model = getattr(student, "torch_model", None)
        if torch_model is not None:
            try:
                _imgsz = int(self.android_app_bench_cfg.imgsz)
                _pnnx_ncnn = self.converter.pnnx_convert(
                    torch_model, run_dir / "export" / "pnnx_ncnn", imgsz=_imgsz
                )
                extra["pnnx_export"] = "ok"
            except Exception as _pnnx_err:
                extra["pnnx_export_skipped"] = str(_pnnx_err)

        size_bytes = int(sizeof_file(onnx_path))
        latency_ms: Dict[str, float] = {}

        backend = str(self.latency_cfg.backend).lower().strip()
        profiles_active = bool(self._active_benchmark_profiles())
        needs_ncnn = self._needs_ncnn_artifact()
        ncnn_final = None

        extra["deploy_onnx_path"] = str(onnx_path)
        extra["deploy_onnx_kind"] = "fp32_raw_baseline"

        if needs_ncnn:
            source_onnx, source_label = self._select_ncnn_source_onnx(
                onnx_fp32=onnx_path,
                onnx_qat=None,
                deploy_onnx=onnx_path,
                deploy_onnx_kind="fp32_raw_baseline",
            )
            ncnn_final = self._prepare_ncnn_from_onnx(
                source_onnx=source_onnx,
                ncnn_dir=run_dir / "ncnn",
                extra=extra,
                source_label=source_label,
                pnnx_ncnn=_pnnx_ncnn if source_onnx == onnx_path else None,
                apply_int8=False,
            )

        tflite_fp32 = self._maybe_export_tflite_fp32_baseline(student, run_dir, extra)
        tflite_models = self._store_tflite_artifacts(run_dir, extra)
        tflite_models_for_profiles = self._baseline_tflite_models_for_profiles(tflite_models)
        if tflite_fp32 is not None:
            extra["baseline_tflite_kind"] = "fp32"
            extra["baseline_tflite_profile_note"] = (
                "TFLite benchmark profiles for the raw baseline use model_fp32.tflite, "
                "even when profile names contain int8/fp16."
            )

        onnx_size_bytes = int(sizeof_file(onnx_path))
        ncnn_size_bytes = int(sizeof_file(ncnn_final.param) + sizeof_file(ncnn_final.bin)) if ncnn_final is not None else None
        tflite_fp32_size_bytes = int(sizeof_file(tflite_fp32)) if tflite_fp32 is not None else None
        extra["deploy_size_bytes_by_backend"] = {
            "onnx": onnx_size_bytes,
            **({"ncnn": ncnn_size_bytes} if ncnn_size_bytes is not None else {}),
            **({"tflite_fp32": tflite_fp32_size_bytes} if tflite_fp32_size_bytes is not None else {}),
        }

        if profiles_active:
            latency_ms, profile_details = self._bench_latency_with_profiles(
                ncnn_param=ncnn_final.param if ncnn_final is not None else None,
                ncnn_bin=ncnn_final.bin if ncnn_final is not None else None,
                onnx_model=onnx_path,
                tflite_models=tflite_models_for_profiles,
                run_dir=run_dir,
            )
            extra["benchmark_profiles"] = profile_details
            extra["deploy_backend"] = "multi_profile"
            extra["primary_latency_backend"] = backend
            size_bytes = ncnn_size_bytes if backend != "ort_android" and ncnn_size_bytes is not None else onnx_size_bytes
        elif backend == "ort_android":
            size_bytes = onnx_size_bytes
            latency_ms = self._bench_latency(None, None, run_dir, onnx_model=onnx_path)
            extra["deploy_backend"] = "ort_android"
        else:
            if ncnn_final is None:
                raise RuntimeError("NCNN model was not prepared for NCNN latency backend")
            size_bytes = int(ncnn_size_bytes)
            latency_ms = self._bench_latency(ncnn_final.param, ncnn_final.bin, run_dir)
            extra["deploy_backend"] = "ncnn"

        deploy_acc_source = "acc_onnx" if "acc_onnx" in extra else "acc_recovered_torch"
        deploy_metrics = self._metrics_from_extra(extra, deploy_acc_source)
        if "map50_95" not in deploy_metrics:
            deploy_metrics = {"map50_95": float(acc)}
        deploy_acc = float(deploy_metrics["map50_95"])
        self._store_eval_metrics(extra, "acc_deploy", deploy_metrics)
        extra["acc_deploy_source"] = deploy_acc_source

        lat_agg = self._latency_aggregate(latency_ms)
        extra["latency_agg_ms"] = float(lat_agg)
        extra["scalar_score"] = self._scalarize(deploy_acc, lat_agg, size_bytes)
        extra["exclude_from_pareto"] = True

        metrics = Metrics(
            acc=float(deploy_acc),
            size_bytes=int(size_bytes),
            latency_ms=latency_ms,
            precision=deploy_metrics.get("precision"),
            recall=deploy_metrics.get("recall"),
            iou=deploy_metrics.get("iou"),
            map50=deploy_metrics.get("map50"),
        )
        return HistoryItem(
            candidate=cand,
            metrics=metrics,
            artifacts_dir=str(run_dir),
            extra=extra,
        )

    def _prepare_android_dataset_subset(
        self,
        *,
        run_dir: Path,
        backend_name: str,
        split: str,
        max_images: int,
        seed: int,
        remote_dir: str,
        remote_subdir: str,
    ) -> AndroidDatasetSubset:
        out_dir = run_dir / f"{backend_name}_dataset"
        return prepare_android_dataset_subset(
            data_yaml=self.dataset_yaml,
            split=split,
            max_images=int(max_images),
            seed=int(seed),
            out_dir=out_dir,
            remote_dir=remote_dir,
            remote_subdir=remote_subdir,
        )

    @staticmethod
    def _android_dataset_cache_suffix(
        *,
        enabled: bool,
        data_yaml: str,
        split: str,
        max_images: int,
        seed: int,
    ) -> str:
        if not enabled:
            return "dataset=synthetic_or_app_default"
        return f"dataset={data_yaml}|split={split}|n={int(max_images)}|seed={int(seed)}"

    def _bench_devices_with_android_ort(self, onnx_model: Path, run_dir: Path) -> Dict[str, float]:
        latency_ms: Dict[str, float] = {}
        if not self.devices:
            return latency_ms

        logs_dir = run_dir / "android_ort_logs"
        ensure_dir(logs_dir)

        dataset_subset: Optional[AndroidDatasetSubset] = None
        if self.ort_android_bench_cfg.push_dataset_images:
            dataset_subset = self._prepare_android_dataset_subset(
                run_dir=run_dir,
                backend_name="android_ort",
                split=self.ort_android_bench_cfg.dataset_split,
                max_images=self.ort_android_bench_cfg.dataset_max_images,
                seed=self.ort_android_bench_cfg.dataset_seed,
                remote_dir=self.ort_android_bench_cfg.remote_dir,
                remote_subdir=self.ort_android_bench_cfg.dataset_remote_subdir,
            )
            (logs_dir / "dataset_subset.json").write_text(
                json.dumps(dataclasses.asdict(dataset_subset), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        dataset_key = self._android_dataset_cache_suffix(
            enabled=bool(self.ort_android_bench_cfg.push_dataset_images),
            data_yaml=self.dataset_yaml,
            split=self.ort_android_bench_cfg.dataset_split,
            max_images=self.ort_android_bench_cfg.dataset_max_images,
            seed=self.ort_android_bench_cfg.dataset_seed,
        )

        for d in self.devices:
            model_hash = self._hash_file_model(onnx_model)
            key = "|".join([
                "ort_android",
                f"serial={d.serial}",
                f"threads={d.threads}",
                f"imgsz={self.ort_android_bench_cfg.imgsz}",
                f"provider={self.ort_android_bench_cfg.provider}",
                dataset_key,
                f"model={model_hash}",
            ])

            if self.latency_cfg.use_cache and (not self.latency_cfg.force_rebench):
                hit = self.cache.get(key)
                if hit is not None:
                    latency_ms[d.name] = float(hit.avg_ms)
                    (logs_dir / f"{d.name}.cache.txt").write_text(
                        f"cache_hit avg_ms={hit.avg_ms} ts={hit.ts}\nkey={key}\n",
                        encoding="utf-8",
                    )
                    continue

            run_kwargs = {}
            if dataset_subset is not None:
                run_kwargs = dict(
                    local_images_dir=dataset_subset.local_dir,
                    local_image_list=dataset_subset.local_list,
                    remote_images_dir=dataset_subset.remote_dir,
                    remote_image_list=dataset_subset.remote_list,
                    dataset_image_count=dataset_subset.count,
                )

            data = self.android_ort.run_once(device=d, local_onnx=onnx_model, **run_kwargs)

            avg_ms = None
            for k in ("avg_ms", "latency_ms", "mean_ms"):
                if k in data:
                    avg_ms = float(data[k])
                    break
            if avg_ms is None:
                raise RuntimeError(f"ORT bench returned no avg latency field: {data}")

            latency_ms[d.name] = avg_ms
            self.cache.set(key, avg_ms)
            (logs_dir / f"{d.name}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return latency_ms

    def _bench_devices_with_android_tflite(self, tflite_model: Path, run_dir: Path, *, delegate: str) -> Dict[str, float]:
        latency_ms: Dict[str, float] = {}
        if not self.devices:
            return latency_ms

        logs_dir = run_dir / "android_tflite_logs"
        ensure_dir(logs_dir)

        dataset_subset: Optional[AndroidDatasetSubset] = None
        if self.ort_android_bench_cfg.push_dataset_images:
            dataset_subset = self._prepare_android_dataset_subset(
                run_dir=run_dir,
                backend_name="android_tflite",
                split=self.ort_android_bench_cfg.dataset_split,
                max_images=self.ort_android_bench_cfg.dataset_max_images,
                seed=self.ort_android_bench_cfg.dataset_seed,
                remote_dir=self.ort_android_bench_cfg.remote_dir,
                remote_subdir=self.ort_android_bench_cfg.dataset_remote_subdir,
            )
            (logs_dir / "dataset_subset.json").write_text(
                json.dumps(dataclasses.asdict(dataset_subset), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        dataset_key = self._android_dataset_cache_suffix(
            enabled=bool(self.ort_android_bench_cfg.push_dataset_images),
            data_yaml=self.dataset_yaml,
            split=self.ort_android_bench_cfg.dataset_split,
            max_images=self.ort_android_bench_cfg.dataset_max_images,
            seed=self.ort_android_bench_cfg.dataset_seed,
        )

        tflite_runner = AndroidTfliteBench(self.tools, self.ort_android_bench_cfg, delegate=delegate)
        for d in self.devices:
            model_hash = self._hash_file_model(tflite_model)
            key = "|".join([
                "tflite_android",
                f"serial={d.serial}",
                f"threads={d.threads}",
                f"imgsz={self.ort_android_bench_cfg.imgsz}",
                f"delegate={delegate}",
                dataset_key,
                f"model={model_hash}",
            ])

            if self.latency_cfg.use_cache and (not self.latency_cfg.force_rebench):
                hit = self.cache.get(key)
                if hit is not None:
                    latency_ms[d.name] = float(hit.avg_ms)
                    (logs_dir / f"{d.name}.cache.txt").write_text(
                        f"cache_hit avg_ms={hit.avg_ms} ts={hit.ts}\nkey={key}\n",
                        encoding="utf-8",
                    )
                    continue

            run_kwargs = {}
            if dataset_subset is not None:
                run_kwargs = dict(
                    local_images_dir=dataset_subset.local_dir,
                    local_image_list=dataset_subset.local_list,
                    remote_images_dir=dataset_subset.remote_dir,
                    remote_image_list=dataset_subset.remote_list,
                    dataset_image_count=dataset_subset.count,
                )

            data = tflite_runner.run_once(device=d, local_tflite=tflite_model, **run_kwargs)

            avg_ms = None
            for k in ("avg_ms", "latency_ms", "mean_ms"):
                if k in data:
                    avg_ms = float(data[k])
                    break
            if avg_ms is None:
                raise RuntimeError(f"TFLite bench returned no avg latency field: {data}")

            latency_ms[d.name] = avg_ms
            self.cache.set(key, avg_ms)
            (logs_dir / f"{d.name}.json").write_text(
                json.dumps(data, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

        return latency_ms

    def _bench_latency(
        self,
        ncnn_param: Optional[Path],
        ncnn_bin: Optional[Path],
        run_dir: Path,
        onnx_model: Optional[Path] = None,
    ) -> Dict[str, float]:
        backend = str(self.latency_cfg.backend).lower().strip()
        if backend == "ort_android":
            if onnx_model is None:
                raise RuntimeError("onnx_model is required for backend=ort_android")
            return self._bench_devices_with_android_ort(onnx_model, run_dir)
        if backend == "android_app":
            if ncnn_param is None or ncnn_bin is None:
                raise RuntimeError("NCNN model is required for backend=android_app")
            return self._bench_devices_with_android_app(ncnn_param, ncnn_bin, run_dir)
        if ncnn_param is None or ncnn_bin is None:
            raise RuntimeError("NCNN model is required for this backend")
        return self._bench_devices_with_cache(ncnn_param, ncnn_bin, run_dir)

    def _select_tflite_profile_model(self, profile: BenchmarkProfileConfig, tflite_models: Optional[Dict[str, Path]]) -> Path:
        artifact = self._profile_artifact(profile).lower().strip()
        models = dict(tflite_models or {})

        aliases = {
            "": "int8",
            "int8": "int8",
            "tflite_int8": "int8",
            "model_int8": "int8",
            "full_integer": "int8",
            "full_integer_quant": "int8",
            "integer_quant": "int8",
            "fp32": "fp32",
            "float32": "fp32",
            "tflite_fp32": "fp32",
            "model_fp32": "fp32",
            "fp16": "fp16",
            "float16": "fp16",
            "tflite_fp16": "fp16",
            "model_fp16": "fp16",
        }
        key = aliases.get(artifact, artifact)
        if key in models and Path(models[key]).exists():
            return Path(models[key])

        if artifact:
            p = Path(self._profile_artifact(profile))
            if p.exists():
                return p

        available = ", ".join(f"{k}={v}" for k, v in sorted(models.items())) or "none"
        raise RuntimeError(f"TFLite artifact {self._profile_artifact(profile)!r} is not available. Available: {available}")

    def _bench_latency_profile(
        self,
        profile: BenchmarkProfileConfig,
        index: int,
        *,
        ncnn_param: Optional[Path],
        ncnn_bin: Optional[Path],
        onnx_model: Optional[Path],
        tflite_models: Optional[Dict[str, Path]] = None,
        run_dir: Path,
    ) -> Dict[str, float]:
        backend = self._profile_backend(profile)
        devices = self._devices_for_profile(profile)
        if not devices:
            raise RuntimeError(f"benchmark profile '{profile.name}' has no matching ready devices")

        profile_name = self._profile_display_name(profile, index)
        profile_dir = run_dir / "benchmark_profiles" / self._profile_log_name(profile_name)
        ensure_dir(profile_dir)

        old_devices = self.devices
        old_ort_cfg = self.ort_android_bench_cfg
        old_ort = self.android_ort
        old_app_cfg = self.android_app_bench_cfg
        old_app = self.android_app
        old_export_cfg = self.export_cfg
        old_latency_cfg = self.latency_cfg

        try:
            self.devices = devices
            self.latency_cfg = dataclasses.replace(self.latency_cfg, backend=backend)

            if backend == "ort_android":
                if onnx_model is None:
                    raise RuntimeError("onnx_model is required for backend=ort_android benchmark profile")
                cfg = self._ort_cfg_for_profile(profile)
                self.ort_android_bench_cfg = cfg
                self.android_ort = AndroidOrtBench(self.tools, cfg)
                return self._bench_devices_with_android_ort(onnx_model, profile_dir)

            if backend == "android_app":
                if ncnn_param is None or ncnn_bin is None:
                    raise RuntimeError("NCNN model is required for backend=android_app benchmark profile")
                cfg = self._android_app_cfg_for_profile(profile)
                self.android_app_bench_cfg = cfg
                self.android_app = AndroidAppBench(self.tools, cfg)
                return self._bench_devices_with_android_app(ncnn_param, ncnn_bin, profile_dir)

            if backend in {"tflite_android", "tflite"}:
                tflite_model = self._select_tflite_profile_model(profile, tflite_models)
                cfg = self._tflite_cfg_for_profile(profile)
                self.ort_android_bench_cfg = cfg
                delegate = self._profile_tflite_delegate(profile)
                return self._bench_devices_with_android_tflite(tflite_model, profile_dir, delegate=delegate)

            if backend == "benchncnn":
                if ncnn_param is None or ncnn_bin is None:
                    raise RuntimeError("NCNN model is required for backend=benchncnn benchmark profile")
                if profile.shape:
                    self.export_cfg = dataclasses.replace(self.export_cfg, bench_shape=str(profile.shape))
                return self._bench_devices_with_cache(ncnn_param, ncnn_bin, profile_dir)

            raise RuntimeError(f"Unsupported benchmark profile backend: {profile.backend!r}")
        finally:
            self.devices = old_devices
            self.ort_android_bench_cfg = old_ort_cfg
            self.android_ort = old_ort
            self.android_app_bench_cfg = old_app_cfg
            self.android_app = old_app
            self.export_cfg = old_export_cfg
            self.latency_cfg = old_latency_cfg

    def _bench_latency_with_profiles(
        self,
        *,
        ncnn_param: Optional[Path],
        ncnn_bin: Optional[Path],
        onnx_model: Optional[Path],
        tflite_models: Optional[Dict[str, Path]] = None,
        run_dir: Path,
    ) -> tuple[Dict[str, float], Dict[str, Any]]:
        profiles = self._active_benchmark_profiles()
        if not profiles:
            legacy = self._bench_latency(ncnn_param, ncnn_bin, run_dir, onnx_model=onnx_model)
            return legacy, {}

        latency_ms: Dict[str, float] = {}
        details: Dict[str, Any] = {}
        for i, profile in enumerate(profiles):
            profile_name = self._profile_display_name(profile, i)
            backend = self._profile_backend(profile)
            try:
                raw = self._bench_latency_profile(
                    profile,
                    i,
                    ncnn_param=ncnn_param,
                    ncnn_bin=ncnn_bin,
                    onnx_model=onnx_model,
                    tflite_models=tflite_models,
                    run_dir=run_dir,
                )
                prefixed: Dict[str, float] = {}
                for device_name, value in raw.items():
                    key = f"{profile_name}/{device_name}"
                    latency_ms[key] = float(value)
                    prefixed[str(device_name)] = float(value)
                details[profile_name] = {
                    "backend": backend,
                    "provider": profile.provider,
                    "delegate": getattr(profile, "delegate", None),
                    "artifact": getattr(profile, "artifact", None),
                    "devices": [d.name for d in self._devices_for_profile(profile)],
                    "latency_ms": prefixed,
                    "required": bool(profile.required),
                }
            except Exception as e:
                details[profile_name] = {
                    "backend": backend,
                    "provider": profile.provider,
                    "delegate": getattr(profile, "delegate", None),
                    "artifact": getattr(profile, "artifact", None),
                    "failed": True,
                    "error": str(e),
                    "required": bool(profile.required),
                }
                if bool(profile.required):
                    raise

        if not latency_ms:
            raise RuntimeError("No benchmark profile produced latency results")
        return latency_ms, details

    def _benchmark_existing_deploy_model(
        self,
        *,
        ncnn_param: Optional[Path] = None,
        ncnn_bin: Optional[Path] = None,
        onnx_model: Optional[Path] = None,
        tflite_models: Optional[Dict[str, Path]] = None,
        run_dir: Path,
    ) -> tuple[Dict[str, float], Dict[str, Any], str]:
        """Измеряет задержку уже экспортированной deploy-модели без повторной сборки."""
        ensure_dir(run_dir)
        backend = str(self.latency_cfg.backend).lower().strip()
        profiles_active = bool(self._active_benchmark_profiles())

        if profiles_active:
            latency_ms, profile_details = self._bench_latency_with_profiles(
                ncnn_param=ncnn_param,
                ncnn_bin=ncnn_bin,
                onnx_model=onnx_model,
                tflite_models=tflite_models,
                run_dir=run_dir,
            )
            return latency_ms, {"benchmark_profiles": profile_details}, "multi_profile"

        if backend == "ort_android":
            if onnx_model is None:
                raise RuntimeError("ONNX model is required for backend=ort_android rebench")
            latency_ms = self._bench_latency(None, None, run_dir, onnx_model=onnx_model)
            return latency_ms, {}, "ort_android"

        if ncnn_param is None or ncnn_bin is None:
            raise RuntimeError(f"NCNN .param/.bin model is required for backend={backend!r} rebench")
        latency_ms = self._bench_latency(ncnn_param, ncnn_bin, run_dir, onnx_model=onnx_model)
        return latency_ms, {}, "ncnn"

    def _bench_devices_with_android_app(self, ncnn_param: Path, ncnn_bin: Path, run_dir: Path) -> Dict[str, float]:
        latency_ms: Dict[str, float] = {}
        if not self.devices:
            return latency_ms

        logs_dir = run_dir / "android_app_logs"
        ensure_dir(logs_dir)

        dataset_subset: Optional[AndroidDatasetSubset] = None
        if self.android_app_bench_cfg.push_dataset_images:
            dataset_subset = self._prepare_android_dataset_subset(
                run_dir=run_dir,
                backend_name="android_app",
                split=self.android_app_bench_cfg.dataset_split,
                max_images=self.android_app_bench_cfg.dataset_max_images,
                seed=self.android_app_bench_cfg.dataset_seed,
                remote_dir=self.android_app_bench_cfg.remote_dir,
                remote_subdir=self.android_app_bench_cfg.dataset_remote_subdir,
            )
            (logs_dir / "dataset_subset.json").write_text(
                json.dumps(dataclasses.asdict(dataset_subset), ensure_ascii=False, indent=2, default=str),
                encoding="utf-8",
            )

        dataset_key = self._android_dataset_cache_suffix(
            enabled=bool(self.android_app_bench_cfg.push_dataset_images),
            data_yaml=self.dataset_yaml,
            split=self.android_app_bench_cfg.dataset_split,
            max_images=self.android_app_bench_cfg.dataset_max_images,
            seed=self.android_app_bench_cfg.dataset_seed,
        )

        for d in self.devices:
            model_hash = self._hash_ncnn_model(ncnn_param, ncnn_bin)
            key = self._cache_key(
                d,
                model_hash,
                shape=f"android_app_imgsz={self.android_app_bench_cfg.imgsz}|{dataset_key}",
            )

            if self.latency_cfg.use_cache and (not self.latency_cfg.force_rebench):
                hit = self.cache.get(key)
                if hit is not None:
                    latency_ms[d.name] = float(hit.avg_ms)
                    (logs_dir / f"{d.name}.cache.txt").write_text(
                        f"cache_hit avg_ms={hit.avg_ms} ts={hit.ts}\nkey={key}\n",
                        encoding="utf-8",
                    )
                    continue

            run_kwargs = {}
            if dataset_subset is not None:
                run_kwargs = dict(
                    local_images_dir=dataset_subset.local_dir,
                    local_image_list=dataset_subset.local_list,
                    remote_images_dir=dataset_subset.remote_dir,
                    remote_image_list=dataset_subset.remote_list,
                    dataset_image_count=dataset_subset.count,
                )

            data = self.android_app.run_once(device=d, local_param=ncnn_param, local_bin=ncnn_bin, **run_kwargs)

            (logs_dir / f"{d.name}.json").write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

            if "avg_ms" not in data:
                raise RuntimeError(f"[{d.name}] android result missing avg_ms: {data}")

            avg = float(data["avg_ms"])
            latency_ms[d.name] = avg
            self.cache.set(key, avg)

        return latency_ms

    def _bench_devices_with_cache(self, ncnn_param: Path, ncnn_bin: Path, run_dir: Path) -> Dict[str, float]:
        latency_ms: Dict[str, float] = {}
        if not self.devices:
            return latency_ms

        model_hash = self._hash_ncnn_model(ncnn_param, ncnn_bin)
        shape = self.export_cfg.bench_shape

        bench_logs_dir = run_dir / "bench_logs"
        ensure_dir(bench_logs_dir)

        for d in self.devices:
            key = self._cache_key(d, model_hash, shape)

            if self.latency_cfg.use_cache and (not self.latency_cfg.force_rebench):
                hit = self.cache.get(key)
                if hit is not None:
                    latency_ms[d.name] = float(hit.avg_ms)
                    (bench_logs_dir / f"{d.name}.cache.txt").write_text(
                        f"cache_hit avg_ms={hit.avg_ms} ts={hit.ts}\nkey={key}\n",
                        encoding="utf-8",
                    )
                    continue

            reps = max(1, int(self.latency_cfg.repeats))
            vals: List[float] = []
            raw_logs: List[str] = []

            for r in range(reps):
                avg_ms, raw = self.bench.bench(
                    d,
                    ncnn=type("NM", (), {"param": ncnn_param, "bin": ncnn_bin})(),
                    shape=shape,
                )
                vals.append(float(avg_ms))
                raw_logs.append(f"=== repeat {r+1}/{reps} avg={avg_ms} ===\n{raw}\n")

            avg = sum(vals) / len(vals)
            latency_ms[d.name] = float(avg)
            self.cache.set(key, float(avg))

            (bench_logs_dir / f"{d.name}.log").write_text("".join(raw_logs), encoding="utf-8")

        return latency_ms

    def _maybe_eval_onnx(self, onnx_path: Path, extra: Dict[str, Any], key: str) -> None:
        if self.eval_exported_onnx_fn is None and self.eval_exported_onnx_metrics_fn is None:
            return
        try:
            metrics = self._eval_exported_onnx_metrics(onnx_path)
            self._store_eval_metrics(extra, key, metrics)
        except Exception as e:
            extra[f"{key}_skipped"] = str(e)

    def _maybe_quantize_and_eval_int8(self, onnx_fp32: Path, run_dir: Path, extra: Dict[str, Any], tag: str) -> Optional[Path]:
        if not self.onnx_ptq_cfg.enabled:
            return None
        if self.quantize_onnx_fn is None:
            extra["onnx_ptq_skipped"] = "quantize_onnx_fn is None"
            return None

        out = run_dir / "export" / (f"model_int8_{tag}.onnx" if tag else "model_int8.onnx")
        try:
            p = self.quantize_onnx_fn(onnx_fp32, out, run_dir)
        except Exception as e:
            extra[f"onnx_ptq_{tag}_failed"] = str(e)
            return None

        if self.eval_exported_onnx_fn is not None or self.eval_exported_onnx_metrics_fn is not None:
            try:
                metrics_int8 = self._eval_exported_onnx_metrics(p)
                if tag == "before_qat":
                    self._store_eval_metrics(extra, "acc_onnx_int8", metrics_int8)
                elif tag == "after_qat":
                    self._store_eval_metrics(extra, "acc_onnx_int8_after_qat", metrics_int8)
            except Exception as e:
                extra[f"acc_onnx_int8_{tag}_skipped"] = str(e)
        return p

    def _collect_tflite_artifacts(self, run_dir: Path) -> Dict[str, Path]:
        out: Dict[str, Path] = {}
        tdir = run_dir / "export" / "tflite"
        int8_name = str(getattr(self.export_cfg, "tflite_int8_name", "model_int8.tflite") or "model_int8.tflite")
        fp32_name = str(getattr(self.export_cfg, "tflite_fp32_name", "model_fp32.tflite") or "model_fp32.tflite")
        fp16_name = str(getattr(self.export_cfg, "tflite_fp16_name", "model_fp16.tflite") or "model_fp16.tflite")
        candidates = {
            "int8": tdir / int8_name,
            "tflite_int8": tdir / int8_name,
            "fp32": tdir / fp32_name,
            "float32": tdir / fp32_name,
            "tflite_fp32": tdir / fp32_name,
            "fp16": tdir / fp16_name,
            "tflite_fp16": tdir / fp16_name,
        }
        for key, path in candidates.items():
            if path.exists():
                out[key] = path
        return out

    def _store_tflite_artifacts(self, run_dir: Path, extra: Dict[str, Any]) -> Dict[str, Path]:
        artifacts = self._collect_tflite_artifacts(run_dir)
        if artifacts:
            extra["tflite_artifacts"] = {k: str(v) for k, v in sorted(artifacts.items())}
            sizes: Dict[str, int] = {}
            for k, v in artifacts.items():
                try:
                    sizes[k] = int(sizeof_file(v))
                except Exception:
                    pass
            if sizes:
                extra["tflite_artifact_size_bytes"] = sizes
        return artifacts

    def _tflite_fp32_enabled(self) -> bool:
        return bool(getattr(self.export_cfg, "tflite", False))

    def _maybe_export_tflite_fp32_baseline(self, student: Any, run_dir: Path, extra: Dict[str, Any]) -> Optional[Path]:
        """Создает FP32 TFLite-модель для исходной базовой модели."""
        if not self._tflite_fp32_enabled():
            return None
        if self.export_tflite_fp32_fn_factory is None:
            msg = "export_tflite_fp32_fn_factory is None"
            extra["tflite_fp32_export"] = "skipped"
            extra["tflite_fp32_skipped"] = msg
            if bool(getattr(self.export_cfg, "tflite_fp32_required", False)):
                raise RuntimeError(msg)
            return None

        tflite_name = str(getattr(self.export_cfg, "tflite_fp32_name", "model_fp32.tflite") or "model_fp32.tflite")
        out = run_dir / "export" / "tflite" / tflite_name
        try:
            export_fn = self.export_tflite_fp32_fn_factory(student, self.export_cfg)
            p = Path(export_fn(out))
            if not p.exists():
                raise RuntimeError(f"TFLite FP32 export did not create file: {p}")
            extra["tflite_fp32_export"] = "ok"
            extra["tflite_fp32_path"] = str(p)
            extra["tflite_fp32_size_bytes"] = int(sizeof_file(p))
            self._store_tflite_artifacts(run_dir, extra)
            return p
        except Exception as exc:
            extra["tflite_fp32_export"] = "failed"
            extra["tflite_fp32_failed"] = str(exc)
            if bool(getattr(self.export_cfg, "tflite_fp32_required", False)):
                raise
            print(f"[warn] TFLite FP32 baseline export failed: {exc}", file=sys.stderr)
            return None

    @staticmethod
    def _baseline_tflite_models_for_profiles(tflite_models: Optional[Dict[str, Path]]) -> Dict[str, Path]:
        """Привязывает TFLite-профили базовой модели к одному FP32 .tflite-файлу."""
        models = dict(tflite_models or {})
        fp32 = models.get("fp32") or models.get("tflite_fp32") or models.get("float32")
        if fp32 is None:
            return models
        for key in ("int8", "tflite_int8", "model_int8", "fp16", "tflite_fp16", "model_fp16"):
            models.setdefault(key, fp32)
        return models

    def _tflite_int8_enabled(self) -> bool:
        return bool(getattr(self.export_cfg, "tflite", False)) and bool(getattr(self.export_cfg, "tflite_int8", False))

    def _maybe_export_tflite_int8(self, student: Any, run_dir: Path, extra: Dict[str, Any]) -> Optional[Path]:
        """Создает дополнительный TFLite INT8-файл, не меняя выбранную ONNX-модель."""
        if not self._tflite_int8_enabled():
            return None
        if self.export_tflite_int8_fn_factory is None:
            msg = "export_tflite_int8_fn_factory is None"
            extra["tflite_int8_export"] = "skipped"
            extra["tflite_int8_skipped"] = msg
            if bool(getattr(self.export_cfg, "tflite_int8_required", False)):
                raise RuntimeError(msg)
            return None

        tflite_name = str(getattr(self.export_cfg, "tflite_int8_name", "model_int8.tflite") or "model_int8.tflite")
        out = run_dir / "export" / "tflite" / tflite_name
        try:
            export_fn = self.export_tflite_int8_fn_factory(student, self.export_cfg)
            p = Path(export_fn(out))
            if not p.exists():
                raise RuntimeError(f"TFLite INT8 export did not create file: {p}")
            extra["tflite_int8_export"] = "ok"
            extra["tflite_int8_path"] = str(p)
            extra["tflite_int8_size_bytes"] = int(sizeof_file(p))
            self._store_tflite_artifacts(run_dir, extra)
            return p
        except Exception as exc:
            extra["tflite_int8_export"] = "failed"
            extra["tflite_int8_failed"] = str(exc)
            if bool(getattr(self.export_cfg, "tflite_int8_required", False)):
                raise
            print(f"[warn] TFLite INT8 export failed: {exc}", file=sys.stderr)
            return None

    def _prune_mode(self) -> str:
        mode = str(getattr(self.trim_cfg, "prune_mode", "one_shot") or "one_shot").strip().lower()
        if mode not in {"one_shot", "staged"}:
            raise ValueError(f"Unsupported trim.prune_mode={mode!r}; expected 'one_shot' or 'staged'")
        return mode

    @staticmethod
    def _local_prune_ratio(prev_total: float, target_total: float) -> float:
        """Переводит общий уровень прунинга в локальный шаг для текущей стадии."""
        prev = float(prev_total)
        target = float(target_total)
        if not (0.0 <= prev < 1.0):
            raise ValueError(f"prev_total must be in [0, 1), got {prev_total}")
        if not (prev < target < 1.0):
            raise ValueError(f"target_total must be in ({prev}, 1), got {target_total}")
        return float(1.0 - ((1.0 - target) / (1.0 - prev)))

    def _build_pruning_plan(self, final_target: float) -> List[Dict[str, float]]:
        target = float(final_target)
        if target <= 0.0:
            return []
        if target >= 1.0:
            raise ValueError(f"candidate prune_ratio must be < 1.0, got {target}")

        if self._prune_mode() == "one_shot":
            targets = [target]
        else:
            raw = [float(v) for v in getattr(self.staged_pruning_cfg, "milestones", ())]
            if any(not (0.0 < v < 1.0) for v in raw):
                raise ValueError(f"staged_pruning.milestones must be in (0, 1), got {raw}")
            targets = sorted({v for v in raw if v < target})
            targets.append(target)

        plan: List[Dict[str, float]] = []
        prev = 0.0
        for stage_idx, stage_target in enumerate(targets, 1):
            local = self._local_prune_ratio(prev, stage_target)
            plan.append(
                {
                    "stage_index": int(stage_idx),
                    "previous_total": float(prev),
                    "target_total": float(stage_target),
                    "local_ratio": float(local),
                }
            )
            prev = stage_target
        return plan

    def _stage_train_cfg(self, *, is_final: bool) -> TrainConfig:
        cfg = self.staged_pruning_cfg
        if is_final:
            epochs = self.train_cfg.short_epochs if cfg.final_epochs is None else int(cfg.final_epochs)
            lr = self.train_cfg.lr if cfg.final_lr is None else float(cfg.final_lr)
        else:
            epochs = int(cfg.intermediate_epochs)
            lr = self.train_cfg.lr if cfg.intermediate_lr is None else float(cfg.intermediate_lr)
        return dataclasses.replace(self.train_cfg, short_epochs=int(epochs), lr=float(lr))

    @staticmethod
    def _collect_student_transform_stats(student: Any, extra: Dict[str, Any]) -> None:
        try:
            if hasattr(student, "yolo") and hasattr(student.yolo, "_xtrim_trim_stats"):
                extra["trim_stats"] = getattr(student.yolo, "_xtrim_trim_stats")
            if hasattr(student, "yolo") and hasattr(student.yolo, "_xtrim_all_stats"):
                extra["all_transform_stats"] = getattr(student.yolo, "_xtrim_all_stats")
        except Exception as e:
            print(e)

    def _staged_target_mode(self) -> str:
        mode = str(getattr(self.staged_pruning_cfg, "target_mode", "ratio_schedule") or "ratio_schedule").strip().lower()
        if mode not in {"ratio_schedule", "match_one_shot_architecture"}:
            raise ValueError(
                f"Unsupported staged_pruning.target_mode={mode!r}; expected "
                "'ratio_schedule' or 'match_one_shot_architecture'"
            )
        return mode

    @staticmethod
    def _interpolate_width_map(
        current: Dict[str, Any],
        final: Dict[str, Any],
        progress: float,
    ) -> Dict[str, int]:
        """Плавно приближает текущие ширины каналов к целевым, не опускаясь ниже финального значения."""
        p = max(0.0, min(1.0, float(progress)))
        out: Dict[str, int] = {}
        for name, final_width in dict(final or {}).items():
            cur = int(dict(current or {}).get(name, final_width))
            fin = int(final_width)
            if cur <= fin:
                out[str(name)] = cur
                continue
            keep = int(fin + math.ceil(((cur - fin) * (1.0 - p)) - 1e-9))
            out[str(name)] = max(fin, min(cur, keep))
        return out

    def _interpolate_architecture(
        self,
        current_arch: Dict[str, Any],
        final_arch: Dict[str, Any],
        progress: float,
    ) -> Dict[str, Any]:
        return {
            "conv_out_channels": self._interpolate_width_map(
                dict(current_arch.get("conv_out_channels", {}) or {}),
                dict(final_arch.get("conv_out_channels", {}) or {}),
                progress,
            ),
            "detect_hidden_channels": self._interpolate_width_map(
                dict(current_arch.get("detect_hidden_channels", {}) or {}),
                dict(final_arch.get("detect_hidden_channels", {}) or {}),
                progress,
            ),
            "total_params": int(final_arch.get("total_params", 0) or 0),
        }

    @staticmethod
    def _architecture_mismatch_summary(
        actual_arch: Dict[str, Any],
        target_arch: Dict[str, Any],
    ) -> Dict[str, Any]:
        def _diff(actual: Dict[str, Any], target: Dict[str, Any]) -> Dict[str, Dict[str, int]]:
            return {
                str(k): {"target": int(v), "actual": int(actual.get(k, -1))}
                for k, v in dict(target or {}).items()
                if int(actual.get(k, -1)) != int(v)
            }

        conv = _diff(
            dict(actual_arch.get("conv_out_channels", {}) or {}),
            dict(target_arch.get("conv_out_channels", {}) or {}),
        )
        detect = _diff(
            dict(actual_arch.get("detect_hidden_channels", {}) or {}),
            dict(target_arch.get("detect_hidden_channels", {}) or {}),
        )
        return {
            "conv_mismatch_count": int(len(conv)),
            "detect_mismatch_count": int(len(detect)),
            "conv_mismatches": conv,
            "detect_mismatches": detect,
            "actual_total_params": int(actual_arch.get("total_params", 0) or 0),
            "target_total_params": int(target_arch.get("total_params", 0) or 0),
        }

    def _build_one_shot_target_architecture(self, cand: CandidateConfig) -> Dict[str, Any]:
        if self.extract_pruning_architecture_fn is None:
            raise RuntimeError(
                "staged_pruning.target_mode='match_one_shot_architecture' requires "
                "extract_pruning_architecture_fn"
            )
        shadow = self.build_candidate_fn(cand)
        try:
            return dict(self.extract_pruning_architecture_fn(shadow) or {})
        finally:
            try:
                del shadow
                import gc
                gc.collect()
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                except Exception:
                    pass
            except Exception:
                pass

    def _build_and_recover_student(
        self,
        cand: CandidateConfig,
        extra: Dict[str, Any],
    ) -> tuple[Any, Optional[float]]:
        mode = self._prune_mode()
        if mode != "staged" or float(cand.prune_ratio) <= 0.0:
            student = self.build_candidate_fn(cand)
            self.warmstart_fn(student)
            self._collect_student_transform_stats(student, extra)

            finetune_logs = self.finetune_fn(student, self.train_cfg)
            if isinstance(finetune_logs, dict):
                extra["finetune_logs"] = finetune_logs
            extra["pruning_mode"] = "one_shot"
            return student, None

        if self.apply_pruning_stage_fn is None:
            raise RuntimeError("trim.prune_mode='staged' requires apply_pruning_stage_fn")

        plan = self._build_pruning_plan(float(cand.prune_ratio))
        if not plan:
            raise RuntimeError("staged pruning requested, but no pruning plan was built")

        target_mode = self._staged_target_mode()
        one_shot_target_arch: Optional[Dict[str, Any]] = None
        if target_mode == "match_one_shot_architecture":
            one_shot_target_arch = self._build_one_shot_target_architecture(cand)

        first = plan[0]
        first_cand = dataclasses.replace(cand, prune_ratio=float(first["local_ratio"]))
        student = self.build_candidate_fn(first_cand)
        self.warmstart_fn(student)

        stages: List[Dict[str, Any]] = []
        final_acc: Optional[float] = None
        eval_each_stage = bool(getattr(self.staged_pruning_cfg, "eval_after_each_stage", True))
        final_target_total = float(plan[-1]["target_total"])

        for i, stage in enumerate(plan):
            is_final = i == len(plan) - 1
            stage_label = f"staged_{i + 1}"
            stage_target_arch: Optional[Dict[str, Any]] = None
            target_progress: Optional[float] = None

            if i == 0:
                stage_trim_stats = getattr(getattr(student, "yolo", None), "_xtrim_trim_stats", None)
            else:
                if target_mode == "match_one_shot_architecture":
                    if self.extract_pruning_architecture_fn is None or one_shot_target_arch is None:
                        raise RuntimeError("missing target architecture callback/state")
                    current_arch = dict(self.extract_pruning_architecture_fn(student) or {})
                    denom = max(1e-12, final_target_total - float(stage["previous_total"]))
                    target_progress = min(
                        1.0,
                        max(0.0, (float(stage["target_total"]) - float(stage["previous_total"])) / denom),
                    )
                    stage_target_arch = self._interpolate_architecture(
                        current_arch,
                        one_shot_target_arch,
                        target_progress,
                    )
                if stage_target_arch is None:
                    stage_trim_stats = self.apply_pruning_stage_fn(
                        student,
                        float(stage["local_ratio"]),
                        stage_label,
                    )
                else:
                    stage_trim_stats = self.apply_pruning_stage_fn(
                        student,
                        float(stage["local_ratio"]),
                        stage_label,
                        target_architecture=stage_target_arch,
                    )

            stage_metrics_before: Optional[Dict[str, float]] = None
            stage_acc_before: Optional[float] = None
            if eval_each_stage:
                stage_metrics_before = self._eval_student_metrics(student)
                stage_acc_before = float(stage_metrics_before["map50_95"])

            stage_train_cfg = self._stage_train_cfg(is_final=is_final)
            stage_logs = self.finetune_fn(student, stage_train_cfg)

            stage_rec: Dict[str, Any] = {
                **stage,
                "stage_label": stage_label,
                "is_final": bool(is_final),
                "train_epochs": int(stage_train_cfg.short_epochs),
                "train_lr": float(stage_train_cfg.lr),
            }
            if target_progress is not None:
                stage_rec["target_progress_to_one_shot"] = float(target_progress)
            if stage_target_arch is not None:
                stage_rec["target_architecture"] = stage_target_arch
            if stage_acc_before is not None:
                stage_rec["acc_before_recovery"] = stage_acc_before
            if stage_metrics_before is not None:
                self._store_stage_metrics(stage_rec, "before_recovery", stage_metrics_before)
            if stage_trim_stats is not None:
                stage_rec["trim_stats"] = stage_trim_stats
            if isinstance(stage_logs, dict):
                stage_rec["finetune_logs"] = stage_logs

            if eval_each_stage:
                stage_metrics_after = self._eval_student_metrics(student)
                stage_acc = float(stage_metrics_after["map50_95"])
                stage_rec["acc_after_recovery"] = stage_acc
                self._store_stage_metrics(stage_rec, "after_recovery", stage_metrics_after)
                if is_final:
                    final_acc = stage_acc

            stages.append(stage_rec)

        self._collect_student_transform_stats(student, extra)
        extra["pruning_mode"] = "staged"
        staged_extra: Dict[str, Any] = {
            "milestones": [float(v) for v in getattr(self.staged_pruning_cfg, "milestones", ())],
            "target_mode": target_mode,
            "plan": plan,
            "stages": stages,
        }
        if one_shot_target_arch is not None:
            staged_extra["one_shot_target_architecture"] = one_shot_target_arch
            if self.extract_pruning_architecture_fn is not None:
                achieved_arch = dict(self.extract_pruning_architecture_fn(student) or {})
                staged_extra["final_architecture"] = achieved_arch
                staged_extra["final_architecture_match"] = self._architecture_mismatch_summary(
                    achieved_arch,
                    one_shot_target_arch,
                )
        extra["staged_pruning"] = staged_extra
        if stages and isinstance(stages[-1].get("finetune_logs"), dict):
            extra["finetune_logs"] = stages[-1]["finetune_logs"]
        return student, final_acc


    def _process_candidate(self, cand: CandidateConfig, run_dir: Path) -> HistoryItem:
        extra: Dict[str, Any] = {}
        student, precomputed_acc = self._build_and_recover_student(cand, extra)

        if self.eval_metrics_fn is not None:
            recovered_metrics = self._eval_student_metrics(student)
        else:
            recovered_acc = float(precomputed_acc) if precomputed_acc is not None else float(self.eval_acc_fn(student))
            recovered_metrics = {"map50_95": recovered_acc}
        acc = float(recovered_metrics["map50_95"])
        extra["acc_kind"] = "mAP50-95"
        self._store_eval_metrics(extra, "acc_recovered_torch", recovered_metrics)

        try:
            from .trim.gumbel_choice import count_mixed_ops, freeze_mixed_ops
            n_mixed = count_mixed_ops(student.torch_model)
            if n_mixed > 0:
                print(f"[export] Freezing {n_mixed} unfrozen MixedOp1x1 before ONNX export")
                freeze_mixed_ops(student.torch_model, verbose=True)
        except Exception as _gumbel_err:
            pass

        onnx_path = run_dir / "export" / "model.onnx"
        exporter = Exporter(self.export_onnx_fn_factory(student, self.export_cfg))
        exporter.export_onnx(onnx_path)

        _pnnx_ncnn: Optional[Any] = None
        torch_model = getattr(student, "torch_model", None)
        if torch_model is not None:
            try:
                _imgsz = int(self.android_app_bench_cfg.imgsz)
                _pnnx_ncnn = self.converter.pnnx_convert(
                    torch_model, run_dir / "export" / "pnnx_ncnn", imgsz=_imgsz
                )
                extra["pnnx_export"] = "ok"
            except Exception as _pnnx_err:
                extra["pnnx_export_skipped"] = str(_pnnx_err)
                print(f"[warn] PNNX export failed, will use onnx2ncnn: {_pnnx_err}", file=sys.stderr)


        self._maybe_eval_onnx(onnx_path, extra, "acc_onnx")
        if "acc_onnx" in extra:
            extra["acc_delta_onnx"] = float(extra["acc_onnx"] - acc)

        onnx_int8_before = self._maybe_quantize_and_eval_int8(onnx_path, run_dir, extra, tag="before_qat")
        if "acc_onnx" in extra and "acc_onnx_int8" in extra:
            extra["acc_drop_int8"] = float(extra["acc_onnx"] - extra["acc_onnx_int8"])

        do_qat = (
            self.qat_cfg.enabled
            and self.finetune_qat_fn is not None
            and ("acc_drop_int8" in extra)
            and (float(extra["acc_drop_int8"]) > float(self.qat_cfg.acc_drop_threshold))
        )
        extra["qat_triggered"] = bool(do_qat)

        onnx_int8_after = None
        onnx_qat = None

        if do_qat:
            qat_logs = self.finetune_qat_fn(student, self.train_cfg)
            if isinstance(qat_logs, dict):
                extra["qat_logs"] = qat_logs

            try:
                from .quant.fake_quant_ultra import set_fake_quant_enabled

                torch_model_qat = getattr(student, "torch_model", None)
                if torch_model_qat is not None:
                    set_fake_quant_enabled(torch_model_qat, False)
                    extra["qat_fake_quant_disabled_before_export"] = True
            except Exception as _fq_err:
                extra["qat_fake_quant_disable_warning"] = str(_fq_err)

            onnx_qat = run_dir / "export" / "model_qat.onnx"
            exporter_qat = Exporter(self.export_onnx_fn_factory(student, self.export_cfg))
            exporter_qat.export_onnx(onnx_qat)

            self._maybe_eval_onnx(onnx_qat, extra, "acc_onnx_after_qat")
            onnx_int8_after = self._maybe_quantize_and_eval_int8(onnx_qat, run_dir, extra, tag="after_qat")

            if "acc_onnx_after_qat" in extra and "acc_onnx_int8_after_qat" in extra:
                extra["acc_drop_int8_after_qat"] = float(extra["acc_onnx_after_qat"] - extra["acc_onnx_int8_after_qat"])

        deploy_onnx = onnx_path
        deploy_onnx_kind = "fp32"

        if onnx_int8_before is not None:
            deploy_onnx = onnx_int8_before
            deploy_onnx_kind = "int8_before_qat"

        if do_qat and onnx_qat is not None:
            deploy_onnx = onnx_qat
            deploy_onnx_kind = "qat_fp32"
            if onnx_int8_after is not None:
                deploy_onnx = onnx_int8_after
                deploy_onnx_kind = "int8_after_qat"

        extra["deploy_onnx_path"] = str(deploy_onnx)
        extra["deploy_onnx_kind"] = deploy_onnx_kind

        deploy_acc_source = "acc_recovered_torch"
        if deploy_onnx_kind == "int8_after_qat" and "acc_onnx_int8_after_qat" in extra:
            deploy_acc_source = "acc_onnx_int8_after_qat"
        elif deploy_onnx_kind == "qat_fp32" and "acc_onnx_after_qat" in extra:
            deploy_acc_source = "acc_onnx_after_qat"
        elif deploy_onnx_kind == "int8_before_qat" and "acc_onnx_int8" in extra:
            deploy_acc_source = "acc_onnx_int8"
        elif deploy_onnx_kind == "fp32" and "acc_onnx" in extra:
            deploy_acc_source = "acc_onnx"

        deploy_metrics = self._metrics_from_extra(extra, deploy_acc_source)
        if "map50_95" not in deploy_metrics:
            deploy_metrics = {"map50_95": float(acc)}
        deploy_acc = float(deploy_metrics["map50_95"])
        self._store_eval_metrics(extra, "acc_deploy", deploy_metrics)
        extra["acc_deploy_source"] = deploy_acc_source

        tflite_int8 = self._maybe_export_tflite_int8(student, run_dir, extra)
        tflite_models = self._store_tflite_artifacts(run_dir, extra)

        size_bytes = int(sizeof_file(deploy_onnx))
        latency_ms: Dict[str, float] = {}

        backend = str(self.latency_cfg.backend).lower().strip()
        profiles_active = bool(self._active_benchmark_profiles())
        needs_ncnn = self._needs_ncnn_artifact()
        ncnn_final = None

        if needs_ncnn:
            source_onnx, source_label = self._select_ncnn_source_onnx(
                onnx_fp32=onnx_path,
                onnx_qat=onnx_qat,
                deploy_onnx=deploy_onnx,
                deploy_onnx_kind=deploy_onnx_kind,
            )
            (run_dir / "convert_backend.txt").write_text(f"onnx2ncnn ({source_label})", encoding="utf-8")
            ncnn_final = self._prepare_ncnn_from_onnx(
                source_onnx=source_onnx,
                ncnn_dir=run_dir / "ncnn",
                extra=extra,
                source_label=source_label,
                pnnx_ncnn=_pnnx_ncnn if source_onnx == onnx_path else None,
            )

        onnx_size_bytes = int(sizeof_file(deploy_onnx))
        ncnn_size_bytes = int(sizeof_file(ncnn_final.param) + sizeof_file(ncnn_final.bin)) if ncnn_final is not None else None
        tflite_int8_size_bytes = int(sizeof_file(tflite_int8)) if tflite_int8 is not None else None
        extra["deploy_size_bytes_by_backend"] = {
            "onnx": onnx_size_bytes,
            **({"ncnn": ncnn_size_bytes} if ncnn_size_bytes is not None else {}),
            **({"tflite_int8": tflite_int8_size_bytes} if tflite_int8_size_bytes is not None else {}),
        }

        if profiles_active:
            latency_ms, profile_details = self._bench_latency_with_profiles(
                ncnn_param=ncnn_final.param if ncnn_final is not None else None,
                ncnn_bin=ncnn_final.bin if ncnn_final is not None else None,
                onnx_model=deploy_onnx,
                tflite_models=tflite_models,
                run_dir=run_dir,
            )
            extra["benchmark_profiles"] = profile_details
            extra["deploy_backend"] = "multi_profile"
            extra["primary_latency_backend"] = backend
            size_bytes = ncnn_size_bytes if backend != "ort_android" and ncnn_size_bytes is not None else onnx_size_bytes
        elif backend == "ort_android":
            size_bytes = onnx_size_bytes
            latency_ms = self._bench_latency(None, None, run_dir, onnx_model=deploy_onnx)
            extra["deploy_backend"] = "ort_android"
        else:
            if ncnn_final is None:
                raise RuntimeError("NCNN model was not prepared for NCNN latency backend")
            size_bytes = int(ncnn_size_bytes)
            latency_ms = self._bench_latency(ncnn_final.param, ncnn_final.bin, run_dir)
            extra["deploy_backend"] = "ncnn"

        # Legacy NCNN demo. Current Android latency is measured through
        # ONNX Runtime/TFLite benchmark profiles in the Android app.
        if self.android_demo_cfg.enabled and ncnn_final is not None:
            try:
                for d in self.devices:
                    try:
                        _ = self.demo.run_once(
                            device=d,
                            demo_cfg=self.android_demo_cfg,
                            ncnn_param=ncnn_final.param,
                            ncnn_bin=ncnn_final.bin,
                            run_dir=run_dir,
                        )
                        extra.setdefault("android_demo_ok", []).append(d.name)
                    except Exception as e:
                        extra.setdefault("android_demo_fail", {})[d.name] = str(e)
            except Exception as e:
                extra["android_demo_skipped"] = str(e)

        lat_agg = self._latency_aggregate(latency_ms)
        extra["latency_agg_ms"] = float(lat_agg)
        extra["scalar_score"] = self._scalarize(deploy_acc, lat_agg, size_bytes)
        extra["latency_aggregate_mode"] = self.latency_cfg.aggregate

        metrics = Metrics(
            acc=float(deploy_acc),
            size_bytes=int(size_bytes),
            latency_ms=latency_ms,
            precision=deploy_metrics.get("precision"),
            recall=deploy_metrics.get("recall"),
            iou=deploy_metrics.get("iou"),
            map50=deploy_metrics.get("map50"),
        )
        return HistoryItem(candidate=cand, metrics=metrics, artifacts_dir=str(run_dir), extra=extra)

    def _archive_history(self) -> None:
        import shutil
        if self.history_path.exists():
            archive_name = f"{now_ts()}.jsonl"
            dest = self.history_archive_dir / archive_name
            shutil.copy2(self.history_path, dest)
            print(f"\n=== History archived -> {dest} ===")

    def _append_history(self, item: HistoryItem) -> None:
        rec = {
            "candidate": dataclasses.asdict(item.candidate),
            "metrics": dataclasses.asdict(item.metrics),
            "artifacts_dir": item.artifacts_dir,
            "extra": item.extra,
        }
        with self.history_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def _load_history(self) -> List[HistoryItem]:
        if not self.history_path.exists():
            return []
        items: List[HistoryItem] = []
        for line in self.history_path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            cand = CandidateConfig(**rec["candidate"])
            met = Metrics(**rec["metrics"])
            items.append(
                HistoryItem(
                    candidate=cand,
                    metrics=met,
                    artifacts_dir=rec["artifacts_dir"],
                    extra=rec.get("extra", {}),
                )
            )
        return items

    def _write_pareto(self, front: List[HistoryItem]) -> None:
        out = []
        for h in front:
            out.append({
                "candidate": dataclasses.asdict(h.candidate),
                "metrics": dataclasses.asdict(h.metrics),
                "artifacts_dir": h.artifacts_dir,
                "extra": h.extra,
            })
        write_json(self.out_root / "pareto.json", out)