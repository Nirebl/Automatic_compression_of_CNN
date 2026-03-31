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
    OrtAndroidBenchConfig,
)
from .android_app_bench import AndroidAppBench, AndroidAppBenchConfig
from .android_ort_bench import AndroidOrtBench
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
        build_candidate_fn: Callable[[CandidateConfig], Any],
        warmstart_fn: Callable[[Any], None],
        finetune_fn: Callable[[Any, TrainConfig], Any],
        finetune_qat_fn: Optional[Callable[[Any, TrainConfig], Any]] = None,
        eval_acc_fn: Callable[[Any], float] = lambda _: 0.0,
        export_onnx_fn_factory: Callable[[Any, ExportConfig], Callable[[Path], None]] = lambda _m, _c: (lambda _p: None),
        eval_exported_onnx_fn: Optional[Callable[[Path], float]] = None,
        quantize_onnx_fn: Optional[Callable[[Path, Path, Path], Path]] = None,
        save_student_pt_fn: Optional[Callable[[Any, Path], None]] = None,
        android_app_bench_cfg: AndroidAppBenchConfig,
        ort_android_bench_cfg: OrtAndroidBenchConfig,
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

        self.build_candidate_fn = build_candidate_fn
        self.warmstart_fn = warmstart_fn
        self.finetune_fn = finetune_fn
        self.finetune_qat_fn = finetune_qat_fn
        self.eval_acc_fn = eval_acc_fn
        self.export_onnx_fn_factory = export_onnx_fn_factory
        self.eval_exported_onnx_fn = eval_exported_onnx_fn
        self.quantize_onnx_fn = quantize_onnx_fn
        self.save_student_pt_fn = save_student_pt_fn

        self.converter = NcnnConverter(tools)
        self.bench = AdbBench(tools)
        self.demo = AdbYoloDemo(tools)

        self.android_app_bench_cfg = android_app_bench_cfg
        self.android_app = AndroidAppBench(tools, android_app_bench_cfg)
        self.ort_android_bench_cfg = ort_android_bench_cfg
        self.android_ort = AndroidOrtBench(tools, ort_android_bench_cfg)

        ensure_dir(self.out_root)
        self.history_path = self.out_root / "history.jsonl"
        self.history_archive_dir = self.out_root / "history"
        ensure_dir(self.history_archive_dir)

        cache_path = self.out_root / self.latency_cfg.cache_file
        self.cache = BenchCache(cache_path)

        self.policy = SearchPolicy.create(search_cfg, search_space)

    def run(self, max_candidates: Optional[int] = None) -> List[HistoryItem]:
        history: List[HistoryItem] = self._load_history()
        self._ensure_reference_baseline(history)

        backend = str(self.latency_cfg.backend).lower().strip()

        ready_devices: List[DeviceConfig] = []
        for d in self.devices:
            if not self.bench.is_device_ready(d):
                print(f"[warn] device {d.name} not ready (adb get-state != device)", file=sys.stderr)
                continue

            if backend == "benchncnn":
                try:
                    self.bench.ensure_benchncnn(d, force_push=False)
                except Exception as e:
                    print(f"[warn] device {d.name} not usable: {e}", file=sys.stderr)
                    continue

            ready_devices.append(d)
        self.devices = ready_devices

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

    def _cache_key(self, device: DeviceConfig, model_hash: str, shape: str) -> str:
        return "|".join([
            "benchncnn",
            f"serial={device.serial}",
            f"loops={device.loops}",
            f"threads={device.threads}",
            f"powersave={device.powersave}",
            f"gpu={device.gpu_device}",
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

        acc = float(self.eval_acc_fn(student))

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

        if backend == "ort_android":
            size_bytes = int(sizeof_file(onnx_path))
            latency_ms = self._bench_latency(None, None, run_dir, onnx_model=onnx_path)
            extra["deploy_backend"] = "ort_android"
            extra["deploy_onnx_path"] = str(onnx_path)
            extra["deploy_onnx_kind"] = "fp32_raw_baseline"
        else:
            ncnn_dir = run_dir / "ncnn"
            ensure_dir(ncnn_dir)

            if _pnnx_ncnn is not None:
                ncnn_float = _pnnx_ncnn
                extra["ncnn_source"] = "pnnx"
            else:
                ncnn_float = self.converter.onnx_to_ncnn(onnx_path, ncnn_dir / "float")
                extra["ncnn_source"] = "onnx_fp32"

            ncnn_final = self.converter.optimize(ncnn_float, ncnn_dir / "opt")
            size_bytes = int(sizeof_file(ncnn_final.param) + sizeof_file(ncnn_final.bin))
            latency_ms = self._bench_latency(ncnn_final.param, ncnn_final.bin, run_dir)
            extra["deploy_backend"] = "ncnn"

        lat_agg = self._latency_aggregate(latency_ms)
        extra["latency_agg_ms"] = float(lat_agg)
        extra["scalar_score"] = self._scalarize(acc, lat_agg, size_bytes)
        extra["exclude_from_pareto"] = True

        metrics = Metrics(
            acc=float(acc),
            size_bytes=int(size_bytes),
            latency_ms=latency_ms,
        )
        return HistoryItem(
            candidate=cand,
            metrics=metrics,
            artifacts_dir=str(run_dir),
            extra=extra,
        )

    def _bench_devices_with_android_ort(self, onnx_model: Path, run_dir: Path) -> Dict[str, float]:
        latency_ms: Dict[str, float] = {}
        if not self.devices:
            return latency_ms

        logs_dir = run_dir / "android_ort_logs"
        ensure_dir(logs_dir)

        for d in self.devices:
            model_hash = self._hash_file_model(onnx_model)
            key = "|".join([
                "ort_android",
                f"serial={d.serial}",
                f"threads={d.threads}",
                f"imgsz={self.ort_android_bench_cfg.imgsz}",
                f"provider={self.ort_android_bench_cfg.provider}",
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

            data = self.android_ort.run_once(device=d, local_onnx=onnx_model)

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

    def _bench_devices_with_android_app(self, ncnn_param: Path, ncnn_bin: Path, run_dir: Path) -> Dict[str, float]:
        latency_ms: Dict[str, float] = {}
        if not self.devices:
            return latency_ms

        logs_dir = run_dir / "android_app_logs"
        ensure_dir(logs_dir)

        for d in self.devices:
            model_hash = self._hash_ncnn_model(ncnn_param, ncnn_bin)
            key = self._cache_key(d, model_hash, shape=f"android_app_imgsz={self.android_app_bench_cfg.imgsz}")

            if self.latency_cfg.use_cache and (not self.latency_cfg.force_rebench):
                hit = self.cache.get(key)
                if hit is not None:
                    latency_ms[d.name] = float(hit.avg_ms)
                    (logs_dir / f"{d.name}.cache.txt").write_text(
                        f"cache_hit avg_ms={hit.avg_ms} ts={hit.ts}\nkey={key}\n",
                        encoding="utf-8",
                    )
                    continue

            data = self.android_app.run_once(device=d, local_param=ncnn_param, local_bin=ncnn_bin)

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
        if self.eval_exported_onnx_fn is None:
            return
        try:
            extra[key] = float(self.eval_exported_onnx_fn(onnx_path))
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

        if self.eval_exported_onnx_fn is not None:
            try:
                acc_int8 = float(self.eval_exported_onnx_fn(p))
                if tag == "before_qat":
                    extra["acc_onnx_int8"] = acc_int8
                elif tag == "after_qat":
                    extra["acc_onnx_int8_after_qat"] = acc_int8
            except Exception as e:
                extra[f"acc_onnx_int8_{tag}_skipped"] = str(e)
        return p


    def _process_candidate(self, cand: CandidateConfig, run_dir: Path) -> HistoryItem:
        student = self.build_candidate_fn(cand)
        self.warmstart_fn(student)

        extra: Dict[str, Any] = {}
        try:
            if hasattr(student, "yolo") and hasattr(student.yolo, "_xtrim_trim_stats"):
                extra["trim_stats"] = getattr(student.yolo, "_xtrim_trim_stats")
            if hasattr(student, "yolo") and hasattr(student.yolo, "_xtrim_all_stats"):
                extra["all_transform_stats"] = getattr(student.yolo, "_xtrim_all_stats")
        except Exception as e:
            print(e)
            pass

        finetune_logs = self.finetune_fn(student, self.train_cfg)
        if isinstance(finetune_logs, dict):
            extra["finetune_logs"] = finetune_logs

        acc = float(self.eval_acc_fn(student))
        extra["acc_kind"] = "mAP50-95"

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

        size_bytes = int(sizeof_file(onnx_path))
        latency_ms: Dict[str, float] = {}

        backend = str(self.latency_cfg.backend).lower().strip()
        ncnn_final = None

        if backend == "ort_android":
            size_bytes = int(sizeof_file(deploy_onnx))
            latency_ms = self._bench_latency(None, None, run_dir, onnx_model=deploy_onnx)
            extra["deploy_backend"] = "ort_android"
        else:
            ncnn_dir = run_dir / "ncnn"
            ensure_dir(ncnn_dir)

            ncnn_float = None
            if _pnnx_ncnn is not None:
                ncnn_float = _pnnx_ncnn
                extra["ncnn_source"] = "pnnx"
                (run_dir / "convert_backend.txt").write_text("pnnx", encoding="utf-8")
            else:
                extra["ncnn_source"] = "onnx_fp32"
                try:
                    ncnn_float = self.converter.onnx_to_ncnn(onnx_path, ncnn_dir / "float")
                    (run_dir / "convert_backend.txt").write_text("onnx2ncnn (onnx_fp32)", encoding="utf-8")
                except Exception as e:
                    extra["ncnn_convert_skipped"] = str(e)
                    (run_dir / "convert_backend.txt").write_text("", encoding="utf-8")

            if ncnn_float is not None:
                ncnn_opt = self.converter.optimize(ncnn_float, ncnn_dir / "opt")
                ncnn_final = ncnn_opt
                if self.ptq_cfg.enabled:
                    auto_calib = run_dir / "ptq" / "calib_images.txt"
                    ptq_cfg_resolved = self.ptq_cfg
                    if auto_calib.exists():
                        ptq_cfg_resolved = dataclasses.replace(self.ptq_cfg, imagelist=str(auto_calib))
                    ncnn_final = self.converter.ptq_int8(ncnn_opt, ncnn_dir / "int8", ptq_cfg_resolved)
                    extra["ncnn_source"] = "ncnn_int8"

                size_bytes = int(sizeof_file(ncnn_final.param) + sizeof_file(ncnn_final.bin))
                latency_ms = self._bench_latency(ncnn_final.param, ncnn_final.bin, run_dir)
                extra["deploy_backend"] = "ncnn"

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
        extra["scalar_score"] = self._scalarize(acc, lat_agg, size_bytes)
        extra["latency_aggregate_mode"] = self.latency_cfg.aggregate

        metrics = Metrics(acc=float(acc), size_bytes=int(size_bytes), latency_ms=latency_ms)
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