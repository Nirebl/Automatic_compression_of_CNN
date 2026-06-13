from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional, Set, Dict

from .types import HistoryItem, CandidateConfig, Metrics
from .pareto import avg_latency, pareto_front


def _is_reference_baseline(h: HistoryItem) -> bool:
    return bool(h.extra.get("is_reference_baseline", False))


def _candidate_history(history: List[HistoryItem]) -> List[HistoryItem]:
    return [h for h in history if not _is_reference_baseline(h)]


def _reference_baseline(history: List[HistoryItem]) -> Optional[HistoryItem]:
    refs = [h for h in history if _is_reference_baseline(h) and not h.extra.get("failed", False)]
    return refs[0] if refs else None


def _baseline_size(history: List[HistoryItem]) -> Optional[int]:
    ref = _reference_baseline(history)
    if ref is not None:
        return ref.metrics.size_bytes

    for h in history:
        if (
            not h.extra.get("failed", False)
            and h.candidate.width_mult == 1.0
            and h.candidate.prune_ratio == 0.0
            and h.candidate.sparse_1x1 == 0.0
            and h.candidate.lowrank_rank == 0
        ):
            return h.metrics.size_bytes
    return None


def _find_baseline_acc(history: List[HistoryItem]) -> float:
    ref = _reference_baseline(history)
    if ref is not None:
        return float(ref.metrics.acc)

    valid = [h for h in history if not h.extra.get("failed", False)]
    exact = [
        h
        for h in valid
        if h.candidate.width_mult == 1.0
        and h.candidate.prune_ratio == 0.0
        and h.candidate.sparse_1x1 == 0.0
        and h.candidate.lowrank_rank == 0
    ]
    if exact:
        return max(h.metrics.acc for h in exact)

    base = [
        h for h in valid
        if h.candidate.width_mult == 1.0 and h.candidate.prune_ratio == 0.0
    ]
    if base:
        return max(h.metrics.acc for h in base)

    return max(h.metrics.acc for h in valid) if valid else 0.0


def _as_float(value: object) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _fmt_acc(value: object) -> str:
    value_f = _as_float(value)
    return f"{value_f:.4f}" if value_f is not None else "---"


def _fmt_delta(value: object) -> str:
    value_f = _as_float(value)
    return f"{value_f:+.4f}" if value_f is not None else "---"


def _deploy_label(kind: object) -> str:
    labels = {
        "fp32_raw_baseline": "RAW FP32",
        "fp32": "ONNX FP32",
        "int8_before_qat": "PTQ INT8",
        "qat_fp32": "QAT FP32",
        "int8_after_qat": "QAT INT8",
    }
    if kind is None:
        return "---"
    return labels.get(str(kind), str(kind))


def _deploy_acc(h: HistoryItem) -> float:
    extra_acc = _as_float(h.extra.get("acc_deploy"))
    return float(extra_acc) if extra_acc is not None else float(h.metrics.acc)


def _deploy_metric(h: HistoryItem, name: str) -> Optional[float]:
    value = _as_float(h.extra.get(f"{name}_deploy"))
    if value is not None:
        return value
    return _as_float(getattr(h.metrics, name, None))


def _baseline_metric(history: List[HistoryItem], name: str) -> Optional[float]:
    """Возвращает дополнительную метрику исходной модели."""
    ref = _reference_baseline(history)
    if ref is not None:
        return _deploy_metric(ref, name)

    valid = [h for h in history if not h.extra.get("failed", False)]
    exact = [
        h
        for h in valid
        if h.candidate.width_mult == 1.0
        and h.candidate.prune_ratio == 0.0
        and h.candidate.sparse_1x1 == 0.0
        and h.candidate.lowrank_rank == 0
    ]
    if exact:
        candidates = [_deploy_metric(h, name) for h in exact]
        candidates = [v for v in candidates if v is not None]
        return max(candidates) if candidates else None
    return None



def _fmt_latency(value: object) -> str:
    value_f = _as_float(value)
    return f"{value_f:.1f}" if value_f is not None else "---"


def _latency_extra_keys(history: List[HistoryItem]) -> List[str]:
    """Находит дополнительные столбцы задержки для профилей вида profile/device."""
    keys: List[str] = []
    seen: Set[str] = set()
    for h in history:
        if h.extra.get("failed", False):
            continue
        for key in h.metrics.latency_ms.keys():
            if key not in seen:
                seen.add(key)
                keys.append(key)
    return keys if len(keys) > 1 else []


def _base_latency_label(key: str) -> str:
    if "/" in key:
        profile, device = key.split("/", 1)
    else:
        profile, device = "", key

    profile_l = profile.lower()
    if "npu" in profile_l or "nnapi" in profile_l:
        return "Lat NPU"
    if "ncnn" in profile_l:
        return "Lat NCNN"
    if "cpu" in profile_l or "xnnpack" in profile_l:
        return "Lat CPU"
    if profile:
        return f"Lat {profile}"
    return f"Lat {device}"


def _latency_column_labels(keys: List[str]) -> Dict[str, str]:
    bases: Dict[str, int] = {}
    for key in keys:
        base = _base_latency_label(key)
        bases[base] = bases.get(base, 0) + 1

    labels: Dict[str, str] = {}
    used: Dict[str, int] = {}
    for key in keys:
        base = _base_latency_label(key)
        label = base
        if bases[base] > 1:
            device = key.split("/", 1)[1] if "/" in key else key
            label = f"{base}:{device}"
        count = used.get(label, 0)
        used[label] = count + 1
        if count:
            label = f"{label}_{count + 1}"
        labels[key] = label
    return labels

def print_results_table(
    history: List[HistoryItem],
    baseline_acc: Optional[float] = None,
    title: str = "RESULTS SUMMARY",
    hide_extra: bool = False,
) -> None:
    if not history:
        print("[results] No candidates evaluated.")
        return

    ref_item = _reference_baseline(history)

    candidate_history = _candidate_history(history)
    ok_history = [h for h in candidate_history if not h.extra.get("failed", False)]
    failed_history = [h for h in candidate_history if h.extra.get("failed", False)]

    pareto_set: Set[str] = {h.candidate.tag for h in pareto_front(ok_history)}

    if baseline_acc is None:
        baseline_acc = _find_baseline_acc(history)

    baseline_size = _baseline_size(history)
    baseline_precision = _baseline_metric(history, "precision")
    baseline_recall = _baseline_metric(history, "recall")
    baseline_iou = _baseline_metric(history, "iou")

    sorted_history = sorted(ok_history, key=_deploy_acc, reverse=True)

    W_TAG = max(24, max(len(h.candidate.tag) for h in candidate_history) if candidate_history else 24)
    W_CFG = 16
    W_ACC = 9
    W_DELTA = 9
    W_DEPLOY = 11
    W_LAT = 10
    W_SIZE = 9
    W_COMPR = 7
    W_P = 1

    latency_extra_keys = _latency_extra_keys(ok_history) if not hide_extra else []
    latency_extra_labels = _latency_column_labels(latency_extra_keys)
    latency_extra_widths = {
        key: max(10, len(label)) for key, label in latency_extra_labels.items()
    }

    def _col(label: str, w: int, align: str = ">") -> str:
        return f"{label:{align}{w}}"

    header_parts = [
        "  #",
        _col("Candidate", W_TAG, "<"),
        _col("Config", W_CFG, "<"),
        _col("Torch", W_ACC),
        _col("ONNX", W_ACC),
        _col("PTQ", W_ACC),
        _col("Δ PTQ", W_DELTA),
        _col("QAT", W_ACC),
        _col("QAT INT8", W_ACC),
        _col("Δ QAT8", W_DELTA),
        _col("Final", W_ACC),
        _col("Δ Final", W_DELTA),
    ]
    if not hide_extra:
        header_parts.extend([
            _col("Init Prec", W_ACC),
            _col("Init Rec", W_ACC),
            _col("Init IoU", W_ACC),
            _col("Prec", W_ACC),
            _col("Recall", W_ACC),
            _col("IoU", W_ACC),
        ])
    header_parts.extend([
        _col("Deploy", W_DEPLOY),
        _col("Lat (ms)", W_LAT),
    ])
    for key in latency_extra_keys:
        label = latency_extra_labels[key]
        header_parts.append(_col(label, latency_extra_widths[key]))
    header_parts.extend([
        _col("Size (MB)", W_SIZE),
        _col("Compr×", W_COMPR),
        _col("P", W_P),
    ])
    header = "  ".join(header_parts)
    sep = "-" * len(header)
    border = "=" * len(header)

    print(f"\n{border}")
    print(f"  {title}")

    if ref_item is not None:
        ref_lat = avg_latency(ref_item.metrics.latency_ms)
        ref_lat_str = f"{ref_lat:.1f}" if ref_lat != float('inf') else "---"
        ref_size_mb = ref_item.metrics.size_bytes / 1024 / 1024
        extra_baseline_metrics = ""
        if not hide_extra:
            if baseline_precision is not None:
                extra_baseline_metrics += f"  |  Prec: {baseline_precision:.4f}"
            if baseline_recall is not None:
                extra_baseline_metrics += f"  |  Recall: {baseline_recall:.4f}"
            if baseline_iou is not None:
                extra_baseline_metrics += f"  |  IoU: {baseline_iou:.4f}"
        print(
            f"  Reference baseline (raw FP32): {baseline_acc:.4f}  |  "
            f"Lat: {ref_lat_str} ms  |  "
            f"Size: {ref_size_mb:.1f} MB  |  "
            f"Pareto-optimal: {len(pareto_set)} / {len(ok_history)}"
            + extra_baseline_metrics
            + (f"  |  Failed: {len(failed_history)}" if failed_history else "")
        )
    else:
        print(
            f"  Baseline mAP (FP32): {baseline_acc:.4f}  |  "
            f"Pareto-optimal: {len(pareto_set)} / {len(ok_history)}"
            + (f"  |  Failed: {len(failed_history)}" if failed_history else "")
        )

    print(border)
    print(header)
    print(sep)

    for i, h in enumerate(sorted_history, 1):
        cand = h.candidate
        m = h.metrics
        extra = h.extra

        config_str = f"w={cand.width_mult:.2f} p={cand.prune_ratio:.2f}"

        lat = avg_latency(m.latency_ms)
        lat_str = f"{lat:.1f}" if lat != float("inf") else "---"
        size_mb = m.size_bytes / 1024 / 1024

        torch_acc = extra.get("acc_recovered_torch")
        onnx_acc = extra.get("acc_onnx")
        ptq_acc = extra.get("acc_onnx_int8")
        qat_acc = extra.get("acc_onnx_after_qat")
        qat_int8_acc = extra.get("acc_onnx_int8_after_qat")
        final_acc = _deploy_acc(h)
        delta_final = final_acc - baseline_acc
        precision = _deploy_metric(h, "precision")
        recall = _deploy_metric(h, "recall")
        iou = _deploy_metric(h, "iou")

        ptq_delta = None
        onnx_acc_f = _as_float(onnx_acc)
        ptq_acc_f = _as_float(ptq_acc)
        if onnx_acc_f is not None and ptq_acc_f is not None:
            ptq_delta = ptq_acc_f - onnx_acc_f

        qat_int8_delta = None
        qat_acc_f = _as_float(qat_acc)
        qat_int8_acc_f = _as_float(qat_int8_acc)
        if qat_acc_f is not None and qat_int8_acc_f is not None:
            qat_int8_delta = qat_int8_acc_f - qat_acc_f

        if baseline_size and baseline_size > 0 and m.size_bytes > 0:
            compr = baseline_size / m.size_bytes
            compr_str = f"{compr:.1f}x"
        else:
            compr_str = "---"

        pareto_flag = "*" if cand.tag in pareto_set else " "
        deploy_str = _deploy_label(extra.get("deploy_onnx_kind"))

        row_parts = [
            f"{i:>3}",
            _col(cand.tag, W_TAG, "<"),
            _col(config_str, W_CFG, "<"),
            _col(_fmt_acc(torch_acc), W_ACC),
            _col(_fmt_acc(onnx_acc), W_ACC),
            _col(_fmt_acc(ptq_acc), W_ACC),
            _col(_fmt_delta(ptq_delta), W_DELTA),
            _col(_fmt_acc(qat_acc), W_ACC),
            _col(_fmt_acc(qat_int8_acc), W_ACC),
            _col(_fmt_delta(qat_int8_delta), W_DELTA),
            _col(f"{final_acc:.4f}", W_ACC),
            _col(f"{delta_final:+.4f}", W_DELTA),
        ]
        if not hide_extra:
            row_parts.extend([
                _col(_fmt_acc(baseline_precision), W_ACC),
                _col(_fmt_acc(baseline_recall), W_ACC),
                _col(_fmt_acc(baseline_iou), W_ACC),
                _col(_fmt_acc(precision), W_ACC),
                _col(_fmt_acc(recall), W_ACC),
                _col(_fmt_acc(iou), W_ACC),
            ])
        row_parts.extend([
            _col(deploy_str, W_DEPLOY),
            _col(lat_str, W_LAT),
        ])
        for key in latency_extra_keys:
            row_parts.append(_col(_fmt_latency(m.latency_ms.get(key)), latency_extra_widths[key]))
        row_parts.extend([
            _col(f"{size_mb:.1f}", W_SIZE),
            _col(compr_str, W_COMPR),
            _col(pareto_flag, W_P),
        ])
        print("  ".join(row_parts))

    print(sep)
    legend_parts = [
        "* = Pareto-optimal",
        "Torch = recovered/pruned PyTorch mAP",
        "ONNX = exported FP32 before QAT",
        "PTQ = INT8 before QAT",
        "QAT = exported FP32 after QAT",
        "QAT INT8 = INT8 after QAT",
        "Final/Δ Final = selected deploy artifact vs raw FP32 baseline",
    ]
    if not hide_extra:
        legend_parts.append("Prec/Recall/IoU = selected deploy artifact metrics")
        if latency_extra_keys:
            legend_parts.append("Lat CPU/NCNN/NPU = per-profile/per-device latency; Lat (ms) = average")
    legend_parts.append("Compr× = baseline_size / deploy_candidate_size")
    print("  " + "  |  ".join(legend_parts))

    if failed_history:
        print(f"\n  {'FAILED CANDIDATES':}")
        print(sep)
        for h in failed_history:
            err = h.extra.get("error", "unknown error")
            err_short = err[:80].replace("\n", " ")
            print(f"  {'F':>3}  {h.candidate.tag:<{W_TAG}}  {err_short}")
        print(sep)

    print(border + "\n")

def plot_pareto(
    history: List[HistoryItem],
    title: str = "Pareto Front",
    save_path: Optional[str] = None,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot_pareto] matplotlib not installed. Run: pip install matplotlib")
        return

    history = _candidate_history(history)
    ok_history = [h for h in history if not h.extra.get("failed", False)]
    if not ok_history:
        print("[plot_pareto] No successful candidates to plot.")
        return

    pareto_set: Set[str] = {h.candidate.tag for h in pareto_front(ok_history)}
    baseline_size = _baseline_size(history) or 1

    fig, ax = plt.subplots(figsize=(10, 6))

    for h in ok_history:
        lat = avg_latency(h.metrics.latency_ms)
        if lat == float("inf"):
            continue
        acc = h.metrics.acc
        size_mb = h.metrics.size_bytes / 1024 / 1024
        compr = baseline_size / h.metrics.size_bytes if h.metrics.size_bytes > 0 else 1.0

        is_pareto = h.candidate.tag in pareto_set
        color = "#e74c3c" if is_pareto else "#3498db"
        marker = "*" if is_pareto else "o"
        zorder = 5 if is_pareto else 3

        bubble = max(30, size_mb * 2)

        ax.scatter(
            lat, acc, s=bubble, c=color, marker=marker,
            alpha=0.85, zorder=zorder, edgecolors="white", linewidths=0.5
        )
        ax.annotate(
            f"{h.candidate.tag}\n({compr:.1f}×)",
            (lat, acc),
            textcoords="offset points",
            xytext=(6, 4),
            fontsize=7,
            color="#2c3e50",
        )

    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="*", color="w", markerfacecolor="#e74c3c",
               markersize=12, label="Pareto-optimal"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor="#3498db",
               markersize=9, label="Dominated"),
    ]
    ax.legend(handles=legend_elements, loc="lower right", fontsize=9)

    ax.set_xlabel("Latency (ms)", fontsize=11)
    ax.set_ylabel("mAP50-95", fontsize=11)
    ax.set_title(title, fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.4)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150)
        print(f"[plot_pareto] Saved to {save_path}")
    else:
        plt.show()


def load_history_jsonl(path: Path) -> List[HistoryItem]:
    items: List[HistoryItem] = []
    if not path.exists():
        return items
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
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