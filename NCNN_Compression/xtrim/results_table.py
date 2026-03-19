from __future__ import annotations

import json
import dataclasses
from pathlib import Path
from typing import List, Optional, Set

from .types import HistoryItem, CandidateConfig, Metrics
from .pareto import avg_latency, pareto_front


def _baseline_size(history: List[HistoryItem]) -> Optional[int]:
    for h in history:
        if (not h.extra.get("failed", False)
                and h.candidate.width_mult == 1.0
                and h.candidate.prune_ratio == 0.0
                and h.candidate.sparse_1x1 == 0.0
                and h.candidate.lowrank_rank == 0):
            return h.metrics.size_bytes
    return None


def _find_baseline_acc(history: List[HistoryItem]) -> float:
    valid = [h for h in history if not h.extra.get("failed", False)]
    exact = [
        h for h in valid
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


def print_results_table(
    history: List[HistoryItem],
    baseline_acc: Optional[float] = None,
    title: str = "RESULTS SUMMARY",
) -> None:
    if not history:
        print("[results] No candidates evaluated.")
        return

    ok_history     = [h for h in history if not h.extra.get("failed", False)]
    failed_history = [h for h in history if h.extra.get("failed", False)]

    pareto_set: Set[str] = {h.candidate.tag for h in pareto_front(ok_history)}

    if baseline_acc is None:
        baseline_acc = _find_baseline_acc(history)

    baseline_size = _baseline_size(history)

    sorted_history = sorted(ok_history, key=lambda h: -h.metrics.acc)

    W_TAG   = max(24, max(len(h.candidate.tag) for h in history))
    W_CFG   = 16
    W_MAP   = 8
    W_DMAP  = 8
    W_PTQ   = 8
    W_DINT8 = 7
    W_QAT   = 8
    W_LAT   = 10
    W_SIZE  = 9
    W_COMPR = 7
    W_P     = 1

    def _col(label: str, w: int, align: str = ">") -> str:
        return f"{label:{align}{w}}"

    header_parts = [
        "  #",
        _col("Candidate",  W_TAG,   "<"),
        _col("Config",     W_CFG,   "<"),
        _col("mAP",        W_MAP),
        _col("Δ mAP",      W_DMAP),
        _col("mAP PTQ",    W_PTQ),
        _col("Δ INT8",     W_DINT8),
        _col("mAP QAT",    W_QAT),
        _col("Lat (ms)",   W_LAT),
        _col("Size (MB)",  W_SIZE),
        _col("Compr×",     W_COMPR),
        _col("P",          W_P),
    ]
    header = "  ".join(header_parts)
    sep    = "-" * len(header)
    border = "=" * len(header)

    print(f"\n{border}")
    print(f"  {title}")
    print(f"  Baseline mAP (FP32): {baseline_acc:.4f}  |  "
          f"Pareto-optimal: {len(pareto_set)} / {len(ok_history)}"
          + (f"  |  Failed: {len(failed_history)}" if failed_history else ""))
    print(border)
    print(header)
    print(sep)

    for i, h in enumerate(sorted_history, 1):
        cand  = h.candidate
        m     = h.metrics
        extra = h.extra

        config_str = f"w={cand.width_mult:.2f} p={cand.prune_ratio:.2f}"

        lat = avg_latency(m.latency_ms)
        lat_str  = f"{lat:.1f}" if lat != float("inf") else "---"
        size_mb  = m.size_bytes / 1024 / 1024

        acc       = m.acc
        delta_map = acc - baseline_acc

        ptq_acc   = extra.get("acc_onnx_int8")
        ptq_str   = f"{ptq_acc:.4f}" if ptq_acc is not None else "---"

        drop_int8 = extra.get("acc_drop_int8")
        if drop_int8 is not None:
            delta_int8_str = f"{-drop_int8:+.4f}"
        else:
            delta_int8_str = "---"

        qat_int8 = extra.get("acc_onnx_int8_after_qat")
        qat_str  = f"{qat_int8:.4f}" if qat_int8 is not None else "---"

        if baseline_size and baseline_size > 0 and m.size_bytes > 0:
            compr = baseline_size / m.size_bytes
            compr_str = f"{compr:.1f}x"
        else:
            compr_str = "---"

        pareto_flag = "*" if cand.tag in pareto_set else " "

        row_parts = [
            f"{i:>3}",
            _col(cand.tag,      W_TAG,   "<"),
            _col(config_str,    W_CFG,   "<"),
            _col(f"{acc:.4f}",  W_MAP),
            _col(f"{delta_map:+.4f}", W_DMAP),
            _col(ptq_str,       W_PTQ),
            _col(delta_int8_str, W_DINT8),
            _col(qat_str,       W_QAT),
            _col(lat_str,       W_LAT),
            _col(f"{size_mb:.1f}", W_SIZE),
            _col(compr_str,     W_COMPR),
            _col(pareto_flag,   W_P),
        ]
        print("  ".join(row_parts))

    print(sep)
    print(f"  * = Pareto-optimal  |  Δ mAP vs baseline  |  Δ INT8 = PTQ - FP32  |  Compr× = baseline_size / candidate_size")

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
        color  = "#e74c3c" if is_pareto else "#3498db"
        marker = "*" if is_pareto else "o"
        zorder = 5 if is_pareto else 3

        bubble = max(30, size_mb * 2)

        ax.scatter(lat, acc, s=bubble, c=color, marker=marker,
                   alpha=0.85, zorder=zorder, edgecolors="white", linewidths=0.5)
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
               markersize=9,  label="Dominated"),
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
        rec  = json.loads(line)
        cand = CandidateConfig(**rec["candidate"])
        met  = Metrics(**rec["metrics"])
        items.append(HistoryItem(
            candidate=cand,
            metrics=met,
            artifacts_dir=rec["artifacts_dir"],
            extra=rec.get("extra", {}),
        ))
    return items
