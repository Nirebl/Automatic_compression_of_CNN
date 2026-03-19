"""Standalone viewer: prints the results table from an existing history.jsonl."""
from __future__ import annotations

import argparse
from pathlib import Path

from xtrim.results_table import load_history_jsonl, print_results_table, plot_pareto


def main() -> int:
    ap = argparse.ArgumentParser(description="Print XTrim results table from history.jsonl")
    ap.add_argument(
        "--history",
        type=str,
        default="outputs/xtrim_no_lr_test/history.jsonl",
        help="Path to history.jsonl",
    )
    ap.add_argument(
        "--title",
        type=str,
        default="RESULTS SUMMARY",
        help="Table title",
    )
    ap.add_argument(
        "--plot",
        action="store_true",
        help="Show interactive Pareto scatter (latency vs mAP, bubble size = model size)",
    )
    ap.add_argument(
        "--plot-save",
        type=str,
        default=None,
        metavar="FILE",
        help="Save Pareto plot to this file instead of displaying (e.g. pareto.png)",
    )
    args = ap.parse_args()

    history = load_history_jsonl(Path(args.history))
    if not history:
        print(f"[show_results] No data found in: {args.history}")
        return 1

    print_results_table(history, title=args.title)

    if args.plot or args.plot_save:
        plot_pareto(history, title=args.title, save_path=args.plot_save)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
