from __future__ import annotations

from typing import Dict, List

from .types import HistoryItem


def avg_latency(lat_map: Dict[str, float]) -> float:
    if not lat_map:
        return float("inf")
    return sum(lat_map.values()) / len(lat_map)


def dominates(a: HistoryItem, b: HistoryItem) -> bool:
    acc_a, acc_b = a.metrics.acc, b.metrics.acc
    size_a, size_b = a.metrics.size_bytes, b.metrics.size_bytes
    lat_a = avg_latency(a.metrics.latency_ms)
    lat_b = avg_latency(b.metrics.latency_ms)

    not_worse = (acc_a >= acc_b) and (lat_a <= lat_b) and (size_a <= size_b)
    strictly_better = (acc_a > acc_b) or (lat_a < lat_b) or (size_a < size_b)
    return not_worse and strictly_better


def pareto_front(items: List[HistoryItem]) -> List[HistoryItem]:
    valid = [h for h in items if not h.extra.get("failed", False)]
    front: List[HistoryItem] = []
    for i in valid:
        dominated = False
        for j in valid:
            if j is i:
                continue
            if dominates(j, i):
                dominated = True
                break
        if not dominated:
            front.append(i)
    return front