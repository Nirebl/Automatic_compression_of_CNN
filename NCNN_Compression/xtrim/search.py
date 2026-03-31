from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from .types import CandidateConfig, HistoryItem, SearchConfig
from .pareto import avg_latency


def _all_candidates(search_space: Dict[str, List[Any]]) -> List[CandidateConfig]:
    widths = [float(x) for x in search_space.get("width_mult", [1.0])]
    prunes = [float(x) for x in search_space.get("prune_ratio", [0.0])]
    ranks = [int(x) for x in search_space.get("lowrank_rank", [0])]
    sparse_vals = [float(x) for x in search_space.get("sparse_1x1", [0.0])]

    out: List[CandidateConfig] = []
    for w in widths:
        for p in prunes:
            for r in ranks:
                for s in sparse_vals:
                    tag = f"w{w}_p{p}_r{r}_s{s}"
                    out.append(CandidateConfig(
                        width_mult=w, prune_ratio=p, lowrank_rank=r,
                        sparse_1x1=s, tag=tag,
                    ))
    return out


def _key(c: CandidateConfig) -> Tuple[float, float, int, float]:
    return (float(c.width_mult), float(c.prune_ratio), int(c.lowrank_rank), float(getattr(c, "sparse_1x1", 0.0)))


def _lat_agg(item: HistoryItem) -> float:
    if "latency_agg_ms" in item.extra:
        try:
            return float(item.extra["latency_agg_ms"])
        except Exception:
            pass
    return float(avg_latency(item.metrics.latency_ms))


def _scalar(item: HistoryItem) -> float:
    if "scalar_score" in item.extra:
        try:
            return float(item.extra["scalar_score"])
        except Exception:
            pass
    acc = float(item.metrics.acc)
    lat = max(1e-6, _lat_agg(item))
    size_mb = max(1e-9, float(item.metrics.size_bytes) / 1e6)
    return acc - math.log1p(lat) - 0.1 * math.log1p(size_mb)


def _objectives(item: HistoryItem) -> Tuple[float, float, float]:
    return (
        float(item.metrics.acc),
        -float(_lat_agg(item)),
        -float(item.metrics.size_bytes),
    )


def _dominates(a: Tuple[float, float, float], b: Tuple[float, float, float]) -> bool:
    not_worse = (a[0] >= b[0]) and (a[1] >= b[1]) and (a[2] >= b[2])
    strictly = (a[0] > b[0]) or (a[1] > b[1]) or (a[2] > b[2])
    return not_worse and strictly


def _fast_non_dominated_sort(objs: List[Tuple[float, float, float]]) -> List[List[int]]:
    n = len(objs)
    S = [set() for _ in range(n)]
    n_dom = [0] * n
    fronts: List[List[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            if _dominates(objs[p], objs[q]):
                S[p].add(q)
            elif _dominates(objs[q], objs[p]):
                n_dom[p] += 1
        if n_dom[p] == 0:
            fronts[0].append(p)

    i = 0
    while i < len(fronts) and fronts[i]:
        nxt: List[int] = []
        for p in fronts[i]:
            for q in S[p]:
                n_dom[q] -= 1
                if n_dom[q] == 0:
                    nxt.append(q)
        i += 1
        if nxt:
            fronts.append(nxt)

    return fronts


def _crowding_distance(front: List[int], objs: List[Tuple[float, float, float]]) -> Dict[int, float]:
    dist = {i: 0.0 for i in front}
    if len(front) <= 2:
        for i in front:
            dist[i] = float("inf")
        return dist

    m = 3
    for k in range(m):
        front_sorted = sorted(front, key=lambda i: objs[i][k])
        dist[front_sorted[0]] = float("inf")
        dist[front_sorted[-1]] = float("inf")

        vmin = objs[front_sorted[0]][k]
        vmax = objs[front_sorted[-1]][k]
        denom = (vmax - vmin) if (vmax - vmin) != 0 else 1.0

        for j in range(1, len(front_sorted) - 1):
            prev_v = objs[front_sorted[j - 1]][k]
            next_v = objs[front_sorted[j + 1]][k]
            dist[front_sorted[j]] += (next_v - prev_v) / denom

    return dist


def _tournament_select(
    rng: random.Random,
    indices: List[int],
    rank: Dict[int, int],
    crowd: Dict[int, float],
    k: int,
) -> int:
    pick = rng.sample(indices, k=min(k, len(indices)))
    best = pick[0]
    for i in pick[1:]:
        if rank[i] < rank[best]:
            best = i
        elif rank[i] == rank[best]:
            if crowd.get(i, 0.0) > crowd.get(best, 0.0):
                best = i
    return best


def _crossover(rng: random.Random, a: CandidateConfig, b: CandidateConfig) -> CandidateConfig:
    w = a.width_mult if rng.random() < 0.5 else b.width_mult
    p = a.prune_ratio if rng.random() < 0.5 else b.prune_ratio
    r = a.lowrank_rank if rng.random() < 0.5 else b.lowrank_rank
    s = getattr(a, "sparse_1x1", 0.0) if rng.random() < 0.5 else getattr(b, "sparse_1x1", 0.0)
    tag = f"w{w}_p{p}_r{r}_s{s}"
    return CandidateConfig(width_mult=float(w), prune_ratio=float(p), lowrank_rank=int(r), sparse_1x1=float(s), tag=tag)


def _mutate(rng: random.Random, c: CandidateConfig, space: Dict[str, List[Any]], prob: float) -> CandidateConfig:
    w_list = [float(x) for x in space.get("width_mult", [c.width_mult])]
    p_list = [float(x) for x in space.get("prune_ratio", [c.prune_ratio])]
    r_list = [int(x) for x in space.get("lowrank_rank", [c.lowrank_rank])]
    s_list = [float(x) for x in space.get("sparse_1x1", [getattr(c, "sparse_1x1", 0.0)])]

    w, p, r, s = c.width_mult, c.prune_ratio, c.lowrank_rank, getattr(c, "sparse_1x1", 0.0)

    if rng.random() < prob:
        w = rng.choice(w_list)
    if rng.random() < prob:
        p = rng.choice(p_list)
    if rng.random() < prob:
        r = rng.choice(r_list)
    if rng.random() < prob:
        s = rng.choice(s_list)

    tag = f"w{w}_p{p}_r{r}_s{s}"
    return CandidateConfig(width_mult=float(w), prune_ratio=float(p), lowrank_rank=int(r), sparse_1x1=float(s), tag=tag)


@dataclass
class SearchPolicy:
    cfg: SearchConfig
    space: Dict[str, List[Any]]
    rng: random.Random

    @classmethod
    def create(cls, cfg: SearchConfig, space: Dict[str, List[Any]]) -> "SearchPolicy":
        return cls(cfg=cfg, space=space, rng=random.Random(int(cfg.seed)))

    def next_candidate(self, history: List[HistoryItem]) -> Optional[CandidateConfig]:
        # reference baseline не должен считаться кандидатом поиска
        search_history = [h for h in history if not h.extra.get("is_reference_baseline", False)]

        all_cands = _all_candidates(self.space)
        tried = {_key(h.candidate) for h in search_history}
        remaining = [c for c in all_cands if _key(c) not in tried]
        if not remaining:
            return None

        method = str(self.cfg.method).lower().strip()

        if len(search_history) < int(self.cfg.init_random):
            return self.rng.choice(remaining)

        if method == "random":
            return self.rng.choice(remaining)

        if method == "grid":
            remaining_sorted = sorted(
                remaining,
                key=lambda c: (
                    c.width_mult,
                    -c.prune_ratio,
                    c.lowrank_rank,
                    -getattr(c, "sparse_1x1", 0.0),
                ),
            )
            return remaining_sorted[0]

        return self._nsga2(search_history, remaining)

    def _nsga2(self, history: List[HistoryItem], remaining: List[CandidateConfig]) -> CandidateConfig:
        pop_n = max(4, int(self.cfg.population))
        hist_sorted = sorted(history, key=_scalar, reverse=True)
        pop = hist_sorted[:min(pop_n, len(hist_sorted))]

        if len(pop) < 4:
            return self.rng.choice(remaining)

        objs = [_objectives(it) for it in pop]
        fronts = _fast_non_dominated_sort(objs)

        rank: Dict[int, int] = {}
        for ri, f in enumerate(fronts):
            for idx in f:
                rank[idx] = ri

        crowd: Dict[int, float] = {}
        for f in fronts:
            cd = _crowding_distance(f, objs)
            crowd.update(cd)

        indices = list(range(len(pop)))

        attempts = max(4, int(self.cfg.offspring))
        for _ in range(attempts):
            p1 = _tournament_select(self.rng, indices, rank, crowd, int(self.cfg.tournament_k))
            p2 = _tournament_select(self.rng, indices, rank, crowd, int(self.cfg.tournament_k))
            a = pop[p1].candidate
            b = pop[p2].candidate

            child = a
            if self.rng.random() < float(self.cfg.crossover_prob):
                child = _crossover(self.rng, a, b)

            child = _mutate(self.rng, child, self.space, float(self.cfg.mutation_prob))

            rk = _key(child)
            if any(_key(c) == rk for c in remaining):
                return child

        remaining_sorted = sorted(remaining, key=lambda c: (c.width_mult, -c.prune_ratio, c.lowrank_rank, -getattr(c, "sparse_1x1", 0.0)))
        return self.rng.choice(remaining_sorted[: max(1, len(remaining_sorted) // 2)])
