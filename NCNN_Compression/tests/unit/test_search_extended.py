from __future__ import annotations

import random

import pytest

from xtrim.search import (
    SearchPolicy,
    _all_candidates,
    _crowding_distance,
    _crossover,
    _fast_non_dominated_sort,
    _lat_agg,
    _mutate,
    _objectives,
    _scalar,
    _tournament_select,
)
from xtrim.types import CandidateConfig, SearchConfig

pytestmark = pytest.mark.unit


def test_search_helpers_cover_fallbacks_and_nsga_math(make_history_item):
    space = {"width_mult": [1.0, 0.5], "prune_ratio": [0.0, 0.2], "lowrank_rank": [0], "sparse_1x1": [0.0]}
    assert len(_all_candidates(space)) == 4

    item = make_history_item(extra={"latency_agg_ms": "bad", "scalar_score": "bad"}, latency={})
    assert _lat_agg(item) == float("inf")
    assert _scalar(item) < item.metrics.acc
    assert _objectives(item)[0] == item.metrics.acc

    objs = [(0.9, -1.0, -10), (0.8, -2.0, -20), (0.7, -3.0, -30)]
    fronts = _fast_non_dominated_sort(objs)
    assert fronts == [[0], [1], [2]]
    assert _crowding_distance([0, 1], objs) == {0: float("inf"), 1: float("inf")}
    crowd = _crowding_distance([0, 1, 2], [(0.1, 0.1, 0.1)] * 3)
    assert crowd[0] == float("inf") and crowd[2] == float("inf")


def test_tournament_crossover_mutation_and_policy_paths(make_history_item):
    rng = random.Random(1)
    a = CandidateConfig(1.0, 0.0, 0, 0.0, "a")
    b = CandidateConfig(0.5, 0.2, 8, 0.5, "b")
    child = _crossover(rng, a, b)
    assert child.tag.startswith("w")

    mut = _mutate(random.Random(0), a, {"width_mult": [0.5], "prune_ratio": [0.2], "lowrank_rank": [8], "sparse_1x1": [0.5]}, prob=1.0)
    assert (mut.width_mult, mut.prune_ratio, mut.lowrank_rank, mut.sparse_1x1) == (0.5, 0.2, 8, 0.5)

    selected = _tournament_select(random.Random(0), [0, 1], rank={0: 1, 1: 0}, crowd={0: 10, 1: 0}, k=2)
    assert selected == 1
    selected2 = _tournament_select(random.Random(0), [0, 1], rank={0: 0, 1: 0}, crowd={0: 0, 1: 10}, k=2)
    assert selected2 == 1

    space = {"width_mult": [1.0, 0.75, 0.5], "prune_ratio": [0.0, 0.2], "lowrank_rank": [0], "sparse_1x1": [0.0]}
    baseline = make_history_item(tag="ref", baseline=True)
    grid = SearchPolicy.create(SearchConfig(method="grid", init_random=0), space)
    assert grid.next_candidate([baseline]).tag == "w0.5_p0.2_r0_s0.0"

    random_policy = SearchPolicy.create(SearchConfig(method="random", init_random=0), space)
    assert random_policy.next_candidate([]) is not None

    init_random = SearchPolicy.create(SearchConfig(method="nsga2", init_random=2), space)
    assert init_random.next_candidate([make_history_item(tag="x", width=1.0)]) is not None

    tried = [make_history_item(tag=f"t{i}", width=w, prune=p, acc=0.9 - i * 0.01, size=100 + i, latency={"p": 10 + i})
             for i, (w, p) in enumerate([(1.0, 0.0), (1.0, 0.2), (0.75, 0.0), (0.75, 0.2)])]
    nsga = SearchPolicy.create(SearchConfig(method="nsga2", init_random=0, population=4, offspring=4), space)
    cand = nsga.next_candidate(tried)
    assert cand is not None
    assert (cand.width_mult, cand.prune_ratio) in {(0.5, 0.0), (0.5, 0.2)}

    all_hist = [make_history_item(tag=str(i), width=w, prune=p) for i, (w, p) in enumerate([(1.0, 0.0), (1.0, 0.2), (0.75, 0.0), (0.75, 0.2), (0.5, 0.0), (0.5, 0.2)])]
    assert grid.next_candidate(all_hist) is None
