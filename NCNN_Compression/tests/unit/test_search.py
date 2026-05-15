from __future__ import annotations

import pytest

from xtrim.search import (
    SearchPolicy,
    _all_candidates,
    _crowding_distance,
    _fast_non_dominated_sort,
)
from xtrim.types import SearchConfig


pytestmark = pytest.mark.unit


def test_all_candidates_builds_cartesian_product_with_tags():
    out = _all_candidates({
        "width_mult": [1.0, 0.75],
        "prune_ratio": [0.0, 0.2],
        "lowrank_rank": [0],
        "sparse_1x1": [0.0, 0.5],
    })

    assert len(out) == 8
    assert out[0].tag == "w1.0_p0.0_r0_s0.0"
    assert out[-1].tag == "w0.75_p0.2_r0_s0.5"


def test_grid_policy_ignores_reference_baseline_and_uses_sort_order(make_history_item):
    policy = SearchPolicy.create(
        SearchConfig(method="grid", init_random=0),
        {"width_mult": [0.75, 1.0], "prune_ratio": [0.0, 0.2], "lowrank_rank": [0], "sparse_1x1": [0.0]},
    )
    baseline = make_history_item(tag="baseline_raw", baseline=True)

    cand = policy.next_candidate([baseline])

    assert cand is not None
    assert (cand.width_mult, cand.prune_ratio) == (0.75, 0.2)


def test_random_policy_is_deterministic_for_same_seed():
    space = {"width_mult": [1.0, 0.75], "prune_ratio": [0.0, 0.2], "lowrank_rank": [0], "sparse_1x1": [0.0]}
    p1 = SearchPolicy.create(SearchConfig(method="random", seed=7, init_random=0), space)
    p2 = SearchPolicy.create(SearchConfig(method="random", seed=7, init_random=0), space)

    assert p1.next_candidate([]) == p2.next_candidate([])


def test_non_dominated_sort_and_crowding_distance():
    objs = [
        (0.9, -10.0, -100.0),
        (0.8, -12.0, -120.0),
        (0.7, -8.0, -90.0),
    ]
    fronts = _fast_non_dominated_sort(objs)
    crowd = _crowding_distance(fronts[0], objs)

    assert set(fronts[0]) == {0, 2}
    assert crowd[0] == float("inf")
    assert crowd[2] == float("inf")
