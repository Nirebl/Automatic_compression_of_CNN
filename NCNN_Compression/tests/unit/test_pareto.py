from __future__ import annotations

import math

import pytest

from xtrim.pareto import avg_latency, dominates, pareto_front


pytestmark = pytest.mark.unit


def test_avg_latency_empty_is_infinite():
    assert math.isinf(avg_latency({}))


def test_dominates_requires_not_worse_and_strict_improvement(make_history_item):
    better = make_history_item(tag="better", acc=0.8, size=80, latency={"a": 8.0})
    worse = make_history_item(tag="worse", acc=0.7, size=100, latency={"a": 10.0})
    equal = make_history_item(tag="equal", acc=0.8, size=80, latency={"a": 8.0})

    assert dominates(better, worse) is True
    assert dominates(worse, better) is False
    assert dominates(better, equal) is False


def test_pareto_front_excludes_failed_items(make_history_item):
    a = make_history_item(tag="a", acc=0.8, size=100, latency={"a": 10.0})
    b = make_history_item(tag="b", acc=0.7, size=90, latency={"a": 8.0})
    dominated = make_history_item(tag="dominated", acc=0.6, size=120, latency={"a": 12.0})
    failed = make_history_item(tag="failed", acc=0.99, size=1, latency={"a": 1.0}, failed=True)

    tags = {h.candidate.tag for h in pareto_front([a, b, dominated, failed])}

    assert tags == {"a", "b"}
