from __future__ import annotations

import json

import pytest

from xtrim.bench_cache import BenchCache


pytestmark = pytest.mark.unit


def test_bench_cache_roundtrip(tmp_path):
    path = tmp_path / "cache.json"
    cache = BenchCache(path)
    cache.set("k", 12.5)
    cache.save()

    loaded = BenchCache(path)
    entry = loaded.get("k")

    assert entry is not None
    assert entry.avg_ms == 12.5
    assert entry.ts


def test_bench_cache_returns_none_for_malformed_record(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text(json.dumps({"bad": {"avg_ms": "not-a-number"}}), encoding="utf-8")

    assert BenchCache(path).get("bad") is None
