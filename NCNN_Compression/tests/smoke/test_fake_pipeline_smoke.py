from __future__ import annotations

import pytest

from xtrim.config import parse_config
from xtrim.results_table import load_history_jsonl


pytestmark = pytest.mark.smoke


def test_fake_pipeline_from_config_to_artifacts(make_orchestrator):
    cfg = {
        "latency": {"backend": "ort_android", "use_cache": False},
        "search": {"method": "grid", "init_random": 0},
        "search_space": {
            "width_mult": [1.0],
            "prune_ratio": [0.0, 0.2],
            "lowrank_rank": [0],
            "sparse_1x1": [0.0],
        },
    }
    parsed = parse_config(cfg)
    latency_cfg = parsed[8]
    search_cfg = parsed[14]
    search_space = parsed[5]

    orch = make_orchestrator(
        latency_cfg=latency_cfg,
        search_cfg=search_cfg,
        search_space=search_space,
    )
    history = orch.run(max_candidates=2)
    reloaded = load_history_jsonl(orch.history_path)

    assert len(history) == 3
    assert len(reloaded) == 3
    assert reloaded[0].extra["is_reference_baseline"] is True
    assert (orch.out_root / "pareto.json").exists()
