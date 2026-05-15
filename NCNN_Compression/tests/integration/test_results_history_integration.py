from __future__ import annotations

import pytest

from xtrim.results_table import load_history_jsonl, print_results_table


pytestmark = pytest.mark.integration


def test_history_roundtrip_and_results_table_output(tmp_path, make_orchestrator, capsys):
    orch = make_orchestrator(out_root=tmp_path / "out")
    history = orch.run(max_candidates=1)

    loaded = load_history_jsonl(orch.history_path)
    print_results_table(loaded, title="TEST TABLE")
    out = capsys.readouterr().out

    assert len(loaded) == len(history)
    assert "TEST TABLE" in out
    assert "Reference baseline" in out
    assert "baseline" not in {h.candidate.tag for h in loaded if not h.extra.get("is_reference_baseline")}
