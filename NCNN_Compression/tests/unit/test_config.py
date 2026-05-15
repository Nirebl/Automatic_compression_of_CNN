from __future__ import annotations

import pytest

from xtrim.config import parse_config
from xtrim.types import DilatedConfig, StagedPruningConfig, TrimConfig


pytestmark = pytest.mark.unit


def test_parse_config_uses_defaults_and_extracts_operator_plan():
    parsed = parse_config({"operator_choice": {"enabled": True, "plan": {"auto": "lowrank", "model.1": "dense"}}})

    trim_cfg = parsed[7]
    op_choice_cfg = parsed[17]
    op_choice_plan = parsed[18]

    assert trim_cfg == TrimConfig()
    assert op_choice_cfg.enabled is True
    assert op_choice_plan == {"auto": "lowrank", "model.1": "dense"}


def test_parse_config_converts_dilated_lists_to_tuples():
    parsed = parse_config({"dilated": {"enabled": True, "rates": [1, 2, 3], "target_layers": ["a", "b"]}})
    dilated_cfg = parsed[-1]

    assert dilated_cfg == DilatedConfig(enabled=True, rates=(1, 2, 3), target_layers=("a", "b"))


def test_parse_config_converts_staged_pruning_milestones_to_tuple():
    parsed = parse_config({"staged_pruning": {"milestones": [0.6, 0.7], "intermediate_epochs": 12}})
    staged_cfg = parsed[-2]

    assert staged_cfg == StagedPruningConfig(milestones=(0.6, 0.7), intermediate_epochs=12)
