from __future__ import annotations

import pytest

from xtrim.types import StagedPruningConfig, TrimConfig

pytestmark = pytest.mark.unit


def test_build_pruning_plan_converts_cumulative_targets_to_local_ratios(make_orchestrator):
    orch = make_orchestrator(
        trim_cfg=TrimConfig(prune_mode="staged"),
        staged_pruning_cfg=StagedPruningConfig(milestones=(0.6, 0.7, 0.75)),
    )

    plan = orch._build_pruning_plan(0.8)

    assert [s["target_total"] for s in plan] == [0.6, 0.7, 0.75, 0.8]
    assert [s["local_ratio"] for s in plan] == pytest.approx([0.6, 0.25, 1.0 / 6.0, 0.2])


def test_staged_candidate_runs_intermediate_recovery_and_records_history(make_orchestrator):
    built_prunes = []
    applied_ratios = []
    train_cfgs = []

    class Student:
        def __init__(self, cand):
            self.candidate = cand
            self.torch_model = None

    def build(cand):
        built_prunes.append(cand.prune_ratio)
        return Student(cand)

    def apply_stage(_student, ratio: float, label: str):
        applied_ratios.append((label, ratio))
        return {"local_ratio": ratio}

    def finetune(_student, train_cfg):
        train_cfgs.append((train_cfg.short_epochs, train_cfg.lr))
        return {"epochs": train_cfg.short_epochs, "lr": train_cfg.lr}

    orch = make_orchestrator(
        search_space={"width_mult": [1.0], "prune_ratio": [0.8], "lowrank_rank": [0], "sparse_1x1": [0.0]},
        trim_cfg=TrimConfig(prune_mode="staged"),
        staged_pruning_cfg=StagedPruningConfig(
            milestones=(0.6, 0.7, 0.75),
            intermediate_epochs=30,
            intermediate_lr=1e-4,
            final_epochs=100,
            final_lr=2e-4,
            eval_after_each_stage=True,
        ),
        build_candidate_fn=build,
        apply_pruning_stage_fn=apply_stage,
        finetune_fn=finetune,
        eval_metrics_fn=lambda _student: {
            "map50_95": 0.7,
            "precision": 0.61,
            "recall": 0.72,
            "iou": 0.76,
            "map50": 0.83,
        },
    )

    history = orch.run(max_candidates=1)
    candidate = history[1]

    assert built_prunes == pytest.approx([0.0, 0.6])
    assert [r for _label, r in applied_ratios] == pytest.approx([0.25, 1.0 / 6.0, 0.2])
    assert train_cfgs == [(30, 1e-4), (30, 1e-4), (30, 1e-4), (100, 2e-4)]
    assert candidate.extra["pruning_mode"] == "staged"
    stages = candidate.extra["staged_pruning"]["stages"]
    assert [s["target_total"] for s in stages] == [0.6, 0.7, 0.75, 0.8]
    assert [s["acc_before_recovery"] for s in stages] == pytest.approx([0.7, 0.7, 0.7, 0.7])
    assert [s["acc_after_recovery"] for s in stages] == pytest.approx([0.7, 0.7, 0.7, 0.7])
    assert [s["precision_before_recovery"] for s in stages] == pytest.approx([0.61, 0.61, 0.61, 0.61])
    assert [s["recall_after_recovery"] for s in stages] == pytest.approx([0.72, 0.72, 0.72, 0.72])
    assert [s["iou_after_recovery"] for s in stages] == pytest.approx([0.76, 0.76, 0.76, 0.76])
    assert stages[-1]["metrics_after_recovery"] == {
        "map50_95": 0.7,
        "precision": 0.61,
        "recall": 0.72,
        "iou": 0.76,
        "map50": 0.83,
    }
    assert candidate.extra["finetune_logs"] == {"epochs": 100, "lr": 2e-4}


def test_match_one_shot_architecture_targets_final_widths(make_orchestrator):
    built_prunes = []
    targeted_widths = []

    class Student:
        def __init__(self, cand, arch):
            self.candidate = cand
            self.arch = arch
            self.torch_model = None

    def build(cand):
        built_prunes.append(cand.prune_ratio)
        if cand.prune_ratio == pytest.approx(0.0):
            return Student(cand, {"conv_out_channels": {"a": 100}, "detect_hidden_channels": {"h": 40}, "total_params": 1000})
        if cand.prune_ratio == pytest.approx(0.8):
            return Student(cand, {"conv_out_channels": {"a": 20}, "detect_hidden_channels": {"h": 10}, "total_params": 200})
        if cand.prune_ratio == pytest.approx(0.6):
            return Student(cand, {"conv_out_channels": {"a": 40}, "detect_hidden_channels": {"h": 20}, "total_params": 400})
        raise AssertionError(f"unexpected build prune_ratio={cand.prune_ratio}")

    def extract(student):
        return dict(student.arch)

    def apply_stage(student, ratio: float, label: str, target_architecture=None):
        assert target_architecture is not None
        targeted_widths.append((label, target_architecture))
        student.arch = {
            "conv_out_channels": dict(target_architecture["conv_out_channels"]),
            "detect_hidden_channels": dict(target_architecture["detect_hidden_channels"]),
            "total_params": target_architecture["total_params"],
        }
        return {"local_ratio": ratio, "target_architecture_applied": True}

    orch = make_orchestrator(
        search_space={"width_mult": [1.0], "prune_ratio": [0.8], "lowrank_rank": [0], "sparse_1x1": [0.0]},
        trim_cfg=TrimConfig(prune_mode="staged"),
        staged_pruning_cfg=StagedPruningConfig(
            milestones=(0.6, 0.7, 0.75),
            target_mode="match_one_shot_architecture",
            intermediate_epochs=1,
            final_epochs=1,
            eval_after_each_stage=False,
        ),
        build_candidate_fn=build,
        apply_pruning_stage_fn=apply_stage,
        extract_pruning_architecture_fn=extract,
    )

    history = orch.run(max_candidates=1)
    candidate = history[1]

    assert built_prunes == pytest.approx([0.0, 0.8, 0.6])
    assert [w[1]["conv_out_channels"]["a"] for w in targeted_widths] == [30, 25, 20]
    assert [w[1]["detect_hidden_channels"]["h"] for w in targeted_widths] == [15, 13, 10]
    staged = candidate.extra["staged_pruning"]
    assert staged["target_mode"] == "match_one_shot_architecture"
    assert staged["final_architecture_match"]["conv_mismatch_count"] == 0
    assert staged["final_architecture_match"]["detect_mismatch_count"] == 0
