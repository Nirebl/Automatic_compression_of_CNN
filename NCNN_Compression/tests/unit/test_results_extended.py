from __future__ import annotations

import builtins
import sys
import types

import pytest

from xtrim.results_table import (
    _baseline_size,
    _find_baseline_acc,
    load_history_jsonl,
    plot_pareto,
    print_results_table,
)

pytestmark = pytest.mark.unit


def test_baseline_fallbacks_and_print_branches(make_history_item, capsys):
    exact = make_history_item(tag="exact", acc=0.8, size=100, width=1.0, prune=0.0)
    base = make_history_item(tag="base", acc=0.7, size=120, width=1.0, prune=0.0, sparse=0.1)
    other = make_history_item(tag="other", acc=0.6, size=80, width=0.5, prune=0.2)
    failed = make_history_item(tag="failed", failed=True, extra={"error": "bad\nline"})

    assert _baseline_size([exact, other]) == 100
    assert _find_baseline_acc([exact, base, other]) == 0.8
    assert _find_baseline_acc([base, other]) == 0.7
    assert _find_baseline_acc([other]) == 0.6
    assert _find_baseline_acc([failed]) == 0.0

    print_results_table([])
    print_results_table([other, failed], baseline_acc=0.5)
    out = capsys.readouterr().out
    assert "No candidates evaluated" in out
    assert "Baseline mAP" in out
    assert "FAILED CANDIDATES" in out
    assert "bad line" in out


def test_plot_pareto_branches_with_fake_matplotlib(make_history_item, tmp_path, capsys, monkeypatch):
    plot_pareto([make_history_item(tag="f", failed=True)])
    assert "No successful candidates" in capsys.readouterr().out

    calls = {"scatter": 0, "save": 0}

    class FakeAx:
        def scatter(self, *_a, **_kw): calls["scatter"] += 1
        def annotate(self, *_a, **_kw): return None
        def set_xlabel(self, *_a, **_kw): return None
        def set_ylabel(self, *_a, **_kw): return None
        def set_title(self, *_a, **_kw): return None
        def grid(self, *_a, **_kw): return None
        def legend(self, *_a, **_kw): return None

    class FakeFig:
        def tight_layout(self): return None
        def savefig(self, *_a, **_kw): calls["save"] += 1

    fake_plt = types.SimpleNamespace(subplots=lambda **_kw: (FakeFig(), FakeAx()), show=lambda: None)
    monkeypatch.setitem(sys.modules, "matplotlib", types.ModuleType("matplotlib"))
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", fake_plt)
    monkeypatch.setitem(sys.modules, "matplotlib.lines", types.SimpleNamespace(Line2D=lambda *_a, **_kw: object()))

    items = [
        make_history_item(tag="a", acc=0.8, size=100, latency={"p": 10.0}),
        make_history_item(tag="b", acc=0.7, size=0, latency={}),
    ]
    plot_pareto(items, save_path=str(tmp_path / "plot.png"))
    assert calls["scatter"] == 1
    assert calls["save"] == 1


def test_plot_pareto_handles_missing_matplotlib(monkeypatch, make_history_item, capsys):
    real_import = builtins.__import__
    def fake_import(name, *args, **kwargs):
        if name == "matplotlib.pyplot":
            raise ImportError("no")
        return real_import(name, *args, **kwargs)
    monkeypatch.setattr(builtins, "__import__", fake_import)
    plot_pareto([make_history_item(tag="a")])
    assert "matplotlib not installed" in capsys.readouterr().out


def test_load_history_missing_returns_empty(tmp_path):
    assert load_history_jsonl(tmp_path / "missing.jsonl") == []


def test_results_table_shows_stage_metrics_and_final_deploy_map(make_history_item, capsys):
    ref = make_history_item(
        tag="baseline_raw",
        acc=0.6193,
        size=83_600_000,
        latency={"phone": 868.2},
        baseline=True,
        extra={
            "deploy_onnx_kind": "fp32_raw_baseline",
            "precision_deploy": 0.7012,
            "recall_deploy": 0.6899,
            "iou_deploy": 0.7634,
        },
    )
    cand = make_history_item(
        tag="w1.0_p0.84_r0_s0.0",
        acc=0.5416,
        size=5_500_000,
        latency={
            "cpu_ort_xnnpack/phone_cpu": 235.4,
            "ncnn_android/phone_ncnn": 210.7,
            "npu_nnapi/phone_npu": 180.2,
        },
        prune=0.84,
        extra={
            "acc_recovered_torch": 0.5416,
            "acc_onnx": 0.4391,
            "acc_onnx_int8": 0.4017,
            "acc_onnx_after_qat": 0.5500,
            "acc_onnx_int8_after_qat": 0.5416,
            "acc_deploy": 0.5416,
            "precision_deploy": 0.6123,
            "recall_deploy": 0.7345,
            "iou_deploy": 0.7456,
            "map50_deploy": 0.8123,
            "deploy_onnx_kind": "int8_after_qat",
        },
    )

    print_results_table([ref, cand])
    out = capsys.readouterr().out

    assert "Torch" in out
    assert "ONNX" in out
    assert "PTQ" in out
    assert "QAT INT8" in out
    assert "Final" in out
    assert "Init Prec" in out
    assert "Init Rec" in out
    assert "Prec" in out
    assert "Recall" in out
    assert "Init IoU" in out
    assert "IoU" in out
    assert "mAP50" not in out
    assert "Lat CPU" in out
    assert "Lat NCNN" in out
    assert "Lat NPU" in out
    assert "Deploy" in out
    assert "0.4391" in out
    assert "0.4017" in out
    assert "0.5500" in out
    assert "0.5416" in out
    assert "0.7012" in out
    assert "0.6899" in out
    assert "0.6123" in out
    assert "0.7345" in out
    assert "0.7634" in out
    assert "0.7456" in out
    assert "0.8123" not in out
    assert "QAT INT8" in out
    assert "selected deploy artifact" in out

    print_results_table([ref, cand], hide_extra=True)
    out = capsys.readouterr().out
    assert "Init Prec" not in out
    assert "Init Rec" not in out
    assert "Prec" not in out
    assert "Recall" not in out
    assert "Init IoU" not in out
    assert "IoU" not in out
    assert "mAP50" not in out
    assert "Lat CPU" not in out
    assert "Lat NCNN" not in out
    assert "Lat NPU" not in out
    assert "0.7012" not in out
    assert "0.6899" not in out
    assert "0.6123" not in out
    assert "0.7345" not in out
    assert "0.7634" not in out
    assert "0.7456" not in out
    assert "0.8123" not in out
