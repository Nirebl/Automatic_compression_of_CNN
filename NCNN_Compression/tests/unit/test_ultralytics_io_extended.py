from __future__ import annotations

import importlib
import json
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from xtrim.types import (
    CandidateConfig,
    DilatedConfig,
    EvalConfig,
    ExportConfig,
    GumbelChoiceConfig,
    KDConfig,
    LowRankConfig,
    ModelConfig,
    OperatorChoiceConfig,
    QATConfig,
    TrainConfig,
    TrimConfig,
)

pytestmark = pytest.mark.unit


def _install_fake_onnx(monkeypatch):
    class Entry:
        def __init__(self):
            self.key = ""
            self.value = ""

    class FakeModel:
        def __init__(self):
            self.metadata_props = []

    store = {"model": FakeModel(), "saved": None}
    onnx = types.ModuleType("onnx")
    onnx.StringStringEntryProto = Entry
    onnx.load = lambda _p: store["model"]
    onnx.save = lambda model, _p: store.__setitem__("saved", model)
    onnx.helper = types.SimpleNamespace(make_string_string_entry=lambda k, v: types.SimpleNamespace(key=k, value=v))
    monkeypatch.setitem(sys.modules, "onnx", onnx)
    monkeypatch.setitem(sys.modules, "onnxsim", types.SimpleNamespace(simplify=lambda model: (model, True)))
    return store


def _install_fake_ultralytics_runtime(monkeypatch):
    class Conv(nn.Module):
        def __init__(self, cin=3, cout=4, k=1):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, bias=False)
            self.bn = nn.BatchNorm2d(cout)
            self.act = nn.Identity()
        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class DWConv(Conv):
        def __init__(self, cin=4, cout=4, k=3):
            nn.Module.__init__(self)
            self.conv = nn.Conv2d(cin, cout, k, padding=k // 2, groups=cin, bias=False)
            self.bn = nn.BatchNorm2d(cout)
            self.act = nn.Identity()

    class C2f(nn.Module):
        def __init__(self):
            super().__init__()
            self.cv = Conv(3, 4, 1)
        def forward(self, x):
            return self.cv(x)

    class Bottleneck(C2f): pass
    class C3k2(C2f): pass
    class Attention(C2f): pass
    class PSA(C2f): pass
    class C2PSA(C2f): pass
    class C2fPSA(C2f): pass

    class Detect(nn.Module):
        def __init__(self):
            super().__init__()
            self.end2end = False
            self.cv2 = nn.Identity()
            self.cv3 = nn.Identity()
        def forward(self, x): return x

    class TinyUltraModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.model = nn.ModuleList([Conv(3, 4, 1), Conv(4, 4, 3), Detect()])
            self.names = ["a", "b"]
            self.stride = torch.tensor([8, 16, 32])
        def forward(self, x):
            for m in self.model:
                x = m(x)
            return x

    class YOLO:
        def __init__(self, weights):
            self.weights = weights
            self.model = TinyUltraModel()
            self._model = self.model
            self.names = ["cat", "dog"]
            self.predictor = object()
        def val(self, **_kwargs):
            return {
                "metrics/mAP50-95(B)": 0.77,
                "metrics/precision(B)": 0.66,
                "metrics/recall(B)": 0.55,
                "metrics/IoU(B)": 0.77,
                "metrics/mAP50(B)": 0.88,
            }

    ultra = types.ModuleType("ultralytics")
    ultra.YOLO = YOLO
    cfg = types.ModuleType("ultralytics.cfg")
    cfg.IterableSimpleNamespace = types.SimpleNamespace
    cfg.get_cfg = lambda overrides=None: types.SimpleNamespace(**(overrides or {}))
    data = types.ModuleType("ultralytics.data")
    data_utils = types.ModuleType("ultralytics.data.utils")
    data_utils.check_det_dataset = lambda _p: {"train": "train", "val": "val"}
    data_build = types.ModuleType("ultralytics.data.build")
    data_build.build_yolo_dataset = lambda *a, **k: "dataset"
    data_build.build_dataloader = lambda *a, **k: [torch.rand(1, 3, 8, 8)]
    nn_mod = types.ModuleType("ultralytics.nn")
    modules = types.ModuleType("ultralytics.nn.modules")
    block = types.ModuleType("ultralytics.nn.modules.block")
    conv = types.ModuleType("ultralytics.nn.modules.conv")
    head = types.ModuleType("ultralytics.nn.modules.head")
    for cls in (C2f, Bottleneck, C3k2, Attention, PSA, C2PSA, C2fPSA):
        setattr(block, cls.__name__, cls)
        setattr(modules, cls.__name__, cls)
    for cls in (Conv, DWConv):
        setattr(conv, cls.__name__, cls)
        setattr(modules, cls.__name__, cls)
    setattr(head, "Detect", Detect)
    setattr(modules, "Detect", Detect)
    for name, mod in {
        "ultralytics": ultra,
        "ultralytics.cfg": cfg,
        "ultralytics.data": data,
        "ultralytics.data.utils": data_utils,
        "ultralytics.data.build": data_build,
        "ultralytics.nn": nn_mod,
        "ultralytics.nn.modules": modules,
        "ultralytics.nn.modules.block": block,
        "ultralytics.nn.modules.conv": conv,
        "ultralytics.nn.modules.head": head,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    for name in [
        "xtrim.yolo.ultralytics_io",
        "xtrim.yolo.kd_finetune",
        "xtrim.yolo.pruning_adapters",
        "xtrim.yolo.detect_head_pruning",
    ]:
        sys.modules.pop(name, None)
    return YOLO, TinyUltraModel


def _import_uio(monkeypatch):
    store = _install_fake_onnx(monkeypatch)
    YOLO, TinyUltraModel = _install_fake_ultralytics_runtime(monkeypatch)
    mod = importlib.import_module("xtrim.yolo.ultralytics_io")
    return mod, YOLO, TinyUltraModel, store


def test_ultralytics_io_build_candidate_runs_all_major_branches(monkeypatch):
    uio, _YOLO, _Model, _store = _import_uio(monkeypatch)
    monkeypatch.setattr(uio, "replace_c2f_with_prunable", lambda *_a, **_kw: 1)
    monkeypatch.setattr(uio, "structured_trim_yolo", lambda *_a, **_kw: {"layers_pruned": 1, "channels_pruned": 2})
    monkeypatch.setattr(uio, "shrink_psa_family_blocks", lambda *_a, **_kw: {"blocks_shrunk": 1})
    monkeypatch.setattr(uio, "shrink_detect_heads", lambda *_a, **_kw: {"heads_shrunk": 1})
    monkeypatch.setattr(uio, "apply_dilation", lambda *_a, **_kw: {"layers_modified": 1})
    monkeypatch.setattr(uio, "apply_lowrank_decomposition", lambda *_a, **_kw: {"layers_decomposed": 1, "compression_ratio": 2.0})
    monkeypatch.setattr(uio, "_build_recalib_loader", lambda *_a, **_kw: [torch.rand(1, 3, 8, 8)])
    monkeypatch.setattr(uio, "recalibrate_bn", lambda *_a, **_kw: {"batches_processed": 1})
    monkeypatch.setattr(uio, "apply_1x1_weight_sparsity", lambda *_a, **_kw: {"layers_sparsified": 1})
    monkeypatch.setattr(uio, "remove_pruning_reparam", lambda *_a, **_kw: 1)
    monkeypatch.setattr(uio, "plan_from_config", lambda *_a, **_kw: {"x": "dense"})
    monkeypatch.setattr(uio, "apply_operator_plan", lambda *_a, **_kw: {"dense": 1, "sparse": 0, "lowrank": 0})

    student = uio.build_ultralytics_candidate(
        CandidateConfig(width_mult=0.75, prune_ratio=0.2, lowrank_rank=2, sparse_1x1=0.5, tag="c"),
        ModelConfig(imgsz=8),
        TrimConfig(exclude_head=False, adapt_c2f_for_pruning=True),
        op_choice_cfg=OperatorChoiceConfig(enabled=True),
        op_choice_plan={"auto": "dense"},
        lowrank_cfg=LowRankConfig(enabled=True, bn_recalib_batches=1, lowrank_1x1=True),
        dilated_cfg=DilatedConfig(enabled=True),
    )
    stats = student.yolo._xtrim_all_stats
    assert stats["width_scale"]["layers_pruned"] == 1
    assert stats["bn_prune"]["channels_pruned"] == 2
    assert stats["lowrank"]["layers_decomposed"] == 1
    assert stats["operator_choice"]["dense"] == 1


def test_ultralytics_io_build_candidate_gumbel_conflict_and_warning_paths(monkeypatch):
    uio, _YOLO, _Model, _store = _import_uio(monkeypatch)
    monkeypatch.setattr(uio, "structured_trim_yolo", lambda *_a, **_kw: {"layers_pruned": 0, "channels_pruned": 0})
    monkeypatch.setattr(uio, "shrink_psa_family_blocks", lambda *_a, **_kw: {"blocks_shrunk": 0})
    monkeypatch.setattr(uio, "insert_mixed_ops", lambda *_a, **_kw: {"layers_replaced": 1})

    s = uio.build_ultralytics_candidate(
        CandidateConfig(tag="g"),
        ModelConfig(imgsz=8),
        TrimConfig(),
        gumbel_cfg=GumbelChoiceConfig(enabled=True),
    )
    assert s.yolo._xtrim_all_stats["gumbel_choice"]["layers_replaced"] == 1

    with pytest.raises(ValueError, match="cannot both"):
        uio.build_ultralytics_candidate(
            CandidateConfig(tag="bad"),
            ModelConfig(imgsz=8),
            TrimConfig(),
            op_choice_cfg=OperatorChoiceConfig(enabled=True),
            gumbel_cfg=GumbelChoiceConfig(enabled=True),
        )

    with pytest.raises(TypeError, match="exclude_name_regex"):
        uio.build_ultralytics_candidate(CandidateConfig(tag="bad2"), ModelConfig(imgsz=8), TrimConfig(exclude_name_regex=1))


def test_ultralytics_io_finetune_eval_export_and_metadata(monkeypatch, tmp_path):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    uio, YOLO, TinyUltraModel, store = _import_uio(monkeypatch)
    y = YOLO("w")
    student = uio.UltralyticsStudent(yolo=y, torch_model=y.model)

    assert uio._torch_device_from_ultralytics_device("cpu").type == "cpu"
    assert uio._torch_device_from_ultralytics_device("0").type == "cpu"
    assert uio.finetune_noop(student, TrainConfig()) is None
    assert uio.finetune_kd(student, TrainConfig(), ModelConfig(imgsz=8), TrimConfig(), KDConfig(enabled=False)) == {}
    assert uio.finetune_qat_recover(student, TrainConfig(), ModelConfig(imgsz=8), TrimConfig(), KDConfig(), QATConfig(enabled=False)) == {}

    monkeypatch.setattr(uio, "finetune_with_kd", lambda **kwargs: {"ok": True})
    monkeypatch.setattr(uio, "count_mixed_ops", lambda _m: 1)
    monkeypatch.setattr(uio, "freeze_mixed_ops", lambda *_a, **_kw: {"frozen": 1})
    logs = uio.finetune_kd(student, TrainConfig(), ModelConfig(imgsz=8), TrimConfig(), KDConfig(enabled=True))
    assert logs["gumbel_freeze"] == {"frozen": 1}
    assert uio.finetune_qat_recover(student, TrainConfig(), ModelConfig(imgsz=8), TrimConfig(), KDConfig(), QATConfig(enabled=True))["ok"] is True

    assert uio._val_kwargs(ModelConfig(imgsz=8), EvalConfig())["imgsz"] == 8
    assert uio._extract_map5095({"metrics/mAP50-95(B)": "0.5"}) == 0.5
    assert uio._extract_map5095(types.SimpleNamespace(results_dict={"metrics/mAP50-95": 0.6})) == 0.6
    assert uio._extract_map5095(types.SimpleNamespace(box=types.SimpleNamespace(map=0.7))) == 0.7
    assert uio._extract_map5095(types.SimpleNamespace(maps=[0.2, 0.4])) == pytest.approx(0.3)
    assert uio._extract_detection_metrics({
        "metrics/mAP50-95(B)": 0.5,
        "metrics/precision(B)": 0.4,
        "metrics/recall(B)": 0.3,
        "metrics/mAP50(B)": 0.6,
        "metrics/IoU(B)": 0.45,
    }) == {"map50_95": 0.5, "precision": 0.4, "recall": 0.3, "iou": 0.45, "map50": 0.6}
    assert uio._extract_detection_metrics(types.SimpleNamespace(box=types.SimpleNamespace(map=0.7, mp=0.6, mr=0.5, iou=0.65, map50=0.8))) == {"map50_95": 0.7, "precision": 0.6, "recall": 0.5, "iou": 0.65, "map50": 0.8}
    assert uio.eval_ultralytics_map(student, ModelConfig(imgsz=8), EvalConfig()) == 0.77
    assert uio.eval_exported_onnx_map(tmp_path / "m.onnx", ModelConfig(imgsz=8), EvalConfig()) == 0.77
    assert uio.eval_ultralytics_metrics(student, ModelConfig(imgsz=8), EvalConfig()) == {"map50_95": 0.77, "precision": 0.66, "recall": 0.55, "iou": 0.77, "map50": 0.88}
    assert uio.eval_exported_onnx_metrics(tmp_path / "m.onnx", ModelConfig(imgsz=8), EvalConfig()) == {"map50_95": 0.77, "precision": 0.66, "recall": 0.55, "iou": 0.77, "map50": 0.88}

    export_calls = {"n": 0}
    def fake_export(_m, _args, path, **kwargs):
        export_calls["n"] += 1
        if export_calls["n"] == 1:
            raise TypeError("old api")
        Path(path).write_bytes(b"onnx")
    monkeypatch.setattr(torch.onnx, "export", fake_export)
    export = uio.make_ultralytics_export_onnx_fn(student, ModelConfig(imgsz=8), ExportConfig())
    out = tmp_path / "model.onnx"
    export(out)
    assert out.exists()

    uio._inject_ultralytics_metadata(out, student, ModelConfig(imgsz=8))
    meta = {p.key: p.value for p in store["model"].metadata_props}
    assert meta["task"] == "detect"
    assert json.loads(meta["names"])["0"] == "cat"

    scripted = tmp_path / "student.pt"
    uio.save_student_torchscript(uio.UltralyticsStudent(yolo=y, torch_model=TinyUltraModel()), scripted)
    assert scripted.exists()


def test_ultralytics_pruning_stage_updates_stats_and_detect_branches(monkeypatch):
    uio, YOLO, TinyUltraModel, _store = _import_uio(monkeypatch)
    y = YOLO("w")
    student = uio.UltralyticsStudent(yolo=y, torch_model=y.model)
    y._xtrim_all_stats = {"params_initial": {"total_params": 1000}}
    calls = []

    monkeypatch.setattr(uio, "structured_trim_yolo", lambda *_a, **kw: {"layers_pruned": 1, "targeted": bool(kw.get("target_out_channels"))})
    monkeypatch.setattr(uio, "shrink_psa_family_blocks", lambda *_a, **_kw: {"blocks_shrunk": 1})
    monkeypatch.setattr(uio, "collect_detect_head_hidden_channels", lambda _m: {"head": 8})
    monkeypatch.setattr(uio, "shrink_detect_heads_to_targets", lambda *_a, **kw: calls.append(("targets", kw["target_hidden_channels"])) or {"heads_shrunk": 2})
    monkeypatch.setattr(uio, "shrink_detect_heads", lambda *_a, **_kw: calls.append(("ratio", None)) or {"heads_shrunk": 3})

    stats = uio.apply_ultralytics_pruning_stage(
        student,
        stage_prune_ratio=0.25,
        model_cfg=ModelConfig(imgsz=8),
        trim_cfg=TrimConfig(exclude_head=True, channel_round=4, min_channels=4),
        stage_label="s1",
    )
    assert stats["layers_pruned"] == 1
    assert stats["compression_ratio_total"] > 0
    assert y._xtrim_all_stats["s1_detect_head"]["skipped"] == "trim.exclude_head=true"
    assert y.model is student.torch_model

    stats = uio.apply_ultralytics_pruning_stage(
        student,
        stage_prune_ratio=0.2,
        model_cfg=ModelConfig(imgsz=8),
        trim_cfg=TrimConfig(exclude_head=False, channel_round=4, min_channels=4),
        stage_label="s2",
        target_architecture={"conv_out_channels": {"model.0": 4}, "detect_hidden_channels": {"head": 8}},
    )
    assert stats["target_architecture_applied"] is True
    assert calls[-1] == ("targets", {"head": 8})

    uio.apply_ultralytics_pruning_stage(
        student,
        stage_prune_ratio=0.2,
        model_cfg=ModelConfig(imgsz=8),
        trim_cfg=TrimConfig(exclude_head=False, channel_round=4, min_channels=4),
        stage_label="s3",
    )
    assert calls[-1] == ("ratio", None)

    with pytest.raises(ValueError, match="stage_prune_ratio"):
        uio.apply_ultralytics_pruning_stage(student, stage_prune_ratio=1.0, model_cfg=ModelConfig(imgsz=8), trim_cfg=TrimConfig(), stage_label="bad")
    with pytest.raises(TypeError, match="include_inner_m_regex"):
        uio.apply_ultralytics_pruning_stage(
            student,
            stage_prune_ratio=0.2,
            model_cfg=ModelConfig(imgsz=8),
            trim_cfg=TrimConfig(include_inner_m_regex=123),
            stage_label="bad2",
        )


def test_ultralytics_tflite_int8_export_selects_int8_and_optional_fp16(monkeypatch, tmp_path):
    uio, YOLO, TinyUltraModel, _store = _import_uio(monkeypatch)
    save_calls = []
    export_calls = []

    class SavingYolo(YOLO):
        def save(self, path):
            save_calls.append(path)
            Path(path).write_bytes(b"checkpoint")

    class ExportYolo:
        def __init__(self, weights):
            self.weights = weights

        def export(self, **kwargs):
            export_calls.append(kwargs)
            Path("nested").mkdir(exist_ok=True)
            Path("nested" / Path("model_float16.tflite")).write_bytes(b"fp16")
            Path("model_full_integer_quant.tflite").write_bytes(b"bad_full_integer_quant")
            Path("model_int8.tflite").write_bytes(b"good_int8")
            Path("model_integer_quant.tflite").write_bytes(b"bad_integer_quant")
            Path("model_float32.tflite").write_bytes(b"fp32")
            return "model_float32.tflite"

    sys.modules["ultralytics"].YOLO = ExportYolo
    y = SavingYolo("w")
    student = uio.UltralyticsStudent(yolo=y, torch_model=y.model)
    export = uio.make_ultralytics_export_tflite_int8_fn(
        student,
        ModelConfig(imgsz=8, data="data.yaml"),
        ExportConfig(tflite_fp16=True, tflite_fp16_name="gpu_fp16.tflite", tflite_int8_device="cpu", tflite_int8_simplify=True),
    )

    out = tmp_path / "model_int8.tflite"
    assert export(out) == out
    assert out.read_bytes() == b"good_int8"
    assert (tmp_path / "gpu_fp16.tflite").read_bytes() == b"fp16"
    assert save_calls and save_calls[0].endswith("xtrim_tflite_export.pt")
    assert export_calls[0]["format"] == "tflite"
    assert export_calls[0]["int8"] is True
    assert export_calls[0]["device"] == "cpu"
    assert export_calls[0]["simplify"] is True


def test_ultralytics_tflite_int8_export_error_paths(monkeypatch, tmp_path):
    uio, YOLO, _TinyUltraModel, _store = _import_uio(monkeypatch)

    class SavingYolo(YOLO):
        def save(self, path):
            Path(path).write_bytes(b"checkpoint")

    class NoFileYolo:
        def __init__(self, weights):
            self.weights = weights

        def export(self, **_kwargs):
            return None

    sys.modules["ultralytics"].YOLO = NoFileYolo
    student = uio.UltralyticsStudent(yolo=SavingYolo("w"), torch_model=SavingYolo("w").model)
    export = uio.make_ultralytics_export_tflite_int8_fn(student, ModelConfig(imgsz=8), ExportConfig())
    with pytest.raises(RuntimeError, match="no .tflite file"):
        export(tmp_path / "missing.tflite")

    class SaveFails(YOLO):
        def save(self, path):
            raise OSError("disk")

    export = uio.make_ultralytics_export_tflite_int8_fn(
        uio.UltralyticsStudent(yolo=SaveFails("w"), torch_model=SaveFails("w").model),
        ModelConfig(imgsz=8),
        ExportConfig(),
    )
    with pytest.raises(RuntimeError, match="Could not save temporary"):
        export(tmp_path / "save_fail.tflite")

    class SavesNothing(YOLO):
        def save(self, path):
            pass

    export = uio.make_ultralytics_export_tflite_int8_fn(
        uio.UltralyticsStudent(yolo=SavesNothing("w"), torch_model=SavesNothing("w").model),
        ModelConfig(imgsz=8),
        ExportConfig(),
    )
    with pytest.raises(RuntimeError, match="Temporary YOLO checkpoint was not created"):
        export(tmp_path / "not_created.tflite")
