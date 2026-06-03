from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch
import torch.nn as nn

from xtrim.types import GumbelChoiceConfig, KDConfig, LatencyLUTConfig, ModelConfig, TrainConfig, TrimConfig

pytestmark = pytest.mark.unit


def _install_fake_ultralytics_cfg(monkeypatch):
    ultra = types.ModuleType("ultralytics")
    cfg = types.ModuleType("ultralytics.cfg")
    data = types.ModuleType("ultralytics.data")
    data_utils = types.ModuleType("ultralytics.data.utils")
    data_build = types.ModuleType("ultralytics.data.build")

    class IterableSimpleNamespace(types.SimpleNamespace):
        pass

    cfg.IterableSimpleNamespace = IterableSimpleNamespace
    cfg.get_cfg = lambda overrides=None: IterableSimpleNamespace(**(overrides or {}))
    data_utils.check_det_dataset = lambda _p: {"train": "train_images"}
    data_build.build_yolo_dataset = lambda *args, **kwargs: "dataset"
    data_build.build_dataloader = lambda dataset, batch, workers, shuffle, rank: [
        {"img": torch.rand(1, 3, 8, 8), "cls": torch.tensor([0])}
    ]

    for name, mod in {
        "ultralytics": ultra,
        "ultralytics.cfg": cfg,
        "ultralytics.data": data,
        "ultralytics.data.utils": data_utils,
        "ultralytics.data.build": data_build,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    sys.modules.pop("xtrim.yolo.kd_finetune", None)
    return importlib.import_module("xtrim.yolo.kd_finetune")


class Detect(nn.Module):
    def __init__(self, *, end2end: bool = False, intact: bool = True):
        super().__init__()
        self.end2end = end2end
        if intact:
            self.cv2 = nn.Identity()
            self.cv3 = nn.Identity()
    def forward(self, x):
        return x


class TinyCriterion:
    def __init__(self):
        self.hyp = {"box": 1.0}
        self.one2many = types.SimpleNamespace(hyp={"cls": 1.0})
        self.one2one = types.SimpleNamespace(hyp={"dfl": 1.0})


class TinyKDModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 3, 1)
        self.detect = Detect()
        self.stride = torch.tensor([8, 16, 32])
    def forward(self, x):
        return self.detect(self.conv(x))
    def init_criterion(self):
        return TinyCriterion()
    def loss(self, _batch, preds):
        return preds.mean(), {}


def test_kd_helper_functions_and_loader(monkeypatch):
    kd = _install_fake_ultralytics_cfg(monkeypatch)

    nested = {"a": torch.ones(1), "b": [torch.zeros(2)]}
    assert len(kd._as_tensor_list(nested)) == 2
    assert kd._attn_map(torch.randn(1, 3, 4, 4)).shape == (1, 1, 4, 4)
    assert kd._attn_map(torch.randn(1, 5)).shape[-1] == 5
    assert kd._mse_resize(torch.randn(1, 1, 4, 4), torch.randn(1, 1, 2, 2)).ndim == 0

    model = TinyKDModel()
    assert kd._select_feature_modules(model, 1)
    assert isinstance(kd._find_detect_module(model), Detect)
    kd._assert_end2end_training_head_is_intact(model)
    with pytest.raises(RuntimeError, match="fused"):
        kd._assert_end2end_training_head_is_intact(nn.Sequential(Detect(end2end=True, intact=False)))

    mgr = kd._HookMgr([model.conv], model.detect)
    mgr.register()
    _ = model(torch.randn(1, 3, 4, 4))
    assert mgr.store.feats and mgr.store.head
    mgr.clear()
    assert mgr.store.feats == [] and mgr.store.head == []
    mgr.remove()

    loader = kd._build_train_loader_ultralytics("d.yaml", imgsz=8, batch=1, workers=0, model_stride=32)
    assert loader[0]["img"].shape == (1, 3, 8, 8)

    obj = types.SimpleNamespace(hyp={"box": 1.0})
    kd._normalize_loss_hyp(obj)
    assert hasattr(obj.hyp, "box")

    crit = kd._init_model_criterion(model)
    assert hasattr(crit.hyp, "box") and model.criterion is crit

    dev = torch.device("cpu")
    assert kd._scalarize({"a": torch.tensor([1.0, 3.0])}, dev).item() == 2.0
    assert kd._scalarize([torch.tensor([2.0])], dev).item() == 2.0
    assert kd._scalarize(3.0, dev).item() == 3.0
    assert kd._to_regular_tensor(torch.ones(1)).shape == (1,)


def test_finetune_with_kd_runs_one_step_with_lut_and_fake_quant(monkeypatch, tmp_path):
    kd = _install_fake_ultralytics_cfg(monkeypatch)
    student = TinyKDModel()
    teacher = TinyKDModel()

    monkeypatch.setattr(kd, "patch_ultralytics_convs_for_fake_quant", lambda _m: 1)
    monkeypatch.setattr(kd, "set_fake_quant_bits", lambda *_a, **_kw: None)
    monkeypatch.setattr(kd, "set_fake_quant_enabled", lambda *_a, **_kw: None)
    monkeypatch.setattr(kd, "bn_sparsity_regularizer", lambda *_a, **_kw: torch.tensor(0.0))

    lut_path = tmp_path / "lut.json"
    lut_path.write_text('{"entries": []}', encoding="utf-8")

    logs = kd.finetune_with_kd(
        student_torch_model=student,
        teacher_torch_model=teacher,
        model_cfg=ModelConfig(imgsz=8),
        trim_cfg=TrimConfig(),
        train_cfg=TrainConfig(short_epochs=1, lr=1e-3),
        kd_cfg=KDConfig(enabled=True, batch=1, workers=0, max_train_batches=1, num_feature_layers=1),
        lut_cfg=LatencyLUTConfig(enabled=True, lut_path=str(lut_path), budget_ms=0.0, lambda_lat=0.1, log_every_n_batches=1),
        gumbel_cfg=GumbelChoiceConfig(enabled=True, tau_schedule="linear"),
        enable_fake_quant=True,
        override_epochs=1,
        override_max_batches=1,
    )
    assert logs["steps"] == 1.0
    assert logs["enable_fake_quant"] == 1.0
    assert logs["latency_penalty"] >= 0.0
