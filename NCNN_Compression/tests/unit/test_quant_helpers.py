from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from xtrim.quant.calib import _list_images_from_source
from xtrim.quant.fake_quant_ultra import (
    _fake_quant_per_channel_symmetric_w,
    _fake_quant_per_tensor_symmetric,
    patch_ultralytics_convs_for_fake_quant,
    set_fake_quant_bits,
    set_fake_quant_enabled,
)


pytestmark = pytest.mark.unit


def test_list_images_from_text_and_directory_and_glob(tmp_path, monkeypatch):
    img_dir = tmp_path / "images"
    img_dir.mkdir()
    a = img_dir / "a.jpg"
    b = img_dir / "b.png"
    ignored = img_dir / "note.txt"
    a.write_bytes(b"jpg")
    b.write_bytes(b"png")
    ignored.write_text("x", encoding="utf-8")

    txt = tmp_path / "list.txt"
    txt.write_text(f"{a}\n\n{b}\n", encoding="utf-8")

    assert _list_images_from_source(str(txt)) == [str(a), str(b)]
    assert _list_images_from_source(str(img_dir)) == [str(a), str(b)]

    monkeypatch.chdir(tmp_path)
    assert _list_images_from_source("images/*.jpg") == [str(Path("images/a.jpg"))]


def test_fake_quant_preserves_shape_and_bounds():
    x = torch.tensor([-2.0, -0.1, 0.0, 0.1, 2.0])
    y = _fake_quant_per_tensor_symmetric(x, bits=4)
    w = torch.randn(4, 3, 1, 1)
    wq = _fake_quant_per_channel_symmetric_w(w, bits=4)

    assert y.shape == x.shape
    assert wq.shape == w.shape
    assert torch.isfinite(y).all()
    assert torch.isfinite(wq).all()


class DummyUltraConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 1, bias=False)
        self.bn = nn.BatchNorm2d(4)
        self.act = nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


def test_patch_ultralytics_convs_can_toggle_fake_quant():
    model = nn.Sequential(DummyUltraConv())
    x = torch.randn(1, 3, 4, 4)

    patched = patch_ultralytics_convs_for_fake_quant(model)
    baseline = model(x)
    set_fake_quant_bits(model, bits_w=4, bits_a=4)
    set_fake_quant_enabled(model, True)
    quantized = model(x)

    assert len(patched) == 1
    assert getattr(model[0], "_xtrim_fq_enabled") is True
    assert quantized.shape == baseline.shape
    assert not torch.equal(quantized, baseline)
