from __future__ import annotations

import importlib
import sys
import types

import pytest
import torch
import torch.nn as nn


pytestmark = pytest.mark.unit


def _install_fake_ultralytics(monkeypatch):
    class Conv(nn.Module):
        def __init__(self, c1, c2, k=1, s=1, act=True):
            super().__init__()
            k = k[0] if isinstance(k, tuple) else k
            self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=k // 2, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act else nn.Identity()

        def forward(self, x):
            return self.act(self.bn(self.conv(x)))

    class DWConv(Conv):
        def __init__(self, c1, c2, k=1, s=1, act=True):
            nn.Module.__init__(self)
            k = k[0] if isinstance(k, tuple) else k
            self.conv = nn.Conv2d(c1, c2, k, stride=s, padding=k // 2, groups=c1, bias=False)
            self.bn = nn.BatchNorm2d(c2)
            self.act = nn.SiLU() if act else nn.Identity()

    class Bottleneck(nn.Module):
        def __init__(self, c1, c2, shortcut=False, g=1, k=((3, 3), (3, 3)), e=1.0):
            super().__init__()
            self.cv1 = Conv(c1, c2, 1)
            self.cv2 = Conv(c2, c2, 3)
            self.add = shortcut and c1 == c2

        def forward(self, x):
            y = self.cv2(self.cv1(x))
            return x + y if self.add else y

    class C2f(nn.Module):
        def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5):
            super().__init__()
            self.c = int(c2 * e)
            self.cv1 = Conv(c1, 2 * self.c, 1)
            self.cv2 = Conv((2 + n) * self.c, c2, 1)
            self.m = nn.ModuleList(Bottleneck(self.c, self.c, shortcut, g) for _ in range(n))

        def forward(self, x):
            y = list(self.cv1(x).split((self.c, self.c), dim=1))
            for block in self.m:
                y.append(block(y[-1]))
            return self.cv2(torch.cat(y, dim=1))

    class C3k2(C2f):
        pass

    class Attention(nn.Module):
        def __init__(self, c, attn_ratio=0.5, num_heads=1):
            super().__init__()
            self.num_heads = num_heads
            self.head_dim = c // num_heads
            self.key_dim = max(1, int(self.head_dim * attn_ratio))
            qkv_out = 2 * self.key_dim * num_heads + self.head_dim * num_heads
            self.qkv = Conv(c, qkv_out, 1, act=False)
            self.proj = Conv(c, c, 1, act=False)
            self.pe = DWConv(c, c, 3, act=False)

        def forward(self, x):
            return self.proj(x)

    class PSABlock(nn.Module):
        def __init__(self, c, attn_ratio=0.5, num_heads=1, shortcut=True):
            super().__init__()
            self.attn = Attention(c, attn_ratio=attn_ratio, num_heads=num_heads)
            self.ffn = nn.Sequential(Conv(c, 2 * c, 1), Conv(2 * c, c, 1, act=False))
            self.add = shortcut

        def forward(self, x):
            return x + self.ffn(x) if self.add else self.ffn(x)

    class PSA(nn.Module):
        def __init__(self, c1=16, c2=16, e=0.5):
            super().__init__()
            self.c = int(c1 * e)
            self.cv1 = Conv(c1, 2 * self.c, 1)
            self.attn = Attention(self.c)
            self.ffn = nn.Sequential(Conv(self.c, 2 * self.c, 1), Conv(2 * self.c, self.c, 1, act=False))
            self.cv2 = Conv(2 * self.c, c1, 1)

    class C2PSA(nn.Module):
        def __init__(self, c1=16, c2=16, n=1, e=0.5):
            super().__init__()
            self.c = int(c1 * e)
            self.cv1 = Conv(c1, 2 * self.c, 1)
            self.m = nn.Sequential(*(PSABlock(self.c) for _ in range(n)))
            self.cv2 = Conv(2 * self.c, c1, 1)

    class C2fPSA(nn.Module):
        def __init__(self, c1=16, c2=16, n=1, e=0.5):
            super().__init__()
            self.c = int(c2 * e)
            self.cv1 = Conv(c1, 2 * self.c, 1)
            self.m = nn.ModuleList(PSABlock(self.c) for _ in range(n))
            self.cv2 = Conv((2 + n) * self.c, c2, 1)

    class Detect(nn.Module):
        def __init__(self, in_ch=16, hidden=16, nc=3, reg_max=4, nl=1, layout="legacy"):
            super().__init__()
            self.nl = nl
            self.nc = nc
            self.reg_max = reg_max
            self.legacy = layout == "legacy"
            self.cv2 = nn.ModuleList(
                [nn.Sequential(Conv(in_ch, hidden, 3), Conv(hidden, hidden, 3), nn.Conv2d(hidden, 4 * reg_max, 1)) for _ in range(nl)]
            )
            if layout == "legacy":
                self.cv3 = nn.ModuleList(
                    [nn.Sequential(Conv(in_ch, hidden, 3), Conv(hidden, hidden, 3), nn.Conv2d(hidden, nc, 1)) for _ in range(nl)]
                )
            else:
                self.cv3 = nn.ModuleList(
                    [
                        nn.Sequential(
                            nn.Sequential(DWConv(in_ch, in_ch, 3), Conv(in_ch, hidden, 1)),
                            nn.Sequential(DWConv(hidden, hidden, 3), Conv(hidden, hidden, 1)),
                            nn.Conv2d(hidden, nc, 1),
                        )
                        for _ in range(nl)
                    ]
                )

    ultra = types.ModuleType("ultralytics")
    ultra_nn = types.ModuleType("ultralytics.nn")
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
        "ultralytics.nn": ultra_nn,
        "ultralytics.nn.modules": modules,
        "ultralytics.nn.modules.block": block,
        "ultralytics.nn.modules.conv": conv,
        "ultralytics.nn.modules.head": head,
    }.items():
        monkeypatch.setitem(sys.modules, name, mod)

    for name in ["xtrim.yolo.pruning_adapters", "xtrim.yolo.detect_head_pruning"]:
        sys.modules.pop(name, None)

    pa = importlib.import_module("xtrim.yolo.pruning_adapters")
    dh = importlib.import_module("xtrim.yolo.detect_head_pruning")
    return types.SimpleNamespace(
        Conv=Conv,
        DWConv=DWConv,
        Bottleneck=Bottleneck,
        C2f=C2f,
        C3k2=C3k2,
        PSA=PSA,
        C2PSA=C2PSA,
        C2fPSA=C2fPSA,
        Detect=Detect,
        pa=pa,
        dh=dh,
    )


def test_c2f_conversion_and_replacement_preserve_shape(monkeypatch):
    fx = _install_fake_ultralytics(monkeypatch)
    block = fx.C2f(16, 16, n=2)
    x = torch.randn(1, 16, 8, 8)

    converted = fx.pa.convert_c2f_to_prunable(block)
    root = nn.Sequential(fx.C2f(16, 16, n=1))
    replaced = fx.pa.replace_c2f_with_prunable(root, verbose=False)

    assert converted.src_block_type == "C2f"
    assert converted(x).shape == block(x).shape
    assert replaced == 1
    assert isinstance(root[0], fx.pa.C2fPrunable)


def test_copy_conv_bn_supports_standard_and_depthwise_and_rejects_grouped(monkeypatch):
    fx = _install_fake_ultralytics(monkeypatch)
    src = fx.Conv(8, 8, 1)
    dst = fx.Conv(8, 4, 1)
    fx.pa._copy_conv_bn_(dst, src, out_idx=[0, 1, 2, 3])
    assert dst.conv.weight.shape[0] == 4

    src_dw = fx.DWConv(4, 4, 3)
    dst_dw = fx.DWConv(2, 2, 3)
    fx.pa._copy_conv_bn_(dst_dw, src_dw, out_idx=[0, 1], in_idx=[0, 1])
    assert dst_dw.conv.weight.shape == (2, 1, 3, 3)

    class Grouped(nn.Module):
        def __init__(self, cin, cout):
            super().__init__()
            self.conv = nn.Conv2d(cin, cout, 1, groups=2, bias=False)
            self.bn = nn.BatchNorm2d(cout)
            self.act = nn.Identity()

    with pytest.raises(NotImplementedError):
        fx.pa._copy_conv_bn_(Grouped(4, 4), Grouped(4, 4), in_idx=[0, 1])


def test_pruning_adapter_helpers_and_psa_shrink(monkeypatch):
    fx = _install_fake_ultralytics(monkeypatch)

    assert fx.pa._safe_num_heads(16, preferred_heads=8, min_head_dim=4) == 4
    assert fx.pa._topk_indices(torch.tensor([0.1, 3.0, 2.0]), 2) == [1, 2]
    assert fx.pa._target_keep(16, prune_ratio=0.5, round_to=4, min_channels=4) == 8

    model = nn.Sequential(fx.PSA(c1=16, c2=16, e=0.5), fx.C2PSA(c1=16, c2=16, n=1, e=0.5))
    stats = fx.pa.shrink_psa_family_blocks(model, prune_ratio=0.5, round_to=4, min_channels=4, verbose=False)

    assert stats["blocks_shrunk"] == 2
    assert model[0].c == 4
    assert model[1].c == 4


def test_detect_head_shrinking_supports_legacy_and_dw_layouts(monkeypatch):
    fx = _install_fake_ultralytics(monkeypatch)

    legacy = fx.Detect(hidden=16, layout="legacy")
    dw = fx.Detect(hidden=16, layout="dw")
    assert fx.dh._cls_branch_layout(legacy.cv3[0]) == "legacy"
    assert fx.dh._cls_branch_layout(dw.cv3[0]) == "dw"

    root = nn.ModuleDict({"head": legacy})
    stats = fx.dh.shrink_detect_heads(root, prune_ratio=0.5, round_to=4, min_channels=4, verbose=False)

    assert stats["heads_shrunk"] == 1
    assert root["head"].cv2[0][0].conv.out_channels == 8
    assert root["head"].cv3[0][0].conv.out_channels == 8
