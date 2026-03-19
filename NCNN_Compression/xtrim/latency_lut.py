from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class LatencyLUT:
    def __init__(self, lut_path: str | Path, verbose: bool = True):
        lut_path = Path(lut_path)
        if not lut_path.exists():
            raise FileNotFoundError(f"LUT file not found: {lut_path}")

        with open(lut_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.device_name: str = data.get("device", "unknown")
        self.unit: str = data.get("unit", "ms")
        self.entries: List[Dict[str, Any]] = data.get("entries", [])

        self._index: Dict[Tuple, float] = {}
        for e in self.entries:
            key = self._entry_key(e)
            self._index[key] = float(e["latency_ms"])

        if verbose:
            print(
                f"[LUT] Loaded {len(self._index)} entries "
                f"from '{lut_path.name}' (device={self.device_name})"
            )

    @staticmethod
    def _entry_key(e: Dict[str, Any]) -> Tuple:
        return (
            str(e.get("op", "conv2d")),
            int(e.get("cin", 0)),
            int(e.get("cout", 0)),
            int(e.get("k", 1)),
            int(e.get("stride", 1)),
            int(e.get("h", 0)),
            int(e.get("w", 0)),
            int(e.get("groups", 1)),
        )

    @staticmethod
    def _bucket(x: int, buckets: List[int]) -> int:
        return min(buckets, key=lambda b: abs(b - x))

    def lookup(
        self,
        op: str,
        cin: int,
        cout: int,
        k: int,
        stride: int,
        h: int,
        w: int,
        groups: int = 1,
    ) -> Optional[float]:
        exact_key = (op, cin, cout, k, stride, h, w, groups)
        if exact_key in self._index:
            return self._index[exact_key]

        candidates = [
            (key, v) for key, v in self._index.items()
            if key[0] == op and key[3] == k and key[4] == stride and key[7] == groups
        ]
        if not candidates:
            return None

        def dist(key: Tuple) -> float:
            dc_in = (math.log2(max(key[1], 1)) - math.log2(max(cin, 1))) ** 2
            dc_out = (math.log2(max(key[2], 1)) - math.log2(max(cout, 1))) ** 2
            dh = (math.log2(max(key[5], 1)) - math.log2(max(h, 1))) ** 2
            dw = (math.log2(max(key[6], 1)) - math.log2(max(w, 1))) ** 2
            return dc_in + dc_out + dh + dw

        best_key, best_val = min(candidates, key=lambda kv: dist(kv[0]))

        key_macs = best_key[1] * best_key[2] * best_key[5] * best_key[6]
        query_macs = cin * cout * h * w
        if key_macs > 0:
            scale = query_macs / key_macs
            return best_val * scale
        return best_val

    def lookup_with_fallback(
        self,
        op: str,
        cin: int,
        cout: int,
        k: int,
        stride: int,
        h: int,
        w: int,
        groups: int = 1,
        macs_per_ms: float = 100_000_000.0,
    ) -> float:
        val = self.lookup(op, cin, cout, k, stride, h, w, groups)
        if val is not None:
            return val

        macs = 2 * cin * cout * k * k * h * w / (stride * stride)
        if groups > 1:
            macs = 2 * (cin // groups) * cout * k * k * h * w / (stride * stride)
        return macs / macs_per_ms


def _get_op_type(m: nn.Conv2d) -> str:
    k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
    if m.groups > 1:
        return "conv2d_dw"
    if k == 1:
        return "conv1x1"
    if k == 3:
        return "conv3x3"
    return f"conv{k}x{k}"


def estimate_model_latency(
    model: nn.Module,
    lut: LatencyLUT,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    macs_per_ms: float = 100_000_000.0,
    verbose: bool = False,
) -> Dict[str, Any]:
    spatial_sizes: Dict[str, Tuple[int, int]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_m, _inp, out: torch.Tensor):
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                spatial_sizes[name] = (out.shape[2], out.shape[3])
        return hook

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(make_hook(name)))

    model.eval()
    try:
        with torch.no_grad():
            dummy = torch.zeros(*input_shape, device=next(model.parameters()).device)
            model(dummy)
    except Exception:
        pass
    finally:
        for h in handles:
            h.remove()

    total_ms = 0.0
    lut_hits = 0
    lut_misses = 0
    per_layer = []

    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue

        cin = m.in_channels
        cout = m.out_channels
        k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        stride = m.stride[0] if isinstance(m.stride, tuple) else m.stride
        groups = m.groups
        op = _get_op_type(m)

        hw = spatial_sizes.get(name)
        if hw is None:
            h = input_shape[2] // stride
            w = input_shape[3] // stride
        else:
            h, w = hw

        exact = lut.lookup(op, cin, cout, k, stride, h, w, groups)
        layer_ms = lut.lookup_with_fallback(
            op, cin, cout, k, stride, h, w, groups, macs_per_ms=macs_per_ms
        )

        if exact is not None:
            lut_hits += 1
        else:
            lut_misses += 1

        total_ms += layer_ms

        if verbose:
            src = "LUT" if exact is not None else "est"
            print(
                f"[LUT] {name}: {op} {cin}->{cout} k={k} s={stride} "
                f"{h}x{w} = {layer_ms:.3f} ms ({src})"
            )

        per_layer.append({
            "name": name,
            "op": op,
            "cin": cin,
            "cout": cout,
            "k": k,
            "stride": stride,
            "h": h,
            "w": w,
            "latency_ms": layer_ms,
            "from_lut": exact is not None,
        })

    if verbose:
        print(
            f"[LUT] Total: {total_ms:.2f} ms "
            f"(hits={lut_hits}, fallback={lut_misses})"
        )

    return {
        "latency_est_ms": total_ms,
        "layers_estimated": len(per_layer),
        "lut_hits": lut_hits,
        "lut_misses": lut_misses,
        "per_layer": per_layer,
    }


def latency_penalty(
    latency_est_ms: float,
    budget_ms: float,
    lambda_lat: float,
) -> float:
    excess = max(0.0, latency_est_ms - budget_ms)
    return float(lambda_lat) * excess


def build_lut_from_model(
    model: nn.Module,
    input_shape: Tuple[int, int, int, int] = (1, 3, 640, 640),
    device_name: str = "local_cpu",
    warmup: int = 3,
    repeats: int = 20,
    verbose: bool = True,
) -> Dict[str, Any]:
    import time

    spatial_sizes: Dict[str, Tuple[int, int]] = {}
    handles = []

    def make_hook(name: str):
        def hook(_m, _inp, out: torch.Tensor):
            if isinstance(out, torch.Tensor) and out.dim() == 4:
                spatial_sizes[name] = (out.shape[2], out.shape[3])
        return hook

    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            handles.append(m.register_forward_hook(make_hook(name)))

    model.eval()
    dummy = torch.zeros(*input_shape, device=next(model.parameters()).device)
    with torch.no_grad():
        model(dummy)
    for h in handles:
        h.remove()

    entries = []
    for name, m in model.named_modules():
        if not isinstance(m, nn.Conv2d):
            continue

        cin = m.in_channels
        cout = m.out_channels
        k = m.kernel_size[0] if isinstance(m.kernel_size, tuple) else m.kernel_size
        stride = m.stride[0] if isinstance(m.stride, tuple) else m.stride
        groups = m.groups
        op = _get_op_type(m)
        hw = spatial_sizes.get(name, (input_shape[2], input_shape[3]))
        h, w = hw

        dev = next(m.parameters()).device
        x_in_ch = cin if groups == 1 else cin
        x_dummy = torch.randn(1, x_in_ch, h, w, device=dev)

        for _ in range(warmup):
            with torch.no_grad():
                m(x_dummy)

        if dev.type == "cuda":
            torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(repeats):
            with torch.no_grad():
                m(x_dummy)
        if dev.type == "cuda":
            torch.cuda.synchronize()
        elapsed = (time.perf_counter() - t0) / repeats * 1000.0

        entries.append({
            "op": op,
            "cin": cin,
            "cout": cout,
            "k": k,
            "stride": stride,
            "h": h,
            "w": w,
            "groups": groups,
            "latency_ms": round(elapsed, 4),
        })

        if verbose:
            print(f"[profile] {name}: {op} {cin}->{cout} {h}x{w} → {elapsed:.3f} ms")

    return {
        "device": device_name,
        "unit": "ms",
        "input_shape": list(input_shape),
        "entries": entries,
    }
