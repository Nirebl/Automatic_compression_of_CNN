from __future__ import annotations

import pytest

from xtrim.types import DeviceConfig, LatencyConfig


pytestmark = pytest.mark.integration


def test_benchncnn_path_uses_cache_after_first_measurement(tmp_path, make_orchestrator, monkeypatch):
    device = DeviceConfig(name="phone", serial="123")
    orch = make_orchestrator(
        devices=[device],
        latency_cfg=LatencyConfig(backend="benchncnn", use_cache=True, repeats=2),
    )

    param = tmp_path / "model.param"
    binf = tmp_path / "model.bin"
    param.write_text("param", encoding="utf-8")
    binf.write_bytes(b"bin")

    calls = []

    def fake_bench(_device, *, ncnn, shape):
        calls.append(1)
        return 10.0 + len(calls), "raw"

    monkeypatch.setattr(orch.bench, "bench", fake_bench)

    first = orch._bench_devices_with_cache(param, binf, tmp_path / "run1")
    second = orch._bench_devices_with_cache(param, binf, tmp_path / "run2")

    assert first == {"phone": 11.5}
    assert second == {"phone": 11.5}
    assert len(calls) == 2
    assert (tmp_path / "run2" / "bench_logs" / "phone.cache.txt").exists()
