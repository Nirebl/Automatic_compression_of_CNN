from __future__ import annotations

import json

import pytest

from xtrim.exporter import Exporter
from xtrim.utils import ensure_dir, sha256_file, sizeof_file, write_json


pytestmark = pytest.mark.unit


def test_utils_file_helpers(tmp_path):
    nested = tmp_path / "a" / "b"
    ensure_dir(nested)
    data = nested / "data.txt"
    data.write_text("abc", encoding="utf-8")
    out = tmp_path / "payload.json"
    write_json(out, {"x": 1})

    assert nested.exists()
    assert sizeof_file(data) == 3
    assert sha256_file(data) == "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    assert json.loads(out.read_text(encoding="utf-8")) == {"x": 1}


def test_exporter_creates_parent_and_checks_output(tmp_path):
    out = tmp_path / "deep" / "model.onnx"
    exporter = Exporter(lambda p: p.write_bytes(b"onnx"))

    assert exporter.export_onnx(out) == out
    assert out.exists()


def test_exporter_raises_when_callback_does_not_create_file(tmp_path):
    exporter = Exporter(lambda _p: None)

    with pytest.raises(RuntimeError, match="did not create file"):
        exporter.export_onnx(tmp_path / "missing" / "model.onnx")
