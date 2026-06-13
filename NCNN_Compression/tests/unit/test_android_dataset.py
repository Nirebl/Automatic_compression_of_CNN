from __future__ import annotations

import pytest

from xtrim.android_dataset import _safe_remote_dir, prepare_android_dataset_subset

pytestmark = pytest.mark.unit


def test_safe_remote_dir_normalizes_base_and_subdir():
    assert _safe_remote_dir("/data/local/tmp/", "/imgs/") == "/data/local/tmp/imgs"
    assert _safe_remote_dir("", "") == "/data/local/tmp/xtrim_bench_images"
    assert _safe_remote_dir("/base", "///") == "/base/xtrim_bench_images"


def test_prepare_android_dataset_subset_copies_images_and_writes_remote_list(tmp_path, monkeypatch):
    img1 = tmp_path / "src1.JPG"
    img2 = tmp_path / "src2"
    img1.write_bytes(b"jpg")
    img2.write_bytes(b"raw")
    calls = []

    def fake_make_calib_imagelist(*, data_yaml, split, max_images, out_txt, seed):
        calls.append((data_yaml, split, max_images, seed))
        out_txt.write_text(f"{img1}\n{img2}\n", encoding="utf-8")

    monkeypatch.setattr("xtrim.android_dataset.make_calib_imagelist", fake_make_calib_imagelist)
    subset = prepare_android_dataset_subset(
        data_yaml="data.yaml",
        split="train",
        max_images=2,
        seed=7,
        out_dir=tmp_path / "out",
        remote_dir="/remote/base/",
        remote_subdir="bench",
    )

    assert calls == [("data.yaml", "train", 2, 7)]
    assert subset.count == 2
    assert subset.remote_dir == "/remote/base/bench"
    assert subset.remote_list == "/remote/base/bench/image_list.txt"
    assert (subset.local_dir / "000000.jpg").read_bytes() == b"jpg"
    assert (subset.local_dir / "000001.jpg").read_bytes() == b"raw"
    assert subset.local_list.read_text(encoding="utf-8").splitlines() == [
        "/remote/base/bench/000000.jpg",
        "/remote/base/bench/000001.jpg",
    ]


def test_prepare_android_dataset_subset_recreates_images_dir_and_errors(tmp_path, monkeypatch):
    out = tmp_path / "out"
    stale = out / "images"
    stale.mkdir(parents=True)
    (stale / "old.jpg").write_bytes(b"old")
    missing = tmp_path / "missing.jpg"

    def write_missing(*, out_txt, **_kwargs):
        out_txt.write_text(f"{missing}\n", encoding="utf-8")

    monkeypatch.setattr("xtrim.android_dataset.make_calib_imagelist", write_missing)
    with pytest.raises(RuntimeError, match="does not exist"):
        prepare_android_dataset_subset(data_yaml="d", split="val", max_images=1, seed=1, out_dir=out, remote_dir="/r")

    def write_empty(*, out_txt, **_kwargs):
        out_txt.write_text("\n", encoding="utf-8")

    monkeypatch.setattr("xtrim.android_dataset.make_calib_imagelist", write_empty)
    with pytest.raises(RuntimeError, match="No dataset images selected"):
        prepare_android_dataset_subset(data_yaml="d", split="val", max_images=1, seed=1, out_dir=tmp_path / "empty", remote_dir="/r")
