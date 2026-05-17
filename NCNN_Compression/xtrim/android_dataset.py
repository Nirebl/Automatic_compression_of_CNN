from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import List

from .quant.calib import make_calib_imagelist
from .utils import ensure_dir


@dataclass(frozen=True)
class AndroidDatasetSubset:
    """Local dataset subset prepared for Android latency benchmarking.

    local_dir contains copied image files with stable short names.
    local_list contains remote Android paths, i.e. paths after adb push.
    """

    local_dir: Path
    local_list: Path
    remote_dir: str
    remote_list: str
    count: int


def _safe_remote_dir(remote_dir: str, subdir: str) -> str:
    base = str(remote_dir or "/data/local/tmp").rstrip("/")
    name = str(subdir or "xtrim_bench_images").strip().strip("/")
    if not name:
        name = "xtrim_bench_images"
    return f"{base}/{name}"


def prepare_android_dataset_subset(
    *,
    data_yaml: str,
    split: str,
    max_images: int,
    seed: int,
    out_dir: Path,
    remote_dir: str,
    remote_subdir: str = "xtrim_bench_images",
) -> AndroidDatasetSubset:
    """Build a small deterministic image subset and list for Android benchmark.

    The source image list is resolved through Ultralytics' check_det_dataset(), so
    regular dataset YAMLs such as coco128.yaml, VisDrone.yaml, or coco.yaml work.
    """

    ensure_dir(out_dir)
    source_txt = out_dir / "source_images.txt"
    make_calib_imagelist(
        data_yaml=str(data_yaml),
        split=str(split),
        max_images=int(max_images),
        out_txt=source_txt,
        seed=int(seed),
    )

    src_paths: List[Path] = [
        Path(line.strip())
        for line in source_txt.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not src_paths:
        raise RuntimeError(f"No dataset images selected from {data_yaml} split={split}")

    local_dir = out_dir / "images"
    if local_dir.exists():
        shutil.rmtree(local_dir)
    ensure_dir(local_dir)

    rdir = _safe_remote_dir(remote_dir, remote_subdir)
    remote_paths: List[str] = []

    for i, src in enumerate(src_paths):
        if not src.exists():
            raise RuntimeError(f"Selected dataset image does not exist: {src}")
        suffix = src.suffix.lower() or ".jpg"
        dst = local_dir / f"{i:06d}{suffix}"
        shutil.copy2(src, dst)
        remote_paths.append(f"{rdir}/{dst.name}")

    local_list = out_dir / "image_list_remote.txt"
    local_list.write_text("\n".join(remote_paths) + "\n", encoding="utf-8")

    return AndroidDatasetSubset(
        local_dir=local_dir,
        local_list=local_list,
        remote_dir=rdir,
        remote_list=f"{rdir}/image_list.txt",
        count=len(remote_paths),
    )
