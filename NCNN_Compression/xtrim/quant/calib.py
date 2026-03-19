from __future__ import annotations

import random
from pathlib import Path
from typing import List

from ..utils import ensure_dir


def _list_images_from_source(src: str) -> List[str]:
    p = Path(src)
    if p.is_file() and p.suffix.lower() in {".txt"}:
        lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        return lines
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
        files = [str(x) for x in sorted(p.rglob("*")) if x.is_file() and x.suffix.lower() in exts]
        return files
    if "*" in src or "?" in src:
        files = [str(x) for x in sorted(Path().glob(src)) if Path(x).is_file()]
        return files
    return []


def make_calib_imagelist(
    *,
    data_yaml: str,
    split: str,
    max_images: int,
    out_txt: Path,
    seed: int = 42,
) -> Path:
    try:
        from ultralytics.data.utils import check_det_dataset
    except Exception as e:
        raise RuntimeError(f"Cannot import ultralytics.data.utils.check_det_dataset: {e}")

    data = check_det_dataset(data_yaml)
    key = str(split).lower()
    if key not in {"train", "val", "test"}:
        key = "train"

    src = data.get(key)
    if not src:
        raise RuntimeError(f"Dataset yaml does not provide split '{key}'. Got keys: {list(data.keys())}")

    imgs = _list_images_from_source(str(src))
    if not imgs:
        raise RuntimeError(f"Could not find images from dataset split '{key}': {src}")

    random.seed(seed)
    if 0 < max_images < len(imgs):
        imgs = random.sample(imgs, k=max_images)

    ensure_dir(out_txt.parent)
    out_txt.write_text("\n".join(imgs) + "\n", encoding="utf-8")
    return out_txt