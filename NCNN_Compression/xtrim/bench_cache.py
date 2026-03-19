from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from .utils import ensure_dir


@dataclass
class CacheEntry:
    avg_ms: float
    ts: str


class BenchCache:
    def __init__(self, path: Path):
        self.path = path
        self._data: Dict[str, Dict[str, Any]] = {}
        self._loaded = False

    def load(self) -> None:
        if self._loaded:
            return
        if self.path.exists():
            self._data = json.loads(self.path.read_text(encoding="utf-8"))
        else:
            self._data = {}
        self._loaded = True

    def get(self, key: str) -> Optional[CacheEntry]:
        self.load()
        rec = self._data.get(key)
        if not rec:
            return None
        try:
            return CacheEntry(avg_ms=float(rec["avg_ms"]), ts=str(rec.get("ts", "")))
        except Exception as e:
            print(e)
            return None

    def set(self, key: str, avg_ms: float) -> None:
        self.load()
        self._data[key] = {"avg_ms": float(avg_ms), "ts": time.strftime("%Y-%m-%d %H:%M:%S")}

    def save(self) -> None:
        self.load()
        ensure_dir(self.path.parent)
        tmp = self.path.with_suffix(self.path.suffix + ".tmp")
        tmp.write_text(json.dumps(self._data, ensure_ascii=False, indent=2), encoding="utf-8")
        tmp.replace(self.path)
