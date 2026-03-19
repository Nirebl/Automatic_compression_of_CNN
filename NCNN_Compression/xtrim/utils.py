from __future__ import annotations

import hashlib
import json
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional


class CmdError(RuntimeError):
    pass


def sh(cmd: List[str], *, cwd: Optional[Path] = None, env: Optional[Dict[str, str]] = None) -> str:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise CmdError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"--- STDOUT ---\n{p.stdout}\n--- STDERR ---\n{p.stderr}\n"
        )
    return p.stdout


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def sizeof_file(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


def write_json(p: Path, obj: Any) -> None:
    p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def now_ts() -> str:
    return time.strftime("%Y%m%d-%H%M%S")


def sha256_file(p: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        while True:
            b = f.read(chunk_size)
            if not b:
                break
            h.update(b)
    return h.hexdigest()