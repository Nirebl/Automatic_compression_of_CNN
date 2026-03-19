from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Optional

from PIL import Image

from ..types import DeviceConfig, ToolsConfig, AndroidDemoConfig
from ..utils import ensure_dir


class CmdError(RuntimeError):
    pass


def sh(cmd: list[str], cwd: Optional[Path] = None) -> str:
    p = subprocess.run(
        cmd,
        cwd=str(cwd) if cwd else None,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if p.returncode != 0:
        raise CmdError(f"Command failed ({p.returncode}): {' '.join(cmd)}\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}\n")
    return p.stdout


def save_ppm_rgb(image_path: Path, out_ppm: Path) -> tuple[int, int]:
    im = Image.open(image_path).convert("RGB")
    w, h = im.size
    data = im.tobytes()

    ensure_dir(out_ppm.parent)
    header = f"P6\n{w} {h}\n255\n".encode("ascii")
    out_ppm.write_bytes(header + data)
    return w, h


class AdbYoloDemo:
    def __init__(self, tools: ToolsConfig):
        self.tools = tools

    def adb(self, serial: str, *args: str) -> str:
        return sh([self.tools.adb, "-s", serial, *args])

    def is_ready(self, device: DeviceConfig) -> bool:
        try:
            st = self.adb(device.serial, "get-state").strip()
            return st == "device"
        except Exception:
            return False

    def ensure_binary(self, device: DeviceConfig) -> None:
        local = Path(self.tools.yolo_detect_local)
        if not local.exists():
            raise RuntimeError(f"Android detect binary not found: {local}")
        self.adb(device.serial, "push", str(local), "/data/local/tmp/xtrim_yolo_detect")
        self.adb(device.serial, "shell", "chmod +x /data/local/tmp/xtrim_yolo_detect")

    def run_once(
        self,
        *,
        device: DeviceConfig,
        demo_cfg: AndroidDemoConfig,
        ncnn_param: Path,
        ncnn_bin: Path,
        run_dir: Path,
    ) -> str:
        if not self.is_ready(device):
            raise RuntimeError(f"Device not ready: {device.name} ({device.serial})")

        self.ensure_binary(device)

        img_path = Path(demo_cfg.sample_image)
        if not img_path.exists():
            raise RuntimeError(f"android_demo.sample_image not found: {img_path}")

        ppm_path = run_dir / "android_demo" / f"{device.name}.ppm"
        ensure_dir(ppm_path.parent)
        w, h = save_ppm_rgb(img_path, ppm_path)

        self.adb(device.serial, "push", str(ncnn_param), "/data/local/tmp/model.param")
        self.adb(device.serial, "push", str(ncnn_bin), "/data/local/tmp/model.bin")
        self.adb(device.serial, "push", str(ppm_path), "/data/local/tmp/input.ppm")

        cmd = (
            "cd /data/local/tmp && "
            "./xtrim_yolo_detect "
            f"--param model.param --bin model.bin "
            f"--image input.ppm "
            f"--imgsz {int(demo_cfg.imgsz)} "
            f"--conf {float(demo_cfg.conf)} "
            f"--iou {float(demo_cfg.iou)} "
            f"--max_det {int(demo_cfg.max_det)}"
        )
        out = self.adb(device.serial, "shell", cmd)

        log_dir = run_dir / "android_demo"
        ensure_dir(log_dir)
        (log_dir / f"{device.name}.log").write_text(
            f"# input_image={img_path}\n# ppm_wh={w}x{h}\n\n{out}\n",
            encoding="utf-8",
        )
        return out
