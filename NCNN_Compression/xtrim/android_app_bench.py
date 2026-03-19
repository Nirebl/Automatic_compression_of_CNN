from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional, List, Tuple

from .types import DeviceConfig, ToolsConfig, AndroidAppBenchConfig


class CmdError(RuntimeError):
    pass


def sh(cmd: List[str]) -> str:
    p = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if p.returncode != 0:
        raise CmdError(
            f"Command failed ({p.returncode}): {' '.join(cmd)}\n"
            f"--- STDOUT ---\n{p.stdout}\n--- STDERR ---\n{p.stderr}\n"
        )
    return p.stdout


class AndroidAppBench:
    def __init__(self, tools: ToolsConfig, cfg: AndroidAppBenchConfig):
        self.tools = tools
        self.cfg = cfg

    def adb(self, serial: str, *args: str) -> str:
        return sh([self.tools.adb, "-s", serial, *args])

    def is_device_ready(self, device: DeviceConfig) -> bool:
        try:
            return self.adb(device.serial, "get-state").strip() == "device"
        except Exception as e:
            print(e)
            return False

    def push_model(self, device: DeviceConfig, local_param: Path, local_bin: Path) -> Tuple[str, str]:
        rdir = self.cfg.remote_dir
        rparam = f"{rdir}/model.param"
        rbin = f"{rdir}/model.bin"
        self.adb(device.serial, "push", str(local_param), rparam)
        self.adb(device.serial, "push", str(local_bin), rbin)
        try:
            self.adb(device.serial, "shell", f"chmod 644 {rparam} {rbin}")
        except Exception as e:
            print(e)
            pass
        return rparam, rbin

    def force_stop(self, device: DeviceConfig) -> None:
        try:
            self.adb(device.serial, "shell", "am", "force-stop", self.cfg.package)
        except Exception as e:
            print(e)

    def clear_logcat(self, device: DeviceConfig) -> None:
        try:
            self.adb(device.serial, "logcat", "-c")
        except Exception as e:
            print(e)

    def _extract_last_json(self, text: str) -> Optional[dict]:
        found: List[dict] = []
        for m in re.finditer(r"\{.*\}", text, flags=re.DOTALL):
            s = m.group(0)
            try:
                found.append(json.loads(s))
            except Exception as e:
                print(e)
                continue
        return found[-1] if found else None

    def run_once(self, *, device: DeviceConfig, local_param: Path, local_bin: Path) -> dict:
        cfg = self.cfg
        if not cfg.enabled:
            raise RuntimeError("AndroidAppBenchConfig.enabled=False")

        if not self.is_device_ready(device):
            raise RuntimeError(f"Device not ready: {device.name} ({device.serial})")

        run_id = uuid.uuid4().hex[:10]

        self.force_stop(device)
        if device.cooling_down > 0:
            time.sleep(float(device.cooling_down))

        rparam, rbin = self.push_model(device, local_param, local_bin)

        if cfg.clear_logcat:
            self.clear_logcat(device)

        comp = f"{cfg.package}/{cfg.activity}"
        am_cmd = [
            "shell", "am", "start", "-W", "-n", comp,
            "--es", "param", rparam,
            "--es", "bin", rbin,
            "--es", "dataset", str(cfg.dataset),
            "--es", "run_id", run_id,
            "--ei", "imgsz", str(int(cfg.imgsz)),
            "--ei", "loops", str(int(cfg.loops)),
            "--ei", "warmup", str(int(cfg.warmup)),
            "--ei", "threads", str(int(cfg.threads)),
            "--ei", "max_det", str(int(cfg.max_det)),
            "--ef", "conf", str(float(cfg.conf)),
            "--ef", "iou", str(float(cfg.iou)),
            "--ez", "optimized", "true" if cfg.optimized else "false",
        ]
        _ = self.adb(device.serial, *am_cmd)

        t0 = time.time()
        last_dump = ""
        while True:
            if time.time() - t0 > float(cfg.timeout_sec):
                raise RuntimeError(
                    f"[{device.name}] Timeout waiting for {cfg.result_tag} in logcat (run_id={run_id}).\n"
                    f"Last dump tail:\n{last_dump[-2000:]}"
                )

            try:
                dump = self.adb(device.serial, "logcat", "-d", "-v", "raw", "-s", f"{cfg.result_tag}:I")
            except Exception as e:
                print(e)
                dump = ""

            last_dump = dump
            data = self._extract_last_json(dump)
            if data is not None:
                if "run_id" in data and str(data["run_id"]) != run_id:
                    time.sleep(float(cfg.poll_interval_sec))
                    continue
                return data

            time.sleep(float(cfg.poll_interval_sec))
