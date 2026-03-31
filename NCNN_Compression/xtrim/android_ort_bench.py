from __future__ import annotations

import json
import re
import time
import uuid
from pathlib import Path
from typing import Optional, List

from .android_app_bench import sh
from .types import DeviceConfig, ToolsConfig, OrtAndroidBenchConfig


class AndroidOrtBench:
    def __init__(self, tools: ToolsConfig, cfg: OrtAndroidBenchConfig):
        self.tools = tools
        self.cfg = cfg

    def adb(self, serial: str, *args: str) -> str:
        return sh([self.tools.adb, "-s", serial, *args])

    def is_device_ready(self, device: DeviceConfig) -> bool:
        try:
            return self.adb(device.serial, "get-state").strip() == "device"
        except Exception:
            return False

    def push_model(self, device: DeviceConfig, local_onnx: Path) -> str:
        rdir = self.cfg.remote_dir
        rmodel = f"{rdir}/model.onnx"
        self.adb(device.serial, "push", str(local_onnx), rmodel)
        try:
            self.adb(device.serial, "shell", f"chmod 644 {rmodel}")
        except Exception:
            pass
        return rmodel

    def force_stop(self, device: DeviceConfig) -> None:
        try:
            self.adb(device.serial, "shell", "am", "force-stop", self.cfg.package)
        except Exception:
            pass

    def clear_logcat(self, device: DeviceConfig) -> None:
        try:
            self.adb(device.serial, "logcat", "-c")
        except Exception:
            pass

    def _extract_last_json(self, text: str) -> Optional[dict]:
        found: List[dict] = []
        for m in re.finditer(r"\{.*\}", text, flags=re.DOTALL):
            s = m.group(0)
            try:
                found.append(json.loads(s))
            except Exception:
                continue
        return found[-1] if found else None

    def run_once(self, *, device: DeviceConfig, local_onnx: Path) -> dict:
        cfg = self.cfg
        if not cfg.enabled:
            raise RuntimeError("OrtAndroidBenchConfig.enabled=False")

        if not self.is_device_ready(device):
            raise RuntimeError(f"Device not ready: {device.name} ({device.serial})")

        run_id = uuid.uuid4().hex[:10]

        self.force_stop(device)
        if device.cooling_down > 0:
            time.sleep(float(device.cooling_down))

        rmodel = self.push_model(device, local_onnx)

        if cfg.clear_logcat:
            self.clear_logcat(device)

        comp = f"{cfg.package}/{cfg.activity}"
        am_cmd = [
            "shell", "am", "start", "-W", "-n", comp,
            "--es", "backend", "ort",
            "--es", "model", rmodel,
            "--es", "provider", str(cfg.provider),
            "--es", "dataset", str(cfg.dataset),
            "--es", "run_id", run_id,
            "--ei", "imgsz", str(int(cfg.imgsz)),
            "--ei", "loops", str(int(cfg.loops)),
            "--ei", "warmup", str(int(cfg.warmup)),
            "--ei", "threads", str(int(cfg.threads)),
            "--ei", "max_det", str(int(cfg.max_det)),
            "--ef", "conf", str(float(cfg.conf)),
            "--ef", "iou", str(float(cfg.iou)),
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
            except Exception:
                dump = ""

            last_dump = dump
            data = self._extract_last_json(dump)
            if data is not None:
                if "run_id" in data and str(data["run_id"]) != run_id:
                    time.sleep(float(cfg.poll_interval_sec))
                    continue
                return data

            time.sleep(float(cfg.poll_interval_sec))