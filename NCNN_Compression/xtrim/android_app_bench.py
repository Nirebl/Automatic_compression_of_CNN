from __future__ import annotations

import json
import re
import subprocess
import time
import uuid
from pathlib import Path
from typing import Optional, List, Tuple

from .types import DeviceConfig, ToolsConfig, AndroidAppBenchConfig
from .runtime_backend import effective_ncnn_gpu_device, resolve_ncnn_runtime


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

    def push_dataset_subset(
        self,
        device: DeviceConfig,
        *,
        local_images_dir: Path,
        local_image_list: Path,
        remote_images_dir: str,
        remote_image_list: str,
    ) -> Tuple[str, str]:
        self.adb(device.serial, "shell", "rm", "-rf", remote_images_dir)
        self.adb(device.serial, "shell", "mkdir", "-p", remote_images_dir)

        for image_path in sorted(Path(local_images_dir).iterdir()):
            if image_path.is_file():
                self.adb(device.serial, "push", str(image_path), f"{remote_images_dir}/{image_path.name}")

        self.adb(device.serial, "push", str(local_image_list), remote_image_list)
        try:
            # Directory must stay executable/traversable for the Android app.
            # chmod -R 644 would make the directory unreadable as a directory.
            self.adb(device.serial, "shell", f"chmod 755 {remote_images_dir} && chmod 644 {remote_images_dir}/*")
        except Exception as e:
            print(e)
            pass
        return remote_images_dir, remote_image_list

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
        decoder = json.JSONDecoder()
        found: List[dict] = []
        for start, ch in enumerate(text):
            if ch != "{":
                continue
            try:
                obj, _end = decoder.raw_decode(text[start:])
            except json.JSONDecodeError:
                continue
            if isinstance(obj, dict):
                found.append(obj)
        return found[-1] if found else None

    def run_once(
        self,
        *,
        device: DeviceConfig,
        local_param: Path,
        local_bin: Path,
        local_images_dir: Optional[Path] = None,
        local_image_list: Optional[Path] = None,
        remote_images_dir: Optional[str] = None,
        remote_image_list: Optional[str] = None,
        dataset_image_count: int = 0,
    ) -> dict:
        cfg = self.cfg
        if not cfg.enabled:
            raise RuntimeError("AndroidAppBenchConfig.enabled=False")

        if not self.is_device_ready(device):
            raise RuntimeError(f"Device not ready: {device.name} ({device.serial})")

        run_id = uuid.uuid4().hex[:10]
        runtime = resolve_ncnn_runtime(device)
        gpu_device = effective_ncnn_gpu_device(device)

        self.force_stop(device)
        if device.cooling_down > 0:
            time.sleep(float(device.cooling_down))

        rparam, rbin = self.push_model(device, local_param, local_bin)

        pushed_image_args: List[str] = []
        if cfg.push_dataset_images:
            if not (local_images_dir and local_image_list and remote_images_dir and remote_image_list):
                raise RuntimeError("push_dataset_images=True, but dataset subset paths were not provided")
            rimages, rlist = self.push_dataset_subset(
                device,
                local_images_dir=local_images_dir,
                local_image_list=local_image_list,
                remote_images_dir=remote_images_dir,
                remote_image_list=remote_image_list,
            )
            pushed_image_args = [
                "--ez", "use_pushed_images", "true",
                "--es", "image_dir", rimages,
                "--es", "image_list", rlist,
                "--ei", "image_count", str(int(dataset_image_count)),
            ]

        if cfg.clear_logcat:
            self.clear_logcat(device)

        comp = f"{cfg.package}/{cfg.activity}"
        am_cmd = [
            "shell", "am", "start", "-W", "-n", comp,
            "--es", "param", rparam,
            "--es", "bin", rbin,
            "--es", "dataset", str(cfg.dataset),
            "--es", "run_id", run_id,
            # Runtime is consumed by the updated Android CliBenchActivity.
            # device/gpu_device are also passed for backward-compatible builds.
            "--es", "runtime", runtime,
            "--es", "backend", runtime,
            "--ei", "device", str(int(gpu_device)),
            "--ei", "gpu_device", str(int(gpu_device)),
            "--ei", "imgsz", str(int(cfg.imgsz)),
            "--ei", "loops", str(int(cfg.loops)),
            "--ei", "warmup", str(int(cfg.warmup)),
            "--ei", "threads", str(int(cfg.threads)),
            "--ei", "max_det", str(int(cfg.max_det)),
            *pushed_image_args,
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
                # Older Android app builds may not echo these fields yet.
                # Add them on the Python side so logs and history remain explicit.
                data.setdefault("runtime", runtime)
                data.setdefault("backend", runtime)
                data.setdefault("use_gpu", runtime == "ncnn_vulkan")
                data.setdefault("gpu_device", gpu_device)
                data.setdefault("device_name", device.name)
                data.setdefault("serial", device.serial)
                return data

            time.sleep(float(cfg.poll_interval_sec))
