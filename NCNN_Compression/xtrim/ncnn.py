from __future__ import annotations

import re
from pathlib import Path
from typing import Optional, Tuple

from .types import ToolsConfig, DeviceConfig, PTQConfig, NcnnModelPaths
from .utils import sh, ensure_dir


def _normalize_shape_arg(shape: str) -> str:
    s = shape.strip()
    if not s.startswith("["):
        s = f"[{s}]"
    return s


def _validate_ncnn_param(param: Path) -> None:
    if not param.exists():
        raise RuntimeError(f"NCNN param not found: {param}")

    lines = param.read_text(encoding="utf-8").splitlines()
    if not lines:
        raise RuntimeError(f"NCNN param is empty: {param}")

    if lines[0].strip() != "7767517":
        raise RuntimeError(
            f"NCNN param bad magic (line 1={lines[0].strip()!r}, expected '7767517'): {param}"
        )

    if len(lines) < 2:
        raise RuntimeError(f"NCNN param truncated — missing layer/blob count header: {param}")

    parts = lines[1].split()
    if len(parts) < 2 or not parts[0].isdigit() or not parts[1].isdigit():
        raise RuntimeError(f"NCNN param bad header line 2 ({lines[1]!r}): {param}")

    n_layers = int(parts[0])
    body_lines = [ln for ln in lines[2:] if ln.strip()]
    if len(body_lines) != n_layers:
        raise RuntimeError(
            f"NCNN param truncated: header says {n_layers} layers "
            f"but found {len(body_lines)} body lines: {param}"
        )

    last_parts = body_lines[-1].split()
    if len(last_parts) < 4:
        raise RuntimeError(
            f"NCNN param last layer line looks truncated ({body_lines[-1]!r}): {param}"
        )


class NcnnConverter:
    def __init__(self, tools: ToolsConfig):
        self.tools = tools

    def pnnx_convert(self, torch_model, out_dir: Path, imgsz: int = 640) -> Optional[NcnnModelPaths]:
        import torch
        try:
            import pnnx
        except ImportError:
            raise RuntimeError("pnnx is not installed. Run: pip install pnnx")

        ensure_dir(out_dir)
        pt_path = out_dir.as_posix() + "/model.pt"
        ncnn_param = out_dir / "model.ncnn.param"
        ncnn_bin = out_dir / "model.ncnn.bin"

        device = next(torch_model.parameters()).device
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=device)

        torch_model.eval()
        _export_flags: list = []
        for m in torch_model.modules():
            if type(m).__name__ in ("Detect", "Segment", "Pose", "OBB", "RTDETRDecoder"):
                _export_flags.append((m, getattr(m, "export", False)))
                m.export = True

        try:
            pnnx.export(
                torch_model,
                pt_path,
                inputs=(dummy,),
                ncnnparam=ncnn_param.as_posix(),
                ncnnbin=ncnn_bin.as_posix(),
                fp16=False,
                check_trace=False,
            )
        except Exception as e:
            import sys as _sys
            _valid = (ncnn_param.exists() and ncnn_param.stat().st_size > 100
                      and ncnn_bin.exists() and ncnn_bin.stat().st_size > 1_000)
            if _valid:
                print(f"[warn] PNNX post-processing error (ignored, NCNN files ok): {e}",
                      file=_sys.stderr)
            else:
                raise
        finally:
            for m, original in _export_flags:
                m.export = original

        if not ncnn_param.exists() or not ncnn_bin.exists():
            raise RuntimeError(f"PNNX did not produce NCNN files in {out_dir}")
        return NcnnModelPaths(param=ncnn_param, bin=ncnn_bin)

    def onnx_to_ncnn(self, onnx_path: Path, out_dir: Path) -> NcnnModelPaths:
        ensure_dir(out_dir)
        out_param = out_dir / "model.param"
        out_bin = out_dir / "model.bin"
        sh([self.tools.onnx2ncnn, str(onnx_path), str(out_param), str(out_bin)])
        return NcnnModelPaths(param=out_param, bin=out_bin)

    def optimize(self, ncnn: NcnnModelPaths, out_dir: Path) -> NcnnModelPaths:
        import shutil
        ensure_dir(out_dir)
        opt_param = out_dir / "opt.param"
        opt_bin = out_dir / "opt.bin"
        try:
            sh([self.tools.ncnnoptimize,
                str(ncnn.param), str(ncnn.bin),
                str(opt_param), str(opt_bin),
                "0"])
            return NcnnModelPaths(param=opt_param, bin=opt_bin)
        except Exception as e:
            (out_dir / "ncnnoptimize_failed.txt").write_text(str(e), encoding="utf-8")
            shutil.copy2(ncnn.param, opt_param)
            shutil.copy2(ncnn.bin, opt_bin)
            return NcnnModelPaths(param=opt_param, bin=opt_bin)

    def ptq_int8(self, ncnn: NcnnModelPaths, out_dir: Path, ptq: PTQConfig) -> NcnnModelPaths:
        ensure_dir(out_dir)
        table = out_dir / "calib.table"
        int8_param = out_dir / "int8.param"
        int8_bin = out_dir / "int8.bin"

        imagelist = Path(ptq.imagelist)
        if not imagelist.exists():
            raise RuntimeError(f"PTQ imagelist not found: {imagelist}")

        sh([
            self.tools.ncnn2table,
            str(ncnn.param),
            str(ncnn.bin),
            str(imagelist),
            str(table),
            f"mean={ptq.mean}",
            f"norm={ptq.norm}",
            f"shape={_normalize_shape_arg(ptq.shape)}",
            f"pixel={ptq.pixel}",
            f"thread={ptq.thread}",
            f"method={ptq.method}",
        ])

        try:
            sh([
                self.tools.ncnn2int8,
                str(ncnn.param),
                str(ncnn.bin),
                str(int8_param),
                str(int8_bin),
                str(table),
            ])
        except Exception as e:
            import sys as _sys
            _min_bin = 1_000_000
            _bin_ok = int8_bin.exists() and int8_bin.stat().st_size > _min_bin
            if _bin_ok:
                try:
                    _validate_ncnn_param(int8_param)
                    _param_ok = True
                except RuntimeError as _ve:
                    _param_ok = False
                    print(f"[error] int8.param validation failed: {_ve}", file=_sys.stderr)
            else:
                _param_ok = False

            if _bin_ok and _param_ok:
                print(
                    f"[warn] ncnn2int8 exited with error but outputs passed structural validation "
                    f"({int8_bin.stat().st_size // 1024} KB) — treating as success. "
                    f"Original error: {e}",
                    file=_sys.stderr,
                )
            else:
                raise RuntimeError(
                    f"ncnn2int8 failed and output files are invalid. "
                    f"bin_ok={_bin_ok}, param_ok={_param_ok}. Original error: {e}"
                ) from e

        _validate_ncnn_param(int8_param)

        return NcnnModelPaths(param=int8_param, bin=int8_bin)


class AdbBench:
    def __init__(self, tools: ToolsConfig):
        self.tools = tools

    def adb(self, serial: str, *args: str) -> str:
        return sh([self.tools.adb, "-s", serial, *args])

    def is_device_ready(self, device: DeviceConfig) -> bool:
        try:
            out = self.adb(device.serial, "get-state").strip()
            return out == "device"
        except Exception:
            return False

    def ensure_benchncnn(self, device: DeviceConfig, force_push: bool = False) -> None:
        if not self.is_device_ready(device):
            raise RuntimeError(f"ADB device not ready: {device.name} ({device.serial})")

        if not force_push:
            try:
                chk = self.adb(device.serial, "shell", "test -x /data/local/tmp/benchncnn && echo OK").strip()
                if chk == "OK":
                    return
            except Exception:
                pass

        local = Path(self.tools.benchncnn_local)
        if not local.exists():
            raise RuntimeError(f"benchncnn binary not found locally: {local}")
        self.adb(device.serial, "push", str(local), "/data/local/tmp/benchncnn")
        self.adb(device.serial, "shell", "chmod +x /data/local/tmp/benchncnn")

    def bench(self, device: DeviceConfig, ncnn: NcnnModelPaths, shape: str) -> Tuple[float, str]:
        if not self.is_device_ready(device):
            raise RuntimeError(f"ADB device not ready: {device.name} ({device.serial})")

        self.adb(device.serial, "push", str(ncnn.param), "/data/local/tmp/model.param")
        self.adb(device.serial, "push", str(ncnn.bin), "/data/local/tmp/model.bin")

        cmd = (
            "cd /data/local/tmp && "
            f"./benchncnn {device.loops} {device.threads} {device.powersave} {device.gpu_device} "
            f"{device.cooling_down} param=model.param bin=model.bin shape={shape}"
        )
        log = self.adb(device.serial, "shell", cmd)

        m = re.search(r"min\s*=\s*([0-9.]+)\s+max\s*=\s*([0-9.]+)\s+avg\s*=\s*([0-9.]+)", log)
        if not m:
            raise RuntimeError(f"Could not parse benchncnn output.\nLOG:\n{log}")
        avg_ms = float(m.group(3))
        return avg_ms, log
