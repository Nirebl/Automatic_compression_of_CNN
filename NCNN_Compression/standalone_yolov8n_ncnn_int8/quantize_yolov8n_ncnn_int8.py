from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path


DEMO_IMAGES = {
    "bus.jpg": "https://ultralytics.com/images/bus.jpg",
    "zidane.jpg": "https://ultralytics.com/images/zidane.jpg",
}
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def log(message: str) -> None:
    print(message, flush=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Standalone pipeline: export yolov8n to NCNN, quantize to INT8, and verify inference."
    )
    parser.add_argument("--workspace", type=Path, default=Path("work"))
    parser.add_argument("--weights", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--table-method", default="kl", choices=["kl", "aciq", "eq"])
    parser.add_argument("--ncnn-tools-dir", type=Path, default=None)
    parser.add_argument("--calib-dir", type=Path, default=None)
    parser.add_argument("--calib-list", type=Path, default=None)
    parser.add_argument("--demo-calibration-repeats", type=int, default=8)
    parser.add_argument("--reuse-export", action="store_true")
    return parser.parse_args()


def require_package(package: str):
    try:
        return __import__(package)
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            f"Missing Python package '{package}'. Install requirements.txt first."
        ) from exc


def run_command(command: list[str], cwd: Path | None = None) -> None:
    rendered = " ".join(str(part) for part in command)
    log(f"$ {rendered}")
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        print(result.stderr, end="", file=sys.stderr)
    if result.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {result.returncode}: {rendered}")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def download_file(url: str, destination: Path) -> None:
    requests = require_package("requests")
    log(f"Downloading {url} -> {destination}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    destination.write_bytes(response.content)


def prepare_demo_images(workspace: Path, repeats: int) -> tuple[Path, Path]:
    image_dir = ensure_dir(workspace / "demo_images")
    for name, url in DEMO_IMAGES.items():
        target = image_dir / name
        if not target.exists():
            download_file(url, target)

    calibration_list = workspace / "calibration_list.txt"
    entries = []
    for _ in range(repeats):
        entries.append(str((image_dir / "bus.jpg").resolve()))
        entries.append(str((image_dir / "zidane.jpg").resolve()))
    calibration_list.write_text("\n".join(entries) + "\n", encoding="utf-8")
    return calibration_list, image_dir / "bus.jpg"


def build_calibration_list(workspace: Path, calib_dir: Path | None, calib_list: Path | None, repeats: int) -> tuple[Path, Path]:
    if calib_list:
        if not calib_list.exists():
            raise FileNotFoundError(f"Calibration list not found: {calib_list}")
        demo_image = workspace / "demo_images" / "bus.jpg"
        if not demo_image.exists():
            _, demo_image = prepare_demo_images(workspace, repeats)
        return calib_list.resolve(), demo_image

    if calib_dir:
        if not calib_dir.exists():
            raise FileNotFoundError(f"Calibration directory not found: {calib_dir}")
        images = sorted(
            path.resolve() for path in calib_dir.iterdir() if path.suffix.lower() in IMAGE_EXTENSIONS
        )
        if not images:
            raise RuntimeError(f"No images found in calibration directory: {calib_dir}")
        list_path = workspace / "calibration_list.txt"
        list_path.write_text("\n".join(str(path) for path in images) + "\n", encoding="utf-8")
        _, demo_image = prepare_demo_images(workspace, repeats)
        return list_path, demo_image

    return prepare_demo_images(workspace, repeats)


def resolve_tool(tool_name: str, tools_dir: Path | None) -> Path:
    suffix = ".exe" if os.name == "nt" else ""
    executable = tool_name + suffix

    env_dir = os.environ.get("NCNN_TOOLS_DIR")
    search_roots = []
    if tools_dir:
        search_roots.append(tools_dir)
    if env_dir:
        search_roots.append(Path(env_dir))

    for root in search_roots:
        root = root.resolve()
        direct = root / executable
        if direct.exists():
            return direct
        for candidate in root.rglob(executable):
            return candidate

    in_path = shutil.which(executable)
    if in_path:
        return Path(in_path).resolve()

    raise FileNotFoundError(
        f"Could not find {executable}. Pass --ncnn-tools-dir or set NCNN_TOOLS_DIR."
    )


def export_to_ncnn(weights: str, export_root: Path, imgsz: int, reuse_export: bool) -> Path:
    ultralytics = require_package("ultralytics")
    YOLO = ultralytics.YOLO

    existing = sorted(export_root.rglob("*_ncnn_model"))
    if reuse_export and existing:
        latest = max(existing, key=lambda path: path.stat().st_mtime)
        log(f"Reusing existing export: {latest}")
        return latest

    ensure_dir(export_root)
    model = YOLO(weights)
    previous_cwd = Path.cwd()
    try:
        os.chdir(export_root)
        exported = Path(model.export(format="ncnn", imgsz=imgsz))
    finally:
        os.chdir(previous_cwd)
    if not exported.is_absolute():
        exported = export_root / exported
    return exported.resolve()


def verify_param_file(param_path: Path) -> None:
    lines = param_path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 3:
        raise RuntimeError(f"{param_path} is too short to be a valid NCNN param file.")

    for line_number, line in enumerate(lines[2:], start=3):
        parts = line.split()
        if len(parts) < 4:
            raise RuntimeError(f"Malformed NCNN param line {line_number}: {line!r}")
        try:
            bottom_count = int(parts[2])
            top_count = int(parts[3])
        except ValueError as exc:
            raise RuntimeError(f"Malformed NCNN param line {line_number}: {line!r}") from exc
        required_tokens = 4 + bottom_count + top_count
        if len(parts) < required_tokens:
            raise RuntimeError(f"Truncated NCNN param line {line_number}: {line!r}")


def quantize_model(
    float_model_dir: Path,
    int8_model_dir: Path,
    calibration_list: Path,
    ncnn2table: Path,
    ncnn2int8: Path,
    imgsz: int,
    method: str,
) -> Path:
    ensure_dir(int8_model_dir)
    float_param = float_model_dir / "model.ncnn.param"
    float_bin = float_model_dir / "model.ncnn.bin"
    metadata = float_model_dir / "metadata.yaml"
    table_path = int8_model_dir.parent / "yolov8n.table"
    int8_param = int8_model_dir / "model.ncnn.param"
    int8_bin = int8_model_dir / "model.ncnn.bin"

    run_command(
        [
            str(ncnn2table),
            str(float_param),
            str(float_bin),
            str(calibration_list),
            str(table_path),
            f"shape=[{imgsz},{imgsz},3]",
            "pixel=RGB",
            "mean=[0.0,0.0,0.0]",
            "norm=[0.0039215686,0.0039215686,0.0039215686]",
            f"method={method}",
        ]
    )

    run_command(
        [
            str(ncnn2int8),
            str(float_param),
            str(float_bin),
            str(int8_param),
            str(int8_bin),
            str(table_path),
        ]
    )

    if metadata.exists():
        shutil.copy2(metadata, int8_model_dir / "metadata.yaml")

    try:
        verify_param_file(int8_param)
    except RuntimeError as exc:
        host = platform.system()
        if host == "Windows":
            raise RuntimeError(
                "Generated INT8 NCNN param file is malformed on native Windows. "
                "This was reproduced locally with YOLOv8n and official ncnn CLI tools. "
                "Run the same script inside Linux/WSL with Linux ncnn binaries."
            ) from exc
        raise

    return int8_model_dir


def summarize_result(result) -> dict:
    names = result.names
    detections = []
    if result.boxes is None:
        return {"num_detections": 0, "classes": [], "detections": []}

    boxes = result.boxes
    for index in range(len(boxes)):
        cls_id = int(boxes.cls[index].item())
        xyxy = [round(float(value), 2) for value in boxes.xyxy[index].tolist()]
        detections.append(
            {
                "class_id": cls_id,
                "class_name": names[cls_id],
                "confidence": round(float(boxes.conf[index].item()), 5),
                "xyxy": xyxy,
            }
        )

    classes = sorted({entry["class_name"] for entry in detections})
    return {
        "num_detections": len(detections),
        "classes": classes,
        "detections": detections,
    }


def verify_model(model_path: Path, image_path: Path, verification_dir: Path, tag: str, imgsz: int, conf: float) -> dict:
    ultralytics = require_package("ultralytics")
    YOLO = ultralytics.YOLO

    ensure_dir(verification_dir)
    model = YOLO(str(model_path), task="detect")
    results = model(source=str(image_path), imgsz=imgsz, conf=conf, device="cpu", verbose=False)
    result = results[0]

    annotated_path = verification_dir / f"{tag}_pred.jpg"
    summary_path = verification_dir / f"{tag}_summary.json"
    result.save(filename=str(annotated_path))

    summary = summarize_result(result)
    summary["model"] = str(model_path)
    summary["image"] = str(image_path)
    summary["annotated_image"] = str(annotated_path)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    if summary["num_detections"] == 0:
        raise RuntimeError(f"{tag} model produced no detections on {image_path.name}.")

    return summary


def write_run_summary(workspace: Path, float_summary: dict, int8_summary: dict | None) -> None:
    payload = {
        "float_model": float_summary,
        "int8_model": int8_summary,
    }
    (workspace / "run_summary.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def main() -> int:
    args = parse_args()
    workspace = ensure_dir(args.workspace.resolve())
    export_root = ensure_dir(workspace / "exports")
    verification_dir = ensure_dir(workspace / "verification")
    int8_model_dir = workspace / "int8_model"

    calibration_list, test_image = build_calibration_list(
        workspace=workspace,
        calib_dir=args.calib_dir,
        calib_list=args.calib_list,
        repeats=args.demo_calibration_repeats,
    )

    ncnn2table = resolve_tool("ncnn2table", args.ncnn_tools_dir)
    ncnn2int8 = resolve_tool("ncnn2int8", args.ncnn_tools_dir)
    log(f"Using ncnn2table: {ncnn2table}")
    log(f"Using ncnn2int8:  {ncnn2int8}")

    float_model_dir = export_to_ncnn(
        weights=args.weights,
        export_root=export_root,
        imgsz=args.imgsz,
        reuse_export=args.reuse_export,
    )
    log(f"Float NCNN model: {float_model_dir}")

    float_summary = verify_model(
        model_path=float_model_dir,
        image_path=test_image,
        verification_dir=verification_dir,
        tag="fp32",
        imgsz=args.imgsz,
        conf=args.conf,
    )
    log(f"FP32 detections: {float_summary['num_detections']} classes={float_summary['classes']}")

    int8_summary = None
    try:
        quantized_dir = quantize_model(
            float_model_dir=float_model_dir,
            int8_model_dir=int8_model_dir,
            calibration_list=calibration_list,
            ncnn2table=ncnn2table,
            ncnn2int8=ncnn2int8,
            imgsz=args.imgsz,
            method=args.table_method,
        )
        log(f"INT8 NCNN model: {quantized_dir}")

        int8_summary = verify_model(
            model_path=quantized_dir,
            image_path=test_image,
            verification_dir=verification_dir,
            tag="int8",
            imgsz=args.imgsz,
            conf=args.conf,
        )
        log(f"INT8 detections: {int8_summary['num_detections']} classes={int8_summary['classes']}")
    except Exception as exc:
        write_run_summary(workspace, float_summary, None)
        raise RuntimeError(f"INT8 stage failed after FP32 verification succeeded: {exc}") from exc

    write_run_summary(workspace, float_summary, int8_summary)
    log("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
