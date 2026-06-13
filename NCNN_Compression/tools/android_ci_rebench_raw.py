import argparse
import csv
import json
import math
import re
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


try:
    import yaml
except ImportError:
    print("ERROR: PyYAML is not installed. Install it with: pip install pyyaml")
    sys.exit(1)


DEFAULT_PACKAGE = "com.example.testyolo"
DEFAULT_ACTIVITY = ".CliBenchActivity"
DEFAULT_RESULT_TAG = "XTRIM_RESULT"
DEFAULT_REMOTE_DIR = "/data/local/tmp/xtrim_ci_rebench"
DEFAULT_TIMEOUT_SEC = 10700


ARTIFACT_PATTERNS = {
    "onnx": ["**/*.onnx"],
    "onnx_int8": ["**/*int8*.onnx", "**/*quant*.onnx", "**/*.onnx"],
    "tflite": ["**/*.tflite"],
    "tflite_int8": ["**/*int8*.tflite"],
    "tflite_fp16": ["**/*fp16*.tflite", "**/*float16*.tflite"],
    "tflite_fp32": ["**/*fp32*.tflite", "**/*float32*.tflite", "**/*.tflite"],
}


T_CRIT_95 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    11: 2.201,
    12: 2.179,
    13: 2.160,
    14: 2.145,
    15: 2.131,
    16: 2.120,
    17: 2.110,
    18: 2.101,
    19: 2.093,
    20: 2.086,
    21: 2.080,
    22: 2.074,
    23: 2.069,
    24: 2.064,
    25: 2.060,
    26: 2.056,
    27: 2.052,
    28: 2.048,
    29: 2.045,
    30: 2.042,
}


def run_cmd(
    cmd: List[str],
    timeout: Optional[int] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    p = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )

    if check and p.returncode != 0:
        raise RuntimeError(
            "Command failed:\n"
            + " ".join(cmd)
            + "\n\nSTDOUT:\n"
            + p.stdout
            + "\n\nSTDERR:\n"
            + p.stderr
        )

    return p


def adb_base(serial: Optional[str]) -> List[str]:
    cmd = ["adb"]
    if serial:
        cmd += ["-s", serial]
    return cmd


def adb(
    serial: Optional[str],
    args: List[str],
    timeout: Optional[int] = None,
    check: bool = True,
) -> subprocess.CompletedProcess:
    return run_cmd(adb_base(serial) + args, timeout=timeout, check=check)


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config is not a YAML object: {path}")

    return data


def first_dict(*items: Any) -> Dict[str, Any]:
    for x in items:
        if isinstance(x, dict):
            return x
    return {}


def get_bench_cfg(cfg: Dict[str, Any], timeout_sec_override: Optional[int]) -> Dict[str, Any]:
    app = first_dict(cfg.get("android_app_bench"))
    ort = first_dict(cfg.get("ort_android_bench"))

    merged: Dict[str, Any] = {}
    merged.update(ort)
    merged.update(app)

    merged.setdefault("package", DEFAULT_PACKAGE)
    merged.setdefault("activity", DEFAULT_ACTIVITY)
    merged.setdefault("remote_dir", DEFAULT_REMOTE_DIR)
    merged.setdefault("result_tag", DEFAULT_RESULT_TAG)

    merged.setdefault("imgsz", 640)
    merged.setdefault("loops", 50)
    merged.setdefault("warmup", 10)
    merged.setdefault("threads", 4)
    merged.setdefault("conf", 0.25)
    merged.setdefault("iou", 0.45)
    merged.setdefault("max_det", 100)
    merged.setdefault("timeout_sec", DEFAULT_TIMEOUT_SEC)
    merged.setdefault("poll_interval_sec", 0.6)
    merged.setdefault("clear_logcat", True)

    if timeout_sec_override is not None:
        merged["timeout_sec"] = timeout_sec_override

    return merged


def normalize_profiles(cfg: Dict[str, Any]) -> List[Dict[str, Any]]:
    raw = cfg.get("benchmark_profiles")
    profiles: List[Dict[str, Any]] = []

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                profiles.append(dict(item))

    elif isinstance(raw, dict):
        for name, item in raw.items():
            if isinstance(item, dict):
                p = dict(item)
                p.setdefault("name", name)
                profiles.append(p)

    if not profiles:
        profiles = [
            {
                "name": "ort_xnnpack",
                "backend": "ort_android",
                "provider": "xnnpack",
                "artifact": "onnx_int8",
            },
            {
                "name": "tflite_int8_cpu",
                "backend": "tflite_android",
                "delegate": "xnnpack",
                "artifact": "tflite_int8",
            },
            {
                "name": "tflite_fp16_gpu",
                "backend": "tflite_android",
                "delegate": "gpu",
                "artifact": "tflite_fp16",
            },
        ]

    clean: List[Dict[str, Any]] = []

    for p in profiles:
        p = dict(p)
        p.setdefault("name", p.get("profile", "unnamed_profile"))
        p.setdefault("artifact", infer_artifact_from_profile(p))
        clean.append(p)

    return clean


def infer_artifact_from_profile(profile: Dict[str, Any]) -> str:
    name = str(profile.get("name", "")).lower()
    backend = str(profile.get("backend", "")).lower()
    delegate = str(profile.get("delegate", "")).lower()

    if "tflite" in name or "tflite" in backend:
        if "int8" in name:
            return "tflite_int8"
        if "fp16" in name or delegate == "gpu":
            return "tflite_fp16"
        if "fp32" in name:
            return "tflite_fp32"
        return "tflite"

    if "onnx" in name or "ort" in name or "ort" in backend:
        if "int8" in name:
            return "onnx_int8"
        return "onnx"

    return "onnx"


def map_app_backend(profile: Dict[str, Any]) -> str:
    backend = str(profile.get("backend", "")).lower()
    name = str(profile.get("name", "")).lower()

    if "tflite" in backend or "tflite" in name:
        return "tflite"

    if "ort" in backend or "onnx" in backend or "ort" in name or "onnx" in name:
        return "ort"

    return backend or "ort"


def find_json_in_text(text: str) -> Optional[Dict[str, Any]]:
    candidates: List[str] = []

    for line in text.splitlines():
        if "{" not in line or "}" not in line:
            continue

        m = re.search(r"(\{.*\})", line)
        if not m:
            continue

        candidates.append(m.group(1))

    for s in reversed(candidates):
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except json.JSONDecodeError:
            continue

    return None


def get_device_info(serial: Optional[str]) -> Dict[str, str]:
    def prop(name: str) -> str:
        try:
            return adb(serial, ["shell", "getprop", name], timeout=10).stdout.strip()
        except Exception:
            return ""

    return {
        "serial": serial or "",
        "brand": prop("ro.product.brand"),
        "manufacturer": prop("ro.product.manufacturer"),
        "model": prop("ro.product.model"),
        "device": prop("ro.product.device"),
        "android": prop("ro.build.version.release"),
        "sdk": prop("ro.build.version.sdk"),
        "soc": prop("ro.soc.model") or prop("ro.board.platform"),
    }


def safe_remote_name(s: str) -> str:
    s = re.sub(r"[^A-Za-z0-9_.-]+", "_", s)
    return s[:180]


def dir_has_model_files_directly(path: Path) -> bool:
    return any(path.glob("*.onnx")) or any(path.glob("*.tflite"))


def dir_has_model_files_recursive(path: Path) -> bool:
    return any(path.rglob("*.onnx")) or any(path.rglob("*.tflite"))


def discover_candidate_dirs(models_root: Path) -> List[Path]:
    if not models_root.exists():
        raise FileNotFoundError(f"models_root does not exist: {models_root}")


    if dir_has_model_files_directly(models_root):
        return [models_root]


    direct_candidates = [
        p for p in sorted(models_root.iterdir())
        if p.is_dir() and dir_has_model_files_recursive(p)
    ]

    if direct_candidates:
        return direct_candidates


    leaf_dirs = set()

    for model_path in list(models_root.rglob("*.onnx")) + list(models_root.rglob("*.tflite")):
        leaf_dirs.add(model_path.parent)

    return sorted(leaf_dirs)


def model_name_matches(path: Path, needle: str, root: Optional[Path] = None) -> bool:
    needle = needle.lower().strip()

    if not needle:
        return True

    name = path.name.lower()

    if needle in name:
        return True

    if root is not None:
        try:
            rel = str(path.relative_to(root)).lower()
        except ValueError:
            rel = str(path).lower()

        return needle in rel

    return needle in str(path).lower()


def candidate_has_model_name(candidate_dir: Path, needle: str) -> bool:
    model_paths = list(candidate_dir.rglob("*.onnx")) + list(candidate_dir.rglob("*.tflite"))
    return any(model_name_matches(p, needle, root=candidate_dir) for p in model_paths)


def find_artifact(
    candidate_dir: Path,
    artifact: str,
    model_name_contains: Optional[str] = None,
) -> Optional[Path]:
    patterns = ARTIFACT_PATTERNS.get(artifact, [f"**/*{artifact}*"])
    found: List[Path] = []

    for pat in patterns:
        found.extend(candidate_dir.rglob(pat))

    found = [p for p in found if p.is_file()]

    if model_name_contains:
        found = [
            p for p in found
            if model_name_matches(p, model_name_contains, root=candidate_dir)
        ]

    if not found:
        return None

    def score(p: Path) -> Tuple[int, int, str]:
        name = p.name.lower()
        full = str(p).lower()

        s = 0

        if "best" in name or "final" in name:
            s += 20

        if "int8" in name and "int8" in artifact:
            s += 15

        if "fp16" in name and "fp16" in artifact:
            s += 15

        if "fp32" in name and "fp32" in artifact:
            s += 15

        if "float16" in name and "fp16" in artifact:
            s += 15

        if "float32" in name and "fp32" in artifact:
            s += 15

        if "saved_model" in full:
            s -= 5

        if "calib" in full or "representative" in full:
            s -= 20

        depth = len(p.relative_to(candidate_dir).parts)

        return s, -depth, str(p)

    found = sorted(found, key=score, reverse=True)
    return found[0]


def push_model(
    serial: Optional[str],
    local_model: Path,
    remote_dir: str,
    candidate: str,
    profile: str,
) -> str:
    adb(serial, ["shell", "mkdir", "-p", remote_dir], timeout=20)

    remote_name = safe_remote_name(f"{candidate}__{profile}__{local_model.name}")
    remote_path = f"{remote_dir.rstrip('/')}/{remote_name}"

    adb(serial, ["push", str(local_model), remote_path], timeout=600)
    return remote_path


def clear_logcat(serial: Optional[str]) -> None:
    adb(serial, ["logcat", "-c"], timeout=10, check=False)


def launch_app_bench(
    serial: Optional[str],
    bench_cfg: Dict[str, Any],
    profile: Dict[str, Any],
    remote_model: str,
) -> subprocess.CompletedProcess:
    package = str(bench_cfg.get("package", DEFAULT_PACKAGE))
    activity = str(bench_cfg.get("activity", DEFAULT_ACTIVITY))
    component = f"{package}/{activity}"

    backend = map_app_backend(profile)
    delegate = str(profile.get("delegate", profile.get("provider", "")) or "")
    provider = str(profile.get("provider", delegate) or "")

    cmd = [
        "shell",
        "am",
        "start",
        "-W",
        "-n",
        component,


        "--es", "model", remote_model,
        "--es", "model_path", remote_model,

        "--es", "backend", backend,
        "--es", "delegate", delegate,
        "--es", "provider", provider,

        "--ei", "imgsz", str(int(bench_cfg.get("imgsz", 640))),
        "--ei", "loops", str(int(bench_cfg.get("loops", 50))),
        "--ei", "warmup", str(int(bench_cfg.get("warmup", 10))),
        "--ei", "threads", str(int(bench_cfg.get("threads", 4))),
        "--ef", "conf", str(float(bench_cfg.get("conf", 0.25))),
        "--ef", "iou", str(float(bench_cfg.get("iou", 0.45))),
        "--ei", "max_det", str(int(bench_cfg.get("max_det", 100))),
    ]

    return adb(serial, cmd, timeout=60, check=False)


def wait_result_from_logcat(
    serial: Optional[str],
    result_tag: str,
    timeout_sec: int,
    poll_interval_sec: float,
) -> Tuple[Optional[Dict[str, Any]], str]:
    started = time.time()
    last_log = ""

    while time.time() - started < timeout_sec:
        p = adb(
            serial,
            ["logcat", "-d", "-s", result_tag],
            timeout=30,
            check=False,
        )

        log = (p.stdout or "") + "\n" + (p.stderr or "")
        last_log = log

        obj = find_json_in_text(log)
        if obj is not None:
            return obj, log

        time.sleep(poll_interval_sec)

    return None, last_log


def extract_samples(app_result: Dict[str, Any]) -> Tuple[List[float], str]:
    sample_keys = [
        "samples_ms",
        "latencies_ms",
        "times_ms",
        "raw_ms",
        "measurements_ms",
        "loop_ms",
    ]

    for key in sample_keys:
        value = app_result.get(key)

        if isinstance(value, list):
            vals: List[float] = []

            for x in value:
                try:
                    vals.append(float(x))
                except Exception:
                    pass

            if vals:
                return vals, key

    for key in ["avg_ms", "latency_ms", "mean_ms"]:
        if key in app_result:
            try:
                return [float(app_result[key])], key
            except Exception:
                pass

    return [], ""


def mean(xs: List[float]) -> float:
    return sum(xs) / len(xs)


def sample_std(xs: List[float]) -> float:
    if len(xs) < 2:
        return 0.0

    m = mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def ci95(xs: List[float]) -> Dict[str, float]:
    n = len(xs)

    if n == 0:
        return {
            "n": 0,
            "mean_ms": float("nan"),
            "std_ms": float("nan"),
            "ci95_half_ms": float("nan"),
            "ci95_low_ms": float("nan"),
            "ci95_high_ms": float("nan"),
        }

    m = mean(xs)
    sd = sample_std(xs)

    if n < 2:
        half = 0.0
    else:
        df = n - 1
        tcrit = T_CRIT_95.get(df, 1.96)
        half = tcrit * sd / math.sqrt(n)

    return {
        "n": n,
        "mean_ms": m,
        "std_ms": sd,
        "ci95_half_ms": half,
        "ci95_low_ms": m - half,
        "ci95_high_ms": m + half,
    }


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def write_summary_csv(history_path: Path, summary_path: Path) -> None:
    groups: Dict[Tuple[str, str, str], List[float]] = defaultdict(list)
    source_keys: Dict[Tuple[str, str, str], List[str]] = defaultdict(list)

    if not history_path.exists():
        return

    with history_path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            rec = json.loads(line)

            if rec.get("status") != "ok":
                continue

            key = (
                str(rec.get("candidate", "")),
                str(rec.get("profile", "")),
                str(rec.get("device_serial", "")),
            )

            samples = rec.get("samples_ms")

            if isinstance(samples, list) and samples:
                vals = [float(x) for x in samples]
                groups[key].extend(vals)
                source_keys[key].append(str(rec.get("samples_source", "samples_ms")))

            elif rec.get("avg_ms") is not None:
                groups[key].append(float(rec["avg_ms"]))
                source_keys[key].append("avg_ms")

    summary_path.parent.mkdir(parents=True, exist_ok=True)

    with summary_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "candidate",
                "profile",
                "device_serial",
                "n",
                "mean_ms",
                "std_ms",
                "ci95_half_ms",
                "ci95_low_ms",
                "ci95_high_ms",
                "samples_source",
            ],
        )

        writer.writeheader()

        for key in sorted(groups.keys()):
            candidate, profile, device_serial = key
            stats = ci95(groups[key])

            writer.writerow(
                {
                    "candidate": candidate,
                    "profile": profile,
                    "device_serial": device_serial,
                    "n": stats["n"],
                    "mean_ms": f"{stats['mean_ms']:.6f}",
                    "std_ms": f"{stats['std_ms']:.6f}",
                    "ci95_half_ms": f"{stats['ci95_half_ms']:.6f}",
                    "ci95_low_ms": f"{stats['ci95_low_ms']:.6f}",
                    "ci95_high_ms": f"{stats['ci95_high_ms']:.6f}",
                    "samples_source": "+".join(sorted(set(source_keys[key]))),
                }
            )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=(
            "Detached Android latency rebench for ready ONNX/TFLite models "
            "via com.example.testyolo/.CliBenchActivity"
        )
    )

    ap.add_argument("--config", required=True, type=Path, help="Pipeline YAML config")
    ap.add_argument("--models-root", required=True, type=Path, help="Folder with ready model artifacts")
    ap.add_argument("--out", required=True, type=Path, help="Output folder for history.jsonl and summary.csv")
    ap.add_argument("--serial", default=None, help="ADB device serial, e.g. 8379a6a3")
    ap.add_argument("--repeat", type=int, default=1, help="How many times to launch app per candidate/profile")
    ap.add_argument("--only-profile", default=None, help="Run only one profile name")
    ap.add_argument("--only-candidate", default=None, help="Run only candidate dir name substring")
    ap.add_argument(
        "--only-model",
        default=None,
        help="Run only model files whose file name or relative path contains this substring",
    )
    ap.add_argument("--timeout-sec", type=int, default=DEFAULT_TIMEOUT_SEC, help="Wait timeout for Android app result")
    ap.add_argument("--keep-remote", action="store_true", help="Do not delete remote pushed models after run")

    args = ap.parse_args()

    cfg = load_yaml(args.config)
    bench_cfg = get_bench_cfg(cfg, timeout_sec_override=args.timeout_sec)
    profiles = normalize_profiles(cfg)

    if args.only_profile:
        profiles = [p for p in profiles if str(p.get("name")) == args.only_profile]

    args.out.mkdir(parents=True, exist_ok=True)

    history_path = args.out / "history.jsonl"
    summary_path = args.out / "summary.csv"

    print(f"[config] {args.config}")
    print(f"[models_root] {args.models_root}")
    print(f"[out] {args.out}")
    print(f"[history] {history_path}")
    print(f"[summary] {summary_path}")

    print("[adb] checking device...")
    adb(args.serial, ["devices"], timeout=10, check=True)

    device_info = get_device_info(args.serial)
    print(f"[device] {device_info}")

    package = str(bench_cfg.get("package", DEFAULT_PACKAGE))
    result_tag = str(bench_cfg.get("result_tag", DEFAULT_RESULT_TAG))
    remote_dir = str(bench_cfg.get("remote_dir", DEFAULT_REMOTE_DIR))
    timeout_sec = int(bench_cfg.get("timeout_sec", DEFAULT_TIMEOUT_SEC))
    poll_interval_sec = float(bench_cfg.get("poll_interval_sec", 0.6))
    clear_before = bool(bench_cfg.get("clear_logcat", True))

    print(f"[timeout_sec] {timeout_sec}")

    candidate_dirs = discover_candidate_dirs(args.models_root)

    if args.only_candidate:
        candidate_dirs = [
            p for p in candidate_dirs
            if args.only_candidate.lower() in p.name.lower()
        ]

    if args.only_model:
        candidate_dirs = [
            p for p in candidate_dirs
            if candidate_has_model_name(p, args.only_model)
        ]

    if args.only_model:
        print(f"[only_model] {args.only_model}")

    print(f"[candidates] {len(candidate_dirs)}")
    for p in candidate_dirs:
        print(f"  - {p.name}: {p}")

    print(f"[profiles] {len(profiles)}")
    for p in profiles:
        print(
            f"  - {p.get('name')} "
            f"artifact={p.get('artifact')} "
            f"backend={p.get('backend')} "
            f"delegate={p.get('delegate')} "
            f"provider={p.get('provider')}"
        )

    for candidate_dir in candidate_dirs:
        candidate = candidate_dir.name

        for profile in profiles:
            profile_name = str(profile.get("name"))
            artifact = str(profile.get("artifact"))

            local_model = find_artifact(
                candidate_dir,
                artifact,
                model_name_contains=args.only_model,
            )

            if local_model is None:
                reason = f"artifact not found: {artifact}"

                if args.only_model:
                    reason += f" with model substring: {args.only_model}"

                rec = {
                    "ts": time.time(),
                    "status": "skip",
                    "reason": reason,
                    "candidate": candidate,
                    "candidate_dir": str(candidate_dir),
                    "profile": profile_name,
                    "artifact": artifact,
                    "device_serial": args.serial,
                    "device_info": device_info,
                }

                append_jsonl(history_path, rec)

                print(f"[skip] candidate={candidate} profile={profile_name} artifact={artifact}")
                continue

            print(f"\n[run] candidate={candidate} profile={profile_name}")
            print(f"[model] {local_model}")

            remote_model = push_model(
                args.serial,
                local_model,
                remote_dir,
                candidate,
                profile_name,
            )

            print(f"[push] {remote_model}")

            try:
                for repeat_idx in range(args.repeat):
                    print(f"[launch] repeat {repeat_idx + 1}/{args.repeat}")

                    if clear_before:
                        clear_logcat(args.serial)

                    adb(
                        args.serial,
                        ["shell", "am", "force-stop", package],
                        timeout=20,
                        check=False,
                    )

                    start_ts = time.time()

                    launch_proc = launch_app_bench(
                        args.serial,
                        bench_cfg,
                        profile,
                        remote_model,
                    )

                    app_result, raw_logcat = wait_result_from_logcat(
                        args.serial,
                        result_tag=result_tag,
                        timeout_sec=timeout_sec,
                        poll_interval_sec=poll_interval_sec,
                    )

                    end_ts = time.time()

                    if app_result is None:
                        rec = {
                            "ts": start_ts,
                            "status": "error",
                            "reason": "no JSON result in logcat",
                            "candidate": candidate,
                            "candidate_dir": str(candidate_dir),
                            "profile": profile_name,
                            "artifact": artifact,
                            "local_model": str(local_model),
                            "remote_model": remote_model,
                            "repeat_idx": repeat_idx,
                            "device_serial": args.serial,
                            "device_info": device_info,
                            "bench_cfg": bench_cfg,
                            "profile_cfg": profile,
                            "launch_stdout": launch_proc.stdout,
                            "launch_stderr": launch_proc.stderr,
                            "logcat_tail": raw_logcat[-4000:],
                            "duration_sec": end_ts - start_ts,
                            "timeout_sec": timeout_sec,
                        }

                        append_jsonl(history_path, rec)

                        print("[error] no JSON result in logcat")
                        continue

                    samples, samples_source = extract_samples(app_result)
                    stats = ci95(samples)

                    avg_ms = None

                    for key in ["avg_ms", "latency_ms", "mean_ms"]:
                        if key in app_result:
                            try:
                                avg_ms = float(app_result[key])
                                break
                            except Exception:
                                pass

                    if avg_ms is None and samples:
                        avg_ms = mean(samples)

                    rec = {
                        "ts": start_ts,
                        "status": "ok",
                        "candidate": candidate,
                        "candidate_dir": str(candidate_dir),
                        "profile": profile_name,
                        "artifact": artifact,
                        "local_model": str(local_model),
                        "remote_model": remote_model,
                        "repeat_idx": repeat_idx,
                        "device_serial": args.serial,
                        "device_info": device_info,
                        "bench_cfg": {
                            "package": bench_cfg.get("package"),
                            "activity": bench_cfg.get("activity"),
                            "result_tag": bench_cfg.get("result_tag"),
                            "remote_dir": bench_cfg.get("remote_dir"),
                            "imgsz": bench_cfg.get("imgsz"),
                            "loops": bench_cfg.get("loops"),
                            "warmup": bench_cfg.get("warmup"),
                            "threads": bench_cfg.get("threads"),
                            "conf": bench_cfg.get("conf"),
                            "iou": bench_cfg.get("iou"),
                            "max_det": bench_cfg.get("max_det"),
                            "timeout_sec": bench_cfg.get("timeout_sec"),
                        },
                        "profile_cfg": profile,
                        "app_result": app_result,
                        "samples_ms": samples,
                        "samples_source": samples_source,
                        "n_samples_in_record": len(samples),
                        "avg_ms": avg_ms,
                        "std_ms_in_record": stats["std_ms"],
                        "ci95_half_ms_in_record": stats["ci95_half_ms"],
                        "ci95_low_ms_in_record": stats["ci95_low_ms"],
                        "ci95_high_ms_in_record": stats["ci95_high_ms"],
                        "duration_sec": end_ts - start_ts,
                        "launch_stdout": launch_proc.stdout,
                        "launch_stderr": launch_proc.stderr,
                    }

                    append_jsonl(history_path, rec)

                    if samples and avg_ms is not None:
                        print(
                            f"[ok] avg={avg_ms:.3f} ms, "
                            f"n={len(samples)}, "
                            f"source={samples_source}, "
                            f"ci95=±{stats['ci95_half_ms']:.3f} ms"
                        )
                    else:
                        print(f"[ok] avg={avg_ms} ms, but no raw samples found")

            finally:
                if not args.keep_remote:
                    adb(
                        args.serial,
                        ["shell", "rm", "-f", remote_model],
                        timeout=20,
                        check=False,
                    )

            write_summary_csv(history_path, summary_path)

    write_summary_csv(history_path, summary_path)

    print("\n[done]")
    print(f"history: {history_path}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
