from __future__ import annotations

import argparse
import json
from pathlib import Path

from xtrim.config import load_yaml, parse_config
from xtrim.orchestrator import XTrimOrchestrator
from xtrim.pareto import avg_latency
from xtrim.results_table import print_results_table
from xtrim.quant.calib import make_calib_imagelist
from xtrim.quant.ort_ptq import ort_static_quantize_yolo
from xtrim.yolo.ultralytics_io import (
    build_ultralytics_candidate,
    eval_ultralytics_map,
    eval_exported_onnx_map,
    make_ultralytics_export_onnx_fn,
    save_student_torchscript,
    warmstart_noop,
    finetune_noop,
    finetune_kd,
    finetune_qat_recover,
)


def _copy_onnx_metadata(src: Path, dst: Path) -> None:
    import onnx
    srcm = onnx.load(str(src))
    dstm = onnx.load(str(dst))
    # очистить и скопировать
    del dstm.metadata_props[:]
    for p in srcm.metadata_props:
        dstm.metadata_props.append(p)
    onnx.save(dstm, str(dst))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True, help="Path to YAML config")
    ap.add_argument("--out", type=str, default="outputs/xtrim", help="Output root")
    ap.add_argument("--max_candidates", type=int, default=10)
    args = ap.parse_args()

    cfg = load_yaml(Path(args.config))
    (
        tools,
        devices,
        train_cfg,
        export_cfg,
        ptq_cfg,
        search_space,
        model_cfg,
        trim_cfg,
        latency_cfg,
        eval_cfg,
        kd_cfg,
        onnx_ptq_cfg,
        qat_cfg,
        android_demo_cfg,
        search_cfg,
        android_app_bench_cfg,
        ort_android_bench_cfg,
        op_choice_cfg,
        op_choice_plan,
        lut_cfg,
        gumbel_cfg,
        lowrank_cfg,
        dilated_cfg,
    ) = parse_config(cfg)

    def build_fn(cand):
        return build_ultralytics_candidate(
            cand, model_cfg, trim_cfg,
            op_choice_cfg=op_choice_cfg,
            op_choice_plan=op_choice_plan,
            gumbel_cfg=gumbel_cfg,
            lowrank_cfg=lowrank_cfg,
            dilated_cfg=dilated_cfg,
        )

    def export_factory(student, export_cfg_):
        return make_ultralytics_export_onnx_fn(student, model_cfg, export_cfg_)

    def eval_student(student):
        return eval_ultralytics_map(student, model_cfg, eval_cfg)

    def eval_onnx(onnx_path: Path) -> float:
        return eval_exported_onnx_map(onnx_path, model_cfg, eval_cfg)

    if bool(train_cfg.kd_enabled) and bool(kd_cfg.enabled):
        def finetune_fn(student, _train_cfg):
            return finetune_kd(
                student, train_cfg, model_cfg, trim_cfg, kd_cfg,
                lut_cfg=lut_cfg, gumbel_cfg=gumbel_cfg,
            )
    else:
        finetune_fn = finetune_noop

    if bool(qat_cfg.enabled) and bool(kd_cfg.enabled):
        def finetune_qat_fn(student, _train_cfg):
            return finetune_qat_recover(student, train_cfg, model_cfg, trim_cfg, kd_cfg, qat_cfg)
    else:
        finetune_qat_fn = None

    def quantize_onnx_fn(onnx_fp32: Path, onnx_int8: Path, run_dir: Path) -> Path:
        calib_txt = run_dir / "ptq" / "calib_images.txt"
        make_calib_imagelist(
            data_yaml=model_cfg.data,
            split=str(onnx_ptq_cfg.calib_split),
            max_images=int(onnx_ptq_cfg.calib_max_images),
            out_txt=calib_txt,
            seed=int(train_cfg.seed),
        )

        # 1) делаем INT8 ONNX
        out_path = ort_static_quantize_yolo(
            onnx_fp32=onnx_fp32,
            onnx_int8=onnx_int8,
            image_list_txt=calib_txt,
            imgsz=int(model_cfg.imgsz),
            per_channel=bool(onnx_ptq_cfg.per_channel),
            quant_format=str(onnx_ptq_cfg.quant_format),
            activation_type=str(onnx_ptq_cfg.activation_type),
            weight_type=str(onnx_ptq_cfg.weight_type),
            calibrate_method=str(onnx_ptq_cfg.calibrate_method),
        )

        # 2) копируем metadata (stride/task/names/imgsz) из FP32 -> INT8
        try:
            _copy_onnx_metadata(onnx_fp32, out_path)
        except Exception:
            pass

        return out_path

    orch = XTrimOrchestrator(
        out_root=Path(args.out),
        tools=tools,
        devices=devices,
        train_cfg=train_cfg,
        export_cfg=export_cfg,
        ptq_cfg=ptq_cfg,
        latency_cfg=latency_cfg,
        onnx_ptq_cfg=onnx_ptq_cfg,
        qat_cfg=qat_cfg,
        android_demo_cfg=android_demo_cfg,
        search_cfg=search_cfg,
        search_space=search_space,
        build_candidate_fn=build_fn,
        warmstart_fn=warmstart_noop,
        finetune_fn=finetune_fn,
        finetune_qat_fn=finetune_qat_fn,
        eval_acc_fn=eval_student,
        export_onnx_fn_factory=export_factory,
        eval_exported_onnx_fn=eval_onnx,
        quantize_onnx_fn=quantize_onnx_fn,
        save_student_pt_fn=save_student_torchscript,
        android_app_bench_cfg=android_app_bench_cfg,
        ort_android_bench_cfg=ort_android_bench_cfg,
    )

    history = orch.run(max_candidates=args.max_candidates)

    print_results_table(history, title="FINAL RESULTS")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())