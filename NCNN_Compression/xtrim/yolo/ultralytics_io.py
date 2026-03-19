from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from onnx import StringStringEntryProto

from ..types import (
    CandidateConfig,
    ModelConfig,
    TrimConfig,
    Sparse1x1Config,
    OperatorChoiceConfig,
    GumbelChoiceConfig,
    ExportConfig,
    EvalConfig,
    TrainConfig,
    KDConfig,
    QATConfig,
    LowRankConfig,
    DilatedConfig,
)
from ..utils import ensure_dir
from ..trim.slim import structured_trim_yolo
from ..trim.lowrank import apply_lowrank_decomposition, recalibrate_bn
from ..trim.sparse_1x1 import apply_1x1_weight_sparsity, remove_pruning_reparam
from ..trim.operator_choice import apply_operator_plan, plan_from_config
from ..trim.gumbel_choice import (
    insert_mixed_ops,
    freeze_mixed_ops,
    count_mixed_ops,
)
from ..trim.dilated import apply_dilation
from .kd_finetune import finetune_with_kd
from types import SimpleNamespace


@dataclass
class UltralyticsStudent:
    yolo: Any
    torch_model: Any


def _torch_device_from_ultralytics_device(dev: str):
    import torch
    if dev == "cpu":
        return torch.device("cpu")
    if dev.isdigit():
        if torch.cuda.is_available():
            return torch.device(f"cuda:{dev}")
        return torch.device("cpu")
    if dev.startswith("cuda"):
        if torch.cuda.is_available():
            return torch.device(dev)
        return torch.device("cpu")
    return torch.device("cpu")


def warmstart_noop(student: UltralyticsStudent) -> None:
    return


def _build_recalib_loader(model_cfg: ModelConfig, batch: int = 4, workers: int = 2, model_stride: int = 32):
    """Build a minimal train data loader for BN recalibration."""
    try:
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset, build_dataloader
        from types import SimpleNamespace
    except Exception as e:
        raise RuntimeError(f"Cannot import ultralytics dataset utilities: {e}")

    data = check_det_dataset(model_cfg.data)
    cfg = get_cfg(overrides={
        "imgsz": int(model_cfg.imgsz),
        "task": "detect",
        "rect": False,
        "cache": False,
        "single_cls": False,
        "fraction": 1.0,
    })
    try:
        cfg.classes = None
    except Exception:
        pass

    img_path = data.get("train") or data.get("val")
    dataset = build_yolo_dataset(
        cfg,
        img_path,
        batch,
        data,
        mode="train",
        rect=False,
        stride=int(model_stride),
    )
    return build_dataloader(dataset, batch, workers, shuffle=True, rank=-1)


def build_ultralytics_candidate(
    cand: CandidateConfig,
    model_cfg: ModelConfig,
    trim_cfg: TrimConfig,
    op_choice_cfg: Optional[OperatorChoiceConfig] = None,
    op_choice_plan: Optional[dict] = None,
    gumbel_cfg: Optional[GumbelChoiceConfig] = None,
    lowrank_cfg: Optional[LowRankConfig] = None,
    dilated_cfg: Optional[DilatedConfig] = None,
) -> UltralyticsStudent:
    try:
        from ultralytics import YOLO
    except Exception as e:
        raise RuntimeError(f"ultralytics is required: pip install ultralytics\nImport error: {e}")

    import torch

    yolo = YOLO(model_cfg.weights)
    torch_model = yolo.model

    dev = _torch_device_from_ultralytics_device(str(model_cfg.device))
    torch_model.to(dev).eval()

    x = torch.randn(1, 3, int(model_cfg.imgsz), int(model_cfg.imgsz), device=dev)

    all_stats = {}

    width_mult = float(getattr(cand, "width_mult", 1.0))
    if width_mult < 1.0:
        width_prune_ratio = 1.0 - width_mult
        print(f"[build] Applying width_mult={width_mult} (uniform prune_ratio={width_prune_ratio:.2f})")
        strategy = str(getattr(trim_cfg, "strategy", "layerwise"))
        protect_last_n = int(getattr(trim_cfg, "protect_last_n", 0) or 0)

        try:
            width_stats = structured_trim_yolo(
                torch_model,
                example_input=x,
                prune_ratio=width_prune_ratio,
                channel_round=int(trim_cfg.channel_round),
                min_channels=int(trim_cfg.min_channels),
                exclude_head=bool(trim_cfg.exclude_head),
                exclude_name_regex=trim_cfg.exclude_name_regex,
                strategy=strategy,
                max_prune_per_layer=width_prune_ratio,
                protect_last_n=protect_last_n,
                verbose=True,
                importance_mode="uniform",
            )
            all_stats["width_scale"] = {
                "width_mult": width_mult,
                "layers_pruned": width_stats.get("layers_pruned", 0),
                "channels_pruned": width_stats.get("channels_pruned", 0),
            }
            print(f"[build] Width scaling: {width_stats.get('layers_pruned', 0)} layers, {width_stats.get('channels_pruned', 0)} channels removed")
        except Exception as e:
            print(f"[build] Warning: width_mult failed: {e}")
            all_stats["width_scale"] = {"error": str(e), "width_mult": width_mult}

    if cand.prune_ratio > 0.0:
        print(f"[build] Applying prune_ratio={cand.prune_ratio} (BN-gamma importance)")
        strategy = str(getattr(trim_cfg, "strategy", "layerwise"))
        max_prune_per_layer = getattr(trim_cfg, "max_prune_per_layer", None)
        protect_last_n = int(getattr(trim_cfg, "protect_last_n", 0) or 0)

        try:
            stats = structured_trim_yolo(
                torch_model,
                example_input=x,
                prune_ratio=float(cand.prune_ratio),
                channel_round=int(trim_cfg.channel_round),
                min_channels=int(trim_cfg.min_channels),
                exclude_head=bool(trim_cfg.exclude_head),
                exclude_name_regex=trim_cfg.exclude_name_regex,
                strategy=strategy,
                max_prune_per_layer=max_prune_per_layer,
                protect_last_n=protect_last_n,
                verbose=True,
                importance_mode="bn_gamma",
            )
            all_stats["bn_prune"] = stats
        except TypeError:
            stats = structured_trim_yolo(
                torch_model,
                example_input=x,
                prune_ratio=float(cand.prune_ratio),
                channel_round=int(trim_cfg.channel_round),
                min_channels=int(trim_cfg.min_channels),
                exclude_head=bool(trim_cfg.exclude_head),
                exclude_name_regex=trim_cfg.exclude_name_regex,
                verbose=True,
            )
            all_stats["bn_prune"] = stats

        with torch.no_grad():
            _ = torch_model(x)


        setattr(yolo, "_xtrim_trim_stats", stats)

    if dilated_cfg is not None and dilated_cfg.enabled:
        print(
            f"[build] Applying dilated conv: rates={dilated_cfg.rates}, "
            f"target_n_blocks={dilated_cfg.target_n_blocks}, "
            f"target_layers={dilated_cfg.target_layers or 'auto'}"
        )
        try:
            dilated_stats = apply_dilation(
                torch_model,
                rates=dilated_cfg.rates,
                exclude_head=bool(dilated_cfg.exclude_head),
                target_layers=dilated_cfg.target_layers,
                target_n_blocks=int(dilated_cfg.target_n_blocks),
                verbose=True,
            )
            all_stats["dilated"] = dilated_stats
            print(f"[build] Dilated: {dilated_stats['layers_modified']} layers modified")
        except Exception as e:
            print(f"[build] Warning: dilated conv failed: {e}")
            all_stats["dilated"] = {"error": str(e)}

    lowrank_rank = int(getattr(cand, "lowrank_rank", 0))
    lr_cfg = lowrank_cfg
    lr_enabled = (lr_cfg is not None and lr_cfg.enabled) and lowrank_rank > 0
    if lr_enabled:
        lr_exclude_head = bool(lr_cfg.exclude_head) if lr_cfg else bool(trim_cfg.exclude_head)
        lr_exclude_stem = bool(lr_cfg.exclude_stem) if lr_cfg else True
        lr_energy_thresh = float(lr_cfg.energy_threshold) if lr_cfg else 0.0
        lr_min_channels = int(lr_cfg.min_channels) if lr_cfg else 32
        lr_max_layers = int(lr_cfg.max_layers) if lr_cfg else 0
        lr_exclude_regex = trim_cfg.exclude_name_regex

        effective_rank = 0 if lr_energy_thresh > 0.0 else lowrank_rank

        lr_include_1x1 = bool(lr_cfg.lowrank_1x1) if lr_cfg is not None else False
        print(
            f"[build] Applying low-rank decomposition: "
            f"{'energy_threshold=' + str(lr_energy_thresh) if lr_energy_thresh > 0 else 'rank=' + str(effective_rank)}, "
            f"exclude_stem={lr_exclude_stem}, max_layers={lr_max_layers}, include_1x1={lr_include_1x1}"
        )
        try:
            lowrank_stats = apply_lowrank_decomposition(
                torch_model,
                rank=effective_rank,
                energy_threshold=lr_energy_thresh,
                exclude_head=lr_exclude_head,
                exclude_stem=lr_exclude_stem,
                exclude_name_regex=lr_exclude_regex,
                min_channels=lr_min_channels,
                max_layers=lr_max_layers,
                include_1x1=lr_include_1x1,
                verbose=True,
            )
            all_stats["lowrank"] = lowrank_stats

            if lowrank_stats["layers_decomposed"] > 0:
                print(
                    f"[build] Low-rank: {lowrank_stats['layers_decomposed']} layers decomposed, "
                    f"compression {lowrank_stats.get('compression_ratio', 1.0):.2f}x"
                )

                n_recalib = int(lr_cfg.bn_recalib_batches) if lr_cfg else 0
                if n_recalib > 0:
                    print(f"[build] BN recalibration: {n_recalib} batches...")
                    try:
                        stride = int(torch_model.stride.max()) if hasattr(torch_model, "stride") else 32
                        recalib_loader = _build_recalib_loader(
                            model_cfg, batch=4, workers=2, model_stride=stride
                        )
                        recalib_stats = recalibrate_bn(
                            torch_model,
                            recalib_loader,
                            n_batches=n_recalib,
                            device=dev,
                            verbose=True,
                        )
                        all_stats["bn_recalib"] = recalib_stats
                    except Exception as e:
                        print(f"[build] Warning: BN recalibration failed: {e}")
                        all_stats["bn_recalib"] = {"error": str(e)}

        except Exception as e:
            print(f"[build] Warning: lowrank failed: {e}")
            all_stats["lowrank"] = {"error": str(e)}

    sparse_amount = float(getattr(cand, "sparse_1x1", 0.0))
    if sparse_amount > 0.0:
        print(f"[build] Applying 1x1 weight sparsity={sparse_amount:.0%}")
        try:
            sparse_stats = apply_1x1_weight_sparsity(
                torch_model,
                sparsity=sparse_amount,
                method="l1",
                exclude_head=bool(trim_cfg.exclude_head),
                exclude_name_regex=trim_cfg.exclude_name_regex,
                min_channels=16,
                verbose=True,
            )
            all_stats["sparse_1x1"] = sparse_stats
        except Exception as e:
            print(f"[build] Warning: sparse_1x1 failed: {e}")
            all_stats["sparse_1x1"] = {"error": str(e)}

        n_reparam = remove_pruning_reparam(torch_model, verbose=False)
        if n_reparam > 0:
            print(f"[build] Removed pruning reparametrization from {n_reparam} layers")

    if (op_choice_cfg is not None and op_choice_cfg.enabled
            and gumbel_cfg is not None and gumbel_cfg.enabled):
        raise ValueError(
            "[build] operator_choice and gumbel_choice cannot both be enabled — "
            "they target the same 1x1 layers. Disable one in config.yaml."
        )

    if op_choice_cfg is not None and op_choice_cfg.enabled:
        print(f"[build] Applying operator_choice (default={op_choice_cfg.default})")
        try:
            plan = plan_from_config(
                torch_model,
                cfg_plan=op_choice_plan or {},
                default=op_choice_cfg.default,
                exclude_head=bool(trim_cfg.exclude_head),
                exclude_name_regex=trim_cfg.exclude_name_regex,
                min_channels=int(op_choice_cfg.min_channels),
            )
            op_stats = apply_operator_plan(
                torch_model,
                plan,
                sparse_sparsity=float(op_choice_cfg.sparse_sparsity),
                sparse_method=str(op_choice_cfg.sparse_method),
                lowrank_rank=int(op_choice_cfg.lowrank_rank),
                verbose=True,
            )
            all_stats["operator_choice"] = op_stats
            print(
                f"[build] operator_choice: dense={op_stats['dense']}, "
                f"sparse={op_stats['sparse']}, lowrank={op_stats['lowrank']}"
            )
        except Exception as e:
            print(f"[build] Warning: operator_choice failed: {e}")
            all_stats["operator_choice"] = {"error": str(e)}

    if gumbel_cfg is not None and gumbel_cfg.enabled:
        print(
            f"[build] Inserting MixedOp1x1 (gumbel_choice, "
            f"rank={gumbel_cfg.lowrank_rank}, sparsity={gumbel_cfg.sparse_sparsity})"
        )
        try:
            gumbel_stats = insert_mixed_ops(
                torch_model,
                lowrank_rank=int(gumbel_cfg.lowrank_rank),
                sparsity=float(gumbel_cfg.sparse_sparsity),
                hard=bool(gumbel_cfg.hard),
                exclude_head=bool(trim_cfg.exclude_head),
                exclude_name_regex=trim_cfg.exclude_name_regex,
                min_channels=int(gumbel_cfg.min_channels),
                verbose=True,
            )
            all_stats["gumbel_choice"] = gumbel_stats
        except Exception as e:
            print(f"[build] Warning: gumbel_choice insertion failed: {e}")
            all_stats["gumbel_choice"] = {"error": str(e)}

    with torch.no_grad():
        _ = torch_model(x)

    yolo.model = torch_model
    if hasattr(yolo, "_model"):
        yolo._model = torch_model

    setattr(yolo, "_xtrim_all_stats", all_stats)

    return UltralyticsStudent(yolo=yolo, torch_model=torch_model)


def finetune_noop(student: UltralyticsStudent, train_cfg: TrainConfig) -> None:
    return


def finetune_kd(
    student: UltralyticsStudent,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    trim_cfg: TrimConfig,
    kd_cfg: KDConfig,
    lut_cfg: Optional["LatencyLUTConfig"] = None,
    gumbel_cfg: Optional[GumbelChoiceConfig] = None,
) -> dict:
    if not kd_cfg.enabled:
        return {}

    from ultralytics import YOLO

    teacher_w = getattr(kd_cfg, "teacher", None) or model_cfg.weights
    teacher = YOLO(teacher_w)
    teacher_model = teacher.model

    dev = next(student.torch_model.parameters()).device
    teacher_model.to(dev).eval()

    logs = finetune_with_kd(
        student_torch_model=student.torch_model,
        teacher_torch_model=teacher_model,
        model_cfg=model_cfg,
        trim_cfg=trim_cfg,
        train_cfg=train_cfg,
        kd_cfg=kd_cfg,
        lut_cfg=lut_cfg,
        gumbel_cfg=gumbel_cfg,
        enable_fake_quant=False,
    )

    n_mixed = count_mixed_ops(student.torch_model)
    if n_mixed > 0:
        print(f"[kd] Freezing {n_mixed} MixedOp1x1 layers after KD training...")
        freeze_stats = freeze_mixed_ops(student.torch_model, verbose=True)
        logs["gumbel_freeze"] = freeze_stats

    setattr(student.yolo, "_xtrim_kd_logs", logs)
    return logs


def finetune_qat_recover(
    student: UltralyticsStudent,
    train_cfg: TrainConfig,
    model_cfg: ModelConfig,
    trim_cfg: TrimConfig,
    kd_cfg: KDConfig,
    qat_cfg: QATConfig,
) -> dict:
    if not qat_cfg.enabled:
        return {}

    from ultralytics import YOLO

    teacher_w = getattr(kd_cfg, "teacher", None) or model_cfg.weights
    teacher = YOLO(teacher_w)
    teacher_model = teacher.model

    dev = next(student.torch_model.parameters()).device
    teacher_model.to(dev).eval()

    bits_w = int(getattr(qat_cfg, "bits_w", 8))
    bits_a = int(getattr(qat_cfg, "bits_a", 8))
    max_batches = int(getattr(qat_cfg, "max_train_batches", 0) or 0)

    logs = finetune_with_kd(
        student_torch_model=student.torch_model,
        teacher_torch_model=teacher_model,
        model_cfg=model_cfg,
        trim_cfg=trim_cfg,
        train_cfg=train_cfg,
        kd_cfg=kd_cfg,
        enable_fake_quant=True,
        fq_bits_w=bits_w,
        fq_bits_a=bits_a,
        override_epochs=int(qat_cfg.epochs),
        override_lr=float(qat_cfg.lr),
        override_max_batches=max_batches,
    )
    setattr(student.yolo, "_xtrim_qat_logs", logs)
    return logs


def _val_kwargs(model_cfg: ModelConfig, eval_cfg: EvalConfig) -> dict:
    return dict(
        data=model_cfg.data,
        imgsz=int(model_cfg.imgsz),
        device=str(model_cfg.device),
        task=str(model_cfg.task),
        split=str(getattr(eval_cfg, "split", "val")),
        conf=float(eval_cfg.conf),
        iou=float(eval_cfg.iou),
        max_det=int(eval_cfg.max_det),
        half=bool(getattr(eval_cfg, "half", False)),
        dnn=bool(getattr(eval_cfg, "dnn", False)),
        rect=bool(getattr(eval_cfg, "rect", False)),
        batch=int(getattr(eval_cfg, "batch", 1)),
        workers=int(getattr(eval_cfg, "workers", 8)),
        verbose=bool(getattr(eval_cfg, "verbose", False)),
        plots=bool(getattr(eval_cfg, "plots", False)),
        augment=bool(getattr(eval_cfg, "augment", False)),
        agnostic_nms=bool(getattr(eval_cfg, "agnostic_nms", False)),
    )


def _extract_map5095(res: Any) -> float:
    if isinstance(res, dict):
        for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "map", "box/map"):
            if k in res:
                try:
                    return float(res[k])
                except Exception as e:
                    print(e)
                    pass

    if isinstance(res, SimpleNamespace):
        for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "box/map"):
            if hasattr(res, k):
                try:
                    return float(getattr(res, k))
                except Exception as e:
                    print(e)
                    pass

    if hasattr(res, "results_dict") and isinstance(getattr(res, "results_dict"), dict):
        d = res.results_dict
        for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95"):
            if k in d:
                try:
                    return float(d[k])
                except Exception as e:
                    print(e)
                    pass

    try:
        if hasattr(res, "box") and res.box is not None and hasattr(res.box, "map"):
            return float(res.box.map)
    except Exception as e:
        print(e)
        pass

    if hasattr(res, "maps"):
        try:
            maps = getattr(res, "maps")
            if isinstance(maps, (list, tuple)) and len(maps) > 0:
                return float(sum(maps) / len(maps))
        except Exception as e:
            print(e)
            pass

    return 0.0


def eval_ultralytics_map(student: UltralyticsStudent, model_cfg: ModelConfig, eval_cfg: EvalConfig) -> float:
    res = student.yolo.val(**_val_kwargs(model_cfg, eval_cfg))
    return _extract_map5095(res)


def eval_exported_onnx_map(onnx_path: Path, model_cfg: ModelConfig, eval_cfg: EvalConfig) -> float:
    from ultralytics import YOLO
    y = YOLO(str(onnx_path))
    res = y.val(**_val_kwargs(model_cfg, eval_cfg))
    return _extract_map5095(res)


def make_ultralytics_export_onnx_fn(
    student: "UltralyticsStudent",
    model_cfg: ModelConfig,
    export_cfg: ExportConfig,
) -> Callable[[Path], None]:
    """
    Экспорт ONNX без yolo.export(), но ВАЖНО:
    - форсим legacy exporter (dynamo=False), чтобы не падать на torch.export
      на некоторых pruned-моделях.
    """

    def _export(onnx_path: Path) -> None:
        m = student.torch_model
        m.eval()

        dev = next(m.parameters()).device
        imgsz = int(model_cfg.imgsz)
        dummy = torch.zeros(1, 3, imgsz, imgsz, device=dev)

        ensure_dir(onnx_path.parent)
        if onnx_path.exists():
            onnx_path.unlink()

        try:
            torch.onnx.export(
                m,
                (dummy,),
                str(onnx_path),
                opset_version=int(export_cfg.opset),
                input_names=["images"],
                output_names=["output0"],
                dynamic_axes=None,
                do_constant_folding=True,
                dynamo=False,
            )
        except TypeError:
            torch.onnx.export(
                m,
                (dummy,),
                str(onnx_path),
                opset_version=int(export_cfg.opset),
                input_names=["images"],
                output_names=["output0"],
                dynamic_axes=None,
                do_constant_folding=True,
            )

        try:
            import onnx
            from onnxsim import simplify
            model = onnx.load(str(onnx_path))
            model_simp, ok = simplify(model)
            if ok:
                onnx.save(model_simp, str(onnx_path))
        except Exception as e:
            print(e)
            pass

        _inject_ultralytics_metadata(onnx_path, student, model_cfg)

    return _export


def _make_kv(key: str, value: str) -> StringStringEntryProto:
    try:
        return onnx.helper.make_string_string_entry(key, value)  # type: ignore[attr-defined]
    except Exception:
        e = StringStringEntryProto()
        e.key = str(key)
        e.value = str(value)
        return e


def _inject_ultralytics_metadata(onnx_path: Path, student: Any, model_cfg: Any) -> None:
    try:
        import onnx
    except Exception as e:
        print(e)
        return

    m = onnx.load(str(onnx_path))

    stride = 32
    try:
        s = getattr(student.torch_model, "stride", None)
        if isinstance(s, (list, tuple)):
            stride = int(max(s))
        elif torch.is_tensor(s):
            stride = int(s.max().item())
        elif s is not None:
            stride = int(s)
    except Exception as e:
        stride = 32
        print(e)

    names = None
    try:
        if hasattr(student, "yolo") and hasattr(student.yolo, "names"):
            names = student.yolo.names
        elif hasattr(student.torch_model, "names"):
            names = student.torch_model.names
    except Exception as e:
        names = None
        print(e)

    if isinstance(names, (list, tuple)):
        names = {str(i): str(n) for i, n in enumerate(names)}
    elif isinstance(names, dict):
        names = {str(k): str(v) for k, v in names.items()}
    else:
        names = None

    def set_meta(k: str, v: str) -> None:
        for p in m.metadata_props:
            if p.key == k:
                p.value = str(v)
                return
        m.metadata_props.append(_make_kv(k, str(v)))

    set_meta("task", "detect")
    set_meta("stride", str(stride))
    set_meta("imgsz", str(int(getattr(model_cfg, "imgsz", 640))))
    set_meta("batch", "1")
    if names is not None:
        set_meta("names", json.dumps(names, ensure_ascii=False))

    onnx.save(m, str(onnx_path))


def save_student_torchscript(student: UltralyticsStudent, pt_path: Path) -> None:
    student.torch_model.eval()
    scripted = torch.jit.script(student.torch_model)
    ensure_dir(pt_path.parent)
    scripted.save(str(pt_path))