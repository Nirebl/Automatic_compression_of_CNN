from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional

import torch
from onnx import StringStringEntryProto

from .detect_head_pruning import (
    shrink_detect_heads,
    shrink_detect_heads_to_targets,
    collect_detect_head_hidden_channels,
)
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
from ..trim.slim import structured_trim_yolo, collect_prunable_out_channels
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
from .pruning_adapters import replace_c2f_with_prunable, shrink_psa_family_blocks
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
    """Создает минимальный train dataloader для пересчета BatchNorm."""
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


def _normalize_optional_regex(v, field_name: str):
    if v is None or v is False or v == "":
        return None
    if isinstance(v, str):
        s = v.strip()
        return s if s else None
    raise TypeError(f"{field_name} must be null/false or a regex string, got {type(v).__name__}")


def extract_ultralytics_pruning_architecture(
    student: UltralyticsStudent,
    trim_cfg: TrimConfig,
) -> dict:
    """Описывает ширины каналов, важные для сравнения архитектур после прунинга."""
    torch_model = student.torch_model
    exclude_name_regex = _normalize_optional_regex(
        getattr(trim_cfg, "exclude_name_regex", None), "trim.exclude_name_regex"
    )
    include_inner_m_regex = _normalize_optional_regex(
        getattr(trim_cfg, "include_inner_m_regex", None), "trim.include_inner_m_regex"
    )
    conv_out_channels = collect_prunable_out_channels(
        torch_model,
        exclude_head=bool(trim_cfg.exclude_head),
        exclude_name_regex=exclude_name_regex,
        skip_inner_m=bool(getattr(trim_cfg, "skip_inner_m", True)),
        skip_cv1_if_parent_has_m=bool(getattr(trim_cfg, "skip_cv1_if_parent_has_m", True)),
        include_inner_m_regex=include_inner_m_regex,
    )
    return {
        "conv_out_channels": {str(k): int(v) for k, v in conv_out_channels.items()},
        "detect_hidden_channels": {
            str(k): int(v) for k, v in collect_detect_head_hidden_channels(torch_model).items()
        },
        "total_params": int(sum(p.numel() for p in torch_model.parameters())),
    }


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

    def _normalize_exclude_name_regex(v):
        if v is None or v is False or v == "":
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        raise TypeError(
            f"trim.exclude_name_regex must be null/false or a regex string, got {type(v).__name__}"
        )

    def _normalize_include_inner_m_regex(v):
        if v is None or v is False or v == "":
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        raise TypeError(
            f"trim.include_inner_m_regex must be null/false or a regex string, got {type(v).__name__}"
        )

    def _count_params(model) -> int:
        return sum(p.numel() for p in model.parameters())

    def _count_trainable_params(model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _top_param_modules(model, topk: int = 20):
        rows = []
        for name, module in model.named_modules():
            own_params = sum(p.numel() for p in module.parameters(recurse=False))
            if own_params > 0:
                rows.append((name, module.__class__.__name__, own_params))
        rows.sort(key=lambda x: x[2], reverse=True)
        return rows[:topk]

    def _print_param_snapshot(model, label: str, topk: int = 20):
        total = _count_params(model)
        trainable = _count_trainable_params(model)
        print(f"[params] {label}: total={total:,} trainable={trainable:,}")

        top_rows = _top_param_modules(model, topk=topk)
        if top_rows:
            print(f"[params] {label}: top-{len(top_rows)} modules by own parameter count")
            for i, (name, cls_name, n) in enumerate(top_rows, 1):
                mod_name = name if name else "<root>"
                print(f"  [{i:02d}] {mod_name:<40} {cls_name:<20} {n:,}")

        return {
            "label": label,
            "total_params": total,
            "trainable_params": trainable,
            "top_modules": [
                {"name": name if name else "<root>", "type": cls_name, "params": int(n)}
                for name, cls_name, n in top_rows
            ],
        }

    yolo = YOLO(model_cfg.weights)
    torch_model = yolo.model

    adapt_c2f_for_pruning = bool(getattr(trim_cfg, "adapt_c2f_for_pruning", False))
    will_prune = (
            float(getattr(cand, "width_mult", 1.0)) < 1.0
            or float(getattr(cand, "prune_ratio", 0.0)) > 0.0
    )

    if adapt_c2f_for_pruning and will_prune:
        replaced = replace_c2f_with_prunable(torch_model, verbose=True)
        print(f"[adapt] total C2f blocks replaced: {replaced}")

    dev = _torch_device_from_ultralytics_device(str(model_cfg.device))
    torch_model.to(dev).eval()

    x = torch.randn(1, 3, int(model_cfg.imgsz), int(model_cfg.imgsz), device=dev)

    all_stats = {}

    exclude_name_regex = _normalize_exclude_name_regex(
        getattr(trim_cfg, "exclude_name_regex", None)
    )
    include_inner_m_regex = _normalize_include_inner_m_regex(
        getattr(trim_cfg, "include_inner_m_regex", None)
    )
    skip_inner_m = bool(getattr(trim_cfg, "skip_inner_m", True))
    skip_cv1_if_parent_has_m = bool(getattr(trim_cfg, "skip_cv1_if_parent_has_m", True))
    def _apply_composite_psa_pruning(stage_label: str, prune_ratio_value: float):
        round_to = max(int(getattr(trim_cfg, "channel_round", 8) or 8), 8)
        min_ch = max(int(getattr(trim_cfg, "min_channels", 8) or 8), 8)
        stats = shrink_psa_family_blocks(
            torch_model,
            prune_ratio=float(prune_ratio_value),
            round_to=round_to,
            min_channels=min_ch,
            verbose=True,
        )
        all_stats[f"{stage_label}_composite_psa"] = stats
        if stats.get("blocks_shrunk", 0):
            print(f"[build] Composite PSA prune ({stage_label}): {stats['blocks_shrunk']} blocks shrunk")
            all_stats[f"params_after_{stage_label}_composite_psa"] = _print_param_snapshot(
                torch_model, f"after_{stage_label}_composite_psa", topk=20
            )
        return stats

    def _apply_detect_head_pruning(stage_label: str, prune_ratio_value: float):
        if bool(getattr(trim_cfg, "exclude_head", True)):
            stats = {
                "heads_shrunk": 0,
                "channels_pruned": 0,
                "skipped": "trim.exclude_head=true",
            }
            all_stats[f"{stage_label}_detect_head"] = stats
            print(f"[build] Detect head prune ({stage_label}): skipped because trim.exclude_head=true")
            return stats

        stats = shrink_detect_heads(
            torch_model,
            prune_ratio=float(prune_ratio_value),
            round_to=max(int(getattr(trim_cfg, "channel_round", 8) or 8), 8),
            min_channels=max(int(getattr(trim_cfg, "min_channels", 8) or 8), 8),
            verbose=True,
        )
        all_stats[f"{stage_label}_detect_head"] = stats

        if stats.get("heads_shrunk", 0):
            print(f"[build] Detect head prune ({stage_label}): {stats['heads_shrunk']} heads shrunk")
            all_stats[f"params_after_{stage_label}_detect_head"] = _print_param_snapshot(
                torch_model, f"after_{stage_label}_detect_head", topk=20
            )

        return stats


    all_stats["trim_cfg_effective"] = {
        "exclude_head": bool(trim_cfg.exclude_head),
        "exclude_name_regex": exclude_name_regex,
        "strategy": str(getattr(trim_cfg, "strategy", "layerwise")),
        "max_prune_per_layer": getattr(trim_cfg, "max_prune_per_layer", None),
        "protect_last_n": int(getattr(trim_cfg, "protect_last_n", 0) or 0),
        "skip_inner_m": skip_inner_m,
        "skip_cv1_if_parent_has_m": skip_cv1_if_parent_has_m,
        "include_inner_m_regex": include_inner_m_regex,
        "adapt_c2f_for_pruning": adapt_c2f_for_pruning,
    }

    all_stats["params_initial"] = _print_param_snapshot(torch_model, "initial", topk=20)
    params_before_any = all_stats["params_initial"]["total_params"]

    width_mult = float(getattr(cand, "width_mult", 1.0))
    if width_mult < 1.0:
        width_prune_ratio = 1.0 - width_mult
        print(f"[build] Applying width_mult={width_mult} (uniform prune_ratio={width_prune_ratio:.2f})")
        strategy = str(getattr(trim_cfg, "strategy", "layerwise"))
        protect_last_n = int(getattr(trim_cfg, "protect_last_n", 0) or 0)

        params_before_width = _count_params(torch_model)

        width_stats = structured_trim_yolo(
            torch_model,
            example_input=x,
            prune_ratio=width_prune_ratio,
            channel_round=int(trim_cfg.channel_round),
            min_channels=int(trim_cfg.min_channels),
            exclude_head=bool(trim_cfg.exclude_head),
            exclude_name_regex=exclude_name_regex,
            strategy=strategy,
            max_prune_per_layer=width_prune_ratio,
            protect_last_n=protect_last_n,
            verbose=True,
            importance_mode="uniform",
            skip_inner_m=skip_inner_m,
            skip_cv1_if_parent_has_m=skip_cv1_if_parent_has_m,
            include_inner_m_regex=include_inner_m_regex,
        )

        params_after_width = _count_params(torch_model)
        width_comp = params_before_width / max(1, params_after_width)

        all_stats["width_scale"] = {
            "width_mult": width_mult,
            "prune_ratio_equivalent": width_prune_ratio,
            "layers_pruned": width_stats.get("layers_pruned", 0),
            "channels_pruned": width_stats.get("channels_pruned", 0),
            "params_before": int(params_before_width),
            "params_after": int(params_after_width),
            "compression_ratio": float(width_comp),
            "raw_stats": width_stats,
        }

        print(
            f"[build] Width scaling: "
            f"{width_stats.get('layers_pruned', 0)} layers, "
            f"{width_stats.get('channels_pruned', 0)} channels removed, "
            f"params {params_before_width:,} -> {params_after_width:,} "
            f"({width_comp:.2f}x)"
        )

        _apply_composite_psa_pruning("width", width_prune_ratio)

        all_stats["params_after_width"] = _print_param_snapshot(
            torch_model, "after_width_mult", topk=20
        )

    if cand.prune_ratio > 0.0:
        print(f"[build] Applying prune_ratio={cand.prune_ratio} (BN-gamma importance)")
        strategy = str(getattr(trim_cfg, "strategy", "layerwise"))
        max_prune_per_layer = getattr(trim_cfg, "max_prune_per_layer", None)
        protect_last_n = int(getattr(trim_cfg, "protect_last_n", 0) or 0)

        params_before_bn = _count_params(torch_model)

        stats = structured_trim_yolo(
            torch_model,
            example_input=x,
            prune_ratio=float(cand.prune_ratio),
            channel_round=int(trim_cfg.channel_round),
            min_channels=int(trim_cfg.min_channels),
            exclude_head=bool(trim_cfg.exclude_head),
            exclude_name_regex=exclude_name_regex,
            strategy=strategy,
            max_prune_per_layer=max_prune_per_layer,
            protect_last_n=protect_last_n,
            verbose=True,
            importance_mode="bn_gamma",
            skip_inner_m=skip_inner_m,
            skip_cv1_if_parent_has_m=skip_cv1_if_parent_has_m,
            include_inner_m_regex=include_inner_m_regex,
        )

        params_after_bn = _count_params(torch_model)
        bn_comp = params_before_bn / max(1, params_after_bn)
        total_comp = params_before_any / max(1, params_after_bn)

        stats = dict(stats or {})
        stats["params_before"] = int(params_before_bn)
        stats["params_after"] = int(params_after_bn)
        stats["compression_ratio_stage"] = float(bn_comp)
        stats["compression_ratio_total"] = float(total_comp)

        all_stats["bn_prune"] = stats

        print(
            f"[build] BN prune: "
            f"params {params_before_bn:,} -> {params_after_bn:,} "
            f"(stage {bn_comp:.2f}x, total {total_comp:.2f}x)"
        )

        _apply_composite_psa_pruning("bn", float(cand.prune_ratio))
        _apply_detect_head_pruning("bn", float(cand.prune_ratio))

        with torch.no_grad():
            _ = torch_model(x)

        with torch.no_grad():
            _ = torch_model(x)

        all_stats["params_after_bn_prune"] = _print_param_snapshot(
            torch_model, "after_prune_ratio", topk=20
        )

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
        lr_exclude_regex = exclude_name_regex

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
                exclude_name_regex=exclude_name_regex,
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
                exclude_name_regex=exclude_name_regex,
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
                exclude_name_regex=exclude_name_regex,
                min_channels=int(gumbel_cfg.min_channels),
                verbose=True,
            )
            all_stats["gumbel_choice"] = gumbel_stats
        except Exception as e:
            print(f"[build] Warning: gumbel_choice insertion failed: {e}")
            all_stats["gumbel_choice"] = {"error": str(e)}

    with torch.no_grad():
        _ = torch_model(x)

    all_stats["params_final"] = _print_param_snapshot(torch_model, "final", topk=20)

    yolo.model = torch_model
    if hasattr(yolo, "_model"):
        yolo._model = torch_model

    setattr(yolo, "_xtrim_all_stats", all_stats)

    return UltralyticsStudent(yolo=yolo, torch_model=torch_model)


def apply_ultralytics_pruning_stage(
    student: UltralyticsStudent,
    *,
    stage_prune_ratio: float,
    model_cfg: ModelConfig,
    trim_cfg: TrimConfig,
    stage_label: str,
    target_architecture: Optional[dict] = None,
) -> dict:
    """Выполняет один дополнительный шаг BN-gamma прунинга для существующей student-модели.

    stage_prune_ratio считается относительно текущей модели, а не относительно исходной.
    """
    import torch

    if not (0.0 < float(stage_prune_ratio) < 1.0):
        raise ValueError(f"stage_prune_ratio must be in (0, 1), got {stage_prune_ratio}")

    def _normalize_regex(v, field_name: str):
        if v is None or v is False or v == "":
            return None
        if isinstance(v, str):
            s = v.strip()
            return s if s else None
        raise TypeError(f"{field_name} must be null/false or a regex string, got {type(v).__name__}")

    def _count_params(model) -> int:
        return sum(p.numel() for p in model.parameters())

    def _count_trainable_params(model) -> int:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    def _top_param_modules(model, topk: int = 20):
        rows = []
        for name, module in model.named_modules():
            own_params = sum(p.numel() for p in module.parameters(recurse=False))
            if own_params > 0:
                rows.append((name, module.__class__.__name__, own_params))
        rows.sort(key=lambda x: x[2], reverse=True)
        return rows[:topk]

    def _param_snapshot(model, label: str, topk: int = 20) -> dict:
        total = _count_params(model)
        trainable = _count_trainable_params(model)
        print(f"[params] {label}: total={total:,} trainable={trainable:,}")
        top_rows = _top_param_modules(model, topk=topk)
        return {
            "label": label,
            "total_params": int(total),
            "trainable_params": int(trainable),
            "top_modules": [
                {"name": name if name else "<root>", "type": cls_name, "params": int(n)}
                for name, cls_name, n in top_rows
            ],
        }

    torch_model = student.torch_model
    torch_model.eval()
    dev = next(torch_model.parameters()).device
    x = torch.randn(1, 3, int(model_cfg.imgsz), int(model_cfg.imgsz), device=dev)

    all_stats = dict(getattr(student.yolo, "_xtrim_all_stats", {}) or {})
    exclude_name_regex = _normalize_regex(
        getattr(trim_cfg, "exclude_name_regex", None), "trim.exclude_name_regex"
    )
    include_inner_m_regex = _normalize_regex(
        getattr(trim_cfg, "include_inner_m_regex", None), "trim.include_inner_m_regex"
    )
    skip_inner_m = bool(getattr(trim_cfg, "skip_inner_m", True))
    skip_cv1_if_parent_has_m = bool(getattr(trim_cfg, "skip_cv1_if_parent_has_m", True))
    strategy = str(getattr(trim_cfg, "strategy", "layerwise"))
    max_prune_per_layer = getattr(trim_cfg, "max_prune_per_layer", None)
    protect_last_n = int(getattr(trim_cfg, "protect_last_n", 0) or 0)

    params_initial = int(
        (all_stats.get("params_initial") or {}).get("total_params", _count_params(torch_model))
    )
    params_before = _count_params(torch_model)

    target_architecture = dict(target_architecture or {})
    target_out_channels = dict(target_architecture.get("conv_out_channels", {}) or {})
    target_detect_hidden = dict(target_architecture.get("detect_hidden_channels", {}) or {})

    print(
        f"[staged] Applying {stage_label}: local prune_ratio={float(stage_prune_ratio):.6f} "
        f"(BN-gamma importance, target_arch={bool(target_architecture)})"
    )
    stats = structured_trim_yolo(
        torch_model,
        example_input=x,
        prune_ratio=float(stage_prune_ratio),
        channel_round=int(trim_cfg.channel_round),
        min_channels=int(trim_cfg.min_channels),
        exclude_head=bool(trim_cfg.exclude_head),
        exclude_name_regex=exclude_name_regex,
        strategy=strategy,
        max_prune_per_layer=max_prune_per_layer,
        protect_last_n=protect_last_n,
        verbose=True,
        importance_mode="bn_gamma",
        skip_inner_m=skip_inner_m,
        skip_cv1_if_parent_has_m=skip_cv1_if_parent_has_m,
        include_inner_m_regex=include_inner_m_regex,
        target_out_channels=target_out_channels or None,
    )

    params_after = _count_params(torch_model)
    stage_comp = params_before / max(1, params_after)
    total_comp = params_initial / max(1, params_after)

    stats = dict(stats or {})
    stats.update(
        {
            "params_before": int(params_before),
            "params_after": int(params_after),
            "compression_ratio_stage": float(stage_comp),
            "compression_ratio_total": float(total_comp),
            "target_architecture_applied": bool(target_architecture),
        }
    )
    all_stats[f"{stage_label}_bn_prune"] = stats

    round_to = max(int(getattr(trim_cfg, "channel_round", 8) or 8), 8)
    min_channels = max(int(getattr(trim_cfg, "min_channels", 8) or 8), 8)
    psa_stats = shrink_psa_family_blocks(
        torch_model,
        prune_ratio=float(stage_prune_ratio),
        round_to=round_to,
        min_channels=min_channels,
        verbose=True,
    )
    all_stats[f"{stage_label}_composite_psa"] = psa_stats

    if bool(getattr(trim_cfg, "exclude_head", True)):
        detect_stats = {
            "heads_shrunk": 0,
            "channels_pruned": 0,
            "skipped": "trim.exclude_head=true",
        }
        print(f"[staged] Detect head prune ({stage_label}): skipped because trim.exclude_head=true")
    elif target_detect_hidden:
        detect_stats = shrink_detect_heads_to_targets(
            torch_model,
            target_hidden_channels=target_detect_hidden,
            round_to=round_to,
            min_channels=min_channels,
            verbose=True,
        )
    else:
        detect_stats = shrink_detect_heads(
            torch_model,
            prune_ratio=float(stage_prune_ratio),
            round_to=round_to,
            min_channels=min_channels,
            verbose=True,
        )
    all_stats[f"{stage_label}_detect_head"] = detect_stats

    with torch.no_grad():
        _ = torch_model(x)

    all_stats[f"params_after_{stage_label}_bn_prune"] = _param_snapshot(
        torch_model, f"after_{stage_label}", topk=20
    )
    all_stats["params_final"] = _param_snapshot(torch_model, "final", topk=20)

    if target_architecture:
        achieved_arch = extract_ultralytics_pruning_architecture(student, trim_cfg)
        conv_target = dict(target_architecture.get("conv_out_channels", {}) or {})
        detect_target = dict(target_architecture.get("detect_hidden_channels", {}) or {})
        conv_actual = dict(achieved_arch.get("conv_out_channels", {}) or {})
        detect_actual = dict(achieved_arch.get("detect_hidden_channels", {}) or {})
        conv_mismatches = {
            k: {"target": int(v), "actual": int(conv_actual.get(k, -1))}
            for k, v in conv_target.items()
            if int(conv_actual.get(k, -1)) != int(v)
        }
        detect_mismatches = {
            k: {"target": int(v), "actual": int(detect_actual.get(k, -1))}
            for k, v in detect_target.items()
            if int(detect_actual.get(k, -1)) != int(v)
        }
        stats["target_match"] = {
            "conv_mismatch_count": int(len(conv_mismatches)),
            "detect_mismatch_count": int(len(detect_mismatches)),
            "conv_mismatches": conv_mismatches,
            "detect_mismatches": detect_mismatches,
            "achieved_total_params": int(achieved_arch.get("total_params", 0)),
            "target_total_params": int(target_architecture.get("total_params", 0) or 0),
        }

    setattr(student.yolo, "_xtrim_trim_stats", stats)
    setattr(student.yolo, "_xtrim_all_stats", all_stats)
    return stats


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



def _try_float(value: Any) -> Optional[float]:
    try:
        if value is None:
            return None
        return float(value)
    except Exception:
        return None


def _extract_from_result_dict(res: Any, keys: tuple[str, ...]) -> Optional[float]:
    dicts = []
    if isinstance(res, dict):
        dicts.append(res)
    if isinstance(res, SimpleNamespace):
        dicts.append(res.__dict__)
    if hasattr(res, "results_dict") and isinstance(getattr(res, "results_dict"), dict):
        dicts.append(getattr(res, "results_dict"))

    for d in dicts:
        for k in keys:
            if k in d:
                value = _try_float(d[k])
                if value is not None:
                    return value
    return None


def _extract_box_attr(res: Any, attrs: tuple[str, ...]) -> Optional[float]:
    box = getattr(res, "box", None)
    if box is None:
        return None
    for attr in attrs:
        if hasattr(box, attr):
            value = _try_float(getattr(box, attr))
            if value is not None:
                return value
    return None


def _extract_detection_metrics(res: Any) -> dict[str, float]:
    """Достает метрики детекции из результата Ultralytics val().

    map50_95 используется как основная метрика качества. Precision, recall, IoU и map50 добавляются только если evaluator их вернул.
    """
    metrics: dict[str, float] = {}

    map50_95 = _try_float(_extract_map5095(res))
    if map50_95 is not None:
        metrics["map50_95"] = map50_95

    precision = _extract_from_result_dict(
        res,
        ("metrics/precision(B)", "metrics/precision", "box/mp", "precision", "mp"),
    )
    if precision is None:
        precision = _extract_box_attr(res, ("mp", "precision"))
    if precision is not None:
        metrics["precision"] = precision

    recall = _extract_from_result_dict(
        res,
        ("metrics/recall(B)", "metrics/recall", "box/mr", "recall", "mr"),
    )
    if recall is None:
        recall = _extract_box_attr(res, ("mr", "recall"))
    if recall is not None:
        metrics["recall"] = recall

    iou = _extract_from_result_dict(
        res,
        (
            "metrics/IoU(B)",
            "metrics/IoU",
            "metrics/iou(B)",
            "metrics/iou",
            "box/iou",
            "iou",
            "mean_iou",
            "miou",
            "mIoU",
        ),
    )
    if iou is None:
        iou = _extract_box_attr(res, ("iou", "mean_iou", "miou"))
    if iou is not None:
        metrics["iou"] = iou

    map50 = _extract_from_result_dict(
        res,
        ("metrics/mAP50(B)", "metrics/mAP50", "box/map50", "map50"),
    )
    if map50 is None:
        map50 = _extract_box_attr(res, ("map50",))
    if map50 is not None:
        metrics["map50"] = map50

    return metrics

def _make_eval_yolo_copy(student: UltralyticsStudent, model_cfg: ModelConfig):
    """Создает временную копию YOLO-модели для валидации.

    Это защищает исходную student-модель от изменений, которые Ultralytics может внести во время val().
    """
    from ultralytics import YOLO

    eval_yolo = YOLO(model_cfg.weights)
    eval_model = deepcopy(student.torch_model)
    eval_yolo.model = eval_model
    if hasattr(eval_yolo, "_model"):
        eval_yolo._model = eval_model
    try:
        eval_yolo.predictor = None
    except Exception:
        pass
    return eval_yolo


def eval_ultralytics_metrics(student: UltralyticsStudent, model_cfg: ModelConfig, eval_cfg: EvalConfig) -> dict[str, float]:
    eval_yolo = _make_eval_yolo_copy(student, model_cfg)
    res = eval_yolo.val(**_val_kwargs(model_cfg, eval_cfg))
    return _extract_detection_metrics(res)


def eval_ultralytics_map(student: UltralyticsStudent, model_cfg: ModelConfig, eval_cfg: EvalConfig) -> float:
    return float(eval_ultralytics_metrics(student, model_cfg, eval_cfg).get("map50_95", 0.0))


def eval_exported_onnx_metrics(onnx_path: Path, model_cfg: ModelConfig, eval_cfg: EvalConfig) -> dict[str, float]:
    from ultralytics import YOLO
    y = YOLO(str(onnx_path))
    res = y.val(**_val_kwargs(model_cfg, eval_cfg))
    return _extract_detection_metrics(res)


def eval_exported_onnx_map(onnx_path: Path, model_cfg: ModelConfig, eval_cfg: EvalConfig) -> float:
    return float(eval_exported_onnx_metrics(onnx_path, model_cfg, eval_cfg).get("map50_95", 0.0))


def make_ultralytics_export_onnx_fn(
    student: "UltralyticsStudent",
    model_cfg: ModelConfig,
    export_cfg: ExportConfig,
) -> Callable[[Path], None]:
    """Создает функцию экспорта YOLO-модели в ONNX.

    Используется legacy exporter, потому что он стабильнее работает с pruned-моделями.
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



def _find_tflite_file(root: Path) -> Optional[Path]:
    files = sorted(root.rglob("*.tflite"), key=lambda p: (len(p.parts), str(p)))
    return files[0] if files else None


def _find_tflite_file_by_priority(root: Path, priority_words: tuple[str, ...]) -> Optional[Path]:
    """Ищет TFLite-файл по строгому порядку приоритета.

    Это важно, потому что конвертер может создать несколько похожих .tflite-файлов, но для INT8 нужен именно проверенный вариант.
    """
    files = sorted(root.rglob("*.tflite"), key=lambda p: (len(p.parts), str(p)))
    if not files:
        return None

    for word in priority_words:
        needle = str(word).lower()
        for path in files:
            if needle in path.name.lower():
                return path

    return files[0]


def make_ultralytics_export_tflite_fp32_fn(
    student: "UltralyticsStudent",
    model_cfg: ModelConfig,
    export_cfg: ExportConfig,
) -> Callable[[Path], Path]:
    """Создает функцию экспорта текущей модели в FP32 TFLite.

    Этот путь используется для исходной baseline-модели без INT8 и FP16 преобразований.
    """

    def _export(tflite_path: Path) -> Path:
        import os
        import shutil
        import tempfile

        from ultralytics import YOLO

        ensure_dir(tflite_path.parent)
        if tflite_path.exists():
            tflite_path.unlink()

        old_cwd = Path.cwd()
        export_device = str(getattr(export_cfg, "tflite_int8_device", "cpu") or "cpu")
        simplify = bool(getattr(export_cfg, "tflite_int8_simplify", False))

        with tempfile.TemporaryDirectory(prefix="xtrim_tflite_fp32_", dir=str(tflite_path.parent)) as tmp:
            tmp_dir = Path(tmp)
            tmp_pt = tmp_dir / "xtrim_tflite_export_fp32.pt"

            try:
                from ..quant.fake_quant_ultra import unpatch_ultralytics_convs_for_fake_quant

                restored = unpatch_ultralytics_convs_for_fake_quant(student.torch_model)
                if restored:
                    print(f"[tflite] restored fake-quant patched Conv.forward in {restored} modules before FP32 export")
            except Exception as exc:
                print(f"[tflite] warning: could not restore fake-quant Conv.forward hooks before FP32 export: {exc}")

            try:
                student.yolo.save(str(tmp_pt))
            except Exception as exc:
                raise RuntimeError(f"Could not save temporary YOLO checkpoint for TFLite FP32 export: {exc}") from exc
            if not tmp_pt.exists():
                raise RuntimeError(f"Temporary YOLO checkpoint was not created: {tmp_pt}")

            export_yolo = YOLO(str(tmp_pt))
            try:
                os.chdir(tmp_dir)
                result = export_yolo.export(
                    format="tflite",
                    imgsz=int(model_cfg.imgsz),
                    int8=False,
                    half=False,
                    data=str(model_cfg.data),
                    batch=1,
                    device=export_device,
                    nms=False,
                    simplify=simplify,
                )
            finally:
                os.chdir(old_cwd)

            def _looks_fp32(path: Path) -> bool:
                name = path.name.lower()
                blocked = ("int8", "quant", "integer", "fp16", "float16", "half")
                return path.suffix.lower() == ".tflite" and not any(x in name for x in blocked)

            files = sorted(tmp_dir.rglob("*.tflite"), key=lambda p: (len(p.parts), str(p)))
            src = next((f for f in files if _looks_fp32(f)), None)

            if src is None and result is not None:
                rp = Path(str(result))
                if not rp.is_absolute():
                    rp = tmp_dir / rp
                if rp.is_file() and _looks_fp32(rp):
                    src = rp
                elif rp.is_dir():
                    nested = sorted(rp.rglob("*.tflite"), key=lambda p: (len(p.parts), str(p)))
                    src = next((f for f in nested if _looks_fp32(f)), None)

            if src is None:
                src = _find_tflite_file(tmp_dir)

            if src is None or not src.exists():
                raise RuntimeError(
                    "Ultralytics TFLite FP32 export finished but no .tflite file was found "
                    f"under temporary export directory: {tmp_dir}"
                )

            shutil.copy2(src, tflite_path)

        if not tflite_path.exists():
            raise RuntimeError(f"TFLite FP32 export did not create file: {tflite_path}")
        return tflite_path

    return _export


def make_ultralytics_export_tflite_int8_fn(
    student: "UltralyticsStudent",
    model_cfg: ModelConfig,
    export_cfg: ExportConfig,
) -> Callable[[Path], Path]:
    """Создает функцию экспорта текущей модели в INT8 TFLite.

    Экспорт идет из временного .pt checkpoint внутри папки кандидата. Это не дает Ultralytics искать файлы рядом с исходным checkpoint и делает результат воспроизводимее.
    """

    def _export(tflite_path: Path) -> Path:
        import os
        import shutil
        import tempfile

        from ultralytics import YOLO

        ensure_dir(tflite_path.parent)
        if tflite_path.exists():
            tflite_path.unlink()

        old_cwd = Path.cwd()
        export_device = str(getattr(export_cfg, "tflite_int8_device", "cpu") or "cpu")
        simplify = bool(getattr(export_cfg, "tflite_int8_simplify", False))

        with tempfile.TemporaryDirectory(prefix="xtrim_tflite_int8_", dir=str(tflite_path.parent)) as tmp:
            tmp_dir = Path(tmp)
            tmp_pt = tmp_dir / "xtrim_tflite_export.pt"

            try:
                from ..quant.fake_quant_ultra import unpatch_ultralytics_convs_for_fake_quant

                restored = unpatch_ultralytics_convs_for_fake_quant(student.torch_model)
                if restored:
                    print(f"[tflite] restored fake-quant patched Conv.forward in {restored} modules before export")
            except Exception as exc:
                print(f"[tflite] warning: could not restore fake-quant Conv.forward hooks before export: {exc}")

            try:
                student.yolo.save(str(tmp_pt))
            except Exception as exc:
                raise RuntimeError(f"Could not save temporary YOLO checkpoint for TFLite export: {exc}") from exc
            if not tmp_pt.exists():
                raise RuntimeError(f"Temporary YOLO checkpoint was not created: {tmp_pt}")

            export_yolo = YOLO(str(tmp_pt))
            try:
                os.chdir(tmp_dir)
                result = export_yolo.export(
                    format="tflite",
                    imgsz=int(model_cfg.imgsz),
                    int8=True,
                    data=str(model_cfg.data),
                    batch=1,
                    device=export_device,
                    nms=False,
                    simplify=simplify,
                )
            finally:
                os.chdir(old_cwd)

            int8_priority = (
                "_int8.tflite",
                "int8",
                "full_integer_quant",
                "integer_quant",
                "quant",
            )
            src = _find_tflite_file_by_priority(tmp_dir, int8_priority)

            if src is None and result is not None:
                rp = Path(str(result))
                if not rp.is_absolute():
                    rp = tmp_dir / rp
                if rp.is_file() and rp.suffix.lower() == ".tflite":
                    src = rp
                elif rp.is_dir():
                    src = _find_tflite_file_by_priority(rp, int8_priority) or _find_tflite_file(rp)

            if src is None or not src.exists():
                raise RuntimeError(
                    "Ultralytics TFLite INT8 export finished but no .tflite file was found "
                    f"under temporary export directory: {tmp_dir}"
                )
            shutil.copy2(src, tflite_path)

            if bool(getattr(export_cfg, "tflite_fp16", False)):
                fp16_name = str(getattr(export_cfg, "tflite_fp16_name", "model_fp16.tflite") or "model_fp16.tflite")
                fp16_path = tflite_path.parent / fp16_name
                fp16_src = _find_tflite_file_by_priority(tmp_dir, ("float16", "fp16"))
                if fp16_src is not None and fp16_src.exists():
                    if fp16_path.exists():
                        fp16_path.unlink()
                    shutil.copy2(fp16_src, fp16_path)
                elif bool(getattr(export_cfg, "tflite_fp16_required", False)):
                    raise RuntimeError("TFLite FP16 export was requested but no float16 .tflite file was produced")

        if not tflite_path.exists():
            raise RuntimeError(f"TFLite INT8 export did not create file: {tflite_path}")
        return tflite_path

    return _export

def save_student_torchscript(student: UltralyticsStudent, pt_path: Path) -> None:
    student.torch_model.eval()
    try:
        scripted = torch.jit.script(student.torch_model)
    except (OSError, RuntimeError):
        first_param = next(student.torch_model.parameters(), None)
        dev = first_param.device if first_param is not None else torch.device("cpu")
        dt = first_param.dtype if first_param is not None else torch.float32
        dummy = torch.randn(1, 3, 640, 640, device=dev, dtype=dt)
        scripted = torch.jit.trace(student.torch_model, dummy, strict=False)
    ensure_dir(pt_path.parent)
    scripted.save(str(pt_path))