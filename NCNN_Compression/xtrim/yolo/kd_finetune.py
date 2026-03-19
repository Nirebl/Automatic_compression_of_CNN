from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, cast

import torch
import torch.nn as nn
import torch.nn.functional as F

from ultralytics.cfg import IterableSimpleNamespace

from ..types import ModelConfig, TrimConfig, TrainConfig, KDConfig, LatencyLUTConfig, GumbelChoiceConfig
from ..trim.slim import bn_sparsity_regularizer
from ..quant.fake_quant_ultra import (
    patch_ultralytics_convs_for_fake_quant,
    set_fake_quant_enabled,
    set_fake_quant_bits,
)


@dataclass
class HookStore:
    feats: List[torch.Tensor]
    head: List[torch.Tensor]


def _as_tensor_list(x: Any) -> List[torch.Tensor]:
    out: List[torch.Tensor] = []
    if isinstance(x, torch.Tensor):
        out.append(x)
    elif isinstance(x, (list, tuple)):
        for t in x:
            out.extend(_as_tensor_list(t))
    elif isinstance(x, dict):
        for v in x.values():
            out.extend(_as_tensor_list(v))
    return out


def _attn_map(feat: torch.Tensor) -> torch.Tensor:
    if feat.dim() == 4:
        a = (feat.float() ** 2).mean(dim=1, keepdim=True)
        n = torch.sqrt((a ** 2).sum(dim=(2, 3), keepdim=True) + 1e-9)
        return a / n
    a = feat.float().view(feat.size(0), 1, 1, -1)
    n = torch.sqrt((a ** 2).sum(dim=(2, 3), keepdim=True) + 1e-9)
    return a / n


def _mse_resize(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    if a.shape[-2:] != b.shape[-2:]:
        b = F.interpolate(b, size=a.shape[-2:], mode="bilinear", align_corners=False)
    return F.mse_loss(a, b)


def _select_feature_modules(model: nn.Module, num_layers: int) -> List[nn.Module]:
    candidates: List[nn.Module] = []
    for _, m in model.named_modules():
        name = m.__class__.__name__.lower()
        if any(k in name for k in ["c2f", "c3", "bottleneck", "sppf"]):
            candidates.append(m)

    uniq: List[nn.Module] = []
    seen = set()
    for m in candidates:
        if id(m) in seen:
            continue
        seen.add(id(m))
        uniq.append(m)

    if not uniq:
        for _, m in model.named_modules():
            if isinstance(m, nn.Conv2d):
                uniq.append(m)
        uniq = uniq[-max(1, num_layers):]

    return uniq[-max(1, num_layers):]


def _find_detect_module(model: nn.Module) -> nn.Module | None:
    for _, m in model.named_modules():
        if m.__class__.__name__.lower() == "detect":
            return m
    return None


class _HookMgr:
    def __init__(self, modules_feat: List[nn.Module], module_head: nn.Module | None):
        self.modules_feat = modules_feat
        self.module_head = module_head
        self.handles = []
        self.store = HookStore(feats=[], head=[])

    def clear(self):
        self.store.feats.clear()
        self.store.head.clear()

    def _feat_hook(self, _m, _inp, out):
        ts = _as_tensor_list(out)
        if ts:
            self.store.feats.append(max(ts, key=lambda t: t.numel()))

    def _head_hook(self, _m, _inp, out):
        ts = _as_tensor_list(out)
        if ts:
            ts = sorted(ts, key=lambda t: t.numel(), reverse=True)[:3]
            self.store.head.extend(ts)

    def register(self):
        for m in self.modules_feat:
            self.handles.append(m.register_forward_hook(self._feat_hook))
        if self.module_head is not None:
            self.handles.append(self.module_head.register_forward_hook(self._head_hook))

    def remove(self):
        for h in self.handles:
            try:
                h.remove()
            except Exception:
                pass
        self.handles.clear()


def _build_train_loader_ultralytics(data_yaml: str, imgsz: int, batch: int, workers: int, model_stride: int):
    try:
        from ultralytics.data.utils import check_det_dataset
        from ultralytics.cfg import get_cfg
        from ultralytics.data.build import build_yolo_dataset, build_dataloader
    except Exception as e:
        raise RuntimeError(f"Cannot import ultralytics dataset utilities: {e}")

    data = check_det_dataset(data_yaml)

    cfg = cast(IterableSimpleNamespace, get_cfg(overrides={
        "imgsz": int(imgsz),
        "task": "detect",
        "rect": False,
        "cache": False,
        "single_cls": False,
        "fraction": 1.0,
    }))

    try:
        cfg.classes = None
    except Exception:
        pass

    img_path = data["train"]

    dataset = build_yolo_dataset(
        cfg,
        img_path,
        int(batch),
        data,
        mode="train",
        rect=False,
        stride=int(model_stride),
    )

    loader = build_dataloader(dataset, int(batch), int(workers), shuffle=True, rank=-1)
    return loader


def _build_detection_loss(student_model: nn.Module):
    try:
        from ultralytics.utils.loss import v8DetectionLoss
    except Exception as e:
        raise RuntimeError(f"Cannot import ultralytics.utils.loss.v8DetectionLoss: {e}")

    try:
        from ultralytics.cfg import get_cfg
    except Exception:
        get_cfg = None

    args = getattr(student_model, "args", None)

    if isinstance(args, dict):
        if get_cfg is not None:
            student_model.args = get_cfg(overrides=args)
        else:
            from types import SimpleNamespace
            student_model.args = SimpleNamespace(**args)
    else:
        need = ("box", "cls", "dfl")
        if (args is None) or (not all(hasattr(args, k) for k in need)):
            if get_cfg is not None:
                cfg = get_cfg()
                if hasattr(args, "__dict__"):
                    for k, v in args.__dict__.items():
                        setattr(cfg, k, v)
                student_model.args = cfg
            else:
                from types import SimpleNamespace
                base = {"box": 7.5, "cls": 0.5, "dfl": 1.5}
                if hasattr(args, "__dict__"):
                    base.update(args.__dict__)
                student_model.args = SimpleNamespace(**base)

    return v8DetectionLoss(student_model)


def _scalarize(x: Any, device: torch.device) -> torch.Tensor:
    if isinstance(x, dict):
        xs = [v for v in x.values() if torch.is_tensor(v)]
        x = sum(xs) if xs else torch.tensor(0.0, device=device)
    elif isinstance(x, (list, tuple)):
        xs = [v for v in x if torch.is_tensor(v)]
        x = sum(xs) if xs else torch.tensor(0.0, device=device)
    elif not torch.is_tensor(x):
        x = torch.tensor(float(x), device=device)

    if x.dim() != 0:
        x = x.mean()
    return x


def _to_regular_tensor(t: torch.Tensor) -> torch.Tensor:
    try:
        if hasattr(t, "is_inference") and t.is_inference():
            cpu = t.detach().to("cpu").contiguous()
            return torch.tensor(cpu.numpy(), device=t.device, dtype=t.dtype)
    except Exception:
        pass
    return t


def _deinference_model(model: nn.Module) -> None:
    for m in model.modules():
        for k, p in list(m._parameters.items()):
            if p is None:
                continue
            try:
                is_inf = hasattr(p, "is_inference") and p.is_inference()
            except Exception:
                is_inf = False
            if is_inf:
                t2 = _to_regular_tensor(p.detach())
                m._parameters[k] = nn.Parameter(t2, requires_grad=True)

        for k, b in list(m._buffers.items()):
            if b is None:
                continue
            try:
                is_inf = hasattr(b, "is_inference") and b.is_inference()
            except Exception:
                is_inf = False
            if is_inf:
                m._buffers[k] = _to_regular_tensor(b.detach())


def finetune_with_kd(
    *,
    student_torch_model: nn.Module,
    teacher_torch_model: nn.Module,
    model_cfg: ModelConfig,
    trim_cfg: TrimConfig,
    train_cfg: TrainConfig,
    kd_cfg: KDConfig,
    lut_cfg: Optional[LatencyLUTConfig] = None,
    gumbel_cfg: Optional[GumbelChoiceConfig] = None,
    enable_fake_quant: bool = False,
    fq_bits_w: int = 8,
    fq_bits_a: int = 8,
    override_epochs: int | None = None,
    override_lr: float | None = None,
    override_max_batches: int | None = None,
) -> Dict[str, float]:
    device = next(student_torch_model.parameters()).device

    student_torch_model.train()
    teacher_torch_model.eval()

    _deinference_model(student_torch_model)

    for p in student_torch_model.parameters():
        p.requires_grad_(True)

    torch.set_grad_enabled(True)

    if not any(p.requires_grad for p in student_torch_model.parameters()):
        raise RuntimeError("Student model has no trainable parameters (all requires_grad=False).")

    if enable_fake_quant:
        patch_ultralytics_convs_for_fake_quant(student_torch_model)
        set_fake_quant_bits(student_torch_model, bits_w=fq_bits_w, bits_a=fq_bits_a)
        set_fake_quant_enabled(student_torch_model, True)
    else:
        set_fake_quant_enabled(student_torch_model, False)

    feat_modules_s = _select_feature_modules(student_torch_model, kd_cfg.num_feature_layers)
    feat_modules_t = _select_feature_modules(teacher_torch_model, kd_cfg.num_feature_layers)
    head_s = _find_detect_module(student_torch_model)
    head_t = _find_detect_module(teacher_torch_model)

    hs = _HookMgr(feat_modules_s, head_s)
    ht = _HookMgr(feat_modules_t, head_t)
    hs.register()
    ht.register()

    stride = 32
    try:
        if hasattr(student_torch_model, "stride"):
            s = getattr(student_torch_model, "stride")
            if isinstance(s, (list, tuple)):
                stride = int(max(s))
            elif torch.is_tensor(s):
                stride = int(s.max().item())
            else:
                stride = int(s)
    except Exception:
        pass

    loader = _build_train_loader_ultralytics(
        data_yaml=model_cfg.data,
        imgsz=int(model_cfg.imgsz),
        batch=int(kd_cfg.batch),
        workers=int(kd_cfg.workers),
        model_stride=stride,
    )
    criterion = _build_detection_loss(student_torch_model)

    lr = float(override_lr) if override_lr is not None else float(train_cfg.lr)
    opt = torch.optim.AdamW(student_torch_model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))

    epochs = int(override_epochs) if override_epochs is not None else max(1, int(train_cfg.short_epochs))
    max_batches = int(override_max_batches) if override_max_batches is not None else int(kd_cfg.max_train_batches)

    _lat_penalty_value: float = 0.0
    _lat_est_ms: float = 0.0
    _lut_obj = None
    _log_every = 50

    if lut_cfg is not None and lut_cfg.enabled:
        try:
            from ..latency_lut import LatencyLUT, estimate_model_latency, latency_penalty
            _lut_obj = LatencyLUT(lut_cfg.lut_path, verbose=True)
            _input_shape = (1, 3, int(model_cfg.imgsz), int(model_cfg.imgsz))
            lut_result = estimate_model_latency(
                student_torch_model, _lut_obj,
                input_shape=_input_shape,
                macs_per_ms=float(lut_cfg.macs_per_ms),
                verbose=False,
            )
            _lat_est_ms = lut_result["latency_est_ms"]
            _lat_penalty_value = latency_penalty(
                _lat_est_ms, float(lut_cfg.budget_ms), float(lut_cfg.lambda_lat)
            )
            _log_every = int(lut_cfg.log_every_n_batches)
            print(
                f"[kd] LUT latency: est={_lat_est_ms:.1f} ms, "
                f"budget={lut_cfg.budget_ms} ms, "
                f"penalty={_lat_penalty_value:.4f} "
                f"(hits={lut_result['lut_hits']}, fallback={lut_result['lut_misses']})"
            )
        except Exception as e:
            print(f"[kd] Warning: LUT setup failed: {e}")

    sum_det = 0.0
    sum_feat = 0.0
    sum_head = 0.0
    sum_bn = 0.0
    sum_lat = 0.0
    steps = 0

    for ep in range(epochs):
        if gumbel_cfg is not None and gumbel_cfg.enabled:
            from ..trim.gumbel_choice import set_gumbel_temperature, tau_linear, tau_exponential
            if str(gumbel_cfg.tau_schedule) == "linear":
                tau = tau_linear(ep, epochs, float(gumbel_cfg.tau_start), float(gumbel_cfg.tau_end))
            else:
                tau = tau_exponential(ep, epochs, float(gumbel_cfg.tau_start), float(gumbel_cfg.tau_end))
            n_updated = set_gumbel_temperature(student_torch_model, tau)
            if n_updated > 0 and ep == 0:
                print(f"[kd][gumbel] tau schedule: {gumbel_cfg.tau_start} -> {gumbel_cfg.tau_end} ({gumbel_cfg.tau_schedule}), {n_updated} layers")
            if n_updated > 0:
                print(f"[kd][gumbel] ep={ep+1}/{epochs} tau={tau:.4f}")

        for bi, batch in enumerate(loader):
            if max_batches > 0 and bi >= max_batches:
                break

            img = batch["img"].to(device, non_blocking=True)
            if img.dtype != torch.float32:
                img = img.float() / 255.0

            batch = {k: (v.to(device, non_blocking=True) if torch.is_tensor(v) else v) for k, v in batch.items()}
            batch["img"] = img

            hs.clear()
            ht.clear()

            with torch.no_grad():
                _ = teacher_torch_model(img)

            opt.zero_grad(set_to_none=True)

            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                preds_s = student_torch_model(img)

                loss_det_raw, _ = criterion(preds_s, batch)
                loss_det = _scalarize(loss_det_raw, device)

                loss_feat = torch.zeros((), device=device)
                n = min(len(hs.store.feats), len(ht.store.feats))
                for i in range(n):
                    a_s = _attn_map(hs.store.feats[i])
                    a_t = _attn_map(ht.store.feats[i]).detach()
                    loss_feat = loss_feat + _mse_resize(a_s, a_t)
                loss_feat = _scalarize(loss_feat, device)

                loss_head = torch.zeros((), device=device)
                n2 = min(len(hs.store.head), len(ht.store.head))
                for i in range(n2):
                    a_s = _attn_map(hs.store.head[i])
                    a_t = _attn_map(ht.store.head[i]).detach()
                    loss_head = loss_head + _mse_resize(a_s, a_t)
                loss_head = _scalarize(loss_head, device)

                loss_bn = bn_sparsity_regularizer(
                    student_torch_model,
                    l1_weight=float(kd_cfg.lambda_bn),
                    exclude_head=bool(trim_cfg.exclude_head),
                    exclude_name_regex=trim_cfg.exclude_name_regex,
                )
                loss_bn = _scalarize(loss_bn, device)

                loss = loss_det + float(kd_cfg.lambda_feat) * loss_feat + float(kd_cfg.lambda_head) * loss_head + loss_bn
                if _lat_penalty_value > 0.0:
                    loss = loss + _lat_penalty_value
                loss = _scalarize(loss, device)

            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()

            sum_det += float(loss_det.detach().cpu().item())
            sum_feat += float(loss_feat.detach().cpu().item())
            sum_head += float(loss_head.detach().cpu().item())
            sum_bn += float(loss_bn.detach().cpu().item())
            sum_lat += _lat_penalty_value
            steps += 1

            if (bi + 1) % 10 == 0:
                tag = "qat" if enable_fake_quant else "kd"
                lat_str = f" lat_pen={sum_lat/steps:.4f}" if _lat_penalty_value > 0.0 else ""
                print(
                    f"[{tag}][ep {ep+1}/{epochs}][{bi+1}] "
                    f"det={sum_det/steps:.4f} feat={sum_feat/steps:.4f} "
                    f"head={sum_head/steps:.4f} bn={sum_bn/steps:.6f}{lat_str}"
                )

            if _lut_obj is not None and (bi + 1) % _log_every == 0:
                print(
                    f"[kd][lat] latency_est_ms={_lat_est_ms:.1f} "
                    f"penalty={_lat_penalty_value:.4f}"
                )

    hs.remove()
    ht.remove()

    if enable_fake_quant:
        set_fake_quant_enabled(student_torch_model, False)

    denom = max(1, steps)
    return {
        "loss_det_avg": sum_det / denom,
        "loss_feat_avg": sum_feat / denom,
        "loss_head_avg": sum_head / denom,
        "loss_bn_avg": sum_bn / denom,
        "latency_est_ms": _lat_est_ms,
        "latency_penalty": _lat_penalty_value,
        "steps": float(steps),
        "lr": float(lr),
        "epochs": float(epochs),
        "enable_fake_quant": float(1.0 if enable_fake_quant else 0.0),
    }
