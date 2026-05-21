"""Shared V16.1 training entrypoint used by task-specific wrappers."""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import numpy as np  # noqa: E402
import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from PIL import Image  # noqa: E402
from torch.utils.data import DataLoader, Subset  # noqa: E402

from harvester.v16_1_dataset import V161Dataset  # noqa: E402
from harvester.v16_1_models import (  # noqa: E402
    V161HeightModel,
    V161HolesModel,
    V161LiquidModel,
    V161NormalModel,
    V161TexcompModel,
    recompose_from_mcly_alpha,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_MODELS_ROOT = _PROJECT_ROOT / "models" / "v16_1"
_PANEL_SIZE = 256


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("Requested CUDA but CUDA is unavailable.")
        return torch.device("cuda")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _resolve_num_workers(requested: int, device: torch.device) -> int:
    if requested >= 0:
        return int(requested)
    if device.type != "cuda":
        return 0
    cpu_count = os.cpu_count() or 4
    return max(2, min(8, cpu_count // 2))


def _resolve_persistent_workers(requested: bool | None, num_workers: int) -> bool:
    if num_workers <= 0:
        return False
    if requested is None:
        return True
    return bool(requested)


def _masked_mean(loss_map: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (loss_map * mask).sum() / mask.sum().clamp_min(1e-8)


def _weighted_l1(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _masked_mean((pred - target).abs(), weight)


def _weighted_l2(pred: torch.Tensor, target: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return _masked_mean((pred - target) ** 2, weight)


def _resize_weight(weight: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    if tuple(weight.shape[-2:]) == tuple(size):
        return weight
    return F.interpolate(weight, size=size, mode="bilinear", align_corners=False)


def _normals_to_rgb(normals: torch.Tensor) -> torch.Tensor:
    return ((normals.clamp(-1.0, 1.0) + 1.0) * 0.5).clamp(0.0, 1.0)


def _to_uint8_hwc(x: torch.Tensor) -> np.ndarray:
    arr = x.detach().cpu().clamp(0.0, 1.0).numpy()
    if arr.ndim == 3:
        arr = np.transpose(arr, (1, 2, 0))
    elif arr.ndim == 2:
        arr = arr[..., None]
    if arr.shape[-1] == 1:
        arr = np.repeat(arr, 3, axis=-1)
    return (arr * 255.0).astype(np.uint8)


def _save_horizontal_panel(panels: list[tuple[str, torch.Tensor]], out_path: Path) -> None:
    images: list[Image.Image] = []
    for _label, tensor in panels:
        arr = _to_uint8_hwc(tensor)
        img = Image.fromarray(arr)
        if img.size != (_PANEL_SIZE, _PANEL_SIZE):
            img = img.resize((_PANEL_SIZE, _PANEL_SIZE), Image.Resampling.BILINEAR)
        images.append(img)
    canvas = Image.new("RGB", (_PANEL_SIZE * len(images), _PANEL_SIZE), color=(0, 0, 0))
    for idx, img in enumerate(images):
        canvas.paste(img, (idx * _PANEL_SIZE, 0))
    canvas.save(out_path)


def _coarse_type_to_rgb(type_grid: torch.Tensor) -> torch.Tensor:
    palette = torch.tensor(
        [
            [0.0, 0.0, 0.0],
            [0.2, 0.45, 0.9],
            [0.1, 0.7, 0.95],
            [0.9, 0.35, 0.1],
            [0.25, 0.85, 0.2],
        ],
        dtype=torch.float32,
        device=type_grid.device,
    )
    rgb = palette[type_grid.long().clamp(0, 4)]
    return rgb.permute(2, 0, 1)


@dataclass(frozen=True)
class TaskSpec:
    name: str
    model_factory: Callable[[], torch.nn.Module]
    loss_fn: Callable[[torch.nn.Module, dict[str, Any], torch.device], tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]]
    save_preview: Callable[[dict[str, Any], dict[str, torch.Tensor], Path], None]


def _height_loss(model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target = batch["height_norm"].to(device, non_blocking=True)
    weight = batch["weight_257"].to(device, non_blocking=True)
    pred = model(inp)
    loss = _weighted_l1(pred, target, weight)
    return loss, {"height": float(loss.item())}, {"pred": pred, "target": target, "weight": weight}


def _normal_loss(model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target = batch["normals"].to(device, non_blocking=True)
    normal_mask = batch["normal_mask"].to(device, non_blocking=True)
    object_weight = batch["weight_257"].to(device, non_blocking=True)
    mddf_mask = batch["mddf_mask"].to(device, non_blocking=True)
    modf_mask = batch["modf_mask"].to(device, non_blocking=True)
    liquid_mask = batch["liquid_mask"].to(device, non_blocking=True)
    pred = model(inp)
    pred_n = F.normalize(pred, dim=1, eps=1e-6)
    target_n = F.normalize(target, dim=1, eps=1e-6)
    cosine = 1.0 - (pred_n * target_n).sum(dim=1, keepdim=True)
    liquid_mask_257 = _resize_weight(liquid_mask, target_n.shape[-2:])
    object_presence = torch.maximum(mddf_mask, modf_mask)
    liquid_weight = 1.0 - (0.85 * liquid_mask_257)
    instance_weight = 1.0 - (0.75 * object_presence)
    train_mask = normal_mask * object_weight * liquid_weight * instance_weight
    vec_l1 = (pred_n - target_n).abs().mean(dim=1, keepdim=True)
    nz_l2 = (pred_n[:, 2:3] - target_n[:, 2:3]) ** 2
    loss_cos = _masked_mean(cosine, train_mask)
    loss_vec = _masked_mean(vec_l1, train_mask)
    loss_nz = _masked_mean(nz_l2, train_mask)
    loss = loss_cos + (0.35 * loss_vec) + (0.15 * loss_nz)
    return loss, {
        "normal": float(loss.item()),
        "normal_cos": float(loss_cos.item()),
        "normal_vec": float(loss_vec.item()),
        "normal_nz": float(loss_nz.item()),
        "normal_mask_cov": float(train_mask.mean().item()),
    }, {
        "pred": pred_n,
        "target": target_n,
        "train_mask": train_mask,
        "object_weight": object_weight,
        "liquid_mask": liquid_mask_257,
        "instance_weight": instance_weight,
    }


def _holes_loss(model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target = batch["holes"].to(device, non_blocking=True)
    weight = batch["weight_16"].to(device, non_blocking=True)
    pred = model(inp)
    bce = F.binary_cross_entropy(pred, target, reduction="none")
    loss = _masked_mean(bce, weight)
    return loss, {"holes": float(loss.item())}, {"pred": pred, "target": target}


def _liquid_loss(model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    target_mask = batch["liquid_mask"].to(device, non_blocking=True)
    target_type = batch["liquid_type_16"].to(device, non_blocking=True)
    type_valid = batch["liquid_type_valid_16"].to(device, non_blocking=True)
    weight_256 = batch["weight_256"].to(device, non_blocking=True)
    weight_16 = batch["weight_16"].to(device, non_blocking=True)
    pred_mask, pred_type = model(inp)
    mask_loss = _weighted_l1(pred_mask, target_mask, weight_256)
    type_ce = F.cross_entropy(pred_type, target_type, reduction="none").unsqueeze(1)
    type_loss = _masked_mean(type_ce, type_valid * weight_16)
    loss = mask_loss + (0.5 * type_loss)
    type_pred = pred_type.argmax(dim=1)
    return loss, {"liquid_mask": float(mask_loss.item()), "liquid_type": float(type_loss.item())}, {"pred_mask": pred_mask, "target_mask": target_mask, "pred_type": type_pred, "target_type": target_type}


def _texcomp_loss(model: torch.nn.Module, batch: dict[str, Any], device: torch.device) -> tuple[torch.Tensor, dict[str, float], dict[str, torch.Tensor]]:
    inp = batch["input"].to(device, non_blocking=True)
    alpha_target = batch["alpha"].to(device, non_blocking=True)
    mcly_ids = batch["mcly_ids"].to(device, non_blocking=True)
    mcly_mask = batch["mcly_mask"].to(device, non_blocking=True)
    weight_256 = batch["weight_256"].to(device, non_blocking=True)
    weight_16 = batch["weight_16"].to(device, non_blocking=True)
    pred_alpha, pred_mask, pred_ids = model(inp)
    alpha_loss = _weighted_l1(pred_alpha, alpha_target, weight_256)
    mask_bce = F.binary_cross_entropy(pred_mask, mcly_mask.permute(0, 3, 1, 2), reduction="none")
    mask_loss = _masked_mean(mask_bce, weight_16.repeat(1, 4, 1, 1))
    pred_ids_r = pred_ids.permute(0, 1, 3, 4, 2)
    id_ce = F.cross_entropy(pred_ids_r.reshape(-1, 16), mcly_ids.reshape(-1), reduction="none")
    id_ce = id_ce.view_as(mcly_ids).permute(0, 3, 1, 2)
    id_loss = _masked_mean(id_ce, mcly_mask.permute(0, 3, 1, 2) * weight_16.repeat(1, 4, 1, 1))
    recomposed = recompose_from_mcly_alpha(pred_alpha, pred_ids, pred_mask)
    recompose_loss = _weighted_l1(recomposed, inp, weight_256)
    loss = alpha_loss + (0.35 * mask_loss) + (0.25 * id_loss) + (0.5 * recompose_loss)
    pred_ids_cls = pred_ids.argmax(dim=2)
    return loss, {
        "alpha": float(alpha_loss.item()),
        "mcly_mask": float(mask_loss.item()),
        "mcly_id": float(id_loss.item()),
        "recompose": float(recompose_loss.item()),
    }, {
        "pred_alpha": pred_alpha,
        "target_alpha": alpha_target,
        "pred_mask": pred_mask,
        "target_mask": mcly_mask.permute(0, 3, 1, 2),
        "pred_ids": pred_ids_cls,
        "target_ids": mcly_ids,
        "recomposed": recomposed,
    }


def _preview_height(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    panels = [
        ("input", batch["input"][0]),
        ("height_gt", batch["height_norm"][0]),
        ("height_pred", outputs["pred"][0]),
        ("weight", batch["weight_257"][0]),
    ]
    _save_horizontal_panel(panels, out_path)


def _preview_normal(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    panels = [
        ("input", batch["input"][0]),
        ("normal_gt", _normals_to_rgb(outputs["target"][0])),
        ("normal_pred", _normals_to_rgb(outputs["pred"][0])),
        ("train_mask", outputs["train_mask"][0]),
        ("liquid_mask", outputs["liquid_mask"][0]),
        ("object_weight", outputs["object_weight"][0]),
    ]
    _save_horizontal_panel(panels, out_path)


def _preview_holes(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    panels = [
        ("input", batch["input"][0]),
        ("holes_gt", outputs["target"][0]),
        ("holes_pred", outputs["pred"][0]),
    ]
    _save_horizontal_panel(panels, out_path)


def _preview_liquid(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    pred_type_rgb = _coarse_type_to_rgb(outputs["pred_type"][0])
    target_type_rgb = _coarse_type_to_rgb(outputs["target_type"][0])
    panels = [
        ("input", batch["input"][0]),
        ("liq_gt", outputs["target_mask"][0]),
        ("liq_pred", outputs["pred_mask"][0]),
        ("type_gt", target_type_rgb),
        ("type_pred", pred_type_rgb),
    ]
    _save_horizontal_panel(panels, out_path)


def _preview_texcomp(batch: dict[str, Any], outputs: dict[str, torch.Tensor], out_path: Path) -> None:
    gt_alpha_painted = batch["alpha"][0, 1:].max(dim=0).values.unsqueeze(0)
    pred_alpha_painted = outputs["pred_alpha"][0, 1:].max(dim=0).values.unsqueeze(0)
    gt_mask = outputs["target_mask"][0].max(dim=0).values.unsqueeze(0)
    pred_mask = outputs["pred_mask"][0].max(dim=0).values.unsqueeze(0)
    panels = [
        ("input", batch["input"][0]),
        ("alpha_gt", gt_alpha_painted),
        ("alpha_pred", pred_alpha_painted),
        ("mask_gt", gt_mask),
        ("mask_pred", pred_mask),
        ("recomposed", outputs["recomposed"][0]),
    ]
    _save_horizontal_panel(panels, out_path)


TASKS: dict[str, TaskSpec] = {
    "height": TaskSpec("height", V161HeightModel, _height_loss, _preview_height),
    "normal": TaskSpec("normal", V161NormalModel, _normal_loss, _preview_normal),
    "holes": TaskSpec("holes", V161HolesModel, _holes_loss, _preview_holes),
    "liquid": TaskSpec("liquid", V161LiquidModel, _liquid_loss, _preview_liquid),
    "texcomp": TaskSpec("texcomp", V161TexcompModel, _texcomp_loss, _preview_texcomp),
}


def _parse_args(task_name: str) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=f"Train V16.1 {task_name} model")
    p.add_argument("--dataset-dir", type=Path, default=_DATASET_ROOT)
    p.add_argument(
        "--curation-manifest",
        type=Path,
        default=None,
        help="Optional curation manifest directory/file produced by build_v16_curation_manifest.py",
    )
    p.add_argument("--builds", nargs="+", default=["3_3_5_12340"])
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--grad-accum-steps", type=int, default=1)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument(
        "--num-workers",
        type=int,
        default=-1,
        help="DataLoader worker count. Use -1 to auto-resolve a CUDA-friendly default.",
    )
    p.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Keep DataLoader worker processes alive between epochs (defaults to true when workers > 0)",
    )
    p.add_argument(
        "--prefetch-factor",
        type=int,
        default=4,
        help="Batches prefetched per worker (effective when --num-workers > 0)",
    )
    p.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--val-fraction", type=float, default=0.1)
    p.add_argument("--max-train-samples", type=int, default=0)
    p.add_argument("--max-val-samples", type=int, default=0)
    p.add_argument("--val-interval", type=int, default=1)
    p.add_argument("--val-preview-interval", type=int, default=5)
    p.add_argument("--run-name", type=str, default=None)
    p.add_argument("--resume-checkpoint", type=Path, default=None)
    p.add_argument("--no-augment", action="store_true")
    p.add_argument("--no-amp", action="store_true")
    p.add_argument(
        "--no-compile",
        action="store_true",
        help="Disable torch.compile (useful for CPU-only or limited toolchains)",
    )
    p.add_argument("--target-vram-gb", type=float, default=0.0)
    return p.parse_args()


def run_task(task_name: str) -> None:
    if task_name not in TASKS:
        raise RuntimeError(f"Unknown V16.1 task: {task_name}")
    args = _parse_args(task_name)
    task = TASKS[task_name]
    if args.grad_accum_steps < 1:
        raise RuntimeError("--grad-accum-steps must be >= 1")
    if args.prefetch_factor < 1:
        raise RuntimeError("--prefetch-factor must be >= 1")
    _seed_all(args.seed)
    device = _resolve_device(args.device)
    resolved_num_workers = _resolve_num_workers(int(args.num_workers), device)
    resolved_persistent_workers = _resolve_persistent_workers(args.persistent_workers, resolved_num_workers)
    run_name = args.run_name or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = _MODELS_ROOT / task_name / "runs" / run_name
    ckpt_dir = run_dir / "checkpoints"
    val_dir = run_dir / "validation"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    val_dir.mkdir(parents=True, exist_ok=True)

    train_ds = V161Dataset(
        args.dataset_dir,
        builds=args.builds,
        split="train",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=not args.no_augment,
        curation_manifest=args.curation_manifest,
    )
    val_ds = V161Dataset(
        args.dataset_dir,
        builds=args.builds,
        split="val",
        val_fraction=args.val_fraction,
        seed=args.seed,
        augment=False,
        curation_manifest=args.curation_manifest,
    )
    if args.max_train_samples > 0:
        train_ds = Subset(train_ds, range(min(args.max_train_samples, len(train_ds))))
    if args.max_val_samples > 0:
        val_ds = Subset(val_ds, range(min(args.max_val_samples, len(val_ds))))
    loader_kwargs: dict[str, Any] = {
        "num_workers": resolved_num_workers,
        "pin_memory": (device.type == "cuda"),
    }
    if resolved_num_workers > 0:
        loader_kwargs["persistent_workers"] = bool(resolved_persistent_workers)
        loader_kwargs["prefetch_factor"] = int(args.prefetch_factor)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kwargs)

    model = task.model_factory().to(device)
    can_compile = hasattr(torch, "compile") and not args.no_compile and device.type == "cuda"
    if can_compile:
        try:
            model = torch.compile(model)
            compile_status = "enabled"
        except Exception as ex:
            compile_status = f"disabled (compile failed: {ex})"
    else:
        compile_status = "disabled"
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    scaler = torch.amp.GradScaler("cuda", enabled=(device.type == "cuda" and not args.no_amp))
    best_val = float("inf")
    start_epoch = 1
    log_entries: list[dict[str, Any]] = []

    if args.resume_checkpoint is not None:
        ckpt = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if "scheduler_state_dict" in ckpt:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        if "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = int(ckpt["epoch"]) + 1
        best_val = float(ckpt.get("best_val", float("inf")))

    config = {
        "task": task_name,
        "run_name": run_name,
        "dataset_dir": str(Path(args.dataset_dir)),
        "curation_manifest": str(args.curation_manifest) if args.curation_manifest else None,
        "builds": list(args.builds),
        "batch_size": args.batch_size,
        "grad_accum_steps": args.grad_accum_steps,
        "effective_batch_size": args.batch_size * args.grad_accum_steps,
        "num_workers": args.num_workers,
        "resolved_num_workers": resolved_num_workers,
        "persistent_workers": args.persistent_workers,
        "resolved_persistent_workers": resolved_persistent_workers,
        "prefetch_factor": args.prefetch_factor,
        "epochs": args.epochs,
        "lr": args.lr,
        "weight_decay": args.weight_decay,
        "seed": args.seed,
        "val_fraction": args.val_fraction,
        "max_train_samples": args.max_train_samples,
        "max_val_samples": args.max_val_samples,
        "no_compile": args.no_compile,
        "compile_status": compile_status,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }
    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")

    print(f"Task: {task_name}", flush=True)
    print(f"Device: {device}", flush=True)
    print(f"Run dir: {run_dir}", flush=True)
    print(f"Dataset: train={len(train_ds)} val={len(val_ds)}", flush=True)
    if args.curation_manifest is not None:
        print(f"Curation manifest: {args.curation_manifest}", flush=True)
    print(
        f"DataLoader: workers={resolved_num_workers} "
        f"persistent_workers={resolved_persistent_workers} "
        f"prefetch_factor={(args.prefetch_factor if resolved_num_workers > 0 else 'n/a')}",
        flush=True,
    )
    print(
        f"Batching: micro={args.batch_size} accum={args.grad_accum_steps} "
        f"effective={args.batch_size * args.grad_accum_steps}",
        flush=True,
    )
    print(f"Parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}", flush=True)
    print(f"torch.compile: {compile_status}", flush=True)

    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        metric_sums: dict[str, float] = {}
        train_loss_sum = 0.0
        optimizer_steps = 0
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        for batch_idx, batch in enumerate(train_loader, start=1):
            with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                loss, metrics, _outputs = task.loss_fn(model, batch, device)
            scaler.scale(loss / args.grad_accum_steps).backward()
            should_step = (batch_idx % args.grad_accum_steps == 0) or (batch_idx == len(train_loader))
            if should_step:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
            train_loss_sum += float(loss.item())
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + float(value)
        scheduler.step()

        n_train = max(len(train_loader), 1)
        entry: dict[str, Any] = {
            "epoch": epoch,
            "train_loss": train_loss_sum / n_train,
            "lr": optimizer.param_groups[0]["lr"],
            "elapsed_s": time.perf_counter() - t0,
            "optimizer_steps": optimizer_steps,
        }
        for key, value in metric_sums.items():
            entry[f"train_{key}"] = value / n_train

        print(
            f"Epoch {epoch:3d}/{args.epochs} | loss={entry['train_loss']:.4f} "
            f"lr={entry['lr']:.2e} opt_steps={optimizer_steps} {entry['elapsed_s']:.1f}s",
            flush=True,
        )

        if epoch % args.val_interval == 0 and len(val_loader) > 0:
            model.eval()
            val_loss_sum = 0.0
            val_metric_sums: dict[str, float] = {}
            preview_batch = None
            preview_outputs = None
            with torch.no_grad():
                for batch in val_loader:
                    with torch.amp.autocast("cuda", enabled=(device.type == "cuda" and not args.no_amp)):
                        loss, metrics, outputs = task.loss_fn(model, batch, device)
                    val_loss_sum += float(loss.item())
                    for key, value in metrics.items():
                        val_metric_sums[key] = val_metric_sums.get(key, 0.0) + float(value)
                    if preview_batch is None:
                        preview_batch = batch
                        preview_outputs = outputs
            n_val = max(len(val_loader), 1)
            entry["val_loss"] = val_loss_sum / n_val
            for key, value in val_metric_sums.items():
                entry[f"val_{key}"] = value / n_val
            print(f"        val | loss={entry['val_loss']:.4f}", flush=True)

            if preview_batch is not None and preview_outputs is not None and args.val_preview_interval > 0 and epoch % args.val_preview_interval == 0:
                task.save_preview(preview_batch, preview_outputs, val_dir / f"epoch_{epoch:04d}.png")

            if entry["val_loss"] < best_val:
                best_val = float(entry["val_loss"])
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "scaler_state_dict": scaler.state_dict(),
                        "best_val": best_val,
                        "task": task_name,
                    },
                    ckpt_dir / f"v16_1_{task_name}_best.pt",
                )

        log_entries.append(entry)
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
                "scaler_state_dict": scaler.state_dict(),
                "best_val": best_val,
                "task": task_name,
            },
            ckpt_dir / f"v16_1_{task_name}_last.pt",
        )
        (run_dir / "training_log.json").write_text(json.dumps(log_entries, indent=2), encoding="utf-8")

    torch.save(
        {
            "epoch": args.epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict(),
            "best_val": best_val,
            "task": task_name,
        },
        ckpt_dir / f"v16_1_{task_name}_final.pt",
    )
    print(f"Done. Run dir: {run_dir}", flush=True)
