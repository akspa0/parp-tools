"""Spec 077 H1 residual-height trainer.

This trains the second model in the coarse-to-fine residual chain. It loads a
frozen H0 ``height_coarse_65`` checkpoint, upsamples it to ``257x257``, and
predicts one signal only: ``height_delta_257``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import torch

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

import train_height_only_prior as direct  # noqa: E402
from harvester.height_residual_chain import (  # noqa: E402
    build_height_chain_input,
    compose_refined_height,
    height_chain_input_channels,
    residual_target,
    upsample_coarse_height,
)
from harvester.v18_models import V18HeightCoarseModel, V18HeightResidualModel  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Spec 077 H1 height_delta_257 residual model.")
    parser.add_argument("--coarse-checkpoint", type=Path, required=True)
    parser.add_argument("--prior", type=Path, nargs="+", required=True)
    parser.add_argument("--v18", type=Path, nargs="*", default=None)
    parser.add_argument("--albedo-path", type=Path, nargs="*", default=None)
    parser.add_argument("--curation-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models/spec077/height-residual"))
    parser.add_argument("--run-name", type=str, default="height_residual_h1")
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--steps", type=int, default=0, help="Optional smoke cap across all epochs; 0 disables.")
    parser.add_argument("--val-steps", type=int, default=0, help="Validation batch cap per epoch; 0 validates all batches.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-tiles", type=int, default=0)
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--prefetch-factor", type=int, default=2)
    parser.add_argument("--persistent-workers", dest="persistent_workers", action="store_true", default=None)
    parser.add_argument("--no-persistent-workers", dest="persistent_workers", action="store_false")
    parser.add_argument("--no-amp", action="store_true", default=False)
    parser.add_argument("--no-compile", action="store_true", default=False)
    parser.add_argument("--no-weight", action="store_true", default=False)
    parser.add_argument("--albedo", action="store_true", default=False)
    parser.add_argument("--density", action="store_true", default=False)
    parser.add_argument("--model-norm", choices=["batch", "group"], default="group")
    parser.add_argument("--decoder-upsample", choices=["bilinear", "nearest"], default="nearest")
    parser.add_argument("--split-mode", choices=["random", "map"], default="random")
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--augment-policy", choices=["shadow-safe", "d4"], default="shadow-safe")
    parser.add_argument("--augment-seed", type=int, default=42)
    parser.add_argument("--multiscale-weight", type=float, default=0.2)
    parser.add_argument("--delta-weight", type=float, default=0.25,
                        help="Extra direct L1 supervision on height_delta_257. Set 0 to use composed-height loss only.")
    parser.add_argument("--gradient-weight", type=float, default=0.05)
    parser.add_argument("--normal-guidance-weight", type=float, default=0.10)
    parser.add_argument("--normal-guidance-spacing", type=float, default=1.0)
    parser.add_argument("--hard-error-weight", type=float, default=0.05)
    parser.add_argument("--hard-error-power", type=float, default=1.0)
    parser.add_argument("--hard-error-max-multiplier", type=float, default=4.0)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    return parser.parse_args(argv)


def _load_coarse_model(path: Path, device: torch.device) -> tuple[torch.nn.Module, dict[str, Any]]:
    checkpoint = torch.load(path, map_location=device)
    model_args = dict(checkpoint.get("model_args", {}))
    model = V18HeightCoarseModel(
        in_channels=int(model_args.get("in_channels", 3)),
        norm=str(model_args.get("norm", "group")),
        decoder_upsample=str(model_args.get("decoder_upsample", "nearest")),
        coarse_size=int(model_args.get("coarse_size", 65)),
    ).to(device)
    direct._load_model_state_dict(model, checkpoint["model_state"])
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)
    return model, model_args


def _checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    epoch: int,
    global_step: int,
    best_val_loss: float | None,
    input_channels: int,
    coarse_model_args: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": "spec077-height-residual-h1-checkpoint-v1",
        "model_kind": "height_delta_257",
        "model_state": direct._model_state_dict(model),
        "model_args": {
            "in_channels": int(input_channels),
            "norm": str(args.model_norm),
            "decoder_upsample": str(args.decoder_upsample),
        },
        "coarse_checkpoint": str(args.coarse_checkpoint),
        "coarse_model_args": coarse_model_args,
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": best_val_loss,
        "run_name": str(args.run_name),
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _base_height(
    *,
    coarse_model: torch.nn.Module,
    batch: dict,
    device: torch.device,
    args: argparse.Namespace,
) -> torch.Tensor:
    with torch.no_grad():
        coarse_input = build_height_chain_input(
            batch,
            device=device,
            use_albedo=bool(args.albedo),
            use_density=bool(args.density),
        )
        coarse = coarse_model(coarse_input)
        return upsample_coarse_height(coarse, size=257).detach()


def _save_residual_preview(
    *,
    batch: dict,
    base_height: torch.Tensor,
    delta: torch.Tensor,
    refined: torch.Tensor,
    out_path: Path,
    max_samples: int = 4,
) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rows: list[list[tuple[str, torch.Tensor]]] = []
    count = min(int(max_samples), int(batch["input_prior"].shape[0]))
    tile_ids = batch.get("meta_tile_id")
    for idx in range(count):
        tile_id = int(tile_ids[idx]) if torch.is_tensor(tile_ids) else int(tile_ids[idx] if isinstance(tile_ids, (list, tuple)) else tile_ids)
        truth = batch["height_257"][idx].cpu()
        base_i = base_height[idx].detach().cpu()
        delta_i = delta[idx].detach().cpu()
        refined_i = refined[idx].detach().cpu()
        lo = min(float(truth.min()), float(base_i.min()), float(refined_i.min()))
        hi = max(float(truth.max()), float(base_i.max()), float(refined_i.max()))
        panels: list[tuple[str, torch.Tensor]] = [
            (f"prior tile {tile_id}", batch["input_prior"][idx, :3].cpu()),
        ]
        if "albedo_rgb" in batch:
            panels.append(("albedo input", batch["albedo_rgb"][idx].cpu()))
        panels.extend([
            ("h0 base", direct._gray3(direct._norm_for_display(base_i.squeeze(0), lo=lo, hi=hi))),
            ("h1 delta", direct._gray3(direct._norm_for_display(delta_i.squeeze(0)))),
            ("refined", direct._gray3(direct._norm_for_display(refined_i.squeeze(0), lo=lo, hi=hi))),
            ("truth", direct._gray3(direct._norm_for_display(truth.squeeze(0), lo=lo, hi=hi))),
            ("abs error", direct._gray3(direct._norm_for_display((refined_i - truth).abs().squeeze(0)))),
            ("loss weight", direct._gray3(direct._norm_for_display(batch["weight_257"][idx, 0].cpu(), lo=0.0, hi=1.0))),
        ])
        rows.append(panels)
    direct._compose_panel_grid(rows).save(out_path)


def _run_validation(
    *,
    coarse_model: torch.nn.Module,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    preview_path: Path | None = None,
) -> float:
    model.eval()
    losses: list[float] = []
    preview = None
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            target = batch["height_257"].to(device, non_blocking=True)
            weight = batch["weight_257"].to(device, non_blocking=True)
            base_height = _base_height(coarse_model=coarse_model, batch=batch, device=device, args=args)
            model_input = build_height_chain_input(
                batch,
                device=device,
                use_albedo=bool(args.albedo),
                use_density=bool(args.density),
                base_height_257=base_height,
            )
            target_normals = batch["normal_xyz"].to(device, non_blocking=True) if "normal_xyz" in batch else None
            normal_mask = batch["normal_mask"].to(device, non_blocking=True) if "normal_mask" in batch else None
            with torch.amp.autocast("cuda", enabled=use_amp):
                delta = model(model_input)
                refined = compose_refined_height(base_height, delta)
                loss, _ = direct.compute_height_loss(
                    refined,
                    target,
                    weight,
                    ms_weight=float(args.multiscale_weight),
                    grad_weight=float(args.gradient_weight),
                    nc_weight=0.0,
                    normal_guidance_weight=float(args.normal_guidance_weight),
                    target_normals=target_normals,
                    normal_guidance_mask=normal_mask,
                    normal_guidance_spacing=float(args.normal_guidance_spacing),
                    hard_error_weight=0.0,
                )
                if float(args.delta_weight) > 0.0:
                    delta_truth = residual_target(target, base_height)
                    delta_loss = direct._masked_mean((delta - delta_truth).abs(), weight)
                    loss = loss + float(args.delta_weight) * delta_loss
            losses.append(float(loss.item()))
            if preview is None:
                preview = (batch, base_height.cpu(), delta.cpu(), refined.cpu())
            if args.val_steps and batch_idx >= int(args.val_steps):
                break
    if preview_path is not None and preview is not None:
        _save_residual_preview(batch=preview[0], base_height=preview[1], delta=preview[2], refined=preview[3], out_path=preview_path)
    return float(sum(losses) / max(1, len(losses)))


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    direct._seed_all(int(args.seed))
    device = direct._resolve_device(args.device)
    dataset = direct._build_training_dataset(args)
    if dataset is None:
        return 2
    if args.augment:
        direct._enable_augment_on_base(
            dataset,
            seed=int(args.augment_seed),
            transforms=direct._AUGMENT_POLICY_TRANSFORMS[str(args.augment_policy)],
        )
    if args.split_mode == "map":
        train_subset, val_subset = direct._split_train_val_by_map(dataset, val_fraction=float(args.val_fraction), seed=int(args.seed))
    else:
        train_subset, val_subset = direct._split_train_val(dataset, val_fraction=float(args.val_fraction), seed=int(args.seed))
    val_subset = direct._AugmentGuardSubset(val_subset)

    num_workers = direct._resolve_num_workers(int(args.num_workers), device)
    persistent_workers = direct._resolve_persistent_workers(args.persistent_workers, num_workers)
    train_loader = direct._build_dataloader(
        train_subset,
        batch_size=int(args.batch_size),
        shuffle=True,
        num_workers=num_workers,
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=persistent_workers,
    )
    val_loader = direct._build_dataloader(
        val_subset,
        batch_size=int(args.batch_size),
        shuffle=False,
        num_workers=num_workers,
        prefetch_factor=int(args.prefetch_factor),
        persistent_workers=persistent_workers,
    )

    coarse_model, coarse_model_args = _load_coarse_model(args.coarse_checkpoint, device)
    expected_coarse_channels = height_chain_input_channels(use_albedo=bool(args.albedo), use_density=bool(args.density))
    if int(coarse_model_args.get("in_channels", expected_coarse_channels)) != expected_coarse_channels:
        raise RuntimeError(
            f"H0 checkpoint expects {coarse_model_args.get('in_channels')} input channels, "
            f"but this H1 run builds {expected_coarse_channels}. Match --albedo/--density to the H0 run."
        )

    input_channels = height_chain_input_channels(use_albedo=bool(args.albedo), use_density=bool(args.density), include_base=True)
    model = V18HeightResidualModel(
        in_channels=input_channels,
        norm=str(args.model_norm),
        decoder_upsample=str(args.decoder_upsample),
    ).to(device)
    if args.resume_checkpoint is not None:
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        direct._load_model_state_dict(model, checkpoint["model_state"])
    if bool(hasattr(torch, "compile") and not args.no_compile):
        try:
            model = torch.compile(model)
        except Exception:
            pass

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.learning_rate), weight_decay=float(args.weight_decay))
    scaler = torch.amp.GradScaler("cuda", enabled=bool(device.type == "cuda" and not args.no_amp))
    run_dir = args.output_dir
    run_dir.mkdir(parents=True, exist_ok=True)
    latest_path = run_dir / f"{args.run_name}_h1_latest.pt"
    best_path = run_dir / f"{args.run_name}_h1_best.pt"
    metrics_path = run_dir / f"{args.run_name}_h1_metrics.json"
    preview_dir = run_dir / f"{args.run_name}_h1_validation_previews"

    history: list[dict[str, Any]] = []
    best_val: float | None = None
    global_step = 0
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    stop_after_steps = int(args.steps) if int(args.steps) > 0 else None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            target = batch["height_257"].to(device, non_blocking=True)
            weight = batch["weight_257"].to(device, non_blocking=True)
            base_height = _base_height(coarse_model=coarse_model, batch=batch, device=device, args=args)
            model_input = build_height_chain_input(
                batch,
                device=device,
                use_albedo=bool(args.albedo),
                use_density=bool(args.density),
                base_height_257=base_height,
            )
            target_normals = batch["normal_xyz"].to(device, non_blocking=True) if "normal_xyz" in batch else None
            normal_mask = batch["normal_mask"].to(device, non_blocking=True) if "normal_mask" in batch else None
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                delta = model(model_input)
                refined = compose_refined_height(base_height, delta)
                loss, _ = direct.compute_height_loss(
                    refined,
                    target,
                    weight,
                    ms_weight=float(args.multiscale_weight),
                    grad_weight=float(args.gradient_weight),
                    nc_weight=0.0,
                    normal_guidance_weight=float(args.normal_guidance_weight),
                    target_normals=target_normals,
                    normal_guidance_mask=normal_mask,
                    normal_guidance_spacing=float(args.normal_guidance_spacing),
                    hard_error_weight=float(args.hard_error_weight),
                    hard_error_power=float(args.hard_error_power),
                    hard_error_max_multiplier=float(args.hard_error_max_multiplier),
                )
                if float(args.delta_weight) > 0.0:
                    delta_truth = residual_target(target, base_height)
                    delta_loss = direct._masked_mean((delta - delta_truth).abs(), weight)
                    loss = loss + float(args.delta_weight) * delta_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            global_step += 1
            epoch_losses.append(float(loss.item()))
            if stop_after_steps is not None and global_step >= stop_after_steps:
                break

        val_loss = _run_validation(
            coarse_model=coarse_model,
            model=model,
            loader=val_loader,
            device=device,
            args=args,
            preview_path=preview_dir / f"epoch_{epoch:04d}.png",
        )
        train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        improved = best_val is None or val_loss < best_val
        if improved:
            best_val = val_loss
        payload = _checkpoint_payload(
            args=args,
            model=model,
            epoch=epoch,
            global_step=global_step,
            best_val_loss=best_val,
            input_channels=input_channels,
            coarse_model_args=coarse_model_args,
        )
        _write_checkpoint(latest_path, payload)
        if improved:
            _write_checkpoint(best_path, payload)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "global_step": global_step})
        metrics_path.write_text(
            json.dumps(
                {
                    "schema": "spec077-height-residual-h1-metrics-v1",
                    "run_name": args.run_name,
                    "model_kind": "height_delta_257",
                    "input_channels": input_channels,
                    "coarse_checkpoint": str(args.coarse_checkpoint),
                    "albedo": bool(args.albedo),
                    "density": bool(args.density),
                    "delta_weight": float(args.delta_weight),
                    "best_val_loss": best_val,
                    "latest_checkpoint": str(latest_path),
                    "best_checkpoint": str(best_path),
                    "epoch_history": history,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        print(f"epoch {epoch}: train_loss={train_loss} val_loss={val_loss} best_val={best_val} step={global_step}", flush=True)
        if stop_after_steps is not None and global_step >= stop_after_steps:
            break
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
