"""Spec 077 H0 coarse-height trainer.

This trains the first model in the coarse-to-fine residual chain. It predicts
one signal only: ``height_coarse_65`` from the same processed-prior/albedo/
density inputs used by the direct height model.
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
    downsample_height_target,
    height_chain_input_channels,
    masked_charbonnier,
    upsample_coarse_height,
)
from harvester.v18_models import V18HeightCoarseModel  # noqa: E402


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Spec 077 H0 height_coarse_65 model.")
    parser.add_argument("--prior", type=Path, nargs="+", required=True)
    parser.add_argument("--v18", type=Path, nargs="*", default=None)
    parser.add_argument("--albedo-path", type=Path, nargs="*", default=None)
    parser.add_argument("--curation-manifest", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("models/spec077/height-coarse"))
    parser.add_argument("--run-name", type=str, default="height_coarse_h0")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--steps", type=int, default=0, help="Optional smoke cap across all epochs; 0 disables.")
    parser.add_argument("--val-steps", type=int, default=0, help="Validation batch cap per epoch; 0 validates all batches.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
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
    parser.add_argument("--coarse-size", type=int, default=65)
    parser.add_argument("--split-mode", choices=["random", "map"], default="random")
    parser.add_argument("--augment", action="store_true", default=False)
    parser.add_argument("--augment-policy", choices=["shadow-safe", "d4"], default="shadow-safe")
    parser.add_argument("--augment-seed", type=int, default=42)
    parser.add_argument("--resume-checkpoint", type=Path, default=None)
    return parser.parse_args(argv)


def _checkpoint_payload(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    epoch: int,
    global_step: int,
    best_val_loss: float | None,
    input_channels: int,
) -> dict[str, Any]:
    return {
        "schema": "spec077-height-coarse-h0-checkpoint-v1",
        "model_kind": "height_coarse_65",
        "model_state": direct._model_state_dict(model),
        "model_args": {
            "in_channels": int(input_channels),
            "norm": str(args.model_norm),
            "decoder_upsample": str(args.decoder_upsample),
            "coarse_size": int(args.coarse_size),
        },
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_val_loss": best_val_loss,
        "run_name": str(args.run_name),
    }


def _write_checkpoint(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _run_validation(
    *,
    model: torch.nn.Module,
    loader,
    device: torch.device,
    args: argparse.Namespace,
    preview_path: Path | None = None,
) -> float:
    model.eval()
    losses: list[float] = []
    preview_batch = None
    preview_pred_full = None
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    with torch.no_grad():
        for batch_idx, batch in enumerate(loader, start=1):
            model_input = build_height_chain_input(
                batch,
                device=device,
                use_albedo=bool(args.albedo),
                use_density=bool(args.density),
            )
            target = batch["height_257"].to(device, non_blocking=True)
            weight = batch["weight_257"].to(device, non_blocking=True)
            target_coarse, weight_coarse = downsample_height_target(target, weight, coarse_size=int(args.coarse_size))
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_coarse = model(model_input)
                loss = masked_charbonnier(pred_coarse - target_coarse, weight_coarse)
            losses.append(float(loss.item()))
            if preview_batch is None:
                preview_batch = batch
                preview_pred_full = upsample_coarse_height(pred_coarse.detach(), size=257).cpu()
            if args.val_steps and batch_idx >= int(args.val_steps):
                break
    if preview_path is not None and preview_batch is not None and preview_pred_full is not None:
        direct._save_deconstruction_preview(
            batch=preview_batch,
            pred=preview_pred_full,
            out_path=preview_path,
            max_samples=4,
            expect_albedo=bool(args.albedo),
            expect_density=bool(args.density),
        )
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

    input_channels = height_chain_input_channels(use_albedo=bool(args.albedo), use_density=bool(args.density))
    model = V18HeightCoarseModel(
        in_channels=input_channels,
        norm=str(args.model_norm),
        decoder_upsample=str(args.decoder_upsample),
        coarse_size=int(args.coarse_size),
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
    latest_path = run_dir / f"{args.run_name}_h0_latest.pt"
    best_path = run_dir / f"{args.run_name}_h0_best.pt"
    metrics_path = run_dir / f"{args.run_name}_h0_metrics.json"
    preview_dir = run_dir / f"{args.run_name}_h0_validation_previews"

    history: list[dict[str, Any]] = []
    best_val: float | None = None
    global_step = 0
    use_amp = bool(device.type == "cuda" and not args.no_amp)
    stop_after_steps = int(args.steps) if int(args.steps) > 0 else None

    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_losses: list[float] = []
        for batch in train_loader:
            model_input = build_height_chain_input(
                batch,
                device=device,
                use_albedo=bool(args.albedo),
                use_density=bool(args.density),
            )
            target = batch["height_257"].to(device, non_blocking=True)
            weight = batch["weight_257"].to(device, non_blocking=True)
            target_coarse, weight_coarse = downsample_height_target(target, weight, coarse_size=int(args.coarse_size))
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                pred_coarse = model(model_input)
                loss = masked_charbonnier(pred_coarse - target_coarse, weight_coarse)
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
        )
        _write_checkpoint(latest_path, payload)
        if improved:
            _write_checkpoint(best_path, payload)
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "global_step": global_step})
        metrics_path.write_text(
            json.dumps(
                {
                    "schema": "spec077-height-coarse-h0-metrics-v1",
                    "run_name": args.run_name,
                    "model_kind": "height_coarse_65",
                    "input_channels": input_channels,
                    "coarse_size": int(args.coarse_size),
                    "albedo": bool(args.albedo),
                    "density": bool(args.density),
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
