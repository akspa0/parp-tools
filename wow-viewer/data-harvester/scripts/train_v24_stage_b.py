"""FR-017: train Stage B (lattice detailer).

Input: Stage A's predicted prior (upsampled to 257 via the exact quincunx
bilinear) + cleaned minimap + V18 channels. Target: the residual
``height_257 - upsampled_prior``. Loss gated to non-liquid, non-object,
non-hole pixels.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24 import stage_a, stage_b, train_common  # noqa: E402
from harvester.v24.tiles import (  # noqa: E402
    HEIGHT_SCALE,
    RESIDUAL_SCALE,
    MultiTileSource,
    TileSource,
)


def _build_source(v24_stores: list[str], v18_stores: list[str] | None) -> TileSource | MultiTileSource:
    if v18_stores and len(v18_stores) != len(v24_stores):
        raise ValueError(
            f"--v18-store count ({len(v18_stores)}) must match --v24-store count ({len(v24_stores)}) when given"
        )
    if len(v24_stores) == 1:
        return TileSource(v24_stores[0], (v18_stores[0] if v18_stores else None))
    pairs = [
        (v24, v18_stores[i] if v18_stores else None) for i, v24 in enumerate(v24_stores)
    ]
    return MultiTileSource(pairs)


def _stage_a_priors(records, checkpoint_path: str | None, device: torch.device):
    """Predict the Stage A prior per tile (or fall back to the merged prior)."""
    if checkpoint_path is None:
        return [(r.prior_outer, r.prior_inner) for r in records]

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    model = stage_a.StageAModel(base=checkpoint["config"]["base"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()

    priors = []
    with torch.no_grad():
        for record in records:
            x_np, q_np = stage_a.build_input(record, include_synth=True)
            x = torch.from_numpy(x_np)[None].to(device)
            q = torch.from_numpy(q_np)[None].to(device)
            outer, inner = model(x, q)
            priors.append(
                (
                    outer[0].cpu().numpy() * HEIGHT_SCALE,
                    inner[0].cpu().numpy() * HEIGHT_SCALE,
                )
            )
    return priors


def _load_tensors(records, priors):
    inputs, targets, valids, prior_ups = [], [], [], []
    for record, (prior_outer, prior_inner) in zip(records, priors, strict=True):
        x, prior_up = stage_b.build_input(record, prior_outer, prior_inner)
        residual, valid = stage_b.build_target(record, prior_up)
        inputs.append(x)
        targets.append(residual)
        valids.append(valid)
        prior_ups.append(prior_up)
    return (
        torch.from_numpy(np.stack(inputs)),
        torch.from_numpy(np.stack(targets)),
        torch.from_numpy(np.stack(valids)),
        np.stack(prior_ups),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v24-store", required=True, nargs="+",
                         help="one or more V24 stores; multiple stores (e.g. different builds) are concatenated")
    parser.add_argument("--v18-store", default=None, nargs="*",
                         help="matching V18 store overrides, one per --v24-store (omit to use each store's own v18_store_path attr)")
    parser.add_argument("--stage-a-checkpoint", default=None,
                        help="omit to train against the merged prior directly")
    parser.add_argument("--output", required=True)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--log-interval", type=int, default=1,
                        help="print per-batch progress every N batches (1 = every batch)")
    parser.add_argument("--patience", type=int, default=0,
                        help="early-stop after N epochs without val improvement (0 = run all epochs)")
    parser.add_argument("--cudnn-benchmark", dest="cudnn_benchmark", action="store_true", default=True,
                        help="fast cuDNN kernel autotune + TF32 for training (default on; no bearing on inference determinism)")
    parser.add_argument("--no-cudnn-benchmark", dest="cudnn_benchmark", action="store_false")
    parser.add_argument("--gpu-resident-data", dest="gpu_resident_data", action="store_true", default=True,
                        help="keep the whole train/val tensor set resident on the GPU when it fits (default on)")
    parser.add_argument("--no-gpu-resident-data", dest="gpu_resident_data", action="store_false")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16"], default="fp16")
    parser.add_argument("--autotune-batch-size", action="store_true", default=False,
                        help="probe --autotune-batch-candidates and pick the largest that fits before training")
    parser.add_argument("--autotune-batch-candidates", nargs="+", type=int,
                        default=[2, 4, 8, 12, 16, 24, 32, 48, 64, 96])
    parser.add_argument("--autotune-safety-factor", type=float, default=0.85)
    args = parser.parse_args()

    train_common.set_determinism(args.seed, strict=False)
    train_common.configure_perf(args.cudnn_benchmark)
    device = train_common.pick_device(args.device)
    run_dir = Path(args.output)
    logger = train_common.RunLogger(run_dir)
    print(f"stage B training: device={device} run_dir={run_dir}")

    source = _build_source(args.v24_store, args.v18_store)
    rows = source.usable_rows()
    if args.limit:
        rows = rows[: args.limit]
    train_rows, val_rows = train_common.split_rows(rows, args.val_fraction, args.seed)
    print(f"tiles: {len(rows)} usable -> {len(train_rows)} train / {len(val_rows)} val")

    started = time.time()

    # Preload V18 data in one sequential Zarr pass — avoids per-tile seek overhead.
    source.preload(train_rows + val_rows)

    def _load_records(rows, label):
        out = []
        n = len(rows)
        step = max(1, n // 20)
        for i, r in enumerate(rows):
            out.append(source.load(r))
            if (i + 1) % step == 0 or (i + 1) == n:
                el = time.time() - started
                print(f"[{label}] {i + 1}/{n} tiles ({100.0 * (i + 1) / n:.0f}%) elapsed={el:.1f}s", flush=True)
        return out

    train_records = _load_records(train_rows, "load train")
    val_records = _load_records(val_rows, "load val")
    train_priors = _stage_a_priors(train_records, args.stage_a_checkpoint, device)
    val_priors = _stage_a_priors(val_records, args.stage_a_checkpoint, device)
    x, target, valid, _ = _load_tensors(train_records, train_priors)
    vx, vtarget, vvalid, vprior_up = _load_tensors(val_records, val_priors)
    print(f"loaded tensors in {time.time() - started:.1f}s", flush=True)

    if args.gpu_resident_data:
        x, target, valid = train_common.gpu_resident((x, target, valid), device)
        vx, vtarget, vvalid = train_common.gpu_resident((vx, vtarget, vvalid), device)
        print(f"train/val tensors resident={x.is_cuda if device.type == 'cuda' else False}", flush=True)

    # Baselines in world units on the validation split (SC-003).
    val_height = np.stack([r.height for r in val_records])
    val_valid_np = vvalid.cpu().numpy() > 0.5
    prior_l1 = float(np.abs((val_height - vprior_up))[val_valid_np].mean())
    from harvester.v24 import lattice  # local import to keep top tidy

    br_outer, br_inner = lattice.sample_lattice_from_height(val_height)
    br_up = lattice.upsample_prior_257(br_outer, br_inner)
    block_reduce_l1 = float(np.abs(val_height - br_up)[val_valid_np].mean())
    print(f"baselines: upsampled-prior L1={prior_l1:.4f}, block_reduce+bilinear L1={block_reduce_l1:.4f}")

    model = stage_b.StageBModel().to(device)
    params = stage_b.parameter_count(model)
    assert params <= 2_000_000, f"Stage B exceeds 2M params: {params}"
    print(f"stage B params: {params}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)

    n = x.shape[0]

    if args.autotune_batch_size:
        candidates = [c for c in sorted(args.autotune_batch_candidates) if c <= n]
        if candidates:
            snapshot = train_common.snapshot_state(model, optimizer)

            def _try_step(bs: int) -> None:
                idx = torch.randperm(n)[:bs].to(x.device)
                xb = x[idx].to(device, non_blocking=True)
                tb = target[idx].to(device, non_blocking=True)
                vb = valid[idx].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    pred = model(xb)
                    loss = stage_b.gated_l1(pred, tb, vb)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

            args.batch_size = train_common.autotune_batch_size(
                device, candidates, _try_step, safety_fraction=args.autotune_safety_factor
            )
            train_common.restore_state(model, optimizer, snapshot)
            scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)
            print(f"autotune selected batch_size={args.batch_size}", flush=True)

    n_batches = (n + args.batch_size - 1) // args.batch_size
    generator = torch.Generator().manual_seed(args.seed)
    best_val = float("inf")
    stopper = train_common.EarlyStopping(patience=args.patience, min_delta=1e-4)
    epoch_started = time.time()

    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(n, generator=generator)
        epoch_loss, batches = 0.0, 0
        for start_idx in range(0, n, args.batch_size):
            idx = perm[start_idx : start_idx + args.batch_size].to(x.device)
            xb = x[idx].to(device, non_blocking=True)
            tb = target[idx].to(device, non_blocking=True)
            vb = valid[idx].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                pred = model(xb)
                loss = stage_b.gated_l1(pred, tb, vb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            batches += 1

            if batches % args.log_interval == 0 or start_idx + args.batch_size >= n:
                elapsed = time.time() - epoch_started
                pct = 100.0 * batches / n_batches
                eta = elapsed / batches * (n_batches - batches) if batches else 0.0
                print(
                    f"epoch {epoch} batch {batches}/{n_batches} ({pct:.0f}%) "
                    f"loss={loss.item() * RESIDUAL_SCALE:.4f} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred = []
            for start_idx in range(0, vx.shape[0], args.batch_size):
                batch = vx[start_idx : start_idx + args.batch_size].to(device, non_blocking=True)
                val_pred.append(model(batch))
            val_pred = torch.cat(val_pred)
        # vtarget/vvalid live wherever gpu_resident() put them; keep val_pred on
        # the same device for the loss, then drop to CPU/numpy for the world-unit metric.
        val_loss = stage_b.gated_l1(val_pred, vtarget, vvalid).item() * RESIDUAL_SCALE

        # Final-height L1 in world units on valid pixels.
        final = vprior_up + val_pred.cpu().numpy() * RESIDUAL_SCALE
        final_l1 = float(np.abs(val_height - final)[val_valid_np].mean())

        train_loss_world = epoch_loss / max(1, batches) * RESIDUAL_SCALE
        logger.log_epoch(
            epoch,
            train_loss=train_loss_world,
            val_residual_l1=val_loss,
            val_final_l1=final_l1,
            prior_l1=prior_l1,
            block_reduce_l1=block_reduce_l1,
            lr=scheduler.get_last_lr()[0],
        )

        if final_l1 < best_val:
            best_val = final_l1
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {"base": 40, "in_channels": stage_b.IN_CHANNELS},
                    "height_scale": HEIGHT_SCALE,
                    "residual_scale": RESIDUAL_SCALE,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_final_l1": final_l1,
                },
                run_dir / "stage_b.pt",
            )

        if stopper.step(epoch, final_l1, train_loss_world):
            print(
                f"early stopping at epoch {epoch}: no val improvement for "
                f"{args.patience} epochs (best val_final_l1={stopper.best:.4f} @ epoch {stopper.best_epoch})"
                + (" [overtraining detected: train falling, val rising]" if stopper.overtraining else ""),
                flush=True,
            )
            break

    metrics = {
        "params": params,
        "best_val_final_l1": best_val,
        "best_epoch": stopper.best_epoch,
        "epochs_run": epoch,
        "early_stopped": stopper.stopped,
        "overtraining_detected": stopper.overtraining,
        "upsampled_prior_l1": prior_l1,
        "block_reduce_bilinear_l1": block_reduce_l1,
        "train_tiles": len(train_rows),
        "val_tiles": len(val_rows),
        "epochs": args.epochs,
        "patience": args.patience,
        "peak_vram_gb": train_common.peak_vram_gb(),
    }
    logger.write_json("stage_b_metrics.json", metrics)
    logger.close()
    print(
        f"done: best_val_final_l1={best_val:.4f} @ epoch {stopper.best_epoch} "
        f"(ran {epoch}/{args.epochs} epochs"
        + (", early stopped" if stopper.stopped else "")
        + (", overtraining detected" if stopper.overtraining else "")
        + f") vs prior={prior_l1:.4f} vs block_reduce={block_reduce_l1:.4f}; "
        f"checkpoint={run_dir / 'stage_b.pt'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
