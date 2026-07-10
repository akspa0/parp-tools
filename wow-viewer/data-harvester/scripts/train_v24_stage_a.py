"""FR-012: train Stage A (minimap -> WDL prior correlation).

Target: the merged WDL prior. Sample weight: per-cell confidence, with
learned-fill cells excluded. The synthetic-WDL cheat channel is dropped with
probability --synth-dropout so the model also learns the minimap-only regime.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24 import stage_a, train_common  # noqa: E402
from harvester.v24.tiles import HEIGHT_SCALE, MultiTileSource, TileSource  # noqa: E402


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


def _load_tensors(source: TileSource | MultiTileSource, rows: list[int], label: str = "load",
                  minimap_only: bool = False):
    inputs, quincunxes, t_outer, t_inner = [], [], [], []
    w_outer, w_inner, s_outer, s_inner = [], [], [], []
    n = len(rows)
    step = max(1, n // 20)
    started = time.time()
    for i, row in enumerate(rows):
        record = source.load(row)
        if minimap_only:
            x = stage_a.build_minimap_only_input(record.cleaned_minimap)
            q = np.zeros((33, 33), dtype=np.float32)
        else:
            x, q = stage_a.build_input(record, include_synth=True)
        inputs.append(x)
        quincunxes.append(q)
        outer, inner, wo, wi = stage_a.build_target(record)
        t_outer.append(outer)
        t_inner.append(inner)
        w_outer.append(wo)
        w_inner.append(wi)
        s_outer.append(record.source_outer)
        s_inner.append(record.source_inner)
        if (i + 1) % step == 0 or (i + 1) == n:
            elapsed = time.time() - started
            pct = 100.0 * (i + 1) / n
            eta = elapsed / (i + 1) * (n - i - 1) if i else 0.0
            print(
                f"[{label}] {i + 1}/{n} tiles ({pct:.0f}%) "
                f"elapsed={elapsed:.1f}s eta={eta:.1f}s",
                flush=True,
            )
    return (
        torch.from_numpy(np.stack(inputs)),
        torch.from_numpy(np.stack(quincunxes)),
        torch.from_numpy(np.stack(t_outer)),
        torch.from_numpy(np.stack(t_inner)),
        torch.from_numpy(np.stack(w_outer)),
        torch.from_numpy(np.stack(w_inner)),
        torch.from_numpy(np.stack(s_outer)),
        torch.from_numpy(np.stack(s_inner)),
    )


def _drop_synth(
    x: torch.Tensor, q: torch.Tensor, drop_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Zero the synth channel (11), presence flag (12), and quincunx anchor."""
    drop_mask = drop_mask.to(x.device)
    x_out = x.clone()
    x_out[drop_mask, 11] = 0.0
    x_out[drop_mask, 12] = 0.0
    q_out = q.clone()
    q_out[drop_mask] = 0.0
    return x_out, q_out


def _eval_split(model, x, q, to, ti, wo, wi, so, si, device, include_synth: bool,
                minimap_only: bool = False):
    """Weighted L1 plus real-vs-synthetic cell split (world units)."""
    model.eval()
    with torch.no_grad():
        if minimap_only:
            xb = x
        elif include_synth:
            xb, qb = x, q
        else:
            xb, qb = _drop_synth(x, q, torch.ones(x.shape[0], dtype=torch.bool))
        po, pi = (model(xb.to(device)) if minimap_only
                  else model(xb.to(device), qb.to(device)))
        # to/wo/etc. may be GPU-resident (train_common.gpu_resident) or CPU;
        # match whichever device val_data actually lives on.
        po, pi = po.to(to.device), pi.to(to.device)

    loss = stage_a.weighted_l1(po, pi, to, ti, wo, wi).item() * HEIGHT_SCALE

    def cell_l1(source_value: int) -> float | None:
        mask_o = (so == source_value).float() * wo.sign()
        mask_i = (si == source_value).float() * wi.sign()
        denom = mask_o.sum() + mask_i.sum()
        if denom.item() == 0:
            return None
        num = (mask_o * (po - to).abs()).sum() + (mask_i * (pi - ti).abs()).sum()
        return (num / denom).item() * HEIGHT_SCALE

    return loss, cell_l1(0), cell_l1(1)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v24-store", required=True, nargs="+",
                         help="one or more V24 stores; multiple stores (e.g. different builds) are concatenated")
    parser.add_argument("--v18-store", default=None, nargs="*",
                         help="matching V18 store overrides, one per --v24-store (omit to use each store's own v18_store_path attr)")
    parser.add_argument("--output", required=True, help="run directory (created)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=94)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--minimap-only", action="store_true", default=False,
                        help="train 3-channel minimap-only Stage A (no alpha/normal/mcnr/quincunx)")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--synth-dropout", type=float, default=0.5)
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
                        default=[16, 32, 64, 96, 128, 192, 256, 384, 512])
    parser.add_argument("--autotune-safety-factor", type=float, default=0.85)
    args = parser.parse_args()

    train_common.set_determinism(args.seed, strict=False)
    train_common.configure_perf(args.cudnn_benchmark)
    device = train_common.pick_device(args.device)
    run_dir = Path(args.output)
    logger = train_common.RunLogger(run_dir)
    print(f"stage A training: device={device} run_dir={run_dir}")

    source = _build_source(args.v24_store, args.v18_store)
    rows = source.usable_rows()
    if args.limit:
        rows = rows[: args.limit]
    train_rows, val_rows = train_common.split_rows(rows, args.val_fraction, args.seed)
    print(f"tiles: {len(rows)} usable -> {len(train_rows)} train / {len(val_rows)} val")

    started_load = time.time()
    # Preload V18 data in one sequential Zarr pass (no per-tile random-access seeks).
    source.preload(train_rows + val_rows)
    train_data = _load_tensors(source, train_rows, label="load train",
                               minimap_only=args.minimap_only)
    val_data = _load_tensors(source, val_rows, label="load val",
                             minimap_only=args.minimap_only)
    print(f"loaded tensors in {time.time() - started_load:.1f}s", flush=True)

    if args.gpu_resident_data:
        train_data = train_common.gpu_resident(train_data, device)
        val_data = train_common.gpu_resident(val_data, device)
        print(f"train/val tensors resident={train_data[0].is_cuda if device.type == 'cuda' else False}", flush=True)

    if args.minimap_only:
        model = stage_a.StageAMinimapOnly().to(device)
    else:
        model = stage_a.StageAModel().to(device)
    params = stage_a.parameter_count(model)
    print(f"stage A params: {params} {'(minimap-only 3ch)' if args.minimap_only else ''}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    amp_dtype = torch.bfloat16 if args.amp_dtype == "bf16" else torch.float16
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda" and amp_dtype == torch.float16)

    minimap_only = args.minimap_only
    x, q, to, ti, wo, wi, so, si = train_data
    n = x.shape[0]

    if args.autotune_batch_size:
        candidates = [c for c in sorted(args.autotune_batch_candidates) if c <= n]
        if candidates:
            snapshot = train_common.snapshot_state(model, optimizer)

            def _try_step(bs: int) -> None:
                idx = torch.randperm(n)[:bs].to(x.device)
                if minimap_only:
                    xb = x[idx].to(device, non_blocking=True)
                else:
                    drop = torch.rand(len(idx)) < args.synth_dropout
                    xb, qb = _drop_synth(x[idx], q[idx], drop)
                    xb, qb = xb.to(device, non_blocking=True), qb.to(device, non_blocking=True)
                tob, tib = to[idx].to(device, non_blocking=True), ti[idx].to(device, non_blocking=True)
                wob, wib = wo[idx].to(device, non_blocking=True), wi[idx].to(device, non_blocking=True)
                optimizer.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                    po, pi = model(xb) if minimap_only else model(xb, qb)
                    loss = stage_a.weighted_l1(po, pi, tob, tib, wob, wib)
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
        for start in range(0, n, args.batch_size):
            idx = perm[start : start + args.batch_size].to(x.device)
            if minimap_only:
                xb = x[idx].to(device, non_blocking=True)
            else:
                drop = torch.rand(len(idx), generator=generator) < args.synth_dropout
                xb, qb = _drop_synth(x[idx], q[idx], drop)
                xb, qb = xb.to(device, non_blocking=True), qb.to(device, non_blocking=True)
            tob, tib = to[idx].to(device, non_blocking=True), ti[idx].to(device, non_blocking=True)
            wob, wib = wo[idx].to(device, non_blocking=True), wi[idx].to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=amp_dtype, enabled=device.type == "cuda"):
                po, pi = model(xb) if minimap_only else model(xb, qb)
                loss = stage_a.weighted_l1(po, pi, tob, tib, wob, wib)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            batches += 1

            if batches % args.log_interval == 0 or start + args.batch_size >= n:
                elapsed = time.time() - epoch_started
                pct = 100.0 * batches / n_batches
                eta = elapsed / batches * (n_batches - batches) if batches else 0.0
                print(
                    f"epoch {epoch} batch {batches}/{n_batches} ({pct:.0f}%) "
                    f"loss={loss.item() * HEIGHT_SCALE:.4f} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )

        scheduler.step()
        if minimap_only:
            val_loss, val_real, val_synth = _eval_split(
                model, *val_data, device=device, include_synth=True,
                minimap_only=True
            )
        else:
            val_loss, val_real, val_synth = _eval_split(
                model, *val_data, device=device, include_synth=True
            )
            val_nosynth, _, _ = _eval_split(
                model, *val_data, device=device, include_synth=False
            )

        train_loss_world = epoch_loss / max(1, batches) * HEIGHT_SCALE
        log_kw = dict(
            epoch=epoch,
            train_loss=train_loss_world,
            val_l1=val_loss,
            val_l1_real_cells=val_real if val_real is not None else -1.0,
            val_l1_synth_cells=val_synth if val_synth is not None else -1.0,
            lr=scheduler.get_last_lr()[0],
        )
        if not minimap_only:
            log_kw["val_l1_minimap_only"] = val_nosynth
        logger.log_epoch(**log_kw)

        if val_loss < best_val:
            best_val = val_loss
            in_channels = (
                stage_a.IN_CHANNELS_MINIMAP_ONLY
                if args.minimap_only
                else stage_a.IN_CHANNELS
            )
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config": {
                        "base": 28,
                        "in_channels": in_channels,
                        "minimap_only": bool(args.minimap_only),
                    },
                    "height_scale": HEIGHT_SCALE,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val_l1": val_loss,
                },
                run_dir / "stage_a.pt",
            )

        if stopper.step(epoch, val_loss, train_loss_world):
            print(
                f"early stopping at epoch {epoch}: no val improvement for "
                f"{args.patience} epochs (best val_l1={stopper.best:.4f} @ epoch {stopper.best_epoch})"
                + (" [overtraining detected: train falling, val rising]" if stopper.overtraining else ""),
                flush=True,
            )
            break

    metrics = {
        "params": params,
        "best_val_l1": best_val,
        "best_epoch": stopper.best_epoch,
        "epochs_run": epoch,
        "early_stopped": stopper.stopped,
        "overtraining_detected": stopper.overtraining,
        "train_tiles": len(train_rows),
        "val_tiles": len(val_rows),
        "epochs": args.epochs,
        "patience": args.patience,
        "peak_vram_gb": train_common.peak_vram_gb(),
    }
    logger.write_json("stage_a_metrics.json", metrics)
    logger.close()
    print(
        f"done: best_val_l1={best_val:.4f} world units @ epoch {stopper.best_epoch} "
        f"(ran {epoch}/{args.epochs} epochs"
        + (", early stopped" if stopper.stopped else "")
        + (", overtraining detected" if stopper.overtraining else "")
        + f"); checkpoint={run_dir / 'stage_a.pt'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
