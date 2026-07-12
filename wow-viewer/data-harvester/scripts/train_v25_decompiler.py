"""V25 terrain convergence trainer (Spec 102, Slice 9).

Trains the full universal pipeline from the lean V25 store:

  raw minimap ──> SegFormer decompiler ──> object mask + clean terrain map
                                       ──> object placements
                     final feats ────────> Stage A WDL prior (33x33)
  GT (or predicted) prior + clean map ──> progressive Sylvester solver (257x257)
                     final feats ────────> MTEX multi-hot + MCLY layers
                     final feats ────────> fractal params -> MCAL alpha maps

Every head the inference CLI needs is supervised here — no ground-truth
signal is fed to a head that inference cannot reproduce (the Stage B solver
is teacher-forced with the GT 33x33 prior by default; pass ``--student-prior``
to feed it Stage A's own prediction instead).

VRAM discipline (SC-102-001): ``--gradient-checkpointing``, ``--8bit-optimizer``,
``--amp-dtype``, and a peak-VRAM report written to ``peak_vram.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

DATA_HARVESTER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_HARVESTER_ROOT / "src"))

from harvester.v24.train_common import (  # noqa: E402
    RunLogger,
    peak_vram_gb,
    pick_device,
    configure_perf,
    set_determinism,
    split_rows,
)
from harvester.v25.dataset import V25TileSource  # noqa: E402
from harvester.v25.fractal import DifferentiableFractalGenerator, FractalParameterHead  # noqa: E402
from harvester.v25.lapnet import V25StageBPredictor  # noqa: E402
from harvester.v25.losses import V25UnifiedLoss  # noqa: E402
from harvester.v25.prior import V25StageAPredictor  # noqa: E402
from harvester.v25.segformer import V25SegformerDecompiler  # noqa: E402
from harvester.v25.texture import MclyDecoder, MtexPredictor  # noqa: E402

# Placement normalization constants (recorded in the checkpoint config).
COORD_SCALE = 17066.0   # half the WoW map extent in yards
ROT_SCALE = 180.0       # rotations in degrees -> [-2, 2] range
PLACEMENT_CLASSES = ("m2", "wmo")


class V25DecompilerDataset(torch.utils.data.Dataset):
    """Tensor view over a preloaded :class:`V25TileSource`."""

    def __init__(self, source: V25TileSource, rows: list[int], max_objects: int = 32):
        self.source = source
        self.rows = rows
        self.max_objects = max_objects
        self.vocab_size = source.vocab_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict:
        record = self.source.load(self.rows[idx])

        minimap = torch.from_numpy(record["minimap"]).permute(2, 0, 1)          # (3, 256, 256)
        clean = torch.from_numpy(record["clean_minimap"]).permute(2, 0, 1)      # (3, 256, 256)
        obj_mask = torch.from_numpy(np.clip(record["object_mask"], 0.0, 1.0)).unsqueeze(0)  # (1, 256, 256)
        height = torch.from_numpy(record["height_257"])                          # (257, 257)
        h_33 = torch.from_numpy(record["wdl_height_33"])                         # (33, 33)
        alpha = torch.from_numpy(record["alpha"]).permute(2, 0, 1)               # (4, 256, 256)
        mcly = torch.from_numpy(record["mcly_layer_mask"])                       # (16, 16, 4)
        vocab_ids = torch.from_numpy(record["mcly_vocab_ids"])                   # (16, 16, 4)

        # Liquid-aware height supervision: a 257-vertex is masked out when any
        # adjacent 256-cell carries liquid (water surface != terrain height).
        liquid_cells = record["liquid_mask"] > 0.5                               # (256, 256)
        liquid_verts = np.zeros((257, 257), dtype=bool)
        liquid_verts[:-1, :-1] |= liquid_cells
        liquid_verts[1:, :-1] |= liquid_cells
        liquid_verts[:-1, 1:] |= liquid_cells
        liquid_verts[1:, 1:] |= liquid_cells
        height_mask = torch.from_numpy((~liquid_verts).astype(np.float32))      # (257, 257)
        h_33_mask = height_mask[::8, ::8].clone()                                # (33, 33)

        # MTEX multi-hot over the tileset vocabulary (OOV bucket included).
        mtex_labels = torch.zeros(self.vocab_size, dtype=torch.float32)
        active = vocab_ids[vocab_ids >= 0]
        if active.numel() > 0:
            mtex_labels[active.long()] = 1.0

        # Placement targets: class = object kind, coords/rots normalized.
        class_ids = torch.zeros(self.max_objects, dtype=torch.long)
        coords = torch.zeros(self.max_objects, 3, dtype=torch.float32)
        rotations = torch.zeros(self.max_objects, 3, dtype=torch.float32)
        exist = torch.zeros(self.max_objects, dtype=torch.float32)
        for i, p in enumerate(record["placements"][: self.max_objects]):
            class_ids[i] = PLACEMENT_CLASSES.index(p["kind"]) if p["kind"] in PLACEMENT_CLASSES else 0
            coords[i] = torch.tensor(
                [p["pos_x"] / COORD_SCALE, p["pos_y"] / COORD_SCALE, p["pos_z"] / COORD_SCALE]
            )
            rotations[i] = torch.tensor(
                [p["rot_x"] / ROT_SCALE, p["rot_y"] / ROT_SCALE, p["rot_z"] / ROT_SCALE]
            )
            exist[i] = 1.0

        return {
            "minimap": minimap,
            "clean_rgb": clean,
            "mask": obj_mask,
            "h_257": height,
            "h_33": h_33,
            "height_mask": height_mask,
            "h_33_mask": h_33_mask,
            "alpha": alpha,
            "mcly_labels": mcly,
            "mtex_labels": mtex_labels,
            "class_ids": class_ids,
            "coords": coords,
            "rotations": rotations,
            "exist": exist,
        }


class V25Pipeline(torch.nn.Module):
    """All trainable V25 components behind one module for optimizer/checkpoint plumbing."""

    def __init__(self, vocab_size: int, num_classes: int, max_objects: int, device: str):
        super().__init__()
        self.decompiler = V25SegformerDecompiler(num_classes=num_classes, max_objects=max_objects)
        self.stage_a = V25StageAPredictor(in_channels=256)
        self.stage_b = V25StageBPredictor(device=device)
        self.mtex = MtexPredictor(in_channels=256, vocab_size=vocab_size)
        self.mcly = MclyDecoder(in_channels=256, num_layers=4)
        self.fractal_head = FractalParameterHead(in_channels=256, num_layers=4)
        self.fractal_gen = DifferentiableFractalGenerator(canvas_size=1024)

    def forward(self, minimap: torch.Tensor, prior_33: torch.Tensor | None = None) -> dict:
        """Full forward. When ``prior_33`` is None the pipeline is fully universal
        (Stage A's own prediction feeds the solver) — this is the inference path.
        """
        dec = self.decompiler(minimap)
        h_33_pred = self.stage_a(dec["final_feats"])

        solver_prior = prior_33 if prior_33 is not None else h_33_pred
        h_257 = self.stage_b(solver_prior, dec["clean_rgb"])

        mtex_logits = self.mtex(dec["final_feats"])
        mcly_logits = self.mcly(dec["final_feats"])

        params = self.fractal_head(dec["final_feats"])
        layers = []
        for l in range(4):
            noise = self.fractal_gen(
                params["offsets"][:, l],
                params["frequency"][:, l],
                params["persistence"][:, l],
                params["amplitude"][:, l],
            )
            layers.append(params["boundaries"][:, l] * noise)
        alpha_256 = torch.stack(layers, dim=1)

        return {
            "mask_logits": dec["mask_logits"],
            "clean_rgb": dec["clean_rgb"],
            "placements": dec["placements"],
            "h_33": h_33_pred,
            "h_257": h_257,
            "mtex_logits": mtex_logits,
            "mcly_logits": mcly_logits,
            "alpha_256": alpha_256,
            "fractal_params": params,
        }


def _batch_to_device(batch: dict, device: torch.device) -> dict:
    return {
        k: v.to(device, non_blocking=True) if isinstance(v, torch.Tensor) else v
        for k, v in batch.items()
    }


def _targets_from_batch(batch: dict, liquid_height_mask: bool = True) -> dict:
    targets = {
        "mask": batch["mask"],
        "clean_rgb": batch["clean_rgb"],
        "h_257": batch["h_257"],
        "h_33": batch["h_33"],
        "placements": {
            "class_ids": batch["class_ids"],
            "coords": batch["coords"],
            "rotations": batch["rotations"],
            "exist": batch["exist"],
        },
        "mtex_labels": batch["mtex_labels"],
        "mcly_labels": batch["mcly_labels"],
        "alpha_256": batch["alpha"],
    }
    if liquid_height_mask:
        targets["height_mask"] = batch["height_mask"]
        targets["h_33_mask"] = batch["h_33_mask"]
    return targets


def run_epoch(
    loader,
    pipeline: V25Pipeline,
    loss_fn: V25UnifiedLoss,
    optimizer,
    scaler,
    device: torch.device,
    amp_dtype: torch.dtype | None,
    student_prior: bool,
    train: bool,
    log_interval: int,
    epoch: int,
    liquid_height_mask: bool = True,
) -> dict:
    pipeline.train(train)
    totals: dict[str, float] = {}
    height_l1_sum = 0.0
    steps = 0

    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for batch_idx, batch in enumerate(loader):
            batch = _batch_to_device(batch, device)
            if train:
                optimizer.zero_grad(set_to_none=True)

            autocast = torch.amp.autocast(
                device.type, dtype=amp_dtype, enabled=amp_dtype is not None
            )
            with autocast:
                prior = None if student_prior else batch["h_33"]
                preds = pipeline(batch["minimap"], prior_33=prior)
                losses = loss_fn(
                    preds,
                    _targets_from_batch(batch, liquid_height_mask=liquid_height_mask),
                    minimap=batch["minimap"],
                )
                loss = losses["loss"]

            if train:
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()

            with torch.no_grad():
                height_l1_sum += F.l1_loss(preds["h_257"].float(), batch["h_257"].float()).item()
            for key, val in losses.items():
                if isinstance(val, torch.Tensor) and val.dim() == 0:
                    totals[key] = totals.get(key, 0.0) + val.item()
            steps += 1

            if train and log_interval and ((batch_idx + 1) % log_interval == 0 or batch_idx + 1 == len(loader)):
                print(
                    f"  [epoch {epoch}] step {batch_idx + 1}/{len(loader)} "
                    f"loss={loss.item():.4f} height={losses['height'].item():.4f} "
                    f"h33={losses.get('h_33', torch.tensor(0.0)).item():.4f} "
                    f"clean={losses.get('clean_rgb', torch.tensor(0.0)).item():.4f} "
                    f"alpha={losses['alpha'].item():.4f}",
                    flush=True,
                )

    out = {k: v / max(steps, 1) for k, v in totals.items()}
    out["height_l1"] = height_l1_sum / max(steps, 1)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="V25 terrain convergence trainer (Spec 102)")
    parser.add_argument("--v25-store", required=True, type=Path, help="lean V25 Zarr store")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DATA_HARVESTER_ROOT.parent / "output" / "train_v25_decompiler",
        help="run output directory (checkpoints, metrics)",
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--device", default=None, help="cpu/cuda override")
    parser.add_argument("--gradient-checkpointing", action="store_true",
                        help="enable SegFormer encoder gradient checkpointing (FR-102-501)")
    parser.add_argument("--8bit-optimizer", action="store_true",
                        help="use bitsandbytes 8-bit AdamW (FR-102-501)")
    parser.add_argument("--amp-dtype", choices=["fp16", "bf16", "none"], default="bf16",
                        help="autocast dtype on CUDA (default bf16 — fp16 overflows to NaN "
                             "in the Sylvester solver's eigen-basis matmuls on real heights)")
    parser.add_argument("--student-prior", action="store_true",
                        help="feed Stage A's own 33x33 prediction to the solver instead of the GT prior")
    parser.add_argument("--no-liquid-height-mask", action="store_true",
                        help="disable masking liquid areas out of the height/prior losses")
    parser.add_argument("--difficulty-buckets", nargs="+", default=None,
                        help="train only on tiles whose baked-in curation difficulty_bucket "
                             "matches (e.g. hard medium); default: all rows")
    parser.add_argument("--max-objects", type=int, default=32)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--val-interval", type=int, default=1)
    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--seed", type=int, default=102)
    parser.add_argument("--limit", type=int, default=None, help="cap loaded tiles (smoke runs)")
    args = parser.parse_args()

    set_determinism(args.seed, strict=False)
    device = pick_device(args.device)
    configure_perf(fast=True)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = RunLogger(args.output_dir)

    print(f"=== V25 trainer: store={args.v25_store} device={device} ===", flush=True)
    source = V25TileSource(args.v25_store)
    all_rows = source.rows_for_buckets(args.difficulty_buckets)
    if "difficulty_bucket" in source.index.column_names:
        from collections import Counter

        hist = Counter(
            str(source.index["difficulty_bucket"][r].as_py()) for r in all_rows
        )
        print(f"curation buckets in corpus: {dict(hist)}", flush=True)
    if args.limit is not None:
        all_rows = all_rows[: args.limit]
    train_rows, val_rows = split_rows(all_rows, args.val_fraction, args.seed)
    print(f"tiles: {len(all_rows)} (train {len(train_rows)} / val {len(val_rows)})", flush=True)

    # FR-102-502: one contiguous Zarr pass before the epoch loop.
    t0 = time.time()
    source.preload(all_rows)
    print(f"preload: {time.time() - t0:.1f}s", flush=True)

    train_ds = V25DecompilerDataset(source, train_rows, max_objects=args.max_objects)
    val_ds = V25DecompilerDataset(source, val_rows, max_objects=args.max_objects)
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        pin_memory=(device.type == "cuda"), num_workers=0,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        pin_memory=(device.type == "cuda"), num_workers=0,
    )

    pipeline = V25Pipeline(
        vocab_size=source.vocab_size,
        num_classes=len(PLACEMENT_CLASSES),
        max_objects=args.max_objects,
        device=device.type,
    )
    if args.gradient_checkpointing:
        pipeline.decompiler.encoder.gradient_checkpointing_enable()
        print("gradient checkpointing: enabled", flush=True)
    pipeline.to(device)

    n_params = sum(p.numel() for p in pipeline.parameters())
    print(f"pipeline parameters: {n_params:,}", flush=True)

    if getattr(args, "8bit_optimizer", False):
        try:
            import bitsandbytes as bnb

            optimizer = bnb.optim.AdamW8bit(pipeline.parameters(), lr=args.lr)
            print("optimizer: 8-bit AdamW (bitsandbytes)", flush=True)
        except ImportError:
            print("bitsandbytes unavailable; falling back to AdamW", flush=True)
            optimizer = torch.optim.AdamW(pipeline.parameters(), lr=args.lr)
    else:
        optimizer = torch.optim.AdamW(pipeline.parameters(), lr=args.lr)

    amp_dtype: torch.dtype | None = None
    scaler = None
    if device.type == "cuda" and args.amp_dtype != "none":
        amp_dtype = torch.float16 if args.amp_dtype == "fp16" else torch.bfloat16
        if amp_dtype is torch.float16:
            scaler = torch.amp.GradScaler("cuda")

    loss_fn = V25UnifiedLoss(use_freq_split=True, freq_cutoff=0.1)

    config = {
        "v25_store": str(args.v25_store),
        "vocab_size": source.vocab_size,
        "num_classes": len(PLACEMENT_CLASSES),
        "placement_classes": list(PLACEMENT_CLASSES),
        "max_objects": args.max_objects,
        "coord_scale": COORD_SCALE,
        "rot_scale": ROT_SCALE,
        "student_prior": args.student_prior,
        "liquid_height_mask": not args.no_liquid_height_mask,
        "difficulty_buckets": args.difficulty_buckets,
        "amp_dtype": args.amp_dtype,
        "gradient_checkpointing": args.gradient_checkpointing,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "seed": args.seed,
        "n_params": n_params,
    }
    logger.write_json("config.json", config)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        start = time.time()
        train_metrics = run_epoch(
            train_loader, pipeline, loss_fn, optimizer, scaler, device, amp_dtype,
            student_prior=args.student_prior, train=True,
            log_interval=args.log_interval, epoch=epoch,
            liquid_height_mask=not args.no_liquid_height_mask,
        )

        val_metrics = {}
        if val_rows and (epoch % args.val_interval == 0 or epoch == args.epochs):
            # Validation always runs the universal (student-prior) path: the
            # solver sees Stage A's own prediction, exactly like inference.
            val_metrics = run_epoch(
                val_loader, pipeline, loss_fn, optimizer=None, scaler=None,
                device=device, amp_dtype=amp_dtype, student_prior=True,
                train=False, log_interval=0, epoch=epoch,
                liquid_height_mask=not args.no_liquid_height_mask,
            )

        elapsed = time.time() - start
        record = {
            "train_loss": train_metrics["loss"],
            "train_height_l1": train_metrics["height_l1"],
            "epoch_seconds": round(elapsed, 1),
        }
        if val_metrics:
            record["val_loss"] = val_metrics["loss"]
            record["val_height_l1"] = val_metrics["height_l1"]
            record["val_h33"] = val_metrics.get("h_33", 0.0)
            record["val_mask"] = val_metrics.get("mask", 0.0)
        vram = peak_vram_gb()
        if vram is not None:
            record["peak_vram_gb"] = round(vram, 3)
        logger.log_epoch(epoch, **record)

        checkpoint = {
            "epoch": epoch,
            "pipeline": pipeline.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics.get("loss"),
        }
        torch.save(checkpoint, args.output_dir / "checkpoint_last.pt")
        if val_metrics and val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            torch.save(checkpoint, args.output_dir / "checkpoint_best.pt")
            print(f"  new best val_loss={best_val:.4f} @ epoch {epoch}", flush=True)

    vram = peak_vram_gb()
    if vram is not None:
        logger.write_json("peak_vram.json", {"peak_vram_gb": vram, "target_gb": 7.0, "pass": vram < 7.0})
        print(f"peak VRAM: {vram:.3f} GB (SC-102-001 target < 7.0)", flush=True)
    print(f"=== training complete; outputs in {args.output_dir} ===", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
