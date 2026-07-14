"""Spec 103 T010 — lean v7 trainer (revived MultiChannelUNetV7 on clean signals).

One U-Net, one signal: [minimap + normals + WDL prior + aux] -> terrain height, residual over
the WDL "trestle" prior. Quick-and-dirty per the plan: no object-mask loss gating; the loss is
the ported v7 combined_loss WITHOUT the PatchGAN (or --loss l1 for pure regression).

Conveniences carried over from the committed spec102 simple trainer: complete-holdout split,
AMP, EMA (the deploy weights), LR warmup + cosine, gradient clipping, early stopping, fully
resumable checkpoints, warm start, blank-tile filter. New here:

- --wdl-prior-dropout P: per-sample chance to fill ch 6 with 0.5 (v7's missing-prior fallback)
  so one model serves prior-present and prior-absent tiles. Val runs twice: with the prior and
  with it dropped (val_no_prior) to watch that robustness directly.
- --height-hints gt|wdl|none: ch 7/8 source (gt = v7-faithful tile bounds; wdl = derived from
  the prior, deployment-consistent).
- --max-object-coverage C: drop tiles whose object_precise_mask covers more than C of the tile
  (1.0 keeps everything, v7-faithful; ~0.02 trains object-free only). Tile *selection*, not
  pixel gating.

FR-011: history.json records the exact command, store identity, split, per-epoch metrics, and
peak VRAM. GT-based val L1 here is a development diagnostic; acceptance is the label-free
harness (validate_spec103_labelfree.py).

Run from wow-viewer/data-harvester/ (USER runs training — AGENTS RULE 0):

    uv run python scripts/train_spec103_v7.py \
        --store ../output/datasets/spec103/synthetic_v1.zarr \
        --output ../output/spec103_v7_synth_v1 --val-key pattern --val-value crater \
        --epochs 60 --batch 4 --wdl-prior-dropout 0.25

    uv run python scripts/train_spec103_v7.py \
        --store ../output/datasets/v18/3_3_5_12340.zarr \
        --output ../output/spec103_v7_real_v1 --val-key map --val-value Azeroth \
        --epochs 80 --batch 8 --wdl-prior-dropout 0.25
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from PIL import Image
from torch.utils.data import DataLoader, Dataset

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.spec103.v7_inputs import (  # noqa: E402
    WORKING_SIZE,
    assemble_v7_input,
    build_v7_targets,
)
from harvester.spec103.v7_losses import combined_loss  # noqa: E402
from harvester.spec103.v7_model import (  # noqa: E402
    MODEL_VARIANT_WDL_TRESTLE_REFLECT,
    MultiChannelUNetV7,
)

OPTIONAL_ARRAYS = ("normal_xyz", "liquid_mask", "liquid_height", "object_precise_mask")


class V7TileDataset(Dataset):
    def __init__(self, group: zarr.Group, rows: list[int], *, prior_dropout: float,
                 height_hints: str, force_drop_prior: bool = False) -> None:
        self.group = group
        self.rows = rows
        self.prior_dropout = float(prior_dropout)
        self.height_hints = height_hints
        self.force_drop_prior = force_drop_prior
        self.present = {name: name in group for name in OPTIONAL_ARRAYS}

    def __len__(self) -> int:
        return len(self.rows)

    def _optional(self, name: str, row: int):
        return np.asarray(self.group[name][row]) if self.present[name] else None

    def __getitem__(self, i: int):
        r = self.rows[i]
        height = np.asarray(self.group["height_257"][r], dtype=np.float32)
        drop = self.force_drop_prior or (self.prior_dropout > 0.0 and float(np.random.random()) < self.prior_dropout)
        x = assemble_v7_input(
            minimap_rgb=np.asarray(self.group["minimap_rgb"][r]),
            height_257=height,
            normal_xyz=self._optional("normal_xyz", r),
            liquid_mask=self._optional("liquid_mask", r),
            liquid_height=self._optional("liquid_height", r),
            object_mask=self._optional("object_precise_mask", r),
            height_hints=self.height_hints,
            drop_wdl_prior=drop,
        )
        target, bounds = build_v7_targets(height)
        return x, target, bounds


def rgb_std_per_tile(rgb) -> np.ndarray:
    n = rgb.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for a in range(0, n, 256):
        b = min(n, a + 256)
        out[a:b] = np.asarray(rgb[a:b]).reshape(b - a, -1).astype(np.float32).std(axis=1)
    return out


def object_coverage_per_tile(group: zarr.Group) -> np.ndarray:
    if "object_precise_mask" not in group:
        return np.zeros(group["minimap_rgb"].shape[0], dtype=np.float32)
    mask = group["object_precise_mask"]
    n = mask.shape[0]
    out = np.zeros(n, dtype=np.float32)
    for a in range(0, n, 256):
        b = min(n, a + 256)
        out[a:b] = (np.asarray(mask[a:b]) > 0.5).reshape(b - a, -1).mean(axis=1)
    return out


def compute_loss(outputs, pred_bounds, target, bounds, inputs, mode: str, detail_weight: float):
    if mode == "l1":
        loss = torch.nn.functional.l1_loss(outputs[:, :2], target)
        return loss, {"l1": float(loss.item())}
    return combined_loss(outputs, pred_bounds, target, bounds, input_context=inputs,
                         detail_head_weight=detail_weight)


@torch.no_grad()
def evaluate(model, loader, device, use_amp: bool, loss_mode: str, detail_weight: float) -> dict:
    model.eval()
    l1_global = l1_local = loss_sum = n = 0.0
    for x, target, bounds in loader:
        x, target, bounds = (t.to(device, non_blocking=True) for t in (x, target, bounds))
        with torch.amp.autocast("cuda", enabled=use_amp):
            outputs, pred_bounds = model(x)
            loss, _ = compute_loss(outputs, pred_bounds, target, bounds, x, loss_mode, detail_weight)
        l1_global += float((outputs[:, 0:1] - target[:, 0:1]).abs().mean()) * x.size(0)
        l1_local += float((outputs[:, 1:2] - target[:, 1:2]).abs().mean()) * x.size(0)
        loss_sum += float(loss) * x.size(0)
        n += x.size(0)
    n = max(n, 1.0)
    return {"loss": loss_sum / n, "l1_global": l1_global / n, "l1_local": l1_local / n}


@torch.no_grad()
def render_preview(model, dataset, preview_indices, device, out_path: Path, epoch: int, val: dict) -> None:
    """minimap | WDL prior | predicted global | GT global, one row per preview tile."""
    model.eval()
    rows = []
    for i in preview_indices:
        x, target, _ = dataset[i]
        outputs, _ = model(x.unsqueeze(0).to(device))
        pred = outputs[0, 0].float().cpu().numpy()
        gt = target[0].numpy()
        prior = x[6].numpy()
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
        rgb = np.clip((x[0:3].numpy() * std + mean) * 255.0, 0, 255).astype(np.uint8).transpose(1, 2, 0)

        def gray(a: np.ndarray) -> np.ndarray:
            lo, hi = float(a.min()), float(a.max())
            g = ((a - lo) / max(hi - lo, 1e-6) * 255.0).astype(np.uint8)
            return np.repeat(g[:, :, None], 3, axis=2)

        gap = np.full((WORKING_SIZE, 2, 3), 255, np.uint8)
        rows.append(np.concatenate([rgb, gap, gray(prior), gap, gray(pred), gap, gray(gt)], axis=1))
    canvas = np.concatenate(rows, axis=0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    img = Image.fromarray(canvas)
    img.save(out_path)
    print(f"[preview] epoch {epoch} l1_global={val['l1_global']:.4f} -> {out_path}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 103 v7 trainer")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--val-key", default="map", help="index.parquet column for the complete holdout (map, pattern, ...)")
    ap.add_argument("--val-value", default="Azeroth")
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup", type=int, default=3)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--ema-decay", type=float, default=0.999)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--curation-manifest", type=Path, default=None,
                    help="spec103_curate_dataset.py output dir or curation_manifest.parquet; restricts "
                         "training+val to kept (clean-terrain) tiles. The recommended path.")
    ap.add_argument("--min-rgb-std", type=float, default=1.0, help="drop blank-minimap tiles below this RGB std")
    ap.add_argument("--max-object-coverage", type=float, default=0.02,
                    help="drop tiles with object coverage above this fraction (spec Principle #5: object "
                         "tiles are impossible height targets). Ignored when --curation-manifest is given. "
                         "Use 1.0 for the v7-faithful keep-all behavior.")
    ap.add_argument("--wdl-prior-dropout", type=float, default=0.25)
    ap.add_argument("--height-hints", choices=["gt", "wdl", "none"], default="gt")
    ap.add_argument("--loss", choices=["v7", "l1"], default="v7")
    ap.add_argument("--detail-head", action="store_true", help="V7.7 3-channel detail head")
    ap.add_argument("--detail-head-weight", type=float, default=0.10)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init-weights", type=Path, default=None)
    ap.add_argument("--no-amp", action="store_true")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available; refusing to train on CPU.")
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda")
    use_amp = not args.no_amp
    args.output.mkdir(parents=True, exist_ok=True)
    torch.cuda.reset_peak_memory_stats(device)

    print(f"[load] {args.store}", flush=True)
    group = zarr.open_group(str(args.store), mode="r")
    for required in ("minimap_rgb", "height_257"):
        if required not in group:
            raise SystemExit(f"store lacks required array {required!r}")
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    if args.val_key not in index[0]:
        raise SystemExit(f"index.parquet has no column {args.val_key!r}")

    coverage = object_coverage_per_tile(group)
    if args.curation_manifest is not None:
        manifest_path = args.curation_manifest
        if manifest_path.is_dir():
            manifest_path = manifest_path / "curation_manifest.parquet"
        curation = pq.read_table(manifest_path).to_pylist()
        kept_ids = {int(r["tile_id"]) for r in curation if r["keep"]}
        keep = np.array([int(row["tile_id"]) in kept_ids for row in index], dtype=bool)
        curation_note = f"curation_manifest={manifest_path.name} kept={len(kept_ids)}"
    else:
        keep = rgb_std_per_tile(group["minimap_rgb"]) >= args.min_rgb_std
        keep &= coverage <= args.max_object_coverage
        curation_note = (f"inline dropped_blank={int((rgb_std_per_tile(group['minimap_rgb']) < args.min_rgb_std).sum())} "
                         f"dropped_objects={int((coverage > args.max_object_coverage).sum())}")
    val_rows = [i for i, row in enumerate(index) if str(row[args.val_key]) == args.val_value and keep[i]]
    train_rows = [i for i, row in enumerate(index) if str(row[args.val_key]) != args.val_value and keep[i]]
    if not val_rows or not train_rows:
        raise SystemExit(f"bad holdout: val={len(val_rows)} train={len(train_rows)} for {args.val_key}={args.val_value!r} "
                         f"(after curation: {int(keep.sum())} tiles kept)")
    print(f"[split] train={len(train_rows)} val={len(val_rows)} holdout {args.val_key}={args.val_value} "
          f"{curation_note} hints={args.height_hints} prior_dropout={args.wdl_prior_dropout} loss={args.loss}", flush=True)

    train_ds = V7TileDataset(group, train_rows, prior_dropout=args.wdl_prior_dropout, height_hints=args.height_hints)
    val_ds = V7TileDataset(group, val_rows, prior_dropout=0.0, height_hints=args.height_hints)
    val_noprior_ds = V7TileDataset(group, val_rows, prior_dropout=0.0, height_hints=args.height_hints, force_drop_prior=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)
    val_noprior_loader = DataLoader(val_noprior_ds, batch_size=args.batch, shuffle=False, num_workers=0, pin_memory=True)

    out_channels = 3 if args.detail_head else 2
    model = MultiChannelUNetV7(
        out_channels=out_channels,
        use_wdl_global_trestle=True,
        use_detail_head=args.detail_head,
        output_size=WORKING_SIZE,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[model] MultiChannelUNetV7 variant={MODEL_VARIANT_WDL_TRESTLE_REFLECT} out_ch={out_channels} params={n_params:,}", flush=True)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    def lr_lambda(e: int) -> float:
        if e < args.warmup:
            return (e + 1) / max(1, args.warmup)
        prog = (e - args.warmup) / max(1, args.epochs - args.warmup)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, prog)))

    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    ema_model = copy.deepcopy(model)
    for p in ema_model.parameters():
        p.requires_grad_(False)

    detail_weight = args.detail_head_weight if args.detail_head else 0.0
    history, best_l1, start_epoch = [], float("inf"), 1
    resume_path = args.output / "checkpoint_last.pt"
    if args.resume and resume_path.exists():
        ck = torch.load(resume_path, map_location=device)
        model.load_state_dict(ck["model"])
        ema_model.load_state_dict(ck.get("ema", ck["model"]))
        if "opt" in ck:
            opt.load_state_dict(ck["opt"])
        if "sched" in ck:
            sched.load_state_dict(ck["sched"])
        if "scaler" in ck:
            scaler.load_state_dict(ck["scaler"])
        start_epoch = int(ck.get("epoch", 0)) + 1
        best_l1 = float(ck.get("best_l1", float("inf")))
        history = ck.get("history", []) or []
        print(f"[resume] {resume_path} -> epoch {start_epoch}, best_l1={best_l1:.4f}", flush=True)
    elif args.resume:
        print(f"[resume] no checkpoint at {resume_path}; starting fresh", flush=True)
    elif args.init_weights is not None:
        ck = torch.load(args.init_weights, map_location=device)
        state = ck.get("model", ck)
        try:
            model.load_state_dict(state)
            loaded = "full"
        except RuntimeError:
            missing = model.load_state_dict(state, strict=False)
            loaded = f"partial (unmatched: {list(missing.missing_keys) + list(missing.unexpected_keys)})"
        ema_model.load_state_dict(model.state_dict())
        print(f"[init] warm-start {loaded} from {args.init_weights}", flush=True)

    # object-densest val tiles make the most telling previews; fall back to first tiles
    preview_indices = list(np.argsort(-coverage[np.asarray(val_rows)])[:4]) if len(val_rows) > 4 else list(range(len(val_rows)))

    run_identity = {
        "command": " ".join(sys.argv),
        "store": str(args.store.resolve()),
        "val_key": args.val_key, "val_value": args.val_value,
        "train_rows": len(train_rows), "val_rows": len(val_rows),
        "curation": (str(args.curation_manifest.resolve()) if args.curation_manifest is not None
                     else f"inline max_object_coverage={args.max_object_coverage} min_rgb_std={args.min_rgb_std}"),
        "model_variant": MODEL_VARIANT_WDL_TRESTLE_REFLECT + ("-v77" if args.detail_head else ""),
        "params": n_params, "loss": args.loss, "height_hints": args.height_hints,
        "wdl_prior_dropout": args.wdl_prior_dropout, "max_object_coverage": args.max_object_coverage,
        "seed": args.seed,
    }

    epochs_no_improve = 0
    for epoch in range(start_epoch, args.epochs + 1):
        model.train()
        t0, run = time.time(), 0.0
        for x, target, bounds in train_loader:
            x, target, bounds = (t.to(device, non_blocking=True) for t in (x, target, bounds))
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                outputs, pred_bounds = model(x)
                loss, _ = compute_loss(outputs, pred_bounds, target, bounds, x, args.loss, detail_weight)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            with torch.no_grad():
                for e, p in zip(ema_model.parameters(), model.parameters()):
                    e.mul_(args.ema_decay).add_(p.detach(), alpha=1.0 - args.ema_decay)
                for eb, b in zip(ema_model.buffers(), model.buffers()):
                    eb.copy_(b)
            run += float(loss)
        sched.step()

        val = evaluate(ema_model, val_loader, device, use_amp, args.loss, detail_weight)
        val_noprior = evaluate(ema_model, val_noprior_loader, device, use_amp, args.loss, detail_weight)
        peak_vram_gb = torch.cuda.max_memory_allocated(device) / (1024 ** 3)
        rec = {"epoch": epoch, "train_loss": run / max(len(train_loader), 1), "lr": opt.param_groups[0]["lr"],
               "val": val, "val_no_prior": val_noprior, "peak_vram_gb": round(peak_vram_gb, 2),
               "secs": round(time.time() - t0, 1)}
        history.append(rec)
        star = ""
        if val["l1_global"] < best_l1:
            best_l1 = val["l1_global"]
            epochs_no_improve = 0
            star = " *best"
            torch.save({"model": ema_model.state_dict(), "val": val, "val_no_prior": val_noprior,
                        "epoch": epoch, "run_identity": run_identity,
                        "model_variant": run_identity["model_variant"],
                        "use_wdl_global_trestle": True, "use_detail_head": args.detail_head,
                        "output_size": WORKING_SIZE, "height_hints": args.height_hints},
                       args.output / "checkpoint_best.pt")
            render_preview(ema_model, val_ds, preview_indices, device,
                           args.output / "val_previews" / f"best_epoch_{epoch:03d}.png", epoch, val)
        else:
            epochs_no_improve += 1
        torch.save({"model": model.state_dict(), "ema": ema_model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "scaler": scaler.state_dict(), "epoch": epoch,
                    "best_l1": best_l1, "history": history}, args.output / "checkpoint_last.pt")
        (args.output / "history.json").write_text(
            json.dumps({**run_identity, "history": history, "best_val_l1_global": best_l1}, indent=2),
            encoding="utf-8")
        print(f"[EPOCH {epoch}/{args.epochs}] loss {rec['train_loss']:.4f}/{val['loss']:.4f}  "
              f"l1_g {val['l1_global']:.4f} l1_l {val['l1_local']:.4f}  "
              f"noprior_l1_g {val_noprior['l1_global']:.4f}  vram {rec['peak_vram_gb']}G  "
              f"lr={rec['lr']:.2e} ({rec['secs']}s){star}", flush=True)
        if args.patience and epochs_no_improve >= args.patience:
            print(f"[early-stop] no val improvement for {args.patience} epochs; stopping at {epoch}", flush=True)
            break
    print(f"[DONE] best_val_l1_global={best_l1:.4f} -> {args.output / 'checkpoint_best.pt'}", flush=True)
    print("[NOTE] GT val L1 is a development diagnostic (spec FR-007); acceptance is the label-free "
          "harness: scripts/validate_spec103_labelfree.py", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
