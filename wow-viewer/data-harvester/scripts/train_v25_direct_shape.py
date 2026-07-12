"""Train the direct minimap -> terrain-shape model and prove it converges.

Predicts only the learnable half of the heightmap (detrended relief), on the
same frozen complete-map holdout split. Reports held-out shape MAE against the
flat-shape floor and the true/predicted shape correlation, and renders
side-by-side proof PNGs (input minimap | true shape | predicted shape) so the
result is visible, not just a number.

Not a residual-cascade decision gate: this is a convergence demonstration for
the honest sub-problem the signal diagnostic identified. Deterministic, small
U-Net, bf16, CUDA.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from harvester.v24.train_common import RunLogger, configure_perf, peak_vram_gb, set_determinism
from harvester.v25.direct_shape import DirectShapeUNet, parameter_count


def git_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=ROOT.parent, capture_output=True, text=True, check=False
    )
    return result.stdout.strip() or "unknown"


def preload(store: Path, split_by_row: dict[int, str], batch: int = 128):
    """Cleaned minimap (3,256,256) in [0,1], detrended target height (256,256),
    and valid (non-liquid) mask (256,256) per tile.

    Target: height_257 area-resized to 256 (cell grid, aligns with the 256
    liquid mask), then per-tile mean over valid pixels subtracted -> pure
    shape. The absolute datum is deliberately discarded; it is not a minimap-
    predictable quantity (see the signal diagnostic).
    """
    group = zarr.open_group(str(store), mode="r")
    clean_array = group["clean_minimap_256"]
    height_array = group["height_257"]
    liquid_array = group["liquid_mask_256"]
    values = {name: {"rgb": [], "shape": [], "mask": []} for name in set(split_by_row.values())}
    rows = sorted(split_by_row)
    for start in range(0, len(rows), batch):
        chunk = rows[start:start + batch]
        lo, hi = chunk[0], chunk[-1] + 1
        clean_block = np.asarray(clean_array[lo:hi], dtype=np.float32) / 255.0
        h_block = np.asarray(height_array[lo:hi], dtype=np.float32)
        liq_block = np.asarray(liquid_array[lo:hi])
        h256 = F.interpolate(
            torch.from_numpy(h_block)[:, None], size=(256, 256), mode="area"
        ).squeeze(1).numpy()
        for row in chunk:
            i = row - lo
            valid = (liq_block[i] <= 127) & np.isfinite(h256[i])
            if valid.sum() < 64:
                continue
            h = h256[i].copy()
            h[~np.isfinite(h)] = 0.0
            tile_mean = float(h[valid].mean())
            shape = (h - tile_mean).astype(np.float32)
            split = split_by_row[row]
            values[split]["rgb"].append(np.moveaxis(clean_block[i], -1, 0))
            values[split]["shape"].append(shape)
            values[split]["mask"].append(valid.astype(np.float32))
    return {
        name: (
            torch.from_numpy(np.stack(v["rgb"])),
            torch.from_numpy(np.stack(v["shape"])),
            torch.from_numpy(np.stack(v["mask"])),
        )
        for name, v in values.items()
        if v["rgb"]
    }


def make_loader(rgb, shape, mask, batch_size, shuffle):
    return torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(rgb, shape, mask), batch_size=batch_size,
        shuffle=shuffle, pin_memory=True, num_workers=0,
    )


def masked_mean(x, mask):
    return (x * mask).sum() / mask.sum().clamp(min=1)


def run_epoch(model, loader, device, optimizer=None):
    training = optimizer is not None
    model.train(training)
    err_sum, valid_sum = 0.0, 0
    ctx = torch.enable_grad() if training else torch.no_grad()
    with ctx:
        for rgb, shape, mask in loader:
            rgb, shape, mask = (t.to(device, non_blocking=True) for t in (rgb, shape, mask))
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model(rgb)
                loss = masked_mean((pred.float() - shape).abs(), mask) / model.height_scale
            if training:
                loss.backward()
                optimizer.step()
            err_sum += float(((pred.float() - shape).abs() * mask).sum().item())
            valid_sum += int(mask.sum().item())
    return err_sum / max(valid_sum, 1)


def shape_correlation(model, loader, device):
    """Mean per-tile Pearson correlation between predicted and true shape over
    valid pixels -- 'does the predicted relief point the same way as truth'."""
    model.eval()
    corrs = []
    with torch.no_grad():
        for rgb, shape, mask in loader:
            rgb, shape, mask = (t.to(device) for t in (rgb, shape, mask))
            pred = model(rgb).float()
            for i in range(rgb.shape[0]):
                m = mask[i] > 0.5
                if m.sum() < 64:
                    continue
                p = pred[i][m]
                t = shape[i][m]
                p = p - p.mean()
                t = t - t.mean()
                denom = (p.norm() * t.norm()).clamp(min=1e-6)
                corrs.append(float((p * t).sum() / denom))
    return float(np.mean(corrs)) if corrs else 0.0


def flat_floor(shape, mask):
    return float(((shape.abs() * mask).sum() / mask.sum().clamp(min=1)).item())


_COLORS = np.array([[13, 71, 161], [255, 255, 255], [183, 28, 28]], dtype=np.float32)  # low/mid/high


def colorize(field, mask=None, symmetric=True):
    f = field.astype(np.float32).copy()
    if mask is not None:
        f[mask < 0.5] = 0.0
    lim = np.percentile(np.abs(f), 99) if symmetric else max(np.abs(f).max(), 1e-6)
    lim = max(lim, 1e-6)
    t = np.clip((f / lim + 1.0) / 2.0, 0.0, 1.0)  # [-lim,lim] -> [0,1]
    idx = t * 2.0
    lo = np.clip(np.floor(idx).astype(int), 0, 1)
    frac = (idx - lo)[..., None]
    rgb = _COLORS[lo] * (1 - frac) + _COLORS[lo + 1] * frac
    return rgb.astype(np.uint8)


def render_proof(model, tensors, device, out_path, n=6):
    from PIL import Image

    rgb, shape, mask = tensors
    model.eval()
    with torch.no_grad():
        pred = model(rgb[:n].to(device)).float().cpu().numpy()
    rows = []
    for i in range(min(n, rgb.shape[0])):
        minimap = np.moveaxis(rgb[i].numpy(), 0, -1)
        minimap = (minimap * 255).clip(0, 255).astype(np.uint8)
        m = mask[i].numpy()
        true_img = colorize(shape[i].numpy(), m)
        pred_img = colorize(pred[i], m)
        rows.append(np.concatenate([minimap, true_img, pred_img], axis=1))
    grid = np.concatenate(rows, axis=0)
    Image.fromarray(grid).save(out_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Train direct minimap->shape model")
    parser.add_argument("--v25-store", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--base", type=int, default=32)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=102)
    args = parser.parse_args()

    if args.device != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("CUDA required; refusing silent CPU fallback")
    set_determinism(args.seed, strict=False)
    configure_perf(True)
    device = torch.device("cuda")
    torch.cuda.reset_peak_memory_stats()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_by_row = {int(r["row"]): str(r["split"]) for r in manifest["rows"]}
    started = time.time()
    prepared = preload(args.v25_store, split_by_row)
    print(f"preload {time.time() - started:.1f}s; "
          f"{ {k: v[0].shape[0] for k, v in prepared.items()} }", flush=True)

    model = DirectShapeUNet(base=args.base).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    logger = RunLogger(args.output_dir)

    train_loader = make_loader(*prepared["train"], args.batch_size, True)
    val_loader = make_loader(*prepared["validation_map"], args.batch_size, False)
    era_loader = make_loader(*prepared["test_era"], args.batch_size, False)
    val_floor = flat_floor(prepared["validation_map"][1], prepared["validation_map"][2])
    era_floor = flat_floor(prepared["test_era"][1], prepared["test_era"][2])

    config = {
        "model": "DirectShapeUNet", "target": "detrended_height_256 (shape only)",
        "input": "clean_minimap_256", "parameters": parameter_count(model),
        "base": args.base, "epochs": args.epochs, "batch_size": args.batch_size,
        "lr": args.lr, "seed": args.seed, "val_flat_floor": val_floor, "era_flat_floor": era_floor,
        "git_revision": git_revision(),
    }
    logger.write_json("config.json", config)
    print(f"params={config['parameters']:,}  "
          f"val flat-shape floor={val_floor:.1f}  era flat-shape floor={era_floor:.1f}", flush=True)

    best_val = float("inf")
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_mae = run_epoch(model, train_loader, device, optimizer)
        val_mae = run_epoch(model, val_loader, device)
        era_mae = run_epoch(model, era_loader, device)
        logger.log_epoch(
            epoch, train_shape_mae=train_mae, val_shape_mae=val_mae, era_shape_mae=era_mae,
            val_improve_pct=round(100 * (1 - val_mae / val_floor), 1),
            era_improve_pct=round(100 * (1 - era_mae / era_floor), 1),
            epoch_seconds=round(time.time() - t0, 2), peak_vram_gb=round(peak_vram_gb() or 0.0, 3),
        )
        ckpt = {"model": model.state_dict(), "config": config, "epoch": epoch, "val_shape_mae": val_mae}
        torch.save(ckpt, args.output_dir / "checkpoint_last.pt")
        if val_mae < best_val:
            best_val = val_mae
            torch.save(ckpt, args.output_dir / "checkpoint_best.pt")

    best = torch.load(args.output_dir / "checkpoint_best.pt", map_location=device, weights_only=False)
    model.load_state_dict(best["model"])
    val_corr = shape_correlation(model, val_loader, device)
    era_corr = shape_correlation(model, era_loader, device)
    best_val_mae = run_epoch(model, val_loader, device)
    best_era_mae = run_epoch(model, era_loader, device)
    report = {
        "best_epoch": best["epoch"],
        "val_shape_mae": best_val_mae, "val_flat_floor": val_floor,
        "val_improvement_pct": round(100 * (1 - best_val_mae / val_floor), 1),
        "val_shape_correlation": round(val_corr, 3),
        "era_shape_mae": best_era_mae, "era_flat_floor": era_floor,
        "era_improvement_pct": round(100 * (1 - best_era_mae / era_floor), 1),
        "era_shape_correlation": round(era_corr, 3),
        "peak_vram_gb": peak_vram_gb(),
    }
    logger.write_json("report.json", report)
    render_proof(model, prepared["validation_map"], device, args.output_dir / "proof_validation.png")
    render_proof(model, prepared["test_era"], device, args.output_dir / "proof_era.png")
    print(json.dumps(report, indent=2), flush=True)
    print(f"proof images -> {args.output_dir}/proof_validation.png, proof_era.png", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
