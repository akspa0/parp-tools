"""Decompose the minimap->height problem into absolute vs shape, per split.

The residual cascade (H0 scalar offset -> H1 coarse relief -> H2 detail) kept
stalling on held-out maps. Before building anything else, measure what signal
is actually there. Two very different sub-problems live inside "predict the
heightmap":

  * ABSOLUTE elevation (the per-tile mean height). WoW maps have arbitrary
    global Z origins, so the same-looking grassy slope can sit at height 50 on
    one map and -200 on another. A top-down minimap cannot see absolute
    datum. This is expected to be nearly unlearnable across held-out maps.
  * SHAPE / relief (the heightmap with its own per-tile mean removed). Cliffs,
    ridges, valley walls, roughness are all visible as texture/shading in a
    top-down minimap. This is expected to be learnable and map-transferable.

If that hypothesis holds, the honest product is minimap -> *detrended* terrain
shape (add the absolute datum from elsewhere), not minimap -> absolute height.

Pure descriptive stats, no training, CPU, sequential Zarr reads.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))


def liquid_vertex_mask(liquid_256: np.ndarray) -> np.ndarray:
    cells = liquid_256 > 127
    covered = np.zeros((257, 257), dtype=bool)
    covered[:-1, :-1] |= cells
    covered[1:, :-1] |= cells
    covered[:-1, 1:] |= cells
    covered[1:, 1:] |= cells
    return ~covered


def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnose minimap->height signal decomposition")
    parser.add_argument("--v25-store", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=128)
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    split_by_row = {int(r["row"]): str(r["split"]) for r in manifest["rows"]}
    group = zarr.open_group(str(args.v25_store), mode="r")
    height_array = group["height_257"]
    liquid_array = group["liquid_mask_256"]

    per_split: dict[str, dict[str, list]] = {}
    rows = sorted(split_by_row)
    for start in range(0, len(rows), args.batch):
        chunk = rows[start : start + args.batch]
        lo, hi = chunk[0], chunk[-1] + 1
        h_block = np.asarray(height_array[lo:hi], dtype=np.float64)
        liq_block = np.asarray(liquid_array[lo:hi])
        for row in chunk:
            i = row - lo
            h = h_block[i]
            valid = liquid_vertex_mask(liq_block[i]) & np.isfinite(h)
            if valid.sum() < 16:
                continue
            hv = h[valid]
            tile_mean = float(hv.mean())
            detrended = hv - tile_mean
            split = split_by_row[row]
            bucket = per_split.setdefault(
                split, {"mean": [], "relief_mae": [], "relief_std": [], "range": []}
            )
            bucket["mean"].append(tile_mean)
            bucket["relief_mae"].append(float(np.abs(detrended).mean()))
            bucket["relief_std"].append(float(detrended.std()))
            bucket["range"].append(float(hv.max() - hv.min()))

    print(f"{'split':16} {'tiles':>6} {'abs_mean':>10} {'abs_std':>9} "
          f"{'relief_MAE':>11} {'relief_std':>11} {'range_med':>10}", flush=True)
    report = {}
    global_mean = None
    for split in ("train", "validation_map", "test_era"):
        if split not in per_split:
            continue
        b = per_split[split]
        means = np.array(b["mean"])
        relief_mae = float(np.mean(b["relief_mae"]))
        if split == "train":
            global_mean = float(means.mean())
        print(f"{split:16} {len(means):>6d} {means.mean():>10.2f} {means.std():>9.2f} "
              f"{relief_mae:>11.3f} {float(np.mean(b['relief_std'])):>11.3f} "
              f"{float(np.median(b['range'])):>10.2f}", flush=True)
        report[split] = {
            "tiles": len(means),
            "abs_height_mean": float(means.mean()),
            "abs_height_std": float(means.std()),
            "relief_mae_floor": relief_mae,
            "relief_std": float(np.mean(b["relief_std"])),
        }

    # The two competing baselines a minimap-only model faces on held-out maps.
    print("\n--- what each held-out split says about the task ---", flush=True)
    for split in ("validation_map", "test_era"):
        if split not in report or global_mean is None:
            continue
        means = np.array(per_split[split]["mean"])
        # Absolute baseline: predict the TRAIN global mean everywhere (best you
        # can do for absolute datum with zero map-specific info).
        abs_baseline_mae = float(np.abs(means - global_mean).mean())
        relief_floor = report[split]["relief_mae_floor"]
        report[split]["abs_baseline_mae_train_global_mean"] = abs_baseline_mae
        print(f"{split}: predicting TRAIN-global-mean for absolute datum -> "
              f"{abs_baseline_mae:.1f} MAE (per-tile-mean error).", flush=True)
        print(f"{split}: relief floor (error if you nail the datum but predict "
              f"flat shape) -> {relief_floor:.1f} MAE.", flush=True)
        print(f"{split}: => absolute datum error ({abs_baseline_mae:.0f}) "
              f"{'DOMINATES' if abs_baseline_mae > relief_floor else 'is below'} "
              f"the relief signal ({relief_floor:.0f}). "
              f"{'Predicting shape cannot help absolute error.' if abs_baseline_mae > relief_floor else ''}",
              flush=True)

    out = args.v25_store.parent.parent / "analysis" / "spec102_minimap_baseline_v1" / "height_signal_decomposition.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote {out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
