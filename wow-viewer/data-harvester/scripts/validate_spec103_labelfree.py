"""Spec 103 T015 — label-free, use-case-faithful validation (spec US2, FR-006, SC-002).

Judges predictions the way they will be used: no ground-truth signal is required. Checks, per
predictions directory from infer_spec103_v7.py:

  1. Border agreement: adjacent reconstructed tiles must agree along their shared 257-grid
     edge within --border-threshold (mean abs, world units). Convention: tile (x, y)'s last
     column adjoins (x+1, y)'s first column; last row adjoins (x, y+1)'s first row.
  2. Plausibility: every height finite, inside [--min-height, --max-height], and per-cell
     gradient below --max-gradient (world units per grid step).
  3. Artifacts: checkerboard (Nyquist-band FFT energy ratio) and chunk blockiness (mean abs
     step at 16-multiple grid lines vs elsewhere) below thresholds.

Optionally, --gt-store adds DEV-ONLY diagnostics (FR-007): L1 vs ground truth and vs the
flat-mean and WDL-prior baselines (SC-004). These never affect the pass/fail verdict.

Run from wow-viewer/data-harvester/ (fast, CPU):

    uv run python scripts/validate_spec103_labelfree.py \
        --predictions ../output/spec103_v7_synth_v1/predictions \
        --report ../output/spec103_v7_synth_v1/labelfree_report.json
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np


def checkerboard_ratio(height: np.ndarray) -> float:
    """Energy fraction in the top (Nyquist) frequency band — high for checkerboard artifacts."""
    spectrum = np.abs(np.fft.rfft2(height.astype(np.float64) - height.mean()))
    total = float(spectrum.sum()) + 1e-9
    rows, cols = spectrum.shape
    band = spectrum[int(rows * 0.45):int(rows * 0.55) + 1, int(cols * 0.9):]
    band_v = spectrum[int(rows * 0.9):, int(cols * 0.9):]
    return float((band.sum() + band_v.sum()) / total)


def blockiness_16(height: np.ndarray) -> float:
    """Ratio of mean abs first difference across 16-multiple grid lines vs all lines (~1 = clean)."""
    d_rows = np.abs(np.diff(height, axis=0))
    d_cols = np.abs(np.diff(height, axis=1))
    idx = np.arange(16, 257 - 1, 16) - 1  # differences straddling chunk boundaries
    on = float(np.concatenate([d_rows[idx, :].ravel(), d_cols[:, idx].ravel()]).mean())
    off = float((d_rows.mean() + d_cols.mean()) / 2.0) + 1e-9
    return on / off


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 103 label-free self-consistency harness")
    ap.add_argument("--predictions", required=True, type=Path)
    ap.add_argument("--report", required=True, type=Path)
    ap.add_argument("--border-threshold", type=float, default=2.0, help="mean abs shared-border gap, world units")
    ap.add_argument("--min-height", type=float, default=-1000.0)
    ap.add_argument("--max-height", type=float, default=3000.0)
    ap.add_argument("--max-gradient", type=float, default=200.0, help="max per-cell height step, world units")
    ap.add_argument("--max-checkerboard", type=float, default=0.05)
    ap.add_argument("--max-blockiness", type=float, default=3.0)
    ap.add_argument("--gt-store", type=Path, default=None, help="DEV-ONLY diagnostics; never gates the verdict")
    args = ap.parse_args()

    manifest = json.loads((args.predictions / "predictions_manifest.json").read_text(encoding="utf-8"))
    heights: dict[tuple[str, int, int], np.ndarray] = {}
    tiles = {}
    for tile in manifest["tiles"]:
        key = (str(tile["map"]), int(tile["tile_x"]), int(tile["tile_y"]))
        heights[key] = np.load(args.predictions / tile["prediction_dir"] / "predicted_height_257.npy")
        tiles[key] = tile["tile_name"]

    per_tile = {}
    failures: list[str] = []
    for key, height in heights.items():
        name = tiles[key]
        finite = bool(np.isfinite(height).all())
        in_range = bool(finite and height.min() >= args.min_height and height.max() <= args.max_height)
        max_grad = float(max(np.abs(np.diff(height, axis=0)).max(), np.abs(np.diff(height, axis=1)).max())) if finite else float("inf")
        checker = checkerboard_ratio(height) if finite else float("inf")
        block = blockiness_16(height) if finite else float("inf")
        checks = {
            "finite": finite,
            "in_range": in_range,
            "max_gradient": max_grad <= args.max_gradient,
            "checkerboard": checker <= args.max_checkerboard,
            "blockiness_16": block <= args.max_blockiness,
        }
        per_tile[name] = {
            "height_min": float(height.min()) if finite else None,
            "height_max": float(height.max()) if finite else None,
            "max_gradient": max_grad, "checkerboard_ratio": round(checker, 5),
            "blockiness_16": round(block, 3), "checks": checks,
        }
        failures += [f"{name}:{check}" for check, ok in checks.items() if not ok]

    borders = []
    for (map_name, x, y), height in heights.items():
        east = heights.get((map_name, x + 1, y))
        if east is not None:
            gap = float(np.abs(height[:, -1] - east[:, 0]).mean())
            borders.append({"between": [tiles[(map_name, x, y)], tiles[(map_name, x + 1, y)]],
                            "edge": "east-west", "mean_abs_gap": round(gap, 4),
                            "passed": gap <= args.border_threshold})
        south = heights.get((map_name, x, y + 1))
        if south is not None:
            gap = float(np.abs(height[-1, :] - south[0, :]).mean())
            borders.append({"between": [tiles[(map_name, x, y)], tiles[(map_name, x, y + 1)]],
                            "edge": "south-north", "mean_abs_gap": round(gap, 4),
                            "passed": gap <= args.border_threshold})
    failures += [f"border:{b['between'][0]}|{b['between'][1]}" for b in borders if not b["passed"]]

    dev_only = None
    if args.gt_store is not None:
        import pyarrow.parquet as pq
        import zarr

        group = zarr.open_group(str(args.gt_store), mode="r")
        index = pq.read_table(args.gt_store / "index.parquet").to_pylist()
        lookup = {(str(r["map"]), int(r["tile_x"]), int(r["tile_y"])): i for i, r in enumerate(index)}
        rows = []
        for key, predicted in heights.items():
            if key not in lookup:
                continue
            gt = np.asarray(group["height_257"][lookup[key]], dtype=np.float32)
            outer = gt[::16, ::16]
            # WDL-prior baseline: outer lattice bilinearly upsampled back to the vertex grid
            import torch
            import torch.nn.functional as F
            prior = F.interpolate(torch.from_numpy(outer).view(1, 1, 17, 17), size=(257, 257),
                                  mode="bilinear", align_corners=True).squeeze().numpy()
            rows.append({
                "tile": tiles[key],
                "l1_vs_gt": float(np.abs(predicted - gt).mean()),
                "prior_baseline_l1": float(np.abs(prior - gt).mean()),
                "flat_mean_baseline_l1": float(np.abs(gt.mean() - gt).mean()),
            })
        dev_only = {
            "note": "development diagnostic only (FR-007); never the acceptance criterion",
            "tiles": rows,
            "mean_l1_vs_gt": float(np.mean([r["l1_vs_gt"] for r in rows])) if rows else None,
            "mean_prior_baseline_l1": float(np.mean([r["prior_baseline_l1"] for r in rows])) if rows else None,
            "mean_flat_baseline_l1": float(np.mean([r["flat_mean_baseline_l1"] for r in rows])) if rows else None,
        }

    report = {
        "schema": "spec103-labelfree-report-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "predictions": str(args.predictions.resolve()),
        "thresholds": {
            "border_mean_abs": args.border_threshold, "height_range": [args.min_height, args.max_height],
            "max_gradient": args.max_gradient, "max_checkerboard": args.max_checkerboard,
            "max_blockiness_16": args.max_blockiness,
        },
        "tile_count": len(per_tile),
        "adjacent_border_pairs": len(borders),
        "per_tile": per_tile,
        "borders": borders,
        "failures": failures,
        "passed": not failures,
        "dev_only_diagnostics": dev_only,
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"[spec103] tiles={len(per_tile)} border_pairs={len(borders)} failures={len(failures)}")
    if borders:
        worst = max(borders, key=lambda b: b["mean_abs_gap"])
        print(f"[spec103] worst border gap: {worst['mean_abs_gap']} between {worst['between']}")
    print(f"[{'PASS' if not failures else 'FAIL'}] report -> {args.report}")
    return 0 if not failures else 1


if __name__ == "__main__":
    raise SystemExit(main())
