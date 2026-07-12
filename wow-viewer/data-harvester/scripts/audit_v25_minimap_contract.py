"""Spec 102 reset: freeze leakage-safe splits and RGB-deployable baselines.

This script performs no model inference and never initializes CUDA.  Targets
are read only for fitting train-only constants and measuring held-out error.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr


ERA_HOLDOUT = "0_5_3_3368"
MAP_HOLDOUT = ("3_3_5_12340", "Northrend")


def assign_split(build: str, map_name: str) -> str:
    if build == ERA_HOLDOUT:
        return "test_era"
    if (build, map_name) == MAP_HOLDOUT:
        return "validation_map"
    return "train"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def liquid_vertex_mask(liquid: np.ndarray) -> np.ndarray:
    cells = np.asarray(liquid) > 127
    valid = np.ones((cells.shape[0], 257, 257), dtype=bool)
    covered = np.zeros_like(valid)
    covered[:, :-1, :-1] |= cells
    covered[:, 1:, :-1] |= cells
    covered[:, :-1, 1:] |= cells
    covered[:, 1:, 1:] |= cells
    return valid & ~covered


def fit_rgb_flat_baseline(rgb_means: np.ndarray, target_means: np.ndarray) -> tuple[float, float]:
    design = np.stack([rgb_means, np.ones_like(rgb_means)], axis=1)
    slope, intercept = np.linalg.lstsq(design, target_means, rcond=None)[0]
    return float(slope), float(intercept)


def iter_contiguous_batches(minimap, height, liquid, count: int, batch_size: int = 64):
    """Read each Zarr array sequentially; never regress to per-row random I/O."""
    for start in range(0, count, batch_size):
        stop = min(start + batch_size, count)
        rgb_batch = np.asarray(minimap[start:stop])
        height_batch = np.asarray(height[start:stop], dtype=np.float64)
        liquid_batch = np.asarray(liquid[start:stop])
        valid_batch = liquid_vertex_mask(liquid_batch) & np.isfinite(height_batch)
        for offset in range(stop - start):
            yield start + offset, rgb_batch[offset], height_batch[offset], valid_batch[offset]


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Spec 102 RGB-only contract and baselines")
    parser.add_argument("--v25-store", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()

    index_path = args.v25_store / "index.parquet"
    table = pq.read_table(index_path, columns=["row", "tile_id", "build", "map", "tile_x", "tile_y"])
    rows = table.to_pylist()
    for record in rows:
        record["split"] = assign_split(str(record["build"]), str(record["map"]))

    counts = Counter(record["split"] for record in rows)
    if not all(counts[name] > 0 for name in ("train", "validation_map", "test_era")):
        raise RuntimeError(f"incomplete frozen split: {dict(counts)}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    split_manifest = {
        "schema": "spec102-minimap-only-split-v1",
        "source_index": str(index_path.resolve()),
        "source_index_sha256": sha256_file(index_path),
        "rules": {"era_holdout": ERA_HOLDOUT, "map_holdout": list(MAP_HOLDOUT)},
        "counts": dict(counts),
        "rows": rows,
    }
    manifest_path = args.output_dir / "split_manifest.json"
    manifest_path.write_text(json.dumps(split_manifest, indent=2), encoding="utf-8")

    group = zarr.open_group(str(args.v25_store), mode="r")
    minimap = group["minimap_rgb"]
    height = group["height_257"]
    liquid = group["liquid_mask_256"]

    by_split = {name: [int(r["row"]) for r in rows if r["split"] == name] for name in counts}
    split_by_row = {int(record["row"]): str(record["split"]) for record in rows}
    train_rgb_means: list[float] = []
    train_target_means: list[float] = []
    train_sum = 0.0
    train_n = 0
    for row, rgb, h, valid in iter_contiguous_batches(minimap, height, liquid, len(rows)):
        if split_by_row[row] != "train":
            continue
        train_sum += float(h[valid].sum())
        train_n += int(valid.sum())
        train_rgb_means.append(float(np.asarray(rgb, dtype=np.float32).mean() / 255.0))
        train_target_means.append(float(h[valid].mean()))
    global_mean = train_sum / max(train_n, 1)
    rgb_slope, rgb_intercept = fit_rgb_flat_baseline(
        np.asarray(train_rgb_means), np.asarray(train_target_means)
    )

    report_splits: dict[str, dict] = {}
    accumulators = {
        name: {"errors": {"zero_height": 0.0, "train_global_mean": 0.0, "rgb_flat": 0.0},
               "tile_mean_errors": {"zero_height": 0.0, "train_global_mean": 0.0, "rgb_flat": 0.0},
               "n": 0, "tiles": 0}
        for name in by_split
    }
    for row, rgb, h, valid in iter_contiguous_batches(minimap, height, liquid, len(rows)):
        split_name = split_by_row[row]
        rgb_mean = float(np.asarray(rgb, dtype=np.float32).mean() / 255.0)
        predictions = {
            "zero_height": 0.0,
            "train_global_mean": global_mean,
            "rgb_flat": rgb_slope * rgb_mean + rgb_intercept,
        }
        target_mean = float(h[valid].mean())
        for name, prediction in predictions.items():
            accumulators[split_name]["errors"][name] += float(np.abs(h[valid] - prediction).sum())
            accumulators[split_name]["tile_mean_errors"][name] += abs(target_mean - prediction)
        accumulators[split_name]["n"] += int(valid.sum())
        accumulators[split_name]["tiles"] += 1

    for split_name, split_rows in by_split.items():
        errors = accumulators[split_name]["errors"]
        tile_mean_errors = accumulators[split_name]["tile_mean_errors"]
        n = accumulators[split_name]["n"]
        tile_count = accumulators[split_name]["tiles"]
        report_splits[split_name] = {
            "tiles": len(split_rows),
            "valid_vertices": n,
            "height_l1": {name: value / max(n, 1) for name, value in errors.items()},
            "tile_mean_mae": {
                name: value / max(tile_count, 1) for name, value in tile_mean_errors.items()
            },
        }

    report = {
        "schema": "spec102-minimap-only-baselines-v1",
        "split_manifest": str(manifest_path.resolve()),
        "split_manifest_sha256": sha256_file(manifest_path),
        "deployment_inputs": ["minimap_rgb"],
        "target_only": ["height_257", "liquid_mask_256"],
        "prohibited_baselines": ["per_tile_target_mean", "wdl_input", "height_derived_prior"],
        "fit": {"train_global_mean": global_mean, "rgb_flat_slope": rgb_slope, "rgb_flat_intercept": rgb_intercept},
        "splits": report_splits,
        "historical_minimap_only": {"reported_l1": 190.31, "comparable": False, "reason": "different split"},
    }
    report_path = args.output_dir / "baseline_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
