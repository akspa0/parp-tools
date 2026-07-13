"""Publish leakage-safe Spec 102 baselines on real terrain-lattice nodes only.

The legacy V25 store is used strictly as a label container. Mixed-parity
height_257 cells and wdl_height_33 are never read as terrain/WDL truth.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import zarr

from harvester.spec102.m0 import PRECISE_MASK_KEY, precise_object_target_256


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def lattice_mask() -> np.ndarray:
    y, x = np.indices((257, 257))
    return (x & 1) == (y & 1)


def liquid_vertex_mask(liquid_256: np.ndarray) -> np.ndarray:
    cells = liquid_256 > 127
    covered = np.zeros((257, 257), dtype=bool)
    covered[:-1, :-1] |= cells
    covered[1:, :-1] |= cells
    covered[:-1, 1:] |= cells
    covered[1:, 1:] |= cells
    return ~covered


def rgb_flat_fit(rgb_means: np.ndarray, target_means: np.ndarray) -> tuple[float, float]:
    design = np.stack([rgb_means, np.ones_like(rgb_means)], axis=1)
    slope, intercept = np.linalg.lstsq(design, target_means, rcond=None)[0]
    return float(slope), float(intercept)


def outer_normal_angle_from_flat(height: np.ndarray) -> tuple[float, int]:
    outer = height[::2, ::2].astype(np.float64)
    dz_dy, dz_dx = np.gradient(outer, 533.33333 / 128.0)
    angle = np.degrees(np.arctan(np.sqrt(dz_dx * dz_dx + dz_dy * dz_dy)))
    finite = np.isfinite(angle)
    return float(angle[finite].sum()), int(finite.sum())


def main() -> int:
    parser = argparse.ArgumentParser(description="Audit Spec 102 real-node numeric baselines")
    parser.add_argument("--v25-store", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch", type=int, default=64)
    args = parser.parse_args()

    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "spec102-curated-split-v2":
        raise RuntimeError("numeric audit refuses an uncurated split manifest")
    all_rows = manifest["rows"]
    rows = [row for row in all_rows if row.get("eligible_h2") is True]
    split_by_row = {int(row["row"]): str(row["split"]) for row in rows}
    group = zarr.open_group(str(args.v25_store), mode="r")
    height_array = group["height_257"]
    rgb_array = group["minimap_rgb"]
    liquid_array = group["liquid_mask_256"]
    m0_target_name = PRECISE_MASK_KEY
    object_array = group[m0_target_name]
    real_nodes = lattice_mask()

    train_rgb: list[float] = []
    train_means: list[float] = []
    train_sum = 0.0
    train_count = 0
    for start in range(0, len(all_rows), args.batch):
        stop = min(start + args.batch, len(all_rows))
        heights = np.asarray(height_array[start:stop], dtype=np.float64)
        rgbs = np.asarray(rgb_array[start:stop], dtype=np.float32)
        liquids = np.asarray(liquid_array[start:stop])
        for offset, row_index in enumerate(range(start, stop)):
            if row_index not in split_by_row:
                continue
            if split_by_row[row_index] != "train":
                continue
            valid = real_nodes & liquid_vertex_mask(liquids[offset]) & np.isfinite(heights[offset])
            values = heights[offset][valid]
            train_sum += float(values.sum())
            train_count += int(values.size)
            train_rgb.append(float(rgbs[offset].mean() / 255.0))
            train_means.append(float(values.mean()))

    global_mean = train_sum / max(train_count, 1)
    rgb_slope, rgb_intercept = rgb_flat_fit(np.asarray(train_rgb), np.asarray(train_means))
    split_names = sorted(set(split_by_row.values()))
    accum = {
        split: {
            "tiles": 0,
            "vertices": 0,
            "vertex_error": 0.0,
            "wdl_samples": 0,
            "wdl_error": 0.0,
            "datum_error": 0.0,
            "mask_pixels": 0,
            "mask_positive": 0,
            "normal_angle_sum": 0.0,
            "normal_samples": 0,
        }
        for split in split_names
    }

    for start in range(0, len(all_rows), args.batch):
        stop = min(start + args.batch, len(all_rows))
        heights = np.asarray(height_array[start:stop], dtype=np.float64)
        rgbs = np.asarray(rgb_array[start:stop], dtype=np.float32)
        liquids = np.asarray(liquid_array[start:stop])
        precise_objects = np.asarray(object_array[start:stop], dtype=np.float32)
        for offset, row_index in enumerate(range(start, stop)):
            if row_index not in split_by_row:
                continue
            split = split_by_row[row_index]
            bucket = accum[split]
            height = heights[offset]
            valid = real_nodes & liquid_vertex_mask(liquids[offset]) & np.isfinite(height)
            target = height[valid]
            rgb_mean = float(rgbs[offset].mean() / 255.0)
            prediction = (rgb_slope * rgb_mean) + rgb_intercept
            object_positive = precise_object_target_256(precise_objects[offset]) > 0.5
            angle_sum, angle_count = outer_normal_angle_from_flat(height)

            bucket["tiles"] += 1
            bucket["vertices"] += int(target.size)
            bucket["vertex_error"] += float(np.abs(target - prediction).sum())
            bucket["datum_error"] += abs(float(target.mean()) - prediction)
            bucket["mask_pixels"] += int(object_positive.size)
            bucket["mask_positive"] += int(object_positive.sum())
            bucket["normal_angle_sum"] += angle_sum
            bucket["normal_samples"] += angle_count

    report_splits = {}
    for split, values in accum.items():
        positive = values["mask_positive"]
        pixels = values["mask_pixels"]
        report_splits[split] = {
            "tiles": values["tiles"],
            "m0_zero_mask": {
                "positive_prevalence": positive / max(pixels, 1),
                "pixel_accuracy": (pixels - positive) / max(pixels, 1),
                "positive_iou": 0.0,
            },
            "h0_rgb_flat_tile_datum_mae": values["datum_error"] / max(values["tiles"], 1),
            "w1": {"available": False, "reason": "real paired WDL arrays are absent"},
            "h2_rgb_flat_real_vertex_l1": values["vertex_error"] / max(values["vertices"], 1),
            "h2_flat_outer_finite_difference_normal_angle_degrees": values["normal_angle_sum"] / max(values["normal_samples"], 1),
            "real_vertex_count": values["vertices"],
            "wdl_sample_count": 0,
        }

    report = {
        "schema": "spec102-numeric-lattice-baselines-v1",
        "source_store": str(args.v25_store.resolve()),
        "source_warning": "identity-checked numeric store; only proven real checkerboard nodes are evaluated and wdl_height_33 is absent",
        "split_manifest": str(args.split_manifest.resolve()),
        "split_manifest_sha256": sha256_file(args.split_manifest),
        "deployment_inputs": {"m0": ["minimap_rgb"], "h0": ["minimap_rgb"]},
        "target_only": {
            "m0": [{"name": m0_target_name, "projection": "four_corner_max_to_256"}],
            "w1": ["wdl_outer_17", "wdl_inner_16"],
            "h2": ["mcvt_vertex_z", "mcvt_vertex_present", "mcvt_triangle_indices"],
            "normal_metric": ["numeric normals derived from mcvt_vertex_z and topology"],
        },
        "prohibited": ["wdl_height_33", "mixed-parity height_257 cells", "target-derived forward inputs"],
        "fit": {"train_global_mean": global_mean, "rgb_flat_slope": rgb_slope, "rgb_flat_intercept": rgb_intercept},
        "splits": report_splits,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
