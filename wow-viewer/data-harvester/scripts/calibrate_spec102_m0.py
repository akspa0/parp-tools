"""Calibrate the frozen M0 mask threshold on the held-out validation map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import zarr

from harvester.spec102.m0 import PRECISE_MASK_KEY, M0ObjectMask, precise_object_target_256


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Spec 102 M0 threshold")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M0 calibration is CUDA-only; CPU fallback is prohibited")
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    model = M0ObjectMask(base_channels=int(checkpoint["config"]["base_channels"])).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "spec102-curated-split-v2":
        raise RuntimeError("M0 calibration refuses an uncurated split manifest")
    rows_by_split = {
        split: np.asarray([
            int(row["row"]) for row in manifest["rows"]
            if row["split"] == split and row.get("eligible_m0") is True
        ])
        for split in ("validation_map", "test_era")
    }
    group = zarr.open_group(str(args.store), mode="r")
    rgb_array = group["minimap_rgb"]
    if PRECISE_MASK_KEY not in group:
        raise RuntimeError(f"M0 calibration requires {PRECISE_MASK_KEY}; fallbacks are prohibited")
    target_array = group[PRECISE_MASK_KEY]
    thresholds = np.arange(0.05, 1.0, 0.05, dtype=np.float32)

    def score(rows: np.ndarray) -> list[dict[str, float]]:
        intersection = np.zeros(len(thresholds), dtype=np.float64)
        union = np.zeros(len(thresholds), dtype=np.float64)
        for start in range(0, len(rows), args.batch_size):
            selected = rows[start : start + args.batch_size]
            rgb = np.asarray(rgb_array[selected], dtype=np.uint8)
            precise = np.asarray(target_array[selected], dtype=np.float32)
            truth = np.stack([precise_object_target_256(mask) for mask in precise]) > 0.5
            tensor = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))).float().cuda() / 255.0
            with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
                probability = torch.sigmoid(model(tensor))[:, 0].float().cpu().numpy()
            for index, threshold in enumerate(thresholds):
                predicted = probability >= threshold
                intersection[index] += np.logical_and(predicted, truth).sum()
                union[index] += np.logical_or(predicted, truth).sum()
        return [
            {"threshold": float(threshold), "iou": float(intersection[index] / max(union[index], 1.0))}
            for index, threshold in enumerate(thresholds)
        ]

    validation = score(rows_by_split["validation_map"])
    best = max(validation, key=lambda row: row["iou"])
    test_rows = score(rows_by_split["test_era"])
    test_at_best = min(test_rows, key=lambda row: abs(row["threshold"] - best["threshold"]))
    report = {
        "schema": "spec102-m0-threshold-calibration-v1",
        "validation_curve": validation,
        "selected_threshold": best["threshold"],
        "selected_validation_iou": best["iou"],
        "test_era_iou_at_selected_threshold": test_at_best["iou"],
        "selection_rule": "maximum validation_map IoU; test_era never selects threshold",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
