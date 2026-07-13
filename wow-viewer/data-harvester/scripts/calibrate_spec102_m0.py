"""Calibrate the frozen M0 mask threshold on the held-out validation map."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr

from harvester.spec102.m0 import STRICT_OBJECT_TARGET_KEY, M0ObjectMask, strict_object_target_256
from harvester.spec102.m0_coverage import validate_m0_coverage_audit
from harvester.spec102.m0_scope import validate_m0_build_local_scope


def main() -> int:
    parser = argparse.ArgumentParser(description="Calibrate Spec 102 M0 threshold")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--coverage-report", required=True, type=Path)
    parser.add_argument("--raw-v18-store", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    store_index = pq.read_table(args.store / "index.parquet").to_pylist()
    scope = validate_m0_build_local_scope(manifest, source_index=store_index)
    validate_m0_coverage_audit(
        args.coverage_report,
        raw_v18_store=args.raw_v18_store,
        store=args.store,
        split_manifest=args.split_manifest,
        expected_scope=scope.audit_binding,
    )
    artifact_binding = scope.artifact_binding(
        store=args.store,
        split_manifest=args.split_manifest,
        coverage_report=args.coverage_report,
    )
    rows_by_split = {split: np.asarray(rows) for split, rows in scope.rows_by_split.items()}
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "spec102-m0-checkpoint-v2":
        raise RuntimeError("M0 calibration checkpoint is not bound to the strict geometry-target contract")
    if checkpoint.get("m0_artifact_binding") != artifact_binding:
        raise RuntimeError("M0 calibration checkpoint does not match the 3.3.5 build-local split")
    if not torch.cuda.is_available():
        raise RuntimeError("M0 calibration is CUDA-only; CPU fallback is prohibited")
    model = M0ObjectMask(base_channels=int(checkpoint["config"]["base_channels"])).cuda().eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    group = zarr.open_group(str(args.store), mode="r")
    rgb_array = group["minimap_rgb"]
    if STRICT_OBJECT_TARGET_KEY not in group:
        raise RuntimeError(f"M0 calibration requires {STRICT_OBJECT_TARGET_KEY}; fallbacks are prohibited")
    target_array = group[STRICT_OBJECT_TARGET_KEY]
    thresholds = np.arange(0.05, 1.0, 0.05, dtype=np.float32)

    def score(rows: np.ndarray) -> list[dict[str, float]]:
        intersection = np.zeros(len(thresholds), dtype=np.float64)
        union = np.zeros(len(thresholds), dtype=np.float64)
        for start in range(0, len(rows), args.batch_size):
            selected = rows[start : start + args.batch_size]
            rgb = np.asarray(rgb_array[selected], dtype=np.uint8)
            strict = np.asarray(target_array[selected], dtype=np.float32)
            truth = np.stack([strict_object_target_256(mask) for mask in strict]) > 0.5
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
    test_rows = score(rows_by_split["test_build_local"])
    test_at_best = min(test_rows, key=lambda row: abs(row["threshold"] - best["threshold"]))
    report = {
        "schema": "spec102-m0-threshold-calibration-v2",
        "gate_scope": "build_local_3_3_5_only",
        "cross_era_evaluated": False,
        "m0_training_scope": scope.audit_binding,
        "m0_artifact_binding": artifact_binding,
        "coverage_report": str(args.coverage_report.resolve()),
        "coverage_report_sha256": artifact_binding["coverage_report_sha256"],
        "coverage_raw_v18_store": str(args.raw_v18_store.resolve()),
        "validation_curve": validation,
        "selected_threshold": best["threshold"],
        "selected_validation_iou": best["iou"],
        "test_build_local_iou_at_selected_threshold": test_at_best["iou"],
        "selection_rule": "maximum validation_map IoU; test_build_local never selects threshold",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
