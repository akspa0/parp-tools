"""Render a self-describing Spec 102 M0 panel from an existing checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from train_spec102_m0 import MaskDataset, write_validation_grid

from harvester.spec102.m0 import STRICT_OBJECT_TARGET_KEY, M0ObjectMask, strict_object_target_256
from harvester.spec102.m0_coverage import validate_m0_coverage_audit
from harvester.spec102.m0_scope import validate_m0_build_local_scope


def main() -> int:
    parser = argparse.ArgumentParser(description="Render labeled Spec 102 M0 validation panel")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--coverage-report", required=True, type=Path)
    parser.add_argument("--raw-v18-store", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("validation_map", "test_build_local"), default="validation_map")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--count", type=int, default=8)
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
    rows = scope.rows_by_split[args.split]
    metadata_by_row = scope.metadata_by_row
    group = zarr.open_group(str(args.store), mode="r")
    rgb = np.asarray(group["minimap_rgb"][:])
    strict = np.asarray(group[STRICT_OBJECT_TARGET_KEY][:], dtype=np.float32)
    masks = np.stack([strict_object_target_256(mask) for mask in strict], axis=0)
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("schema") != "spec102-m0-checkpoint-v2":
        raise RuntimeError("checkpoint is not Spec 102 M0")
    if checkpoint.get("m0_artifact_binding") != artifact_binding:
        raise RuntimeError("checkpoint was not trained on this 3.3.5 build-local scope")
    if not torch.cuda.is_available():
        raise RuntimeError("M0 validation renderer requires CUDA; silent CPU fallback is prohibited")
    model = M0ObjectMask(base_channels=int(checkpoint["config"]["base_channels"])).cuda()
    model.load_state_dict(checkpoint["model"], strict=True)
    write_validation_grid(
        args.output,
        model,
        MaskDataset(rgb, masks, rows),
        torch.device("cuda"),
        metadata_by_row=metadata_by_row,
        epoch=int(checkpoint["epoch"]),
        split=args.split,
        threshold=args.threshold,
        checkpoint_label=args.checkpoint.name,
        count=args.count,
    )
    print(f"Wrote labeled validation panel: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
