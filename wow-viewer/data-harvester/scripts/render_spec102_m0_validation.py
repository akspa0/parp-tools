"""Render a self-describing Spec 102 M0 panel from an existing checkpoint."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import zarr

from harvester.spec102.m0 import PRECISE_MASK_KEY, M0ObjectMask, precise_object_target_256
from train_spec102_m0 import MaskDataset, write_validation_grid


def main() -> int:
    parser = argparse.ArgumentParser(description="Render labeled Spec 102 M0 validation panel")
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--split-manifest", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--split", choices=("validation_map", "test_era"), default="validation_map")
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--count", type=int, default=8)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("M0 validation renderer requires CUDA; silent CPU fallback is prohibited")
    manifest = json.loads(args.split_manifest.read_text(encoding="utf-8"))
    if manifest.get("schema") != "spec102-curated-split-v2":
        raise RuntimeError("renderer refuses an uncurated split manifest")
    rows = [
        int(row["row"]) for row in manifest["rows"]
        if row["split"] == args.split and row.get("eligible_m0") is True
    ]
    metadata_by_row = {int(row["row"]): row for row in manifest["rows"]}
    group = zarr.open_group(str(args.store), mode="r")
    rgb = np.asarray(group["minimap_rgb"][:])
    precise = np.asarray(group[PRECISE_MASK_KEY][:], dtype=np.float32)
    masks = np.stack([precise_object_target_256(mask) for mask in precise], axis=0)
    checkpoint = torch.load(args.checkpoint, map_location="cuda", weights_only=False)
    if checkpoint.get("schema") != "spec102-m0-checkpoint-v1":
        raise RuntimeError("checkpoint is not Spec 102 M0")
    model = M0ObjectMask().cuda()
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
