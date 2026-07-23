"""Spec 118 US3: two-mode visible-object inference with a frozen ``ObjectSegmentNet`` checkpoint.

- ``--inputs`` mode: loose PNG tile(s) or directories, NO store and NO ground truth required --
  runs unchanged on a hand-painted OOD minimap (FR-009). The audit record marks
  ``ground_truth: "unavailable"``; it never fabricates reference data.
- ``--store`` mode: batch over a v50 store; when the strict source array is present it also scores
  per-class IoU/recall against ground truth.

Both modes emit one colorized class PNG per tile (none=black, doodad=green, building=red) and a
``v118-object-infer-v1`` audit record (data-model.md Infer Audit Record) binding the checkpoint
sha256 to every prediction. The two modes are mutually exclusive (Spec 116 pattern).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from harvester.spec118.object_contract import CLASS_NAMES, sha256_file
from harvester.spec118.object_segment_model import (
    ObjectSegmentNet,
    derive_class_target,
    per_class_iou_recall,
    visible_object_iou,
)

AUDIT_SCHEMA = "v118-object-infer-v1"
SOURCE_ARRAY = "object_geometry_visible_source_257"

#: Colorized class palette for the review PNGs (none/doodad/building).
CLASS_COLORS = ((0, 0, 0), (0, 200, 0), (220, 40, 40))


class ObjectInferError(ValueError):
    """Raised when inference cannot proceed under the declared contract."""


def load_frozen_model(checkpoint: Path, *, device: str = "cpu"):
    """Reconstruct an architecturally-identical ObjectSegmentNet from a checkpoint's object_config."""
    import torch

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    object_config = ckpt.get("object_config", {})
    if not isinstance(object_config, dict) or "base" not in object_config:
        raise ObjectInferError(
            "checkpoint carries no object_config.base; cannot reconstruct the model "
            "(checkpoints written before the object_config field existed are unsupported)"
        )
    base = int(object_config["base"])
    model = ObjectSegmentNet(base=base)
    try:
        model.load_state_dict(ckpt["model"])
    except RuntimeError as exc:
        raise ObjectInferError(
            f"checkpoint state_dict does not fit ObjectSegmentNet(base={base}): {exc}"
        ) from exc
    model.eval()
    return model.to(torch.device(device)), sha256_file(checkpoint), ckpt


def predict_class_map(model, rgb: np.ndarray, *, device: str = "cpu") -> tuple[np.ndarray, np.ndarray]:
    """One (H,W,3) uint8/float tile -> (predicted class ids (H,W) int64, softmax (3,H,W) float32)."""
    import torch

    if rgb.shape[:2] != (256, 256):
        raise ObjectInferError(f"expected a 256x256 tile, got {rgb.shape}")
    x = torch.from_numpy(rgb.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(torch.device(device))
    with torch.no_grad():
        probs = torch.softmax(model(x).squeeze(0), dim=0).cpu().numpy()
    return probs.argmax(0).astype(np.int64), probs


def _write_class_png(class_map: np.ndarray, path: Path) -> None:
    from PIL import Image

    palette = np.zeros((256, 256, 3), dtype=np.uint8)
    for class_id, color in enumerate(CLASS_COLORS):
        palette[class_map == class_id] = color
    Image.fromarray(palette).save(path)


def _tile_audit(
    *,
    tile_id: str,
    class_map: np.ndarray,
    ground_truth: np.ndarray | None,
) -> dict[str, Any]:
    histogram = {
        name: int((class_map == class_id).sum()) for class_id, name in enumerate(CLASS_NAMES)
    }
    record: dict[str, Any] = {
        "tile": tile_id,
        "predicted_class_histogram": histogram,
        "marked_fraction": float((class_map > 0).mean()),
    }
    if ground_truth is None:
        record["ground_truth"] = "unavailable"
    else:
        record["ground_truth"] = "strict_visible_object_source"
        record["per_class"] = per_class_iou_recall(class_map, ground_truth)
        record["visible_object_iou"] = visible_object_iou(class_map, ground_truth)
    return record


def infer_loose_inputs(
    *,
    checkpoint: Path,
    inputs: list[Path],
    output: Path,
    device: str = "cpu",
    write: bool = False,
) -> dict:
    """OOD/loose-image mode: no store, no ground truth (FR-009)."""
    from harvester.v50.direct_geometry_infer import discover_tiles, load_tile_rgb

    tiles = discover_tiles(inputs)
    if not tiles:
        raise ObjectInferError(f"no minimap tiles found under: {[str(i) for i in inputs]}")
    plan = {
        "schema": AUDIT_SCHEMA,
        "mode": "loose_inputs",
        "checkpoint": {"path": str(checkpoint)},
        "tile_count": len(tiles),
        "ground_truth": "unavailable",
    }
    if not write:
        return plan

    model, checkpoint_sha, _ckpt = load_frozen_model(checkpoint, device=device)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    for tile in tiles:
        rgb = load_tile_rgb(tile)
        class_map, _probs = predict_class_map(model, rgb, device=device)
        png_path = output / f"{tile.stem}_object_classes.png"
        _write_class_png(class_map, png_path)
        records.append({
            **_tile_audit(tile_id=tile.name, class_map=class_map, ground_truth=None),
            "class_png": png_path.name,
            "input_sha256": sha256_file(tile),
        })
    audit = {
        **plan,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "tiles": records,
    }
    (output / "object_infer_audit.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return audit


def infer_store(
    *,
    checkpoint: Path,
    store: Path,
    dumps: Path,
    device: str = "cpu",
    write: bool = False,
) -> dict:
    """Store-batch mode: predict over every row; score against ground truth where present."""
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(store), mode="r")
    if "minimap_rgb" not in group:
        raise ObjectInferError(f"store is missing minimap_rgb: {store}")
    index_rows = pq.read_table(store / "index.parquet").to_pylist()
    has_truth = SOURCE_ARRAY in group
    plan = {
        "schema": AUDIT_SCHEMA,
        "mode": "store",
        "checkpoint": {"path": str(checkpoint)},
        "store": str(store),
        "tile_count": len(index_rows),
        "ground_truth": "strict_visible_object_source" if has_truth else "unavailable",
    }
    if not write:
        return plan

    model, checkpoint_sha, _ckpt = load_frozen_model(checkpoint, device=device)
    dumps.mkdir(parents=True, exist_ok=True)
    records = []
    for row in range(len(index_rows)):
        rgb = np.asarray(group["minimap_rgb"][row], dtype=np.uint8)
        class_map, _probs = predict_class_map(model, rgb, device=device)
        truth = (
            derive_class_target(np.asarray(group[SOURCE_ARRAY][row])) if has_truth else None
        )
        meta = index_rows[row]
        tile_id = f"{meta.get('map', 'unknown')}_{meta.get('tile_x', row)}_{meta.get('tile_y', 0)}"
        png_path = dumps / f"{tile_id}_object_classes.png"
        _write_class_png(class_map, png_path)
        records.append({
            **_tile_audit(tile_id=tile_id, class_map=class_map, ground_truth=truth),
            "source_row_index": row,
            "class_png": png_path.name,
        })
    audit = {
        **plan,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha},
        "tiles": records,
    }
    (dumps / "object_infer_audit.json").write_text(
        json.dumps(audit, indent=2, default=str) + "\n", encoding="utf-8"
    )
    return audit


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Spec 118 US3: visible-object inference (two mutually exclusive modes)")
    ap.add_argument("--checkpoint", required=True, type=Path, help="frozen ObjectSegmentNet checkpoint_best.pt")
    ap.add_argument("--inputs", nargs="+", type=Path, default=None,
                    help="loose PNG tile(s) or directories (OOD mode: no store, no ground truth)")
    ap.add_argument("--output", type=Path, default=None, help="output dir for --inputs mode")
    ap.add_argument("--store", type=Path, default=None, help="v50 store for batch mode")
    ap.add_argument("--dumps", type=Path, default=None, help="output dir for --store mode")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--write", action="store_true", help="write PNGs + audit (default: print plan only)")
    args = ap.parse_args(argv)

    loose = args.inputs is not None
    store_mode = args.store is not None
    if loose == store_mode:
        print("REFUSING: exactly one of --inputs (loose) or --store (batch) is required")
        return 2
    try:
        if loose:
            if args.output is None:
                print("REFUSING: --inputs mode requires --output")
                return 2
            result = infer_loose_inputs(
                checkpoint=args.checkpoint, inputs=args.inputs, output=args.output,
                device=args.device, write=args.write,
            )
        else:
            if args.dumps is None:
                print("REFUSING: --store mode requires --dumps")
                return 2
            result = infer_store(
                checkpoint=args.checkpoint, store=args.store, dumps=args.dumps,
                device=args.device, write=args.write,
            )
    except ObjectInferError as exc:
        print(f"REFUSING: {exc}")
        return 2
    print(json.dumps(result if args.write else result, indent=2, default=str)[:4000], flush=True)
    if not args.write:
        print("DRY RUN ONLY -- pass --write to emit class PNGs and the audit record.", flush=True)
    return 0


__all__ = [
    "AUDIT_SCHEMA",
    "CLASS_COLORS",
    "ObjectInferError",
    "load_frozen_model",
    "predict_class_map",
    "infer_loose_inputs",
    "infer_store",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
