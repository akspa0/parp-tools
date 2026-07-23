"""Spec 118 US3: bridge a frozen object segmenter's generated output into the existing
``--feature-store`` contract (research.md D-06, FR-011).

``direct_geometry_train.py``/``geometry_detailer_train.py`` already validate ``--feature-store``
purely structurally: ``schema == "v115-feature-map-v1"``, ``class_count >= 1``, a ``feature_map``
array present, full row coverage via ``source_row_index``. This bridge writes the segmenter's two
object-class softmax channels (doodad, building -- the ``none`` channel is redundant as 1 - sum),
so **no changes to either trainer are required or made**, mirroring
``harvester.spec117.lattice_bridge`` exactly (including the ``object_config.base`` reconstruction
that checkpoint's raw ``architecture.config_sha256`` cannot provide).

The model's PREDICTED map -- never the ground-truth mask -- is what enters the chain as an input
channel (FR-014). The source store is never mutated (FR-013).
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.spec118.object_contract import BRIDGE_CLASS_COUNT as CLASS_COUNT
from harvester.v50.model_stage_contract import sha256_file

FEATURE_STORE_SCHEMA = "v115-feature-map-v1"
FEATURE_ARRAY = "feature_map"
PIXELS = 256


class ObjectBridgeError(ValueError):
    """Raised when a frozen object-segmenter checkpoint cannot be bridged into a feature store."""


def objects_to_feature_map(
    *,
    store: Path,
    checkpoint: Path,
    output: Path,
    device: str = "cpu",
    write: bool = False,
) -> dict:
    """Run a frozen ``ObjectSegmentNet`` checkpoint over every row's minimap and optionally persist.

    Returns the materialization plan (or the written store summary if ``write=True``).
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import zarr

    from harvester.spec118.object_segment_model import ObjectSegmentNet

    source_group = zarr.open_group(str(store), mode="r")
    if "minimap_rgb" not in source_group:
        raise ObjectBridgeError(f"store is missing minimap_rgb: {store}")
    index_path = store / "index.parquet"
    if not index_path.exists():
        raise ObjectBridgeError(f"store has no index.parquet: {store}")
    index_rows = pq.read_table(index_path).to_pylist()
    row_count = len(index_rows)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_sha = sha256_file(checkpoint)
    # `architecture.config_sha256` only hashes the config; `object_config` carries the raw `base`
    # needed to reconstruct an architecturally-identical ObjectSegmentNet before load_state_dict.
    object_config = ckpt.get("object_config", {})
    base = int(object_config.get("base", 24)) if isinstance(object_config, dict) else 24

    plan = {
        "schema": "v118-object-bridge-plan-v1",
        "source_store": str(store),
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha, "base": base},
        "selected_rows": row_count,
        "output_array": {
            "name": FEATURE_ARRAY,
            "shape": [row_count, CLASS_COUNT, PIXELS, PIXELS],
            "dtype": "float16",
        },
        "channels": ["doodad_softmax", "building_softmax"],
    }
    if not write:
        return plan

    if output.exists() and any(output.iterdir()):
        raise ObjectBridgeError(f"refusing to overwrite non-empty feature-map store {output}; choose a new path")

    model = ObjectSegmentNet(base=base)
    model.load_state_dict(ckpt["model"])
    model.eval()
    dev = torch.device(device)
    model = model.to(dev)

    output.mkdir(parents=True, exist_ok=False)
    group = zarr.open_group(str(output), mode="w")
    feature_array = group.create_array(
        FEATURE_ARRAY,
        shape=(row_count, CLASS_COUNT, PIXELS, PIXELS),
        chunks=(1, CLASS_COUNT, PIXELS, PIXELS),
        dtype=np.float16,
    )

    for row in range(row_count):
        rgb = np.asarray(source_group["minimap_rgb"][row], dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            probs = torch.softmax(model(rgb_tensor).squeeze(0), dim=0).cpu().numpy()  # (3, 256, 256)
        feature_array[row] = probs[1:].astype(np.float16)  # doodad, building; none dropped

    derived_index = [
        {
            "source_row_index": row,
            "map": str(index_rows[row].get("map")),
            "tile_x": int(index_rows[row].get("tile_x", -1)),
            "tile_y": int(index_rows[row].get("tile_y", -1)),
        }
        for row in range(row_count)
    ]
    pq.write_table(pa.Table.from_pylist(derived_index), output / "index.parquet")

    created_utc = datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
    group.attrs.update(
        {
            "schema": FEATURE_STORE_SCHEMA,
            "created_utc": created_utc,
            "class_count": CLASS_COUNT,
            "source_signal": "object_geometry_visible",
            "source_store": str(store),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "bridge": "spec118_objects_to_feature_map_v1",
            "channels": ["doodad_softmax", "building_softmax"],
        }
    )

    return {**plan, "schema": FEATURE_STORE_SCHEMA, "created_utc": created_utc, "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Spec 118 US3: bridge a frozen object segmenter into a v115-feature-map-v1 store (dry-run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="source v50 curriculum Zarr store")
    ap.add_argument("--checkpoint", required=True, type=Path, help="frozen ObjectSegmentNet checkpoint_best.pt")
    ap.add_argument("--output", required=True, type=Path, help="derived feature-map store output directory")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--write", action="store_true", help="write the derived store (default: print plan only)")
    args = ap.parse_args(argv)

    result = objects_to_feature_map(
        store=args.store, checkpoint=args.checkpoint, output=args.output, device=args.device, write=args.write,
    )
    print(json.dumps(result, indent=2, default=str), flush=True)
    if not args.write:
        print("DRY RUN ONLY -- pass --write to emit the derived feature-map store.", flush=True)
        return 0
    print(f"wrote feature-map store: {args.output}", flush=True)
    return 0


__all__ = [
    "ObjectBridgeError",
    "objects_to_feature_map",
    "main",
    "FEATURE_STORE_SCHEMA",
    "FEATURE_ARRAY",
    "CLASS_COUNT",
]


if __name__ == "__main__":
    raise SystemExit(main())
