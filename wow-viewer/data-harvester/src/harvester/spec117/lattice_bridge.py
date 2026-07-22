"""Spec 117 US3(i): bridge a frozen lattice predictor's generated output into the existing
``--feature-store`` contract (research.md D-01).

``direct_geometry_train.py``/``geometry_detailer_train.py`` already validate ``--feature-store``
purely structurally: ``schema == "v115-feature-map-v1"``, ``class_count >= 1``, a ``feature_map``
array present, full row coverage via ``source_row_index``. Neither trainer inspects the channel's
semantics (they never assume the channels sum to a probability simplex). A ``class_count=1`` scalar
height-prior channel satisfies that contract exactly as written -- so this bridge writes it, and
**no changes to either trainer are required or made**, mirroring
``harvester.spec116.structure_feature_bridge`` exactly.

The lattice's native 545 samples are two REGULAR grids at the same 16-world-unit stride (17x17
outer covering the full 0..256 span, 16x16 inner offset by 8 and covering 8..248) -- not one
unified irregular grid. Rather than fit an irregular/quincunx grid (extra complexity, extra
dependency), this bridge independently bilinear-upsamples each regular grid to 256x256
(``align_corners=True``, since both grids are evenly spaced) and averages them. This is a
documented approximation of "a dense field from the sparse lattice," not a precision reconstruction
-- adequate for a coarse structural PRIOR channel, consistent with this project's
time-to-signal-over-rigor preference.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.spec117.lattice_contract import INNER_DIM, OUTER_DIM
from harvester.v50.model_stage_contract import sha256_file

FEATURE_STORE_SCHEMA = "v115-feature-map-v1"
FEATURE_ARRAY = "feature_map"
CLASS_COUNT = 1
PIXELS = 256


class LatticeBridgeError(ValueError):
    """Raised when a frozen lattice checkpoint cannot be bridged into a feature-map store."""


def _upsample_grid(grid: np.ndarray, size: int):
    """Bilinear-resize one regular (rows, cols) grid to (size, size)."""
    import torch
    import torch.nn.functional as functional

    tensor = torch.from_numpy(np.asarray(grid, dtype=np.float32))[None, None]
    resized = functional.interpolate(tensor, size=(size, size), mode="bilinear", align_corners=True)
    return resized[0, 0].numpy()


def lattice_to_feature_map(
    *,
    store: Path,
    checkpoint: Path,
    output: Path,
    device: str = "cpu",
    write: bool = False,
) -> dict:
    """Run a frozen ``LatticeNet`` checkpoint over every row's minimap and optionally persist.

    Returns the materialization plan (or the written store summary if ``write=True``). The source
    store is never mutated; the derived output is immutable once written and bound to the
    checkpoint's sha256.
    """
    import pyarrow as pa
    import pyarrow.parquet as pq
    import torch
    import zarr

    from harvester.spec117.lattice_model import LatticeNet

    source_group = zarr.open_group(str(store), mode="r")
    if "minimap_rgb" not in source_group:
        raise LatticeBridgeError(f"store is missing minimap_rgb: {store}")
    index_path = store / "index.parquet"
    if not index_path.exists():
        raise LatticeBridgeError(f"store has no index.parquet: {store}")
    index_rows = pq.read_table(index_path).to_pylist()
    row_count = len(index_rows)

    ckpt = torch.load(checkpoint, map_location="cpu", weights_only=False)
    checkpoint_sha = sha256_file(checkpoint)
    # `architecture` (identity_for_path's config_sha256) only hashes the config; `lattice_config`
    # carries the raw `base` needed to reconstruct an architecturally-identical LatticeNet.
    lattice_config = ckpt.get("lattice_config", {})
    base = int(lattice_config.get("base", 24)) if isinstance(lattice_config, dict) else 24

    plan = {
        "schema": "v117-lattice-bridge-plan-v1",
        "source_store": str(store),
        "checkpoint": {"path": str(checkpoint), "sha256": checkpoint_sha, "base": base},
        "selected_rows": row_count,
        "output_array": {
            "name": FEATURE_ARRAY,
            "shape": [row_count, CLASS_COUNT, PIXELS, PIXELS],
            "dtype": "float16",
        },
    }
    if not write:
        return plan

    if output.exists() and any(output.iterdir()):
        raise LatticeBridgeError(f"refusing to overwrite non-empty feature-map store {output}; choose a new path")

    model = LatticeNet(base=base)
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

    outer_count = OUTER_DIM * OUTER_DIM
    for row in range(row_count):
        rgb = np.asarray(source_group["minimap_rgb"][row], dtype=np.float32) / 255.0
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(dev)
        with torch.no_grad():
            predicted = model(rgb_tensor).squeeze(0).cpu().numpy()  # (545,) in [0, 1]
        outer = predicted[:outer_count].reshape(OUTER_DIM, OUTER_DIM)
        inner = predicted[outer_count:].reshape(INNER_DIM, INNER_DIM)
        dense = (_upsample_grid(outer, PIXELS) + _upsample_grid(inner, PIXELS)) / 2.0
        feature_array[row] = dense[None, :, :].astype(np.float16)

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
            "source_signal": "wdl_lattice",
            "source_store": str(store),
            "checkpoint_path": str(checkpoint),
            "checkpoint_sha256": checkpoint_sha,
            "bridge": "spec117_lattice_to_feature_map_v1",
            "upsample_method": "independent_bilinear_average_outer_inner",
        }
    )

    return {**plan, "schema": FEATURE_STORE_SCHEMA, "created_utc": created_utc, "output": str(output)}


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="Spec 117 US3(i): bridge a frozen lattice checkpoint into a v115-feature-map-v1 store (dry-run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="source v50 curriculum Zarr store")
    ap.add_argument("--checkpoint", required=True, type=Path, help="frozen LatticeNet checkpoint_best.pt")
    ap.add_argument("--output", required=True, type=Path, help="derived feature-map store output directory")
    ap.add_argument("--device", default="cpu", help="cpu (default) or cuda")
    ap.add_argument("--write", action="store_true", help="write the derived store (default: print plan only)")
    args = ap.parse_args(argv)

    result = lattice_to_feature_map(
        store=args.store, checkpoint=args.checkpoint, output=args.output, device=args.device, write=args.write,
    )
    print(json.dumps(result, indent=2, default=str), flush=True)
    if not args.write:
        print("DRY RUN ONLY -- pass --write to emit the derived feature-map store.", flush=True)
        return 0
    print(f"wrote feature-map store: {args.output}", flush=True)
    return 0


__all__ = [
    "LatticeBridgeError",
    "lattice_to_feature_map",
    "main",
    "FEATURE_STORE_SCHEMA",
    "FEATURE_ARRAY",
    "CLASS_COUNT",
]


if __name__ == "__main__":
    raise SystemExit(main())
