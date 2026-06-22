#!/usr/bin/env python3
"""Recover normal_xyz from V16 store into V18 store, then apply MCNR mask patch.

The previous patch run deleted normal_xyz from the 0.5.3 V18 store before failing.
This script recovers it from the V16 store (same tiles, same data) and applies the fix.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/recover_and_patch_v18.py --build 0_5_3_3368
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr
import zarr.storage


_V16_ROOT = Path(__file__).resolve().parent.parent.parent / "output" / "datasets" / "v16"
_V18_ROOT = Path(__file__).resolve().parent.parent.parent / "output" / "datasets" / "v18"


def _make_checkerboard_mask(size: int = 257) -> np.ndarray:
    mask = np.zeros((size, size), dtype=bool)
    for y in range(size):
        for x in range(size):
            if x % 2 == y % 2:
                mask[y, x] = True
    return mask


def recover_and_patch(build: str) -> None:
    v16_path = _V16_ROOT / f"{build}.zarr"
    v18_path = _V18_ROOT / f"{build}.zarr"

    if not v16_path.exists():
        print(f"SKIP: V16 store not found: {v16_path}")
        return
    if not v18_path.exists():
        print(f"SKIP: V18 store not found: {v18_path}")
        return

    print(f"Recovering normal_xyz from V16 -> V18 for {build}")

    v16_store = zarr.storage.LocalStore(str(v16_path), read_only=True)
    v16_root = zarr.open_group(store=v16_store, mode="r")

    v18_store = zarr.storage.LocalStore(str(v18_path), read_only=False)
    v18_root = zarr.open_group(store=v18_store, mode="a")

    if "normal_xyz" not in v16_root:
        print(f"  SKIP: V16 has no normal_xyz")
        return

    v16_index = pq.read_table(str(v16_path / "index.parquet"))
    v18_index = pq.read_table(str(v18_path / "index.parquet"))

    def _tile_key(table, i):
        return f"{table.column('map')[i].as_py()}_{table.column('tile_x')[i].as_py()}_{table.column('tile_y')[i].as_py()}"

    v16_tiles = {_tile_key(v16_index, i): i for i in range(v16_index.num_rows)}
    v18_tiles = {_tile_key(v18_index, i): i for i in range(v18_index.num_rows)}

    common_tiles = sorted(set(v16_tiles.keys()) & set(v18_tiles.keys()))
    print(f"  V16 tiles: {len(v16_tiles)}, V18 tiles: {len(v18_tiles)}, common: {len(common_tiles)}")

    if "normal_xyz" in v18_root:
        print(f"  normal_xyz already exists in V18, skipping recovery")
    else:
        n_v18 = len(v18_tiles)
        normals = np.zeros((n_v18, 257, 257, 3), dtype=np.float32)
        recovered = 0
        for tile_name in common_tiles:
            v18_idx = v18_tiles[tile_name]
            v16_idx = v16_tiles[tile_name]
            normals[v18_idx] = v16_root["normal_xyz"][v16_idx]
            recovered += 1

        v18_root.create_array("normal_xyz", data=normals, chunks=(1, 257, 257, 3), overwrite=True)
        print(f"  Recovered normal_xyz: {recovered}/{n_v18} tiles")

    checkerboard = _make_checkerboard_mask(257)
    normals = v18_root["normal_xyz"][:]
    n_tiles = normals.shape[0]

    gap_before = int(np.any(np.abs(normals[:, ~checkerboard]) > 1e-6, axis=-1).sum())
    normals[:, ~checkerboard] = 0.0
    gap_after = int(np.any(np.abs(normals[:, ~checkerboard]) > 1e-6, axis=-1).sum())
    print(f"  Zeroed gap positions: {gap_before} -> {gap_after}")

    del v18_root["normal_xyz"]
    v18_root.create_array("normal_xyz", data=normals, chunks=(1, 257, 257, 3), overwrite=True)

    if "mcnr_mask_257" in v18_root:
        del v18_root["mcnr_mask_257"]
    mask_broadcast = np.broadcast_to(checkerboard, (n_tiles, 257, 257)).copy()
    v18_root.create_array("mcnr_mask_257", data=mask_broadcast, chunks=(1, 257, 257), overwrite=True)
    print(f"  Added mcnr_mask_257: shape={mask_broadcast.shape}")

    v16_store.close()
    v18_store.close()
    print(f"  DONE")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=str, required=True)
    args = parser.parse_args()
    recover_and_patch(args.build)


if __name__ == "__main__":
    main()
