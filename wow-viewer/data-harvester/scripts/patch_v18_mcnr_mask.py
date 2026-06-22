#!/usr/bin/env python3
"""Patch existing V18 Zarr stores: fix normal_xyz to raw MCNR + add mcnr_mask_257.

The old C# AssembleNormals interpolated checkerboard gaps before writing.
This script:
1. Zeros out gap positions in normal_xyz (x%2 != y%2) to restore raw state
2. Adds mcnr_mask_257 (True where x%2 == y%2)

The normal_mask array is already correct (50% coverage) and is NOT modified.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/patch_v18_mcnr_mask.py --builds 0_5_3_3368 3_3_5_12340
    uv run python scripts/patch_v18_mcnr_mask.py --all
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import zarr
import zarr.storage


_V18_ROOT = Path(__file__).resolve().parent.parent.parent / "output" / "datasets" / "v18"


def _make_checkerboard_mask(size: int = 257) -> np.ndarray:
    """Create MCNR checkerboard mask: True where x%2 == y%2."""
    mask = np.zeros((size, size), dtype=bool)
    for y in range(size):
        for x in range(size):
            if x % 2 == y % 2:
                mask[y, x] = True
    return mask


def patch_build(zarr_path: Path, dry_run: bool = False) -> int:
    """Patch one Zarr store. Returns number of tiles patched."""
    print(f"\nPatching: {zarr_path}")

    store = zarr.storage.LocalStore(str(zarr_path), read_only=False)
    root = zarr.open_group(store=store, mode="a")

    if "normal_xyz" not in root:
        print(f"  SKIP: no normal_xyz array")
        store.close()
        return 0

    if "normal_mask" not in root:
        print(f"  SKIP: no normal_mask array")
        store.close()
        return 0

    n_tiles = root["normal_xyz"].shape[0]
    checkerboard = _make_checkerboard_mask(257)

    normal_mask_coverage = float(root["normal_mask"][:].mean())
    print(f"  Tiles: {n_tiles}, normal_mask coverage: {normal_mask_coverage:.3f}")

    if normal_mask_coverage > 0.99:
        print(f"  normal_mask coverage {normal_mask_coverage:.3f} ~1.0 (fully interpolated)")
        print(f"  Will zero gap positions based on MCNR checkerboard pattern")

    if dry_run:
        print(f"  DRY RUN: would zero gap positions and add mcnr_mask_257")
        store.close()
        return 0

    normals = root["normal_xyz"][:]

    gap_count_before = int(np.any(np.abs(normals[:, ~checkerboard]) > 1e-6, axis=-1).sum())
    normals[:, ~checkerboard] = 0.0
    gap_count_after = int(np.any(np.abs(normals[:, ~checkerboard]) > 1e-6, axis=-1).sum())

    print(f"  Zeroed gap positions: {gap_count_before} -> {gap_count_after}")

    del root["normal_xyz"]
    root.create_array("normal_xyz", data=normals, chunks=(1, 257, 257), overwrite=True)

    if "mcnr_mask_257" in root:
        del root["mcnr_mask_257"]
    mask_broadcast = np.broadcast_to(checkerboard, (n_tiles, 257, 257)).copy()
    root.create_array("mcnr_mask_257", data=mask_broadcast, chunks=(1, 257, 257), overwrite=True)

    print(f"  Added mcnr_mask_257: shape={mask_broadcast.shape}")
    print(f"  DONE")
    store.close()
    return n_tiles


def main() -> None:
    parser = argparse.ArgumentParser(description="Patch V18 Zarr stores with MCNR mask")
    parser.add_argument("--builds", type=str, nargs="+", default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.all:
        builds = [p.stem for p in sorted(_V18_ROOT.glob("*.zarr"))]
    elif args.builds:
        builds = args.builds
    else:
        parser.error("Specify --builds or --all")

    total = 0
    for build in builds:
        zarr_path = _V18_ROOT / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"SKIP {build}: {zarr_path} not found")
            continue
        total += patch_build(zarr_path, dry_run=args.dry_run)

    print(f"\nTotal tiles patched: {total}")


if __name__ == "__main__":
    main()
