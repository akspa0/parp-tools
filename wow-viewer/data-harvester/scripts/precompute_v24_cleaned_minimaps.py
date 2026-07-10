"""Pre-compute cleaned minimaps for a V24 store and store them as a new array.

Run ONCE after building a V24 store (or whenever you want to add cleaned minimaps).
After this, ``TileSource.load()`` reads the pre-computed array instead of raw
minimap — no per-load ``clean_minimap()`` cost at training time.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/precompute_v24_cleaned_minimaps.py \\
        --v24-store ../output/datasets/v24/3_3_5_12340_openworld_curated.zarr \\
        --workers 4
"""

from __future__ import annotations

import argparse
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24.clean_minimap import clean_minimap  # noqa: E402
from harvester.v24 import store as v24_store  # noqa: E402


def _clean_one(args: tuple) -> tuple[int, np.ndarray | None, str]:
    """Load V18 tile, clean it, return (row, cleaned_array_or_None, status)."""
    v18_path, v18_row, row = args
    try:
        v18 = zarr.open_group(str(v18_path), mode="r")
        minimap = np.asarray(v18["minimap_rgb"][v18_row])
        obj_mask = np.asarray(v18["object_precise_mask"][v18_row], dtype=np.float32)
        has_no_obj = "no_object_minimap" in v18
        rendered = (
            np.asarray(v18["no_object_minimap"][v18_row])
            if has_no_obj
            else None
        )
        cleaned, meta = clean_minimap(minimap, obj_mask, rendered)
        return row, cleaned.astype(np.float32), meta["source"]
    except Exception as exc:
        return row, None, str(exc)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v24-store", required=True, help="V24 Zarr store path")
    parser.add_argument("--workers", type=int, default=1,
                        help="parallel workers for cleaning (default 1 = sequential)")
    parser.add_argument("--force", action="store_true",
                        help="overwrite existing cleaned_minimap_256 array if present")
    args = parser.parse_args()

    store_path = Path(args.v24_store)
    print(f"Opening V24 store: {store_path}", flush=True)

    # Open store in read-write mode
    root = zarr.open_group(str(store_path), mode="a")
    index = v24_store.read_index(store_path)

    # Check existing
    if "cleaned_minimap_256" in root and not args.force:
        arr = root["cleaned_minimap_256"]
        n_existing = arr.shape[0] if arr.shape else 0
        print(f"cleaned_minimap_256 already exists with {n_existing} tiles. "
              f"Use --force to overwrite.", flush=True)
        return 1

    v18_path = root.attrs.get("v18_store_path")
    if not v18_path:
        print("V24 store has no v18_store_path attr", file=sys.stderr, flush=True)
        return 1
    v18_path = Path(v18_path)
    print(f"V18 store: {v18_path}", flush=True)

    n_tiles = len(index["tile_id"])
    print(f"Tiles to clean: {n_tiles}", flush=True)

    # Build args for each tile
    job_args = []
    for row in range(n_tiles):
        v18_row = int(index["v18_row"][row])
        job_args.append((v18_path, v18_row, row))

    # Create output array (uint8, 256x256x3 — same as V18 minimap)
    dtype = np.float32
    shape = (n_tiles, 256, 256, 3)

    # Check if we need to recreate
    if "cleaned_minimap_256" in root and args.force:
        del root["cleaned_minimap_256"]

    # zarr v3 API: create_dataset with keyword-only args after name
    arr = root.create_dataset(
        "cleaned_minimap_256",
        shape=shape,
        dtype=dtype,
        chunks=(1, 256, 256, 3),
        compressor=None,  # no compression — fast reads
        fill_value=0,
    )
    print(f"Created cleaned_minimap_256: shape={shape} dtype={dtype}", flush=True)

    # Process
    stats = {"ok": 0, "failed": 0, "sources": {}}
    started = time.time()

    if args.workers > 1:
        executor = ProcessPoolExecutor(max_workers=args.workers)
        futures = {executor.submit(_clean_one, ja): ja[2] for ja in job_args}
        for future in as_completed(futures):
            row, cleaned, source = future.result()
            if cleaned is not None:
                arr[row] = cleaned
                stats["ok"] += 1
                stats["sources"][source] = stats["sources"].get(source, 0) + 1
            else:
                stats["failed"] += 1
                print(f"  row {row} FAILED: {source}", flush=True)
            if stats["ok"] % 200 == 0 and stats["ok"] > 0:
                elapsed = time.time() - started
                pct = 100.0 * stats["ok"] / n_tiles
                print(f"  {stats['ok']}/{n_tiles} ({pct:.0f}%) cleaned in {elapsed:.0f}s", flush=True)
    else:
        # Sequential (default) — shows progress per 5%
        v18 = zarr.open_group(str(v18_path), mode="r")
        step = max(1, n_tiles // 20)
        for row in range(n_tiles):
            v18_row = int(index["v18_row"][row])
            try:
                minimap = np.asarray(v18["minimap_rgb"][v18_row])
                obj_mask = np.asarray(v18["object_precise_mask"][v18_row], dtype=np.float32)
                has_no_obj = "no_object_minimap" in v18
                rendered = (
                    np.asarray(v18["no_object_minimap"][v18_row])
                    if has_no_obj
                    else None
                )
                cleaned, meta = clean_minimap(minimap, obj_mask, rendered)
                arr[row] = cleaned.astype(dtype)
                stats["ok"] += 1
                stats["sources"][meta["source"]] = stats["sources"].get(meta["source"], 0) + 1
            except Exception as exc:
                stats["failed"] += 1
                print(f"  row {row} FAILED: {exc}", flush=True)

            if (row + 1) % step == 0 or (row + 1) == n_tiles:
                elapsed = time.time() - started
                pct = 100.0 * (row + 1) / n_tiles
                eta = elapsed / (row + 1) * (n_tiles - row - 1)
                print(f"  [{row + 1}/{n_tiles} ({pct:.0f}%)] elapsed={elapsed:.0f}s eta={eta:.0f}s "
                      f"ok={stats['ok']} failed={stats['failed']}", flush=True)

    elapsed = time.time() - started
    print(f"\nDone: {stats['ok']}/{n_tiles} cleaned in {elapsed:.0f}s", flush=True)
    print(f"Sources: {stats['sources']}", flush=True)
    if stats["failed"]:
        print(f"WARNING: {stats['failed']} tiles failed", file=sys.stderr, flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
