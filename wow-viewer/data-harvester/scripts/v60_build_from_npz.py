"""Build a unified v60 Zarr store from NPZ harvest shards (Spec 134 US1).

Reads NPZ shards from the harvest tool's output (harvest-map-mpq), builds a single
v60-format Zarr store with all signals including terrain_shadow_256, and writes a
unified index across all builds and maps.

The v60 store is the training dataset — NOT the archaeology pipeline. Archaeology
(spec 127/132) is a separate analysis tool that reads from stores or NPZ. This
builder creates the canonical training store directly from raw harvest output.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/v60_build_from_npz.py \\
        --npz-dir ../output/archaeology/0_5_3_3368/npz/Azeroth \\
        --npz-dir ../output/archaeology/0_5_3_3368/npz/Kalimdor \\
        --output ../output/datasets/v60/v60.1/unified.zarr
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
import uuid
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.contracts import DEFAULT_RELEASE_V60, STORE_SCHEMA_V60  # noqa: E402


def _parse_npz_coords(stem: str) -> tuple[str, int, int]:
    """Parse tile coordinates from a harvest NPZ filename.
    
    Formats:
        MapName_XX_YY_harvest.npz -> (MapName, XX, YY)
        MapName_XXX_YYY_harvest.npz -> (MapName_XXX, XXX...) -> (first part, last two)
    """
    base = stem.replace("_harvest", "")
    parts = base.split("_")
    if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
        tile_x = int(parts[-2])
        tile_y = int(parts[-1])
        map_name = "_".join(parts[:-2])
    else:
        map_name = "unknown"
        tile_x = -1
        tile_y = -1
    return map_name, tile_x, tile_y


def _build_id_from_npz_dir(npz_dir: Path) -> str:
    """Extract build ID from the NPZ directory path structure.
    
    Expected: .../<build_id>/npz/<map_name>/*.npz
    """
    parent = npz_dir.parent
    if parent.name == "npz":
        grandparent = parent.parent
        return grandparent.name
    return "unknown"


def _replace_directory(staging_path: Path, output_path: Path) -> None:
    """Move staging_path onto output_path, retrying transient failures."""
    last_error: OSError | None = None
    for attempt in range(6):
        try:
            if output_path.exists():
                shutil.rmtree(output_path)
            staging_path.rename(output_path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2 * (2**attempt))
    raise RuntimeError(
        f"could not replace {output_path} with {staging_path} after retrying: {last_error}"
    )


def build_v60_from_npz(
    npz_dirs: list[Path],
    output_path: Path,
    *,
    release: str = DEFAULT_RELEASE_V60,
) -> dict:
    """Build a unified v60 Zarr store from NPZ harvest shard directories.

    Each NPZ directory should contain ``*_harvest.npz`` files from the harvest tool.
    The build ID is inferred from the directory path structure.

    Returns a summary dict with row count, signal count, and source info.
    """
    # Discover all NPZ shards across all directories
    npz_files: list[tuple[Path, str, int, int, str]] = []  # (path, map, tx, ty, build_id)
    for npz_dir in npz_dirs:
        if not npz_dir.exists():
            print(f"WARNING: NPZ directory not found: {npz_dir}", flush=True)
            continue
        build_id = _build_id_from_npz_dir(npz_dir)
        for path in sorted(npz_dir.glob("*_harvest.npz")):
            map_name, tx, ty = _parse_npz_coords(path.stem)
            npz_files.append((path, map_name, tx, ty, build_id))

    if not npz_files:
        raise ValueError(f"no NPZ harvest files found in {npz_dirs}")

    print(f"Found {len(npz_files)} NPZ shards across {len(npz_dirs)} directories", flush=True)

    # Load all NPZ data and determine signal names
    all_tiles: list[dict] = []
    all_signal_names: set[str] = set()
    for npz_path, map_name, tx, ty, build_id in npz_files:
        try:
            data = dict(np.load(npz_path))
        except Exception as e:
            print(f"  WARNING: {npz_path.name}: {e}", flush=True)
            continue
        data["_map"] = map_name
        data["_tile_x"] = tx
        data["_tile_y"] = ty
        data["_build_id"] = build_id
        all_tiles.append(data)
        for key in data:
            if isinstance(data[key], np.ndarray) and not key.startswith("_"):
                all_signal_names.add(key)

    if not all_tiles:
        raise ValueError("no NPZ shards could be loaded")

    all_signal_names = sorted(all_signal_names)
    print(f"Signals: {len(all_signal_names)} across {len(all_tiles)} tiles", flush=True)

    # Build index
    index_rows: list[dict] = []
    for tile_id, tile in enumerate(all_tiles):
        index_rows.append({
            "build_id": tile["_build_id"],
            "map": tile["_map"],
            "tile_x": int(tile["_tile_x"]),
            "tile_y": int(tile["_tile_y"]),
            "tile_id": tile_id,
        })

    # Build the v60 store
    if output_path.exists():
        shutil.rmtree(output_path)

    staging_path = output_path.parent / f".{output_path.name}.staging-{uuid.uuid4().hex}"
    staging_path.parent.mkdir(parents=True, exist_ok=True)

    written_signals = 0
    unavailable_signals: list[dict] = []

    try:
        root = zarr.open_group(str(staging_path), mode="w")

        # Write index
        index_table = pa.Table.from_pylist(index_rows)
        pq.write_table(index_table, str(staging_path / "index.parquet"))
        print(f"  Wrote index.parquet with {len(index_rows)} rows", flush=True)

        # Write each signal as a 3D array [tile_id, ...shape]
        for signal_name in all_signal_names:
            arrays: list[np.ndarray] = []
            missing = 0
            shape = None
            dtype = None

            for tile in all_tiles:
                arr = tile.get(signal_name)
                if arr is not None:
                    a = np.asarray(arr)
                    if shape is None:
                        shape = a.shape
                        dtype = a.dtype
                    if a.shape != shape:
                        print(f"  WARNING: {signal_name}: shape mismatch {a.shape} vs {shape}"
                              f" for tile {tile['_map']}_{tile['_tile_x']}_{tile['_tile_y']}",
                              flush=True)
                        missing += 1
                        continue
                    if a.dtype != dtype:
                        a = a.astype(dtype)
                    arrays.append(np.ascontiguousarray(a))
                else:
                    missing += 1

            if missing == len(all_tiles) or not arrays:
                unavailable_signals.append({
                    "name": signal_name,
                    "reason": "no_source_data:not_present_in_any_npz_shard",
                })
                print(f"  SKIP {signal_name}: zero tiles have this signal", flush=True)
                continue

            if missing > 0:
                print(f"  {signal_name}: {missing}/{len(all_tiles)} tiles missing (zero-filled)",
                      flush=True)
                # Zero-fill missing tiles
                for _ in range(missing):
                    arrays.append(np.zeros(shape, dtype=dtype))

            stacked = np.stack(arrays, axis=0)
            root.create_dataset(signal_name, data=stacked, shape=stacked.shape,
                                dtype=dtype, overwrite=True)
            written_signals += 1
            print(f"  Wrote {signal_name}: shape={stacked.shape} dtype={dtype}", flush=True)

        # Write manifest
        manifest = {
            "store_schema": STORE_SCHEMA_V60,
            "release": release,
            "row_count": len(all_tiles),
            "signal_count": written_signals,
            "source_dirs": [str(d) for d in npz_dirs],
            "unavailable_signals": unavailable_signals,
        }
        root.attrs.update(manifest)
        print(f"  Manifest: {written_signals} signals, {len(all_tiles)} rows", flush=True)

    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    _replace_directory(staging_path, output_path)

    return {
        "store_path": str(output_path),
        "row_count": len(all_tiles),
        "signal_count": written_signals,
        "source_dirs": [str(d) for d in npz_dirs],
        "unavailable_signals": unavailable_signals,
    }


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build unified v60 Zarr store from NPZ harvest shards"
    )
    parser.add_argument("--npz-dir", required=True, type=Path, action="append",
                        dest="npz_dirs", metavar="DIR",
                        help="Directory of *_harvest.npz shards; repeatable for multiple "
                             "builds/maps. Each dir should be <build>/npz/<map>/")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output v60 Zarr store path")
    parser.add_argument("--release", default=DEFAULT_RELEASE_V60,
                        help=f"Release identifier (default: {DEFAULT_RELEASE_V60})")
    args = parser.parse_args()

    result = build_v60_from_npz(args.npz_dirs, args.output, release=args.release)

    print(f"\n[DONE] v60 unified store: {result['store_path']}")
    print(f"       {result['row_count']} rows, {result['signal_count']} signals")
    print(f"       {len(result['source_dirs'])} source directories")
    if result["unavailable_signals"]:
        print(f"       {len(result['unavailable_signals'])} signals unavailable:")
        for u in result["unavailable_signals"][:5]:
            print(f"         {u['name']}: {u['reason']}")
        if len(result["unavailable_signals"]) > 5:
            print(f"         ... and {len(result['unavailable_signals']) - 5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())