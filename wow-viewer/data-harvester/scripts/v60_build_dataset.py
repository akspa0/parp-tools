#!/usr/bin/env python3
"""Build the unified v60 Zarr datastore — one shot, no intermediate files (Spec 134 US1).

Streams tile data directly from the C# harvest tool into a single unified v60 Zarr
store for every build and map. No NPZ intermediates. Same pattern as the v50 pipeline:
``harvest-stream`` writes raw binary tile blobs to stdout, Python reads them and writes
the Zarr store directly.

The script discovers all WoW clients under ``--client-root``, runs ``discover-maps``
on each to find the actual terrain maps available, then streams each build/map and
accumulates every tile into one unified store. Uses ``--dry-run`` to preview what
would be harvested without running anything.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/v60_build_dataset.py --client-root <path> --output <path>
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
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

from harvester.raw_reader import read_tile_blob  # noqa: E402
from harvester.v50.classify import SignalTier, compute_signal_tier  # noqa: E402
from harvester.v50.contracts import DEFAULT_RELEASE_V60, STORE_SCHEMA_V60  # noqa: E402

HARVEST_PROJECT = Path(__file__).resolve().parents[2] / "tools" / "harvest" / "WowViewer.Tool.Harvest"
DLL_SEARCH = [
    HARVEST_PROJECT / "bin" / "Debug" / tfm / "WowViewer.Tool.Harvest.dll"
    for tfm in ("net10.0", "net9.0", "net8.0")
]


def _find_harvest_dll() -> Path:
    for candidate in DLL_SEARCH:
        if candidate.exists():
            return candidate
    # Build it
    result = subprocess.run(
        ["dotnet", "build", str(HARVEST_PROJECT / "WowViewer.Tool.Harvest.csproj"),
         "-c", "Debug", "-nologo"],
        capture_output=True, text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(f"harvest tool build failed:\n{result.stderr}")
    for candidate in DLL_SEARCH:
        if candidate.exists():
            return candidate
    raise RuntimeError("harvest tool DLL not found after build")


def _discover_clients(client_root: str) -> list[tuple[str, str]]:
    r"""Enumerate client root, return list of (build_id, client_path) for every WoW client root.

    Looks for ``World of Warcraft`` subdirectory inside each folder.
    The build_id is the folder name.
    """
    root = Path(client_root)
    if not root.exists():
        print(f"  WARNING: client root not found: {root}", flush=True)
        return []
    clients: list[tuple[str, str]] = []
    for entry in sorted(root.iterdir()):
        if not entry.is_dir():
            continue
        wow_path = entry / "World of Warcraft"
        if wow_path.is_dir():
            clients.append((entry.name, str(wow_path)))
        else:
            if (entry / "Data").is_dir() or (entry / "WTF").is_dir():
                clients.append((entry.name, str(entry)))
    return clients


def _discover_maps(harvest_dll: Path, client_path: str) -> list[str]:
    """Run discover-maps for a client and return the usable map names.

    discover-maps emits a JSON array of HarvestMapDiscoveryResult records. We keep
    maps that are marked Include (terrain-trainable) and have a usable tile.
    """
    import json

    cmd = [
        "dotnet", str(harvest_dll),
        "discover-maps",
        "--client-root", client_path,
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            stderr = result.stderr[-300:]
            print(f"  WARNING: discover-maps failed: {stderr}", flush=True)
            return []
        # The JSON is the last non-empty block of stdout (listfile/md5 lines precede it).
        json_start = result.stdout.find("[")
        if json_start < 0:
            print(f"  WARNING: discover-maps produced no JSON for {client_path}", flush=True)
            return []
        try:
            records = json.loads(result.stdout[json_start:])
        except json.JSONDecodeError as e:
            print(f"  WARNING: discover-maps JSON parse failed: {e}", flush=True)
            return []
        maps = [
            str(record["map"])
            for record in records
            if record.get("include") and record.get("hasUsableTile")
        ]
        return maps
    except subprocess.TimeoutExpired:
        print(f"  WARNING: discover-maps timed out for {client_path}", flush=True)
        return []


def _stream_build_map(
    harvest_dll: Path,
    client_path: str,
    build_id: str,
    map_name: str,
) -> list[dict]:
    """Run harvest-stream for one build/map and return all tile dicts."""
    if not Path(client_path).exists():
        print(f"  SKIP: client not found: {client_path}", flush=True)
        return []

    cmd = [
        "dotnet", str(harvest_dll),
        "harvest-stream",
        "--client-root", client_path,
        "--map", map_name,
        "--stream-profile", "v22",
    ]

    print(f"  Streaming {build_id} / {map_name} ...", flush=True)
    result = subprocess.run(cmd, capture_output=True, text=False, timeout=7200)

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")[-500:]
        print(f"  WARNING: harvest-stream failed for {build_id}/{map_name}: {stderr}", flush=True)
        return []

    # Parse the raw binary stream from stdout
    stdout = result.stdout
    tiles: list[dict] = []
    offset = 0
    while offset + 8 <= len(stdout):
        magic = stdout[offset:offset + 4]
        length = struct.unpack("<i", stdout[offset + 4:offset + 8])[0]
        if magic == b"ENDS":
            break
        if magic != b"ARRY":
            offset += 1
            continue
        blob = stdout[offset + 8:offset + 8 + length]
        if len(blob) < length:
            break
        try:
            tile = read_tile_blob(blob)
            if tile:
                tile["_build_id"] = build_id
                tile["_map"] = map_name
                tiles.append(tile)
        except Exception as e:
            print(f"  WARNING: tile decode error: {e}", flush=True)
        offset += 8 + length

    print(f"  Got {len(tiles)} tiles", flush=True)
    return tiles


def _replace_directory(staging: Path, target: Path) -> None:
    for attempt in range(6):
        try:
            if target.exists():
                shutil.rmtree(target)
            staging.rename(target)
            return
        except OSError as exc:
            time.sleep(0.2 * (2**attempt))
    raise RuntimeError(f"could not replace {target} with {staging}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build unified v60 Zarr datastore — directly from C# harvest tool, no intermediates"
    )
    parser.add_argument("--client-root", required=True,
                        help="Path to the directory containing WoW client build folders "
                             "(e.g. the root where your World of Warcraft client directories live)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output v60 Zarr store path (e.g. ../output/datasets/v60/v60.1/unified.zarr)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be harvested without running anything")
    parser.add_argument("--release", default=DEFAULT_RELEASE_V60)
    args = parser.parse_args()

    client_root = str(args.client_root).replace("\\", "/")

    clients = _discover_clients(client_root)
    if not clients:
        raise SystemExit(f"ERROR: no WoW client roots found under {client_root}")

    # Find the harvest DLL
    print("Finding harvest tool...", flush=True)
    harvest_dll = _find_harvest_dll()
    print(f"  DLL: {harvest_dll}", flush=True)

    if args.dry_run:
        print("Discovered clients:")
        for build_id, client_path in clients:
            maps = _discover_maps(harvest_dll, client_path)
            print(f"  {build_id}: {client_path} -> {len(maps)} maps: {maps[:5]}{'...' if len(maps) > 5 else ''}")
        print(f"\nOutput: {args.output}")
        return 0

    # Stream all builds/maps and accumulate tiles
    all_tiles: list[dict] = []
    for build_id, client_path in clients:
        maps = _discover_maps(harvest_dll, client_path)
        if not maps:
            print(f"  SKIP {build_id}: no discoverable maps", flush=True)
            continue
        print(f"  {build_id}: {len(maps)} maps to stream", flush=True)
        for map_name in maps:
            tiles = _stream_build_map(harvest_dll, client_path, build_id, map_name)
            all_tiles.extend(tiles)

    if not all_tiles:
        raise SystemExit("ERROR: no tiles harvested from any build")

    print(f"\nTotal tiles: {len(all_tiles)}", flush=True)

    # Determine signal names
    all_signal_names = sorted(set(
        k for tile in all_tiles for k in tile
        if isinstance(tile[k], np.ndarray) and not k.startswith("_")
    ))
    print(f"Signals: {len(all_signal_names)}", flush=True)

    # Compute sidecar metadata (per-tile scalars, stored in index.parquet, not as Zarr arrays).
    # These are derived signals that Python tools compute — they must be part of the v60 store.
    print("Computing sidecar metadata...", flush=True)
    sidecar_index: list[dict] = []
    for tile_id, tile in enumerate(all_tiles):
        height = tile.get("height_257")
        if height is not None:
            h = np.asarray(height, dtype=np.float32)
            if h.size > 0:
                levels = int(np.unique(h).size)
                height_range = float(np.max(h) - np.min(h))
                tier = compute_signal_tier(height_range=height_range, surviving_levels=levels)
                sidecar_index.append({
                    "surviving_height_levels": levels,
                    "signal_class": tier.tier.value,
                    "signal_class_evidence": tier.evidence,
                })
            else:
                sidecar_index.append({
                    "surviving_height_levels": 0,
                    "signal_class": "na",
                    "signal_class_evidence": "zero-height tile",
                })
        else:
            sidecar_index.append({
                "surviving_height_levels": 0,
                "signal_class": "na",
                "signal_class_evidence": "no height data",
            })

    # Build index (with sidecar metadata)
    index_rows = []
    for tile_id, tile in enumerate(all_tiles):
        sidecar = sidecar_index[tile_id] if tile_id < len(sidecar_index) else {}
        index_rows.append({
            "build_id": tile["_build_id"],
            "map": tile["_map"],
            "tile_x": int(tile.get("tile_x", -1)),
            "tile_y": int(tile.get("tile_y", -1)),
            "tile_id": tile_id,
            "surviving_height_levels": sidecar.get("surviving_height_levels", 0),
            "signal_class": sidecar.get("signal_class", "na"),
            "signal_class_evidence": sidecar.get("signal_class_evidence", ""),
        })

    # Write the unified v60 store
    output_path = args.output
    if output_path.exists():
        shutil.rmtree(output_path)

    staging = output_path.parent / f".{output_path.name}.staging-{uuid.uuid4().hex}"
    staging.parent.mkdir(parents=True, exist_ok=True)

    written_signals = 0
    unavailable_signals: list[dict] = []

    try:
        group = zarr.open_group(str(staging), mode="w")

        # Write index
        index_table = pa.Table.from_pylist(index_rows)
        pq.write_table(index_table, str(staging / "index.parquet"))
        print(f"  Wrote index.parquet with {len(index_rows)} rows", flush=True)

        # Write each signal
        for signal_name in all_signal_names:
            arrays: list[np.ndarray] = []
            shape = None
            dtype = None
            missing = 0

            for tile in all_tiles:
                arr = tile.get(signal_name)
                if arr is not None:
                    a = np.asarray(arr)
                    if shape is None:
                        shape = a.shape
                        dtype = a.dtype
                    if a.shape != shape:
                        print(f"  WARNING: {signal_name}: shape mismatch {a.shape} vs {shape}, "
                              f"skipping tile", flush=True)
                        missing += 1
                        continue
                    arrays.append(np.ascontiguousarray(a.astype(dtype) if a.dtype != dtype else a))
                else:
                    missing += 1

            if not arrays:
                unavailable_signals.append({
                    "name": signal_name,
                    "reason": "no_source_data:not_present_in_any_streamed_tile",
                })
                print(f"  SKIP {signal_name}: present in zero tiles", flush=True)
                continue

            if missing > 0:
                print(f"  {signal_name}: {missing}/{len(all_tiles)} tiles missing (zero-filled)", flush=True)
                for _ in range(missing):
                    arrays.append(np.zeros(shape, dtype=dtype))

            stacked = np.stack(arrays, axis=0)
            group.create_dataset(signal_name, data=stacked, shape=stacked.shape,
                                dtype=dtype, overwrite=True)
            written_signals += 1
            print(f"  Wrote {signal_name}: shape={stacked.shape} dtype={dtype}", flush=True)

        # Write manifest
        manifest = {
            "store_schema": STORE_SCHEMA_V60,
            "release": args.release,
            "row_count": len(all_tiles),
            "signal_count": written_signals,
            "builds": sorted({tile["_build_id"] for tile in all_tiles}),
            "unavailable_signals": unavailable_signals,
        }
        group.attrs.update(manifest)
        print(f"  Manifest: {written_signals} signals, {len(all_tiles)} tiles", flush=True)

    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise

    _replace_directory(staging, output_path)

    print(f"\n[DONE] v60 unified store: {output_path}")
    print(f"       {len(all_tiles)} tiles, {written_signals} signals, {len(BUILDS)} builds")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())