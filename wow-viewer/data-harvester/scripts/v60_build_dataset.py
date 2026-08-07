#!/usr/bin/env python3
"""Build the unified v60 Zarr datastore (Spec 134 US1).

Consolidates the existing v50.1 Zarr stores (which are already built and validated,
including 0.5.3 Kalimdor) into a single unified v60 store, and harvests only the
builds/maps that v50 does NOT already have (e.g. 1.0.0, 3.3.5) via harvest-stream.

This avoids re-harvesting maps that already work — the v50 datastore already has
0.5.3 and 4.0.0 built. We only stream the new builds.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/v60_build_dataset.py --client-root <path> --output <path>
"""

from __future__ import annotations

import argparse
import hashlib
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
from harvester.v50.classify import compute_signal_tier  # noqa: E402
from harvester.v50.contracts import DEFAULT_RELEASE_V60, STORE_SCHEMA_V60  # noqa: E402

HARVEST_PROJECT = Path(__file__).resolve().parents[2] / "tools" / "harvest" / "WowViewer.Tool.Harvest"
DLL_SEARCH = [
    HARVEST_PROJECT / "bin" / "Debug" / tfm / "WowViewer.Tool.Harvest.dll"
    for tfm in ("net10.0", "net9.0", "net8.0")
]

# Builds that already have v50.1 stores on disk — we consolidate these, not re-harvest.
# Maps that already exist in v50 for a build are skipped during streaming.
V50_STORE_ROOT = Path(__file__).resolve().parents[2] / "output" / "datasets" / "v50" / "v50.1"


def _find_harvest_dll() -> Path:
    # Always rebuild so the DLL reflects the current source.
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

    Searches recursively so era-folder layouts work (e.g. ``Vanilla/0.x/<client>/World of Warcraft``
    and ``Vanilla/1.x/<client>/World of Warcraft``). The build_id is the client folder name.
    """
    root = Path(client_root)
    if not root.exists():
        print(f"  WARNING: client root not found: {root}", flush=True)
        return []
    clients: list[tuple[str, str]] = []
    for wow_path in sorted(root.rglob("World of Warcraft")):
        if not wow_path.is_dir():
            continue
        clients.append((wow_path.parent.name, str(wow_path)))
    # Also accept a client folder that IS the root (has Data/WTF directly).
    if not clients:
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and ((entry / "Data").is_dir() or (entry / "WTF").is_dir()):
                clients.append((entry.name, str(entry)))
    return clients


def _discover_maps(harvest_dll: Path, client_path: str) -> list[str]:
    """Run discover-maps for a client and return the usable map names (JSON output)."""
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
        json_start = result.stdout.find("[")
        if json_start < 0:
            print(f"  WARNING: discover-maps produced no JSON for {client_path}", flush=True)
            return []
        try:
            records = json.loads(result.stdout[json_start:])
        except json.JSONDecodeError as e:
            print(f"  WARNING: discover-maps JSON parse failed: {e}", flush=True)
            return []
        return [
            str(record["map"])
            for record in records
            if record.get("include") and record.get("hasUsableTile")
        ]
    except subprocess.TimeoutExpired:
        print(f"  WARNING: discover-maps timed out for {client_path}", flush=True)
        return []


def _is_per_build_store(store: Path) -> bool:
    """True for a per-build/map store like 0_5_3_3368-Kalimdor.zarr.

    Excludes superseded datastores (coarse-mit_*, curriculum-*, feature-map-*,
    terrain-feature-labels-*, etc.) whose build part is not a real build ID.
    A real build ID contains digits (e.g. 0_5_3_3368, 4_0_0_11927).
    """
    if not (store / "index.parquet").exists():
        return False
    name = store.name
    if not name.endswith(".zarr") or "-" not in name:
        return False
    build = name[:-len(".zarr")].split("-", 1)[0]
    return any(ch.isdigit() for ch in build)


def _existing_v50_maps() -> dict[str, set[str]]:
    """Return {build_id: {map_name}} for v50.1 per-build stores already on disk."""
    result: dict[str, set[str]] = {}
    if not V50_STORE_ROOT.exists():
        return result
    for store in sorted(V50_STORE_ROOT.glob("*.zarr")):
        if not _is_per_build_store(store):
            continue
        name = store.name
        # e.g. 0_5_3_3368-Kalimdor.zarr -> build=0_5_3_3368, map=Kalimdor
        build, map_name = name[:-len(".zarr")].split("-", 1)
        result.setdefault(build, set()).add(map_name)
    return result


def _stream_tiles(
    harvest_dll: Path,
    client_path: str,
    build_id: str,
    map_name: str,
) -> list[dict]:
    """Run harvest-stream for one build/map and return all tile dicts.

    Streams the raw binary output incrementally (Popen + read) instead of buffering
    the whole map in memory. Uses a reader thread + queue timeout to detect a wedged
    stream instead of hanging forever.
    """
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
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    import queue
    import threading

    STREAM_IDLE_TIMEOUT = 300.0

    def _read_exact(n: int) -> bytes | None:
        chunks = bytearray()
        while len(chunks) < n:
            want = n - len(chunks)
            result_q: queue.Queue = queue.Queue(maxsize=1)

            def _reader() -> None:
                try:
                    result_q.put(proc.stdout.read(want))
                except Exception as exc:  # noqa: BLE001
                    result_q.put(exc)

            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            try:
                got = result_q.get(timeout=STREAM_IDLE_TIMEOUT)
            except queue.Empty:
                print(f"  WARNING: no output for {STREAM_IDLE_TIMEOUT:.0f}s — stream appears "
                      f"wedged (got {len(tiles)} tiles so far)", flush=True)
                return None
            if isinstance(got, Exception):
                print(f"  WARNING: stream read error: {got}", flush=True)
                return None
            if not got:
                return None
            chunks.extend(got)
        return bytes(chunks)

    tiles: list[dict] = []
    try:
        while True:
            header = _read_exact(8)
            if header is None or len(header) < 8:
                break
            magic = header[:4]
            length = struct.unpack("<i", header[4:8])[0]
            if magic == b"ENDS":
                break
            if magic != b"ARRY":
                continue
            if length <= 0 or length > 512 * 1024 * 1024:
                print(f"  WARNING: implausible tile length {length}, aborting stream", flush=True)
                break
            blob = _read_exact(length)
            if blob is None or len(blob) < length:
                break
            try:
                tile = read_tile_blob(blob)
                if tile:
                    tile["_build_id"] = build_id
                    tile["_map"] = map_name
                    tiles.append(tile)
            except Exception as e:
                print(f"  WARNING: tile decode error: {e}", flush=True)
    finally:
        proc.stdout.close()
        proc.wait()

    if proc.returncode != 0:
        stderr = proc.stderr.read().decode("utf-8", errors="replace")[-500:]
        print(f"  WARNING: harvest-stream failed for {build_id}/{map_name}: {stderr}", flush=True)
    proc.stderr.close()

    print(f"  Got {len(tiles)} tiles", flush=True)
    return tiles


def _write_per_build_store(
    store_path: Path,
    tiles: list[dict],
    build_id: str,
    map_name: str,
) -> int:
    """Write one build/map's tiles to a per-build Zarr store. Returns signal count."""
    if not tiles:
        return 0
    if store_path.exists():
        shutil.rmtree(store_path)
    store_path.parent.mkdir(parents=True, exist_ok=True)

    group = zarr.open_group(str(store_path), mode="w")

    signal_names = sorted(set(
        k for tile in tiles for k in tile
        if isinstance(tile[k], np.ndarray) and not k.startswith("_")
    ))

    index_rows = []
    for tile_id, tile in enumerate(tiles):
        height = tile.get("height_257")
        levels = 0
        signal_class = "na"
        evidence = "no height data"
        if height is not None:
            h = np.asarray(height, dtype=np.float32)
            if h.size > 0:
                levels = int(np.unique(h).size)
                height_range = float(np.max(h) - np.min(h))
                tier = compute_signal_tier(height_range=height_range, surviving_levels=levels)
                signal_class = tier.tier.value
                evidence = tier.evidence
        index_rows.append({
            "build_id": build_id,
            "map": map_name,
            "tile_x": int(tile.get("tile_x", -1)),
            "tile_y": int(tile.get("tile_y", -1)),
            "tile_id": tile_id,
            "surviving_height_levels": levels,
            "signal_class": signal_class,
            "signal_class_evidence": evidence,
        })

    pq.write_table(pa.Table.from_pylist(index_rows), str(store_path / "index.parquet"))

    written = 0
    for signal_name in signal_names:
        arrays = [np.asarray(tile[signal_name]) for tile in tiles if signal_name in tile]
        if not arrays:
            continue
        shape = arrays[0].shape
        dtype = arrays[0].dtype
        stacked = np.stack(
            [a.astype(dtype) if a.shape == shape else np.zeros(shape, dtype=dtype) for a in arrays],
            axis=0,
        )
        group.create_dataset(signal_name, data=stacked, shape=stacked.shape, dtype=dtype, overwrite=True)
        written += 1

    group.attrs.update({
        "store_schema": STORE_SCHEMA_V60,
        "release": DEFAULT_RELEASE_V60,
        "build_id": build_id,
        "map": map_name,
        "row_count": len(tiles),
        "signal_count": written,
    })
    return written


def _merge_into_unified(
    per_build_stores: list[Path],
    output_path: Path,
    release: str,
) -> dict:
    """Merge all per-build stores into a single unified v60 store (streaming, bounded memory)."""
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    staging = output_path.parent / f".{output_path.name}.staging-{uuid.uuid4().hex}"
    group = zarr.open_group(str(staging), mode="w")

    all_signal_names: set[str] = set()
    for store in per_build_stores:
        g = zarr.open_group(str(store), mode="r")
        all_signal_names.update(g.array_keys())
    all_signal_names = sorted(all_signal_names)

    # First pass: build index + determine shapes (no arrays held in memory).
    index_rows: list[dict] = []
    signal_shapes: dict[str, tuple] = {}
    signal_dtypes: dict[str, np.dtype] = {}
    unavailable: list[dict] = []

    for store in per_build_stores:
        g = zarr.open_group(str(store), mode="r")
        idx = pq.read_table(store / "index.parquet").to_pylist()
        for row_id, row in enumerate(idx):
            index_rows.append(row)
            for name in all_signal_names:
                if name in g and row_id < g[name].shape[0]:
                    arr = np.asarray(g[name][row_id])
                    if name not in signal_shapes:
                        signal_shapes[name] = arr.shape
                        signal_dtypes[name] = arr.dtype

    pq.write_table(pa.Table.from_pylist(index_rows), str(staging / "index.parquet"))
    total_rows = len(index_rows)
    print(f"  Wrote unified index.parquet with {total_rows} rows", flush=True)

    written = 0
    for name in all_signal_names:
        if name not in signal_shapes:
            unavailable.append({"name": name, "reason": "no_source_data:not_present_in_any_store"})
            print(f"  SKIP {name}: present in zero tiles", flush=True)
            continue
        shape = signal_shapes[name]
        dtype = signal_dtypes[name]
        group.create_dataset(
            name,
            shape=(total_rows, *shape),
            dtype=dtype,
            chunks=(1, *shape),
            overwrite=True,
        )
        written += 1

    # Second pass: write each signal row-by-row.
    for name in all_signal_names:
        if name not in signal_shapes:
            continue
        shape = signal_shapes[name]
        dtype = signal_dtypes[name]
        target = group[name]
        row = 0
        for store in per_build_stores:
            g = zarr.open_group(str(store), mode="r")
            idx = pq.read_table(store / "index.parquet").to_pylist()
            for row_id in range(len(idx)):
                if name in g and row_id < g[name].shape[0]:
                    arr = np.asarray(g[name][row_id])
                    if arr.shape == shape:
                        target[row] = arr.astype(dtype) if arr.dtype != dtype else arr
                    else:
                        target[row] = np.zeros(shape, dtype=dtype)
                else:
                    target[row] = np.zeros(shape, dtype=dtype)
                row += 1
        print(f"  Wrote {name}: shape=({total_rows}, *{shape}) dtype={dtype}", flush=True)

    group.attrs.update({
        "store_schema": STORE_SCHEMA_V60,
        "release": release,
        "row_count": len(index_rows),
        "signal_count": written,
        "builds": sorted({row["build_id"] for row in index_rows}),
        "unavailable_signals": unavailable,
    })

    for attempt in range(6):
        try:
            if output_path.exists():
                shutil.rmtree(output_path)
            staging.rename(output_path)
            break
        except OSError as exc:
            time.sleep(0.2 * (2**attempt))
    else:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"could not replace {output_path}")

    return {
        "store_path": str(output_path),
        "row_count": len(index_rows),
        "signal_count": written,
        "builds": sorted({row["build_id"] for row in index_rows}),
        "unavailable_signals": unavailable,
    }


def _merge_into_unified_dedup(
    per_build_stores: list[Path],
    output_path: Path,
    release: str,
) -> dict:
    """Merge all per-build stores into a single unified v60 store with deduplication.

    Many signal arrays (height_257, normal_xyz, minimap_rgb, terrain_shadow_256) are
    byte-identical across builds for the same map when the terrain wasn't changed.
    This stores each UNIQUE array once (keyed by content hash) and keeps a per-row
    pointer into the canonical set, so identical data is never stored twice. Full
    lineage (build_id, map, tile_x, tile_y) is preserved in the index, so every
    build's data stays queryable and attributable.

    Layout:
      index.parquet          — one row per tile: build_id, map, tile_x, tile_y,
                               surviving_height_levels, signal_class, evidence
      <signal>/canonical     — [unique_count, *shape] unique arrays
      <signal>/row_index     — [row_count] int32 pointer into canonical
      <signal>/row_hash      — [row_count] str content hash (lineage/audit)
    """
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    staging = output_path.parent / f".{output_path.name}.staging-{uuid.uuid4().hex}"
    group = zarr.open_group(str(staging), mode="w")

    all_signal_names: set[str] = set()
    for store in per_build_stores:
        g = zarr.open_group(str(store), mode="r")
        all_signal_names.update(g.array_keys())
    all_signal_names = sorted(all_signal_names)

    # First pass: build the unified index (lineage) and per-signal row arrays.
    # We hold one signal's arrays at a time, not all of them, to bound memory.
    index_rows: list[dict] = []
    for store in per_build_stores:
        g = zarr.open_group(str(store), mode="r")
        idx = pq.read_table(store / "index.parquet").to_pylist()
        for row in idx:
            index_rows.append(row)

    pq.write_table(pa.Table.from_pylist(index_rows), str(staging / "index.parquet"))
    total_rows = len(index_rows)
    print(f"  Wrote unified index.parquet with {total_rows} rows", flush=True)

    unavailable: list[dict] = []
    written = 0
    total_unique = 0
    total_naive = 0

    for name in all_signal_names:
        # Two-pass, memory-bounded dedup. Pass 1 computes each row's content hash and
        # the canonical (unique) set WITHOUT holding all row arrays in memory — only
        # hashes and one array at a time. Pass 2 writes the canonical arrays and the
        # per-row pointers. This keeps memory flat regardless of dataset size.
        #
        # A v50 store's index.parquet can have MORE rows than a signal array when the
        # signal is unavailable for some tiles (per-row unavailability). Out-of-range
        # rows are treated as unavailable (None).
        shape = None
        dtype = None
        hash_to_idx: dict[str, int] = {}
        row_hash: list[str] = []
        row_index = np.zeros(total_rows, dtype=np.int32)
        unique_count = 0

        # Pass 1: hash every row, build the unique set. Parallelized per store with a
        # thread pool so Zarr reads + hashing overlap across tiles (I/O-bound).
        import concurrent.futures

        def _hash_store_rows(store: Path) -> tuple[list[str], tuple | None, np.dtype | None]:
            g = zarr.open_group(str(store), mode="r")
            idx = pq.read_table(store / "index.parquet").to_pylist()
            if name not in g:
                return [""] * len(idx), None, None
            arr_len = g[name].shape[0]
            hashes: list[str] = []
            local_shape = None
            local_dtype = None
            for row_id in range(len(idx)):
                if row_id >= arr_len:
                    hashes.append("")
                    continue
                arr = np.asarray(g[name][row_id])
                if local_shape is None:
                    local_shape = arr.shape
                    local_dtype = arr.dtype
                a = arr.astype(local_dtype) if arr.dtype != local_dtype else arr
                hashes.append(hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest())
            return hashes, local_shape, local_dtype

        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            futures = {pool.submit(_hash_store_rows, store): store for store in per_build_stores}
            # Collect in STORE ORDER so row_hash aligns with pass 2's store iteration.
            results = {store: fut.result() for fut, store in futures.items()}
            for store in per_build_stores:
                hashes, local_shape, local_dtype = results[store]
                if local_shape is not None and shape is None:
                    shape = local_shape
                    dtype = local_dtype
                row_hash.extend(hashes)
                for h in hashes:
                    if h and h not in hash_to_idx:
                        hash_to_idx[h] = unique_count
                        unique_count += 1

        if unique_count == 0:
            unavailable.append({"name": name, "reason": "no_source_data:not_present_in_any_store"})
            print(f"  SKIP {name}: present in zero tiles", flush=True)
            continue

        # Pre-allocate canonical + pointers.
        sig = group.create_group(name)
        sig.create_array("canonical", shape=(unique_count, *shape), dtype=dtype, overwrite=True)
        sig.create_array("row_index", data=row_index, overwrite=True)
        hash_arr = np.array(row_hash, dtype="U64")
        sig.create_array("row_hash", data=hash_arr, overwrite=True)

        # Pass 2: write each unique array once, and fill row_index pointers.
        canonical = sig["canonical"]
        written_hashes: set[str] = set()
        r = 0
        for store in per_build_stores:
            g = zarr.open_group(str(store), mode="r")
            idx = pq.read_table(store / "index.parquet").to_pylist()
            if name not in g:
                r += len(idx)
                continue
            arr_len = g[name].shape[0]
            for row_id in range(len(idx)):
                if row_id >= arr_len:
                    row_index[r] = -1
                    r += 1
                    continue
                arr = np.asarray(g[name][row_id])
                a = arr.astype(dtype) if arr.dtype != dtype else arr
                h = row_hash[r]
                idx_u = hash_to_idx[h]
                row_index[r] = idx_u
                if h not in written_hashes:
                    canonical[idx_u] = a
                    written_hashes.add(h)
                r += 1
        sig["row_index"][:] = row_index

        total_unique += unique_count
        total_naive += total_rows
        written += 1
        print(f"  Wrote {name}: {unique_count} unique / {total_rows} rows "
              f"(saved {total_rows - unique_count} copies)", flush=True)

    group.attrs.update({
        "store_schema": STORE_SCHEMA_V60,
        "release": release,
        "row_count": total_rows,
        "signal_count": written,
        "dedup": True,
        "unique_arrays": total_unique,
        "naive_arrays": total_naive,
        "builds": sorted({row["build_id"] for row in index_rows}),
        "unavailable_signals": unavailable,
    })

    for attempt in range(6):
        try:
            if output_path.exists():
                shutil.rmtree(output_path)
            staging.rename(output_path)
            break
        except OSError as exc:
            time.sleep(0.2 * (2**attempt))
    else:
        shutil.rmtree(staging, ignore_errors=True)
        raise RuntimeError(f"could not replace {output_path}")

    return {
        "store_path": str(output_path),
        "row_count": total_rows,
        "signal_count": written,
        "builds": sorted({row["build_id"] for row in index_rows}),
        "unavailable_signals": unavailable,
        "unique_arrays": total_unique,
        "naive_arrays": total_naive,
    }


def _copy_v50_store_into_work(
    v50_store: Path,
    work_dir: Path,
) -> Path | None:
    """Copy an existing v50.1 per-build store into the work dir.

    Returns the work-dir path, or None if the store is not a per-build store
    (e.g. a superseded coarse-mit_* / curriculum-* datastore).
    """
    if not _is_per_build_store(v50_store):
        return None
    name = v50_store.name
    build, map_name = name[:-len(".zarr")].split("-", 1)
    dest = work_dir / f"{build}-{map_name}.zarr"
    if dest.exists():
        shutil.rmtree(dest)
    shutil.copytree(v50_store, dest)
    print(f"  Consolidated v50 store: {v50_store.name} -> {dest}", flush=True)
    return dest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build unified v60 Zarr datastore — consolidate v50 stores + harvest new builds"
    )
    parser.add_argument("--client-root", required=True,
                        help="Path to the directory containing WoW client build folders")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output v60 Zarr store path (e.g. ../output/datasets/v60/v60/unified.zarr)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be harvested without running anything")
    parser.add_argument("--release", default=DEFAULT_RELEASE_V60)
    parser.add_argument("--dedup", action="store_true",
                        help="Enable deduplication (store unique arrays once with per-row pointers). "
                             "SLOW on large corpora; off by default.")
    parser.add_argument("--skip-builds", default="",
                        help="Comma-separated build IDs to skip entirely (e.g. '3.3.5.12340'). "
                             "Matches by substring against the client folder name.")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel harvest workers (default 4). Each worker streams "
                             "one build/map at a time.")
    args = parser.parse_args()

    skip_builds = [b.strip().lower() for b in args.skip_builds.split(",") if b.strip()]
    if skip_builds:
        print(f"Skipping builds: {skip_builds}", flush=True)

    client_root = str(args.client_root).replace("\\", "/")

    clients = _discover_clients(client_root)
    if not clients:
        raise SystemExit(f"ERROR: no WoW client roots found under {client_root}")

    print("Finding harvest tool...", flush=True)
    harvest_dll = _find_harvest_dll()
    print(f"  DLL: {harvest_dll}", flush=True)

    # Which builds/maps already exist in the v50 datastore?
    existing_v50 = _existing_v50_maps()
    print(f"Existing v50 stores: {len(existing_v50)} builds", flush=True)
    for build, maps in existing_v50.items():
        print(f"  {build}: {sorted(maps)}", flush=True)

    if args.dry_run:
        print("\nDiscovered clients:")
        for build_id, client_path in clients:
            maps = _discover_maps(harvest_dll, client_path)
            print(f"  {build_id}: {client_path} -> {len(maps)} maps: {maps[:5]}{'...' if len(maps) > 5 else ''}")
        print(f"\nOutput: {args.output}")
        return 0

    work_dir = args.output.parent / ".v60-work"
    work_dir.mkdir(parents=True, exist_ok=True)
    per_build_stores: list[Path] = []

    try:
        # 1. Consolidate existing v50 stores (already built, no re-harvest).
        for v50_store in sorted(V50_STORE_ROOT.glob("*.zarr")):
            dest = _copy_v50_store_into_work(v50_store, work_dir)
            if dest is not None:
                per_build_stores.append(dest)

        # 2. Harvest builds/maps NOT already in v50, in parallel.
        import concurrent.futures

        # Build the job list first (discover maps is quick; streaming is the slow part).
        jobs: list[tuple[str, str, str]] = []  # (build_id, client_path, map_name)
        for build_id, client_path in clients:
            if skip_builds and any(s in build_id.lower() for s in skip_builds):
                print(f"  SKIP {build_id}: excluded via --skip-builds", flush=True)
                continue
            maps = _discover_maps(harvest_dll, client_path)
            if not maps:
                print(f"  SKIP {build_id}: no discoverable maps", flush=True)
                continue
            already = existing_v50.get(build_id, set())
            to_stream = [m for m in maps if m not in already]
            if not to_stream:
                print(f"  SKIP {build_id}: all maps already in v50 ({sorted(already)})", flush=True)
                continue
            print(f"  {build_id}: {len(to_stream)} new maps to stream "
                  f"(skipping {len(already)} already in v50)", flush=True)
            for map_name in to_stream:
                jobs.append((build_id, client_path, map_name))

        def _harvest_one(job: tuple[str, str, str]) -> Path | None:
            build_id, client_path, map_name = job
            store = work_dir / f"{build_id}-{map_name}.zarr"
            if store.exists():
                print(f"  SKIP {build_id}/{map_name}: already harvested", flush=True)
                return store
            tiles = _stream_tiles(harvest_dll, client_path, build_id, map_name)
            if not tiles:
                return None
            n = _write_per_build_store(store, tiles, build_id, map_name)
            print(f"  Wrote per-build store: {store} ({len(tiles)} tiles, {n} signals)", flush=True)
            return store

        workers = max(1, args.workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for store in pool.map(_harvest_one, jobs):
                if store is not None:
                    per_build_stores.append(store)

        if not per_build_stores:
            raise SystemExit("ERROR: no stores to merge")

        print(f"\nMerging {len(per_build_stores)} stores into unified v60 store...", flush=True)
        if args.dedup:
            result = _merge_into_unified_dedup(per_build_stores, args.output, args.release)
        else:
            result = _merge_into_unified(per_build_stores, args.output, args.release)

        print(f"\n[DONE] v60 unified store: {result['store_path']}")
        print(f"       {result['row_count']} tiles, {result['signal_count']} signals, "
              f"{len(result['builds'])} builds")
        if "unique_arrays" in result:
            print(f"       dedup: {result['unique_arrays']} unique arrays vs "
                  f"{result['naive_arrays']} naive "
                  f"(saved {result['naive_arrays'] - result['unique_arrays']} copies)")
        if result["unavailable_signals"]:
            print(f"       {len(result['unavailable_signals'])} signals unavailable:")
            for u in result["unavailable_signals"][:5]:
                print(f"         {u['name']}: {u['reason']}")
    finally:
        # Keep the work dir so a later run can resume; it's small relative to the store.
        pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())