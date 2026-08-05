#!/usr/bin/env python3
"""Build the unified v60 Zarr datastore — one shot, no intermediate NPZ files (Spec 134 US1).

Streams tile data directly from the C# harvest tool into per-build Zarr stores
(the v50 pattern), then merges them into a single unified v60 Zarr store. No NPZ
intermediates. Each build/map is written incrementally as it streams, so output
appears on disk immediately and memory stays bounded.

The script discovers all WoW clients under ``--client-root``, runs ``discover-maps``
on each to find the actual terrain maps available, then streams each build/map into
its own per-build Zarr store. After all builds complete, it merges them into the
unified v60 store. Uses ``--dry-run`` to preview what would be harvested.

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
from harvester.v50.classify import compute_signal_tier  # noqa: E402
from harvester.v50.contracts import DEFAULT_RELEASE_V60, STORE_SCHEMA_V60  # noqa: E402

HARVEST_PROJECT = Path(__file__).resolve().parents[2] / "tools" / "harvest" / "WowViewer.Tool.Harvest"
DLL_SEARCH = [
    HARVEST_PROJECT / "bin" / "Debug" / tfm / "WowViewer.Tool.Harvest.dll"
    for tfm in ("net10.0", "net9.0", "net8.0")
]


def _find_harvest_dll() -> Path:
    # Always rebuild so the DLL reflects the current source (e.g. the per-tile timeout fix).
    # Returning a stale DLL silently runs old behavior.
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
    r"""Enumerate client root, return list of (build_id, client_path) for every WoW client root."""
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


def _stream_tiles(
    harvest_dll: Path,
    client_path: str,
    build_id: str,
    map_name: str,
) -> list[dict]:
    """Run harvest-stream for one build/map and return all tile dicts.

    Streams the raw binary output incrementally (Popen + read) instead of buffering
    the whole map in memory — a full map's binary stream is gigabytes.
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

    # If the C# tool hangs on a tile, a blocking read would wedge forever. select.select
    # does not work on Windows pipes (only sockets), so use a reader thread + queue with
    # a timeout to detect a wedged stream and report it instead of hanging.
    import queue
    import threading

    STREAM_IDLE_TIMEOUT = 300.0  # seconds of no output before we declare the stream wedged

    def _read_exact(n: int) -> bytes | None:
        """Read exactly n bytes from stdout, or None on EOF/timeout."""
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

    # Determine signal names present in this build/map
    signal_names = sorted(set(
        k for tile in tiles for k in tile
        if isinstance(tile[k], np.ndarray) and not k.startswith("_")
    ))

    # Index rows with sidecar metadata
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
    """Merge all per-build stores into a single unified v60 store."""
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    staging = output_path.parent / f".{output_path.name}.staging-{uuid.uuid4().hex}"
    group = zarr.open_group(str(staging), mode="w")

    # Discover all signals across all per-build stores
    all_signal_names: set[str] = set()
    for store in per_build_stores:
        g = zarr.open_group(str(store), mode="r")
        all_signal_names.update(g.array_keys())
    all_signal_names = sorted(all_signal_names)

    # Build unified index and per-signal arrays
    index_rows: list[dict] = []
    signal_arrays: dict[str, list[np.ndarray]] = {name: [] for name in all_signal_names}
    signal_shapes: dict[str, tuple] = {}
    signal_dtypes: dict[str, np.dtype] = {}
    unavailable: list[dict] = []

    for store in per_build_stores:
        g = zarr.open_group(str(store), mode="r")
        idx = pq.read_table(store / "index.parquet").to_pylist()
        for row_id, row in enumerate(idx):
            index_rows.append(row)
            for name in all_signal_names:
                if name in g:
                    arr = np.asarray(g[name][row_id])
                    if name not in signal_shapes:
                        signal_shapes[name] = arr.shape
                        signal_dtypes[name] = arr.dtype
                    if arr.shape == signal_shapes[name]:
                        signal_arrays[name].append(arr)
                    else:
                        signal_arrays[name].append(np.zeros(signal_shapes[name], dtype=signal_dtypes[name]))
                else:
                    if name not in signal_shapes:
                        signal_shapes[name] = (1,)
                        signal_dtypes[name] = np.float32
                    signal_arrays[name].append(np.zeros(signal_shapes[name], dtype=signal_dtypes[name]))

    # Write index
    pq.write_table(pa.Table.from_pylist(index_rows), str(staging / "index.parquet"))
    print(f"  Wrote unified index.parquet with {len(index_rows)} rows", flush=True)

    written = 0
    for name in all_signal_names:
        arrays = signal_arrays[name]
        if not arrays or all(a.size == 0 for a in arrays):
            unavailable.append({"name": name, "reason": "no_source_data:not_present_in_any_store"})
            print(f"  SKIP {name}: present in zero tiles", flush=True)
            continue
        shape = signal_shapes[name]
        dtype = signal_dtypes[name]
        stacked = np.stack(arrays, axis=0)
        group.create_dataset(name, data=stacked, shape=stacked.shape, dtype=dtype, overwrite=True)
        written += 1
        print(f"  Wrote {name}: shape={stacked.shape} dtype={dtype}", flush=True)

    group.attrs.update({
        "store_schema": STORE_SCHEMA_V60,
        "release": release,
        "row_count": len(index_rows),
        "signal_count": written,
        "builds": sorted({row["build_id"] for row in index_rows}),
        "unavailable_signals": unavailable,
    })

    # Atomic replace
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


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build unified v60 Zarr datastore — directly from C# harvest tool, no intermediates"
    )
    parser.add_argument("--client-root", required=True,
                        help="Path to the directory containing WoW client build folders")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output v60 Zarr store path (e.g. ../output/datasets/v60/v60/unified.zarr)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would be harvested without running anything")
    parser.add_argument("--resume", action="store_true",
                        help="Resume: keep per-build stores in a stable work dir and skip "
                             "builds/maps already harvested")
    parser.add_argument("--release", default=DEFAULT_RELEASE_V60)
    args = parser.parse_args()

    client_root = str(args.client_root).replace("\\", "/")

    clients = _discover_clients(client_root)
    if not clients:
        raise SystemExit(f"ERROR: no WoW client roots found under {client_root}")

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

    # Per-build stores live beside the output. With --resume they persist in a stable
    # work dir so a wedged/interrupted run can be resumed without re-harvesting.
    if args.resume:
        work_dir = args.output.parent / ".v60-work"
        work_dir.mkdir(parents=True, exist_ok=True)
    else:
        work_dir = args.output.parent / f".v60-work-{uuid.uuid4().hex}"
        work_dir.mkdir(parents=True, exist_ok=True)

    per_build_stores: list[Path] = []

    try:
        for build_id, client_path in clients:
            maps = _discover_maps(harvest_dll, client_path)
            if not maps:
                print(f"  SKIP {build_id}: no discoverable maps", flush=True)
                continue
            print(f"  {build_id}: {len(maps)} maps to stream", flush=True)
            for map_name in maps:
                store = work_dir / f"{build_id}-{map_name}.zarr"
                if args.resume and store.exists():
                    print(f"  SKIP {build_id}/{map_name}: already harvested", flush=True)
                    per_build_stores.append(store)
                    continue
                tiles = _stream_tiles(harvest_dll, client_path, build_id, map_name)
                if not tiles:
                    continue
                n = _write_per_build_store(store, tiles, build_id, map_name)
                per_build_stores.append(store)
                print(f"  Wrote per-build store: {store} ({len(tiles)} tiles, {n} signals)", flush=True)

        if not per_build_stores:
            raise SystemExit("ERROR: no tiles harvested from any build")

        print(f"\nMerging {len(per_build_stores)} per-build stores into unified v60 store...", flush=True)
        result = _merge_into_unified(per_build_stores, args.output, args.release)

        print(f"\n[DONE] v60 unified store: {result['store_path']}")
        print(f"       {result['row_count']} tiles, {result['signal_count']} signals, "
              f"{len(result['builds'])} builds")
        if result["unavailable_signals"]:
            print(f"       {len(result['unavailable_signals'])} signals unavailable:")
            for u in result["unavailable_signals"][:5]:
                print(f"         {u['name']}: {u['reason']}")
    finally:
        # Only clean up the temp work dir on a non-resume run; resume keeps it for later.
        if not args.resume:
            shutil.rmtree(work_dir, ignore_errors=True)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())