#!/usr/bin/env python3
"""Harvest all clients into a single deduplicated v61 Zarr store, on the fly.

Streams each build/map directly from the C# harvest tool and deduplicates as it
goes: every signal array is content-hashed, and identical arrays (across builds
and maps) are stored ONCE in a canonical set with per-row pointers. This hits
two birds with one stone — harvest and dedup happen in a single pass, so there
is no separate slow merge/dedup step.

Multi-threaded across (build, map) pairs. A global content-hash registry (guarded
by a lock) is shared by all workers. Each tile has a short per-tile timeout so a
hanging tile is skipped instead of blocking the stream forever.

Layout:
  index.parquet          — one row per tile: build_id, map, tile_x, tile_y,
                           surviving_height_levels, signal_class, evidence
  <signal>/canonical     — [unique_count, *shape] unique arrays (stored once)
  <signal>/row_index     — [row_count] int32 pointer into canonical
  <signal>/row_hash      — [row_count] U64 content hash (lineage/audit)

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/v61_harvest.py --client-root H:/CLIENTS/Vanilla --output <store.zarr>
"""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import shutil
import struct
import subprocess
import sys
import threading
import queue
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

STREAM_IDLE_TIMEOUT = 60.0


def _find_harvest_dll() -> Path:
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
    root = Path(client_root)
    if not root.exists():
        print(f"  WARNING: client root not found: {root}", flush=True)
        return []
    clients: list[tuple[str, str]] = []
    for wow_path in sorted(root.rglob("World of Warcraft")):
        if wow_path.is_dir():
            clients.append((wow_path.parent.name, str(wow_path)))
    if not clients:
        for entry in sorted(root.iterdir()):
            if entry.is_dir() and ((entry / "Data").is_dir() or (entry / "WTF").is_dir()):
                clients.append((entry.name, str(entry)))
    return clients


def _discover_maps(harvest_dll: Path, client_path: str) -> list[str]:
    import json

    cmd = ["dotnet", str(harvest_dll), "discover-maps", "--client-root", client_path]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            print(f"  WARNING: discover-maps failed: {result.stderr[-200:]}", flush=True)
            return []
        json_start = result.stdout.find("[")
        if json_start < 0:
            return []
        try:
            records = json.loads(result.stdout[json_start:])
        except json.JSONDecodeError:
            return []
        return [
            str(record["map"])
            for record in records
            if record.get("include") and record.get("hasUsableTile")
        ]
    except subprocess.TimeoutExpired:
        print(f"  WARNING: discover-maps timed out for {client_path}", flush=True)
        return []


class DedupStore:
    """Thread-safe on-the-fly dedup writer into a single Zarr store.

    Each signal has a canonical array (unique arrays, appended as discovered) plus
    per-row pointers. A global content-hash registry maps hash -> canonical index,
    so identical arrays across builds/maps are stored once.
    """

    def __init__(self, output_path: Path, release: str):
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.staging = output_path.parent / f".{output_path.name}.staging-{uuid.uuid4().hex}"
        self.group = zarr.open_group(str(self.staging), mode="w")
        self.release = release
        self.lock = threading.Lock()
        self.index_rows: list[dict] = []
        # signal -> {hash: canonical_index}
        self.hash_to_idx: dict[str, dict[str, int]] = {}
        # signal -> canonical array (zarr Array)
        self.canonical: dict[str, zarr.Array] = {}
        # signal -> count of unique arrays appended so far
        self.unique_count: dict[str, int] = {}
        # signal -> shape/dtype
        self.shapes: dict[str, tuple] = {}
        self.dtypes: dict[str, np.dtype] = {}
        # signal -> list of row pointers (parallel to index_rows)
        self.row_index: dict[str, list[int]] = {}
        self.row_hash: dict[str, list[str]] = {}

    def _ensure_signal(self, name: str, shape: tuple, dtype: np.dtype) -> None:
        if name in self.canonical:
            return
        self.shapes[name] = shape
        self.dtypes[name] = dtype
        sig = self.group.create_group(name)
        self.canonical[name] = sig.create_array(
            "canonical", shape=(0, *shape), dtype=dtype, chunks=(1, *shape), overwrite=True
        )
        self.unique_count[name] = 0
        self.hash_to_idx[name] = {}
        self.row_index[name] = []
        self.row_hash[name] = []

    def add_tile(self, tile: dict) -> None:
        """Register one tile's signals into the dedup store (thread-safe)."""
        build_id = tile["_build_id"]
        map_name = tile["_map"]
        height = tile.get("height_257")
        levels, signal_class, evidence = 0, "na", "no height data"
        if height is not None:
            h = np.asarray(height, dtype=np.float32)
            if h.size > 0:
                levels = int(np.unique(h).size)
                tier = compute_signal_tier(height_range=float(np.max(h) - np.min(h)),
                                           surviving_levels=levels)
                signal_class, evidence = tier.tier.value, tier.evidence

        with self.lock:
            row_id = len(self.index_rows)
            self.index_rows.append({
                "build_id": build_id, "map": map_name,
                "tile_x": int(tile.get("tile_x", -1)), "tile_y": int(tile.get("tile_y", -1)),
                "tile_id": row_id, "surviving_height_levels": levels,
                "signal_class": signal_class, "signal_class_evidence": evidence,
            })
            for name, arr in tile.items():
                if not isinstance(arr, np.ndarray) or name.startswith("_"):
                    continue
                a = np.asarray(arr)
                self._ensure_signal(name, a.shape, a.dtype)
                h = hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
                idx_map = self.hash_to_idx[name]
                if h in idx_map:
                    self.row_index[name].append(idx_map[h])
                else:
                    idx = self.unique_count[name]
                    self.canonical[name].append(a[np.newaxis, ...])
                    self.unique_count[name] += 1
                    idx_map[h] = idx
                    self.row_index[name].append(idx)
                self.row_hash[name].append(h)

    def finalize(self, output_path: Path) -> dict:
        """Write index + row pointers, then atomically move staging into place."""
        pq.write_table(pa.Table.from_pylist(self.index_rows),
                       str(self.staging / "index.parquet"))
        total_rows = len(self.index_rows)
        written = 0
        total_unique = 0
        for name in self.canonical:
            sig = self.group[name]
            sig.create_array("row_index", data=np.array(self.row_index[name], dtype=np.int32),
                             overwrite=True)
            sig.create_array("row_hash", data=np.array(self.row_hash[name], dtype="U64"),
                             overwrite=True)
            written += 1
            total_unique += self.unique_count[name]
        self.group.attrs.update({
            "store_schema": STORE_SCHEMA_V60, "release": self.release,
            "row_count": total_rows, "signal_count": written,
            "dedup": True, "unique_arrays": total_unique,
            "builds": sorted({r["build_id"] for r in self.index_rows}),
        })
        for attempt in range(6):
            try:
                if output_path.exists():
                    shutil.rmtree(output_path)
                self.staging.rename(output_path)
                break
            except OSError as exc:
                time.sleep(0.2 * (2**attempt))
        else:
            shutil.rmtree(self.staging, ignore_errors=True)
            raise RuntimeError(f"could not replace {output_path}")
        return {"row_count": total_rows, "signal_count": written, "unique_arrays": total_unique}


def _stream_tiles(harvest_dll: Path, client_path: str, build_id: str, map_name: str) -> list[dict]:
    if not Path(client_path).exists():
        return []
    cmd = [
        "dotnet", str(harvest_dll), "harvest-stream",
        "--client-root", client_path, "--map", map_name, "--stream-profile", "v22",
    ]
    print(f"  [{build_id}] streaming {map_name} ...", flush=True)
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    def _read_exact(n: int) -> bytes | None:
        chunks = bytearray()
        while len(chunks) < n:
            want = n - len(chunks)
            q: queue.Queue = queue.Queue(maxsize=1)

            def _reader() -> None:
                try:
                    q.put(proc.stdout.read(want))
                except Exception as exc:  # noqa: BLE001
                    q.put(exc)

            t = threading.Thread(target=_reader, daemon=True)
            t.start()
            try:
                got = q.get(timeout=STREAM_IDLE_TIMEOUT)
            except queue.Empty:
                print(f"  [{build_id}] {map_name}: no output for {STREAM_IDLE_TIMEOUT:.0f}s, "
                      f"skipping (got {len(tiles)} tiles)", flush=True)
                return None
            if isinstance(got, Exception) or not got:
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
                print(f"  [{build_id}] {map_name}: tile decode error: {e}", flush=True)
    finally:
        proc.stdout.close()
        proc.wait()
    proc.stderr.close()
    print(f"  [{build_id}] {map_name}: {len(tiles)} tiles", flush=True)
    return tiles


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Harvest all clients into a single deduplicated v61 Zarr store, on the fly"
    )
    parser.add_argument("--client-root", required=True,
                        help="Root containing WoW client build folders (recursively searched)")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output v61 Zarr store path (e.g. ../output/datasets/v61/v61.zarr)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Parallel harvest workers (default 4)")
    parser.add_argument("--skip-builds", default="",
                        help="Comma-separated build substrings to skip")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    skip_builds = [b.strip().lower() for b in args.skip_builds.split(",") if b.strip()]

    clients = _discover_clients(args.client_root)
    if not clients:
        raise SystemExit(f"ERROR: no WoW client roots found under {args.client_root}")
    print(f"Discovered {len(clients)} clients", flush=True)

    print("Finding harvest tool...", flush=True)
    harvest_dll = _find_harvest_dll()
    print(f"  DLL: {harvest_dll}", flush=True)

    jobs: list[tuple[str, str, str]] = []
    for build_id, client_path in clients:
        if skip_builds and any(s in build_id.lower() for s in skip_builds):
            print(f"  SKIP {build_id}: excluded", flush=True)
            continue
        maps = _discover_maps(harvest_dll, client_path)
        if not maps:
            print(f"  SKIP {build_id}: no discoverable maps", flush=True)
            continue
        print(f"  {build_id}: {len(maps)} maps", flush=True)
        for map_name in maps:
            jobs.append((build_id, client_path, map_name))

    if args.dry_run:
        print(f"\nWould harvest {len(jobs)} build/map pairs into {args.output}")
        return 0

    store = DedupStore(args.output, DEFAULT_RELEASE_V60)

    def _harvest_one(job: tuple[str, str, str]) -> int:
        build_id, client_path, map_name = job
        tiles = _stream_tiles(harvest_dll, client_path, build_id, map_name)
        for tile in tiles:
            store.add_tile(tile)
        return len(tiles)

    total_tiles = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        futures = {pool.submit(_harvest_one, job): job for job in jobs}
        for fut in concurrent.futures.as_completed(futures):
            job = futures[fut]
            try:
                total_tiles += fut.result()
            except Exception as e:  # noqa: BLE001
                print(f"  ERROR {job[0]}/{job[2]}: {e}", flush=True)

    result = store.finalize(args.output)
    print(f"\n[DONE] v61 store: {args.output}")
    print(f"       {result['row_count']} tiles, {result['signal_count']} signals, "
          f"{result['unique_arrays']} unique arrays (deduped on the fly)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())