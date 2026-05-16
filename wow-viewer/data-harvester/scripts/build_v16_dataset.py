"""Build V16 consolidated Zarr dataset directly from game client archives.

Single-pass pipeline: C# harvester streams NPZ blobs → Python reads from pipe → Zarr.
NO intermediate files on disk. The Zarr store IS the dataset.

Usage:
    cd wow-viewer/data-harvester

    # Build one build (all maps):
    uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340

    # Build multiple builds:
    uv run python scripts/build_v16_dataset.py build --builds 3_3_5_12340 4_0_0_11927

    # Limit tiles (for testing):
    uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --limit 100

    # Only specific maps:
    uv run python scripts/build_v16_dataset.py build --build 3_3_5_12340 --maps Azeroth Northrend

    # Check stats:
    uv run python scripts/build_v16_dataset.py stats --build 3_3_5_12340
"""

from __future__ import annotations

import argparse
import shutil
import struct
import subprocess
import sys
import time
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
import zarr.codecs
import zarr.storage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HARVEST_TOOL_DIR = _PROJECT_ROOT / "tools" / "harvest" / "WowViewer.Tool.Harvest" / "bin" / "Debug" / "net10.0"
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"
_CLIENT_ROOTS = _PROJECT_ROOT.parent / "output" / "tmp" / "wowarchive-clients"

OUTPUT_ARRAY_NAMES = {
    "height_257": "height_257",
    "mcnr_normal_xyz": "normal_xyz",
    "mcal_alpha_pack_256": "alpha_256",
    "hole_mask_16": "holes_16",
    "unified_liquid_mask": "liquid_mask",
    "unified_liquid_height": "liquid_height",
    "object_mask_257": "object_mask",
    "minimap_rgb_256": "minimap_rgb",
    "mcsh_shadow_mask_256": "shadow_mask",
    "mcly_texture_ids": "mcly_texture_ids",
    "mcly_layer_mask": "mcly_layer_mask",
}

DTYPES = {
    "height_257": np.float32, "normal_xyz": np.float32, "normal_mask": np.bool_,
    "alpha_256": np.float32, "holes_16": np.bool_, "liquid_mask": np.float32,
    "liquid_height": np.float32, "object_mask": np.bool_, "minimap_rgb": np.uint8,
    "shadow_mask": np.float32, "mcly_texture_ids": np.int32, "mcly_layer_mask": np.float32,
}

FILL_VALUES = {
    "height_257": 0.0, "normal_xyz": 0.0, "normal_mask": False,
    "alpha_256": 0.0, "holes_16": False, "liquid_mask": 0.0,
    "liquid_height": 0.0, "object_mask": False, "minimap_rgb": 0,
    "shadow_mask": 0.0, "mcly_texture_ids": -1, "mcly_layer_mask": 0.0,
}

SHAPES = {
    "height_257": (257, 257), "normal_xyz": (257, 257, 3), "normal_mask": (257, 257),
    "alpha_256": (256, 256, 4), "holes_16": (16, 16), "liquid_mask": (256, 256),
    "liquid_height": (256, 256), "object_mask": (257, 257), "minimap_rgb": (256, 256, 3),
    "shadow_mask": (256, 256), "mcly_texture_ids": (16, 16, 4), "mcly_layer_mask": (16, 16, 4),
}

CHUNK_SIZES = {
    "height_257": (64, 257, 257), "normal_xyz": (64, 257, 257, 3),
    "normal_mask": (256, 257, 257), "alpha_256": (64, 256, 256, 4),
    "holes_16": (1024, 16, 16), "liquid_mask": (64, 256, 256),
    "liquid_height": (64, 256, 256), "object_mask": (256, 257, 257),
    "minimap_rgb": (64, 256, 256, 3), "shadow_mask": (64, 256, 256),
    "mcly_texture_ids": (1024, 16, 16, 4), "mcly_layer_mask": (256, 16, 16, 4),
}

ALL_ARRAY_KEYS = [
    "height_257", "normal_xyz", "normal_mask", "alpha_256", "holes_16",
    "liquid_mask", "liquid_height", "object_mask", "minimap_rgb",
    "shadow_mask", "mcly_texture_ids", "mcly_layer_mask",
]

REQUIRED_KEYS = {"minimap_rgb_256", "height_257"}
NPZB_MAGIC = b"NPZB"
ENDS_MAGIC = b"ENDS"


def _find_harvest_tool() -> Path:
    exe = _HARVEST_TOOL_DIR / "WowViewer.Tool.Harvest.exe"
    if exe.exists():
        return exe
    for p in sorted((_PROJECT_ROOT / "tools" / "harvest" / "WowViewer.Tool.Harvest" / "bin" / "Debug").glob("*/WowViewer.Tool.Harvest.exe")):
        if p.exists():
            return p
    raise FileNotFoundError("Harvest tool not found. Build it first.")


def _find_client_root(build: str) -> Path | None:
    parent = _CLIENT_ROOTS / build
    if not parent.exists():
        return None
    for child in parent.iterdir():
        if child.is_dir() and ((child / "WoW.exe").exists() or (child / "Data").exists()):
            return child
    return None


def _read_npblobs_from_stream(proc: subprocess.Popen) -> list[dict[str, np.ndarray]]:
    """Read length-prefixed NPZ blobs from the harvester's stdout pipe."""
    tiles = []
    buf = proc.stdout
    while True:
        header = buf.read(8)
        if not header or len(header) < 8:
            break
        magic = header[:4]
        if magic == ENDS_MAGIC:
            break
        if magic != NPZB_MAGIC:
            # Skip until we find a valid header
            continue
        length = struct.unpack("<I", header[4:8])[0]
        if length == 0 or length > 50_000_000:
            break
        blob = buf.read(length)
        if not blob or len(blob) < length:
            break
        try:
            data = dict(np.load(BytesIO(blob), allow_pickle=False))
            tiles.append(data)
        except Exception:
            continue
    return tiles


def _process_tile_data(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, bool]] | None:
    if "minimap_rgb_256" not in data or "height_257" not in data:
        return None

    tile_arrays: dict[str, np.ndarray] = {}
    has_signals: dict[str, bool] = {}

    for src_key, dst_key in OUTPUT_ARRAY_NAMES.items():
        if src_key in data:
            tile_arrays[dst_key] = _normalize_array(data[src_key], dst_key)
            has_signals[dst_key] = True
        else:
            shape = SHAPES[dst_key]
            dtype = DTYPES[dst_key]
            fill = FILL_VALUES[dst_key]
            tile_arrays[dst_key] = np.full(shape, fill, dtype=dtype)
            has_signals[dst_key] = False

    if "mcnr_normal_xyz" in data:
        nrm = data["mcnr_normal_xyz"].astype(np.float32)
        normal_mask = (np.abs(nrm).sum(axis=-1) > 1e-6)
        zero_mask = ~normal_mask
        nrm[zero_mask] = [0.0, 0.0, 1.0]
        norms = np.linalg.norm(nrm, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        nrm = nrm / norms
        tile_arrays["normal_xyz"] = nrm.astype(np.float32)
        has_signals["normal_xyz"] = True
    else:
        tile_arrays["normal_xyz"] = np.zeros((257, 257, 3), dtype=np.float32)
        normal_mask = np.zeros((257, 257), dtype=np.bool_)
        has_signals["normal_xyz"] = False

    tile_arrays["normal_mask"] = normal_mask.astype(np.bool_)
    has_signals["normal_mask"] = True

    return tile_arrays, has_signals


def _normalize_array(arr: np.ndarray, dst_key: str) -> np.ndarray:
    arr = arr.astype(DTYPES.get(dst_key, np.float32))
    if dst_key == "alpha_256":
        if arr.max() > 1.5:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
    elif dst_key == "liquid_mask":
        if arr.max() > 1.5:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
    elif dst_key in ("holes_16", "object_mask"):
        arr = arr.astype(np.bool_)
    return arr


def _default_maps_for_build(build: str) -> list[str]:
    if build.startswith("0_"):
        return ["Azeroth", "Kalimdor", "Kalidar"]
    elif build.startswith("3_"):
        return ["Azeroth", "Kalimdor", "Expansion01", "Northrend"]
    elif build.startswith("4_"):
        return ["Azeroth", "Kalimdor", "Expansion01", "Northrend", "development_nonweighted", "Deephome"]
    return ["Azeroth"]


def _write_index(rows: list[dict], output_path: Path) -> None:
    schema_fields = [
        pa.field("tile_id", pa.int64()),
        pa.field("build", pa.string()),
        pa.field("map", pa.string()),
        pa.field("tile_x", pa.int32()),
        pa.field("tile_y", pa.int32()),
        pa.field("height_mean", pa.float32()),
        pa.field("height_std", pa.float32()),
    ]
    bool_fields = [k for k in rows[0] if k.startswith("has_")] if rows else []
    for bf in bool_fields:
        schema_fields.append(pa.field(bf, pa.bool_()))

    schema = pa.schema(schema_fields)
    col_data = {k: [] for k in schema.names}
    for row in rows:
        for k in schema.names:
            col_data[k].append(row.get(k, False if k.startswith("has_") else 0))

    table = pa.table(col_data, schema=schema)
    pq.write_table(table, str(output_path / "index.parquet"))


def cmd_build(args: argparse.Namespace) -> None:
    builds = args.builds or [args.build]
    harvest_tool = _find_harvest_tool()
    print(f"Harvest tool: {harvest_tool}")

    maps_override = getattr(args, "maps", None)
    limit = args.limit

    for build in builds:
        client_root = _find_client_root(build)
        if client_root is None:
            print(f"SKIP build {build}: no client root found at {_CLIENT_ROOTS / build}")
            continue

        output_path = _DATASET_ROOT / f"{build}.zarr"
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True, exist_ok=True)

        build_version = build.replace("_", ".")
        map_names = maps_override or _default_maps_for_build(build)

        print(f"\n{'='*60}")
        print(f"Building V16 dataset for {build}")
        print(f"Client: {client_root}")
        print(f"Maps: {map_names}")
        print(f"Output: {output_path}")

        _build_zarr_streaming(
            harvest_tool=harvest_tool,
            client_root=client_root,
            build=build,
            build_version=build_version,
            map_names=map_names,
            output_path=output_path,
            limit=limit,
        )


def _build_zarr_streaming(
    harvest_tool: Path,
    client_root: Path,
    build: str,
    build_version: str,
    map_names: list[str],
    output_path: Path,
    limit: int | None,
) -> None:
    codec = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    store = zarr.storage.LocalStore(str(output_path), read_only=False)
    root = zarr.group(store=store)

    arrays: dict[str, zarr.Array] = {}
    index_rows: list[dict] = []
    valid = 0
    t0 = time.perf_counter()
    capacity = 50000

    for key in ALL_ARRAY_KEYS:
        shape = (capacity,) + SHAPES[key]
        chunks = CHUNK_SIZES.get(key, (64,) + SHAPES[key])
        arrays[key] = root.create_array(
            key, shape=shape, chunks=chunks, dtype=DTYPES[key],
            compressors=[codec], fill_value=FILL_VALUES.get(key, 0),
        )

    for map_name in map_names:
        print(f"\n  Streaming map: {map_name}")

        cmd = [
            str(harvest_tool), "harvest-stream",
            "--client-root", str(client_root),
            "--map", map_name,
        ]
        if build_version:
            cmd.extend(["--build", build_version])

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )

        tile_count = 0
        while True:
            header = proc.stdout.read(8)
            if not header or len(header) < 8:
                break

            magic = header[:4]
            if magic == ENDS_MAGIC:
                break
            if magic != NPZB_MAGIC:
                # Out of sync — drain and break
                proc.terminate()
                break

            length = struct.unpack("<I", header[4:8])[0]
            if length == 0 or length > 50_000_000:
                break

            blob = proc.stdout.read(length)
            if not blob or len(blob) < length:
                break

            try:
                data = dict(np.load(BytesIO(blob), allow_pickle=False))
            except Exception:
                continue

            result = _process_tile_data(data)
            if result is None:
                continue

            tile_arrays, has_signals = result

            h_mean = float(np.mean(tile_arrays["height_257"]))
            h_std = float(np.std(tile_arrays["height_257"])) + 1e-8

            # Parse tile coords from metadata or filename
            meta_raw = data.get("metadata.json")
            tx, ty = 0, 0
            if meta_raw is not None:
                try:
                    import json
                    meta = json.loads(meta_raw) if isinstance(meta_raw, str) else json.loads(meta_raw.tobytes().decode())
                    source = meta.get("source_adt_path", "")
                    # Parse "Azeroth_28_27.adt" → (28, 27)
                    parts = source.replace(".adt", "").rsplit("_", 2)
                    if len(parts) >= 2:
                        try:
                            ty = int(parts[-1])
                            tx = int(parts[-2])
                        except (ValueError, IndexError):
                            pass
                    actual_map = meta.get("map_name", map_name)
                except Exception:
                    actual_map = map_name
            else:
                actual_map = map_name

            row = {
                "tile_id": valid, "build": build, "map": actual_map,
                "tile_x": tx, "tile_y": ty,
                "height_mean": h_mean, "height_std": h_std,
            }
            for key, present in has_signals.items():
                row[f"has_{key}"] = present
            index_rows.append(row)

            # Grow arrays if needed
            if valid >= capacity - 1:
                capacity += 50000
                for key in ALL_ARRAY_KEYS:
                    arrays[key].resize((capacity,) + SHAPES[key])

            for key in ALL_ARRAY_KEYS:
                arrays[key][valid] = tile_arrays[key]

            valid += 1
            tile_count += 1
            if valid % 50 == 0:
                elapsed = time.perf_counter() - t0
                rate = valid / max(elapsed, 0.01)
                print(f"    [{valid} tiles] {rate:.1f} tiles/s, {elapsed:.0f}s")

            if limit is not None and valid >= limit:
                proc.terminate()
                break

        proc.wait()
        print(f"    Map {map_name}: {tile_count} tiles streamed")

        if limit is not None and valid >= limit:
            break

    # Trim arrays to actual size
    for key in ALL_ARRAY_KEYS:
        arrays[key].resize((valid,) + SHAPES[key])

    if index_rows:
        _write_index(index_rows, output_path)

    store.close()

    total_bytes = sum(f.stat().st_size for f in output_path.rglob("*") if f.is_file())
    liq_count = sum(1 for r in index_rows if r.get("has_liquid_mask", False))
    elapsed = time.perf_counter() - t0
    print(f"\nDone. {valid} tiles -> {output_path}")
    print(f"Size: {total_bytes / 1024 / 1024:.1f} MB, Liquid: {liq_count}/{valid}")
    print(f"Time: {elapsed:.0f}s ({valid / max(elapsed, 0.01):.1f} tiles/s)")


def cmd_stats(args: argparse.Namespace) -> None:
    builds = args.builds or [args.build]
    for build in builds:
        zarr_path = _DATASET_ROOT / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"SKIP {build}: no Zarr store at {zarr_path}")
            continue
        store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
        root = zarr.open_group(store=store, mode="r")
        n = root["height_257"].shape[0]
        print(f"\n{build}: {n} tiles")
        for k in sorted(root.array_keys()):
            a = root[k]
            print(f"  {k}: shape={a.shape} dtype={a.dtype}")
        store.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V16 consolidated Zarr dataset")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--build", type=str, help="Single build key (e.g. 3_3_5_12340)")
    common.add_argument("--builds", nargs="+", help="Multiple build keys")

    build_p = sub.add_parser("build", parents=[common])
    build_p.add_argument("--limit", type=int, default=None, help="Max tiles to extract")
    build_p.add_argument("--maps", nargs="+", default=None, help="Specific maps to extract")

    stats_p = sub.add_parser("stats", parents=[common])

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "stats":
        cmd_stats(args)


if __name__ == "__main__":
    main()