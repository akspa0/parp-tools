"""Build the V22 Zarr dataset from a stream of decoded C# V22 tile records.

Usage::

    uv run python scripts/build_v22_dataset.py build \\
        --stream /path/to/v22_stream.bin \\
        --output output/datasets/v22/3_3_5_12340.zarr

The C# harvester writes the binary V22 stream (``RawArraySerializer.StreamProfile.V22``).
This script parses the stream and writes the canonical Zarr dataset. No game
client reparse, no Python-side patch derivation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "data-harvester"))

from harvester.v22_zarr_io import (  # noqa: E402  (sys.path inserted above)
    V22ZarrWriter,
    V22TileRecord,
)


def _parse_stream(stream_path: Path) -> list[V22TileRecord]:
    """Parse a binary V22 stream into tile records.

    The stream has outer ARRY+length wrappers from ``WriteHarvestStreamBlob``
    and inner blobs from ``RawArraySerializer.StreamProfile.V22``::

        ARRY <blob_len> [ARRY <meta_len> <meta_json> <arrays> ENDS]  # tile 0
        ARRY <blob_len> [ARRY <meta_len> <meta_json> <arrays> ENDS]  # tile 1
        ENDS <4 zero bytes>                                          # terminator
    """
    records: list[V22TileRecord] = []
    data = stream_path.read_bytes()
    offset = 0
    tile_id = 0
    while offset < len(data):
        if data[offset:offset + 4] == b"ENDS":
            break
        if data[offset:offset + 4] != b"ARRY":
            break
        offset += 4
        blob_len = int.from_bytes(data[offset:offset + 4], "little")
        offset += 4
        blob = data[offset:offset + blob_len]
        offset += blob_len
        _parse_tile_blob(blob, records, tile_id)
        tile_id += 1
    return records


def _ensure_str_list(val: Any) -> tuple[str, ...]:
    if val is None:
        return ()
    if isinstance(val, list):
        return tuple(str(v) for v in val)
    return tuple(str(s).strip('"') for s in str(val).strip("[]").split(",") if s.strip())


def _parse_tile_blob(blob: bytes, records: list[V22TileRecord], tile_id: int) -> None:
    pos = 0
    if blob[pos:pos + 4] != b"ARRY":
        return
    pos += 4
    meta_len = int.from_bytes(blob[pos:pos + 4], "little")
    pos += 4
    meta = json.loads(blob[pos:pos + meta_len].decode("utf-8"))
    pos += meta_len

    per_tile: dict[str, np.ndarray] = {}
    while pos + 4 <= len(blob):
        magic = blob[pos:pos + 4]
        if magic == b"ENDS":
            pos += 4
            break
        name_len = int.from_bytes(blob[pos:pos + 4], "little")
        pos += 4
        name = blob[pos:pos + name_len].decode("utf-8")
        pos += name_len
        rank = int.from_bytes(blob[pos:pos + 4], "little")
        pos += 4
        shape = tuple(int.from_bytes(blob[pos + 4 * i:pos + 4 * (i + 1)], "little") for i in range(rank))
        pos += 4 * rank
        dtype = blob[pos:pos + 8].rstrip(b"\x00").decode("ascii")
        pos += 8
        data_len = int.from_bytes(blob[pos:pos + 8], "little")
        pos += 8
        payload = blob[pos:pos + data_len]
        pos += data_len
        if dtype in {"<f4", "<f8"}:
            arr = np.frombuffer(payload, dtype=dtype).reshape(shape) if shape else np.asarray(0, dtype=dtype)
        elif dtype in {"<i4", "<u4"}:
            arr = np.frombuffer(payload, dtype=dtype).reshape(shape) if shape else np.asarray(0, dtype=dtype)
        elif dtype == "|u1":
            arr = np.frombuffer(payload, dtype=np.uint8).reshape(shape) if shape else np.asarray(0, dtype=np.uint8)
        elif dtype == "|b1":
            arr = np.frombuffer(payload, dtype=bool).reshape(shape) if shape else np.asarray(False, dtype=bool)
        else:
            arr = np.frombuffer(payload, dtype=dtype).reshape(shape) if shape else np.asarray(0, dtype=dtype)
        per_tile[name] = arr

        record = V22TileRecord(
            tile_id=tile_id,
            build=meta.get("build_key", ""),
            map=meta.get("map_name", ""),
            tile_x=int(meta.get("tile_x", 0)),
            tile_y=int(meta.get("tile_y", 0)),
            per_tile=per_tile,
            placement_mddf=per_tile.get("mddf_placement_data"),
            placement_modf=per_tile.get("modf_placement_data"),
            mddf_asset_paths=_ensure_str_list(meta.get("placement_mddf_asset_paths")),
            modf_asset_paths=_ensure_str_list(meta.get("placement_modf_asset_paths")),
            mtex_texture_paths=_ensure_str_list(meta.get("mtex_texture_paths")),
        )
        records.append(record)
        tile_id += 1
    return records


def _build(store_path: Path, stream_path: Path) -> Path:
    records = _parse_stream(stream_path)
    writer = V22ZarrWriter(store_path)
    for r in records:
        writer.add_tile(r)
    return writer.finalize()


def _stats(store_path: Path) -> dict[str, object]:
    if not store_path.exists():
        return {"exists": False, "path": str(store_path)}
    import zarr
    import zarr.storage
    grp = zarr.open_group(zarr.storage.LocalStore(str(store_path), read_only=True), mode="r")
    return {
        "exists": True,
        "path": str(store_path),
        "tile_count": int(grp.attrs.get("tile_count", 0)),
        "builds": list(grp.attrs.get("builds", [])),
        "root_arrays": sorted(grp.array_keys()),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="V22 Zarr dataset builder")
    sub = parser.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="Parse a V22 stream and write a Zarr store")
    build.add_argument("--stream", required=True, type=Path)
    build.add_argument("--output", required=True, type=Path)

    stats = sub.add_parser("stats", help="Print a summary of an existing V22 Zarr store")
    stats.add_argument("--store", required=True, type=Path)

    args = parser.parse_args()
    if args.command == "build":
        out = _build(args.output, args.stream)
        print(f"wrote {out}")
        return 0
    if args.command == "stats":
        print(json.dumps(_stats(args.store), indent=2, default=str))
        return 0
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
