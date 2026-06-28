"""Build the spec 077 teacher deconstruction prior dataset (T017).

Reads a V18 Zarr store (``output/datasets/v18/<build>.zarr``) and writes a
sibling ``output/datasets/teacher_prior/<build>.zarr`` with the four
phase-1 channels described in ``teacher_prior.py``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
import zarr.codecs
import zarr.storage

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.teacher_prior import (  # noqa: E402
    PRIOR_CHANNELS,
    MaskSource,
    build_prior_tensor,
    make_tile_record,
)

DEFAULT_CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")


def _load_index_rows(v18_path: Path) -> list[dict]:
    index_path = v18_path / "index.parquet"
    if not index_path.exists():
        return []
    table = pq.read_table(str(index_path))
    return [
        {col: table.column(col)[idx].as_py() for col in table.column_names}
        for idx in range(table.num_rows)
    ]


def _read_array(v18_path: Path, key: str) -> np.ndarray | None:
    store = zarr.storage.LocalStore(str(v18_path), read_only=True)
    root = zarr.open_group(store, mode="r")
    if key not in root:
        return None
    return np.asarray(root[key][:])


def _write_zarr(
    output_path: Path,
    raw_minimap: np.ndarray,
    teacher_mask: np.ndarray,
    teacher_confidence: np.ndarray,
    processed_prior: np.ndarray,
    metadata: dict,
) -> Path:
    if output_path.exists():
        shutil.rmtree(output_path)
    store = zarr.storage.LocalStore(str(output_path), read_only=False)
    root = zarr.group(store=store)
    if raw_minimap.size:
        root.create_array(
            "raw_minimap_rgb_256",
            data=raw_minimap,
            chunks=(min(8, raw_minimap.shape[0]), 256, 256, 3),
            compressors=DEFAULT_CODEC,
        )
    if teacher_mask.size:
        root.create_array(
            "teacher_object_mask_256",
            data=teacher_mask,
            chunks=(min(8, teacher_mask.shape[0]), 256, 256),
            compressors=DEFAULT_CODEC,
        )
    if teacher_confidence.size:
        root.create_array(
            "teacher_object_confidence_256",
            data=teacher_confidence,
            chunks=(min(8, teacher_confidence.shape[0]), 256, 256),
            compressors=DEFAULT_CODEC,
        )
    if processed_prior.size:
        root.create_array(
            "processed_minimap_prior_256",
            data=processed_prior,
            chunks=(min(8, processed_prior.shape[0]), 256, 256, processed_prior.shape[3]),
            compressors=DEFAULT_CODEC,
        )
    root.attrs.update(dict(metadata.items()))
    return output_path


def _write_tiles_parquet(records: list, path: Path) -> None:
    if not records:
        path.write_text("", encoding="utf-8")
        return
    table = pa.table(
        {
            "build": [r.build for r in records],
            "map": [r.map_name for r in records],
            "tile_id": [r.tile_id for r in records],
            "tile_x": [r.tile_x for r in records],
            "tile_y": [r.tile_y for r in records],
            "raw_minimap_key": [r.raw_minimap_key for r in records],
            "teacher_object_mask_key": [r.teacher_object_mask_key for r in records],
            "teacher_object_confidence_key": [r.teacher_object_confidence_key for r in records],
            "processed_prior_key": [r.processed_prior_key for r in records],
            "has_teacher_objects": [r.has_teacher_objects for r in records],
            "teacher_object_cov": [r.teacher_object_cov for r in records],
            "filtered_mask_source": [r.filtered_mask_source for r in records],
        }
    )
    pq.write_table(table, str(path))


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build spec 077 teacher deconstruction prior dataset from a V18 Zarr store."
    )
    parser.add_argument("--v18-path", type=Path, required=True,
                        help="Path to <build>.zarr V18 store (input).")
    parser.add_argument("--output-root", type=Path, required=True,
                        help="Directory under which <build>.zarr teacher-prior store is written.")
    parser.add_argument("--max-tiles", type=int, default=None,
                        help="Optional cap on number of tiles processed.")
    parser.add_argument("--start-tile-id", type=int, default=0,
                        help="Skip tiles with tile_id < this value.")
    return parser.parse_args(argv)


def main_with_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    if not args.v18_path.exists():
        print(f"V18 store not found: {args.v18_path}", file=sys.stderr)
        return 2

    build = args.v18_path.stem.replace(".zarr", "")
    index_rows = _load_index_rows(args.v18_path)
    if not index_rows:
        print(f"No index.parquet under {args.v18_path}", file=sys.stderr)
        return 2

    raw_minimap = _read_array(args.v18_path, "minimap_rgb")
    if raw_minimap is None or raw_minimap.size == 0:
        print(f"No minimap_rgb array under {args.v18_path}", file=sys.stderr)
        return 2
    obj_filtered = _read_array(args.v18_path, "object_filtered_mask")
    obj_precise = _read_array(args.v18_path, "object_precise_mask")
    obj_mask = _read_array(args.v18_path, "object_mask")

    n_tiles = raw_minimap.shape[0]
    rows: list[dict] = []
    raw_list: list[np.ndarray] = []
    mask_list: list[np.ndarray] = []
    conf_list: list[np.ndarray] = []
    prior_list: list[np.ndarray] = []
    records: list = []
    for tile_id in range(n_tiles):
        if tile_id < args.start_tile_id:
            continue
        if args.max_tiles is not None and len(rows) >= args.max_tiles:
            break
        if tile_id >= len(index_rows):
            break
        index_row = index_rows[tile_id]
        tile_x = int(index_row.get("tile_x") or 0)
        tile_y = int(index_row.get("tile_y") or 0)
        map_name = str(index_row.get("map") or "")

        minimap = raw_minimap[tile_id]
        tensor, mask_uint8, confidence, source = build_prior_tensor(
            minimap,
            obj_filtered[tile_id] if obj_filtered is not None else None,
            obj_precise[tile_id] if obj_precise is not None else None,
            obj_mask[tile_id] if obj_mask is not None else None,
        )
        records.append(
            make_tile_record(
                build=build,
                map_name=map_name,
                tile_id=tile_id,
                tile_x=tile_x,
                tile_y=tile_y,
                mask_uint8=mask_uint8,
                source=source,
                index=len(rows),
            )
        )
        raw_list.append(minimap)
        mask_list.append(mask_uint8)
        conf_list.append(confidence)
        prior_list.append(tensor)
        rows.append(index_row)

    if not rows:
        print("No tiles processed", file=sys.stderr)
        return 2

    raw_arr = np.stack(raw_list, axis=0).astype(np.uint8, copy=False)
    mask_arr = np.stack(mask_list, axis=0).astype(np.uint8, copy=False)
    conf_arr = np.stack(conf_list, axis=0).astype(np.uint8, copy=False)
    prior_arr = np.stack(prior_list, axis=0).astype(np.uint8, copy=False)

    args.output_root.mkdir(parents=True, exist_ok=True)
    output_path = args.output_root / f"{build}.zarr"
    metadata = {
        "schema": "spec-077-teacher-prior",
        "schema_version": "1",
        "build": build,
        "source_v18_path": str(args.v18_path),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tile_count": len(rows),
        "phase1_prior_channels": list(PRIOR_CHANNELS),
        "mask_preference_chain": [
            "object_filtered_mask",
            "object_precise_mask",
            "object_mask",
        ],
        "fill_strategy": "per_tile_median_of_non_object_pixels",
    }
    _write_zarr(output_path, raw_arr, mask_arr, conf_arr, prior_arr, metadata)
    _write_tiles_parquet(records, output_path / "tiles.parquet")
    print(f"Wrote {len(rows)} teacher-prior tiles to {output_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main_with_args())
