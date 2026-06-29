"""Audit whether teacher object masks are visibly supported by minimap pixels.

This is a Spec 077 diagnostic/curation helper. It does not create object
masks. It checks an already-built teacher-prior store and buckets tiles where
ADT-derived object masks appear mismatched with the baked minimap.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
import zarr.storage


def _open_zarr_array(store_path: Path, key: str) -> np.ndarray | None:
    store = zarr.storage.LocalStore(str(store_path), read_only=True)
    root = zarr.open_group(store, mode="r")
    if key not in root:
        return None
    return np.asarray(root[key][:])


def _read_tiles_parquet(path: Path) -> list[dict]:
    if not path.exists():
        return []
    table = pq.read_table(str(path))
    return [
        {col: table.column(col)[idx].as_py() for col in table.column_names}
        for idx in range(table.num_rows)
    ]


def _score_tile(raw_rgb: np.ndarray, mask: np.ndarray, *, min_mask_coverage: float, visible_delta: float) -> dict[str, float | str | bool]:
    mask_bool = mask.astype(np.float32) > 0.5
    coverage = float(mask_bool.mean())
    if coverage <= 0.0:
        return {
            "bucket": "empty",
            "keep": True,
            "mask_coverage": 0.0,
            "visible_delta": 0.0,
            "masked_rgb_std": 0.0,
        }
    raw = raw_rgb.astype(np.float32)
    masked = raw[mask_bool]
    unmasked = raw[~mask_bool]
    if unmasked.size == 0:
        baseline = np.array([128.0, 128.0, 128.0], dtype=np.float32)
    else:
        baseline = np.median(unmasked.reshape(-1, 3), axis=0).astype(np.float32)
    delta = float(np.mean(np.abs(masked.reshape(-1, 3) - baseline[None, :])))
    masked_std = float(np.mean(np.std(masked.reshape(-1, 3), axis=0))) if masked.size else 0.0

    if coverage < min_mask_coverage:
        bucket = "tiny"
        keep = False
    elif delta < visible_delta:
        bucket = "weak"
        keep = False
    else:
        bucket = "visible"
        keep = True
    return {
        "bucket": bucket,
        "keep": keep,
        "mask_coverage": coverage,
        "visible_delta": delta,
        "masked_rgb_std": masked_std,
    }


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit teacher-prior masks against raw minimap visibility.")
    parser.add_argument("--library", type=Path, nargs="+", required=True, help="One or more <build>.zarr teacher-prior stores.")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for visibility_audit.parquet and kept_tiles.parquet.")
    parser.add_argument("--min-mask-coverage", type=float, default=0.001,
                        help="Masks below this tile coverage are bucketed as tiny and rejected.")
    parser.add_argument("--visible-delta", type=float, default=18.0,
                        help="Mean RGB delta from non-mask median required to call a mask visibly supported.")
    return parser.parse_args(argv)


def main_with_args(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rows: list[dict] = []
    library_paths = list(args.library)
    for library in library_paths:
        if not library.exists():
            print(f"Teacher-prior store not found: {library}", file=sys.stderr)
            return 2
        raw = _open_zarr_array(library, "raw_minimap_rgb_256")
        mask = _open_zarr_array(library, "teacher_object_mask_256")
        if raw is None or mask is None:
            print(f"Teacher-prior store is missing raw_minimap_rgb_256 or teacher_object_mask_256: {library}", file=sys.stderr)
            return 2
        tiles = _read_tiles_parquet(library / "tiles.parquet")
        if not tiles:
            print(f"No tiles.parquet under {library}", file=sys.stderr)
            return 2

        build = library.stem.replace(".zarr", "")
        for row_index, tile in enumerate(tiles):
            if row_index >= raw.shape[0] or row_index >= mask.shape[0]:
                break
            score = _score_tile(
                raw[row_index],
                mask[row_index],
                min_mask_coverage=float(args.min_mask_coverage),
                visible_delta=float(args.visible_delta),
            )
            rows.append({
                "build": str(tile.get("build", build)),
                "map": str(tile.get("map", tile.get("map_name", ""))),
                "tile_id": int(tile.get("tile_id", row_index)),
                "row_index": int(row_index),
                "mask_source": str(tile.get("filtered_mask_source", "")),
                "bucket": str(score["bucket"]),
                "keep": bool(score["keep"]),
                "mask_coverage": float(score["mask_coverage"]),
                "visible_delta": float(score["visible_delta"]),
                "masked_rgb_std": float(score["masked_rgb_std"]),
            })

    args.output_dir.mkdir(parents=True, exist_ok=True)
    audit_table = pa.Table.from_pylist(rows)
    pq.write_table(audit_table, str(args.output_dir / "visibility_audit.parquet"))
    kept_table = pa.table({
        "build": [row["build"] for row in rows],
        "tile_id": [row["tile_id"] for row in rows],
        "keep": [row["keep"] for row in rows],
        "bucket": [row["bucket"] for row in rows],
    })
    pq.write_table(kept_table, str(args.output_dir / "kept_tiles.parquet"))
    counts = Counter(row["bucket"] for row in rows)
    summary = {
        "schema": "spec-077-teacher-prior-visibility-audit",
        "libraries": [str(path) for path in library_paths],
        "tile_count": len(rows),
        "bucket_counts": dict(counts),
        "kept_count": sum(1 for row in rows if row["keep"]),
        "min_mask_coverage": float(args.min_mask_coverage),
        "visible_delta": float(args.visible_delta),
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Wrote visibility audit for {len(rows)} tiles to {args.output_dir}")
    return 0


def main() -> None:
    raise SystemExit(main_with_args())


if __name__ == "__main__":
    main()
