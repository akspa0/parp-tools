"""Patch renderer-truth object masks into existing V16 Zarr stores.

This script reads MdxViewer validation capture artifacts (PNG images) and
writes them as arrays into the existing V16 Zarr stores. The new arrays are:

  - object_visibility_mask: (N, 256, 256) float32 — renderer-truth object mask
  - no_object_minimap: (N, 256, 256, 3) uint8 — terrain-only rendered minimap

These arrays enable V16.2 training to use renderer-truth object guidance
instead of only the harvester-side approximate masks.

Usage:
    uv run python scripts/patch_v16_renderer_truth.py \
        --build 3_3_5_12340 \
        --capture-dir output/tmp/mdxviewer_validation_smoke/3_3_5_12340_Azeroth_30_48 \
        --dataset-root ../output/datasets/v16 \
        --allow-zarr-write

    uv run python scripts/patch_v16_renderer_truth.py \
        --build 0_5_3_3368 \
        --capture-dir output/tmp/mdxviewer_validation_smoke/0_5_3_3368_Azeroth_30_48 \
        --dataset-root ../output/datasets/v16 \
        --allow-zarr-write
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import PIL.Image
import zarr
import zarr.storage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v16"


def _require_explicit_zarr_write(args: argparse.Namespace) -> None:
    if not args.allow_zarr_write:
        raise RuntimeError(
            "Refusing to mutate Zarr store without --allow-zarr-write. "
            "This flag confirms you understand the store will be modified in-place."
        )


def _load_png_as_grayscale(path: Path, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Load a PNG as a single-channel float32 array, resized to target_size."""
    img = PIL.Image.open(str(path)).convert("L")
    img = img.resize(target_size, PIL.Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0
    return arr


def _load_png_as_rgb(path: Path, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    """Load a PNG as an RGB uint8 array, resized to target_size."""
    img = PIL.Image.open(str(path)).convert("RGB")
    img = img.resize(target_size, PIL.Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _discover_capture_tiles(capture_dir: Path) -> dict[str, dict[str, Path]]:
    """Discover MdxViewer capture artifacts in the capture directory.

    Returns a dict mapping tile_name -> {
        "visibility_mask": Path (object_visibility_mask.png),
        "no_objects": Path (no_objects.png),
    }
    """
    images_dir = capture_dir / "images"
    if not images_dir.exists():
        raise RuntimeError(f"No images directory found at {images_dir}")

    tiles: dict[str, dict[str, Path]] = {}
    for png in sorted(images_dir.glob("*.png")):
        name = png.stem
        if name.endswith("_object_visibility_mask"):
            tile_name = name[: -len("_object_visibility_mask")]
            tiles.setdefault(tile_name, {})["visibility_mask"] = png
        elif name.endswith("_no_objects"):
            tile_name = name[: -len("_no_objects")]
            tiles.setdefault(tile_name, {})["no_objects"] = png

    return tiles


def _read_index_rows(store_path: Path) -> list[dict]:
    """Read index.parquet from the Zarr store directory."""
    import pyarrow.parquet as pq

    idx_path = store_path / "index.parquet"
    if not idx_path.exists():
        raise RuntimeError(f"No index.parquet at {idx_path}")
    table = pq.read_table(str(idx_path))
    return [{col: table.column(col)[i].as_py() for col in table.column_names} for i in range(table.num_rows)]


def _tile_name_from_entry(entry: dict) -> str:
    """Build a tile name like 'Azeroth_30_48' from an index entry."""
    map_name = str(entry.get("map", "")).strip()
    tile_x = entry.get("tile_x")
    tile_y = entry.get("tile_y")
    if tile_x is not None and tile_y is not None and map_name:
        return f"{map_name}_{int(tile_x)}_{int(tile_y)}"
    return ""


def cmd_patch_renderer_truth(args: argparse.Namespace) -> None:
    _require_explicit_zarr_write(args)

    build = args.build
    dataset_root = Path(args.dataset_root)
    store_path = dataset_root / f"{build}.zarr"

    if not store_path.exists():
        raise RuntimeError(f"No store at {store_path}")

    capture_dir = Path(args.capture_dir)
    if not capture_dir.exists():
        raise RuntimeError(f"No capture directory at {capture_dir}")

    print(f"Patching renderer-truth signals for {build}")
    print(f"Store: {store_path}")
    print(f"Captures: {capture_dir}")

    # Discover capture tiles
    tiles = _discover_capture_tiles(capture_dir)
    print(f"Found {len(tiles)} capture tiles: {sorted(tiles.keys())[:10]}...")

    if not tiles:
        raise RuntimeError("No capture tiles found. Check capture directory structure.")

    # Read index
    index_rows = _read_index_rows(store_path)
    print(f"Store has {len(index_rows)} tiles in index")

    # Build capture -> tile_id mapping
    capture_to_tile_id: dict[str, int] = {}
    for i, entry in enumerate(index_rows):
        tile_name = _tile_name_from_entry(entry)
        if tile_name in tiles:
            capture_to_tile_id[tile_name] = i

    print(f"Matched {len(capture_to_tile_id)} capture tiles to store index")

    if not capture_to_tile_id:
        raise RuntimeError(
            "No capture tiles matched store index. "
            "Check that capture tile names match index map/tile_x/tile_y."
        )

    # Prepare arrays
    n_tiles = len(index_rows)
    visibility_masks = np.zeros((n_tiles, 256, 256), dtype=np.float32)
    no_object_minimaps = np.zeros((n_tiles, 256, 256, 3), dtype=np.uint8)
    has_visibility = np.zeros(n_tiles, dtype=bool)
    has_no_object = np.zeros(n_tiles, dtype=bool)

    # Load capture data
    matched_count = 0
    for tile_name, tile_id in capture_to_tile_id.items():
        caps = tiles[tile_name]
        if "visibility_mask" in caps:
            visibility_masks[tile_id] = _load_png_as_grayscale(caps["visibility_mask"])
            has_visibility[tile_id] = True
        if "no_objects" in caps:
            no_object_minimaps[tile_id] = _load_png_as_rgb(caps["no_objects"])
            has_no_object[tile_id] = True
        matched_count += 1

    print(f"Loaded renderer-truth data for {matched_count} tiles")
    print(f"  visibility_mask coverage: {has_visibility.sum()}/{n_tiles}")
    print(f"  no_object_minimap coverage: {has_no_object.sum()}/{n_tiles}")

    # Backup index
    idx_path = store_path / "index.parquet"
    if not args.no_backup:
        backup_path = store_path / "index.parquet.bak.renderer_truth"
        if not backup_path.exists():
            shutil.copy2(idx_path, backup_path)
            print(f"Backed up {idx_path} -> {backup_path}")

    # Write arrays to store
    codec = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    store = zarr.storage.LocalStore(str(store_path), read_only=False)
    root = zarr.open_group(store=store, mode="a")
    try:
        root.create_array(
            "object_visibility_mask",
            data=visibility_masks,
            chunks=(1, 256, 256),
            compressors=codec,
            overwrite=True,
        )
        root.create_array(
            "no_object_minimap",
            data=no_object_minimaps,
            chunks=(1, 256, 256, 3),
            compressors=codec,
            overwrite=True,
        )
    finally:
        store.close()

    print(f"Wrote object_visibility_mask: {visibility_masks.shape} {visibility_masks.dtype}")
    print(f"Wrote no_object_minimap: {no_object_minimaps.shape} {no_object_minimaps.dtype}")

    # Update index
    for i, row in enumerate(index_rows):
        row["has_object_visibility_mask"] = bool(has_visibility[i])
        row["has_no_object_minimap"] = bool(has_no_object[i])

    _write_index(index_rows, store_path)

    # Write patch report
    report = {
        "build": build,
        "patched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tiles": n_tiles,
        "matched_capture_tiles": matched_count,
        "visibility_mask_coverage": int(has_visibility.sum()),
        "no_object_minimap_coverage": int(has_no_object.sum()),
        "capture_dir": str(capture_dir),
        "matched_tiles": sorted(capture_to_tile_id.keys()),
    }
    report_path = store_path / "renderer_truth_patch_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Patch report: {report_path}")


def _write_index(rows: list[dict], store_path: Path) -> None:
    """Write index.parquet from a list of row dicts."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if not rows:
        return

    # Collect all keys
    all_keys: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                all_keys.append(key)
                seen.add(key)

    columns = {}
    for key in all_keys:
        values = [row.get(key) for row in rows]
        # Try to infer type
        if all(v is None or isinstance(v, bool) for v in values):
            columns[key] = pa.array(values, type=pa.bool_())
        elif all(v is None or isinstance(v, int) for v in values):
            columns[key] = pa.array(values, type=pa.int64())
        elif all(v is None or isinstance(v, float) for v in values):
            columns[key] = pa.array(values, type=pa.float64())
        elif all(v is None or isinstance(v, str) for v in values):
            columns[key] = pa.array(values, type=pa.string())
        else:
            columns[key] = pa.array([str(v) for v in values], type=pa.string())

    table = pa.table(columns)
    idx_path = store_path / "index.parquet"
    pq.write_table(table, str(idx_path))


def main() -> None:
    p = argparse.ArgumentParser(description="Patch renderer-truth object masks into V16 Zarr stores")
    p.add_argument("--build", required=True, help="Build identifier (e.g. 3_3_5_12340)")
    p.add_argument("--capture-dir", required=True, help="MdxViewer validation capture directory")
    p.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    p.add_argument("--allow-zarr-write", action="store_true", help="Required confirmation flag")
    p.add_argument("--no-backup", action="store_true", help="Skip backing up index.parquet")
    args = p.parse_args()
    cmd_patch_renderer_truth(args)


if __name__ == "__main__":
    main()
