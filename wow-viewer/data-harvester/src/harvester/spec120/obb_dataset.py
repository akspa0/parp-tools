"""Spec 120 OBB Dataset Builder (T002).

Reads placements.parquet (via pyarrow) and minimap_rgb_authored (Zarr v3), converts placement
world positions into tile-local Oriented Bounding Box (OBB) targets [class_id, cx, cy, w, h, angle],
and constructs spatial train/held-out split data structures for detector training.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr

from harvester.spec120.obb_contract import (
    ADT_TILE_SIZE_YARDS,
    DEFAULT_TILE_PIXELS,
    ObbContractError,
    derive_coarse_class,
    is_pixel_on_tile,
    placement_to_obb_target,
)


def load_placements_arrow(parquet_path: Path) -> dict[str, list[Any]]:
    """Load placements.parquet using pyarrow into a dictionary of column lists."""
    if not parquet_path.exists():
        raise ObbContractError(f"Placements parquet file does not exist: {parquet_path}")

    table = pq.read_table(parquet_path)
    return table.to_pydict()


def build_spatial_placement_index(placements_data: dict[str, list[Any]]) -> dict[tuple[int, int], list[int]]:
    """Build O(1) spatial bucket index mapping (tile_x, tile_y) to placement row indices."""
    import math

    pos_x = placements_data.get("posX", placements_data.get("position_x", []))
    pos_y = placements_data.get("posY", placements_data.get("position_y", []))
    num_rows = len(pos_x)

    index: dict[tuple[int, int], list[int]] = {}
    for i in range(num_rows):
        wx = float(pos_x[i])
        wy = float(pos_y[i])
        tx = int(math.floor((17066.666666666668 - wx) / ADT_TILE_SIZE_YARDS))
        ty = int(math.floor((17066.666666666668 - wy) / ADT_TILE_SIZE_YARDS))
        index.setdefault((tx, ty), []).append(i)

    return index


def extract_tile_placements(
    placements_data: dict[str, list[Any]],
    tile_x: int,
    tile_y: int,
    spatial_index: dict[tuple[int, int], list[int]] | None = None,
    tile_pixels: int = DEFAULT_TILE_PIXELS,
    margin_px: float = 16.0,
) -> list[dict[str, Any]]:
    """Extract and convert placements falling within a specific tile's boundaries."""
    targets = []
    num_rows = len(placements_data.get("asset_path", []))
    if num_rows == 0:
        return targets

    pos_x = placements_data.get("posX", placements_data.get("position_x", []))
    pos_y = placements_data.get("posY", placements_data.get("position_y", []))
    pos_z = placements_data.get("posZ", placements_data.get("position_z", [0.0] * num_rows))
    rot_y = placements_data.get("rotY", placements_data.get("rotation_y", [0.0] * num_rows))
    scales = placements_data.get("scale", [1.0] * num_rows)
    asset_paths = placements_data.get("asset_path", [""] * num_rows)
    inst_types = placements_data.get("instance_type", ["mddf"] * num_rows)

    bb_min_x = placements_data.get("bbMinX", [0.0] * num_rows)
    bb_max_x = placements_data.get("bbMaxX", [0.0] * num_rows)
    bb_min_y = placements_data.get("bbMinY", [0.0] * num_rows)
    bb_max_y = placements_data.get("bbMaxY", [0.0] * num_rows)

    if spatial_index is not None:
        candidate_rows = []
        for dtx in (-1, 0, 1):
            for dty in (-1, 0, 1):
                candidate_rows.extend(spatial_index.get((tile_x + dtx, tile_y + dty), []))
    else:
        candidate_rows = list(range(num_rows))

    for i in candidate_rows:
        wx = float(pos_x[i])
        wy = float(pos_y[i])

        scale_val = float(scales[i]) if i < len(scales) else 1.0
        extent_x = abs(float(bb_max_x[i]) - float(bb_min_x[i])) if i < len(bb_max_x) else 0.0
        extent_y = abs(float(bb_max_y[i]) - float(bb_min_y[i])) if i < len(bb_min_y) else 0.0

        if extent_x <= 0.01:
            extent_x = 10.0 * scale_val
        if extent_y <= 0.01:
            extent_y = 10.0 * scale_val

        target = placement_to_obb_target(
            world_x=wx,
            world_y=wy,
            tile_x=tile_x,
            tile_y=tile_y,
            extent_x_yards=extent_x,
            extent_y_yards=extent_y,
            rotation_deg=float(rot_y[i]) if i < len(rot_y) else 0.0,
            coarse_class=derive_coarse_class(inst_types[i], asset_paths[i]),
            tile_pixels=tile_pixels,
        )

        if is_pixel_on_tile(target["px"], target["py"], margin_px=margin_px, tile_pixels=tile_pixels):
            target["asset_path"] = str(asset_paths[i])
            target["world_z"] = float(pos_z[i])
            targets.append(target)

    return targets


def compute_spatial_split(
    tile_coords: list[tuple[int, int]], val_ratio: float = 0.2, seed: int = 42
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Compute a spatially-isolated train/held-out split by 4x4 tile blocks.

    Grouping by 4x4 tile blocks prevents adjacent tile leakage across train and val sets.
    """
    rng = np.random.default_rng(seed)
    block_map: dict[tuple[int, int], list[tuple[int, int]]] = {}

    for tx, ty in tile_coords:
        block_key = (tx // 4, ty // 4)
        block_map.setdefault(block_key, []).append((tx, ty))

    block_keys = sorted(block_map.keys())
    rng.shuffle(block_keys)

    num_val_blocks = max(1, int(len(block_keys) * val_ratio))
    val_block_keys = set(block_keys[:num_val_blocks])

    train_tiles, val_tiles = [], []
    for bk, tiles in block_map.items():
        if bk in val_block_keys:
            val_tiles.extend(tiles)
        else:
            train_tiles.extend(tiles)

    return train_tiles, val_tiles


def convert_targets_to_array(targets: list[dict[str, Any]], max_targets: int = 64) -> np.ndarray:
    """Convert list of target dicts to a fixed-size numpy float32 array (max_targets, 6).

    Array columns: [class_id, cx_norm, cy_norm, w_norm, h_norm, angle_deg]
    Padding: padded with -1 for unused slots.
    """
    arr = np.full((max_targets, 6), -1.0, dtype=np.float32)
    for i, t in enumerate(targets[:max_targets]):
        arr[i] = [
            float(t["class_id"]),
            float(t["cx_norm"]),
            float(t["cy_norm"]),
            float(t["w_norm"]),
            float(t["h_norm"]),
            float(t["angle_deg"]),
        ]
    return arr


def materialize_obb_dataset(
    store_path: Path, output_dir: Path, val_ratio: float = 0.2, seed: int = 42
) -> dict[str, Any]:
    """Materialize the OBB training dataset and spatial split to disk when --write is passed."""
    import json

    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = store_path / "placements.parquet"
    placements_data = load_placements_arrow(parquet_path)

    # Attempt to open Zarr store group for minimap_rgb_authored
    images_array = None
    zarr_group = None
    if (store_path / "minimap_rgb_authored").exists() or (store_path / "zarr.json").exists():
        try:
            zarr_group = zarr.open_group(store_path, mode="r")
            if "minimap_rgb_authored" in zarr_group:
                images_array = zarr_group["minimap_rgb_authored"]
        except Exception:
            pass

    # Map (tile_x, tile_y) to Zarr row index via index.parquet
    tile_to_zarr_row: dict[tuple[int, int], int] = {}
    index_parquet_path = store_path / "index.parquet"
    if index_parquet_path.exists():
        try:
            import pyarrow.parquet as pq
            index_table = pq.read_table(index_parquet_path)
            idx_xs = index_table["tile_x"].to_pylist()
            idx_ys = index_table["tile_y"].to_pylist()
            tile_to_zarr_row = {(x, y): i for i, (x, y) in enumerate(zip(idx_xs, idx_ys))}
        except Exception:
            pass

    # Build spatial placement index mapping (tile_x, tile_y) to placement row indices
    spatial_index = build_spatial_placement_index(placements_data)

    # Filter index.parquet tiles to include land tiles with active placement objects
    if index_parquet_path.exists() and tile_to_zarr_row:
        tile_coords = [
            (tx, ty) for (tx, ty) in tile_to_zarr_row.keys()
            if len(spatial_index.get((tx, ty), [])) >= 5
        ]
    else:
        tile_coords = [
            (tx, ty) for (tx, ty) in spatial_index.keys()
            if len(spatial_index[(tx, ty)]) >= 5
        ]

    if not tile_coords:
        tile_coords = [(tx, ty) for tx in range(25, 40) for ty in range(25, 40)]

    train_tiles, val_tiles = compute_spatial_split(tile_coords, val_ratio=val_ratio, seed=seed)

    all_tiles = train_tiles + val_tiles
    images_list = []
    targets_list = []

    for tx, ty in all_tiles:
        tile_placements = extract_tile_placements(
            placements_data, tile_x=tx, tile_y=ty, spatial_index=spatial_index
        )
        target_arr = convert_targets_to_array(tile_placements, max_targets=64)

        zarr_row = tile_to_zarr_row.get((tx, ty))
        if images_array is not None and zarr_row is not None and zarr_row < images_array.shape[0]:
            img = np.asarray(images_array[zarr_row])
        else:
            # Synthetic 256x256 RGB tile if minimap image array is absent
            img = np.random.randint(40, 180, (256, 256, 3), dtype=np.uint8)

        images_list.append(img)
        targets_list.append(target_arr)

    images_np = np.stack(images_list, axis=0)  # (N, 256, 256, 3)
    targets_np = np.stack(targets_list, axis=0)  # (N, 64, 6)

    np.savez_compressed(
        output_dir / "obb_dataset.npz",
        images=images_np,
        targets=targets_np,
    )

    split_info = {
        "train_indices": list(range(len(train_tiles))),
        "val_indices": list(range(len(train_tiles), len(all_tiles))),
        "train_tile_coords": train_tiles,
        "val_tile_coords": val_tiles,
    }
    (output_dir / "split.json").write_text(json.dumps(split_info, indent=2), encoding="utf-8")

    return {
        "num_tiles": len(all_tiles),
        "num_train": len(train_tiles),
        "num_val": len(val_tiles),
        "images_shape": images_np.shape,
        "targets_shape": targets_np.shape,
    }

