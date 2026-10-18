"""Headless Batch-Export of Clean Minimap / Elevation Pairs (Spec 274).

Scans across Zarr v3 map stores (e.g. Wrath 3.0.1.8303 and Retail 3.3.5 stores),
extracting synchronized pairs:
1. real_minimap_rgb: (256, 256, 3) uint8 (real authored minimap tile)
2. synthetic_hillshade_dxt1: (256, 256, 3) uint8 (object-free hillshade with DXT1 parity)
3. wdl_lattice: (33, 33) float32 (interlaced WDL macro height lattice)
4. height_257: (257, 257) float32 (clean ground elevation in yards)
5. index.parquet: coordinate lineage and split assignment
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from harvester.v60.dxt1_parity import dxt1_encode_decode_rgb

logger = logging.getLogger(__name__)

CLEAN_PAIR_SCHEMA = "v60-clean-minimap-elevation-pair-v1"


def combine_wdl_lattice(outer_17: np.ndarray, inner_16: np.ndarray) -> np.ndarray:
    """Combine 17x17 outer and 16x16 inner WDL height grids into a 33x33 lattice."""
    lattice = np.zeros((33, 33), dtype=np.float32)
    lattice[0::2, 0::2] = outer_17
    lattice[1::2, 1::2] = inner_16

    # Interpolate cross midpoints from outer neighbors
    lattice[0::2, 1::2] = 0.5 * (lattice[0::2, :-2:2] + lattice[0::2, 2::2])
    lattice[1::2, 0::2] = 0.5 * (lattice[:-2:2, 0::2] + lattice[2::2, 0::2])
    return lattice


def find_zarr_map_stores(root: Path | str) -> List[Path]:
    """Find all Zarr v3 / v2 map stores under a root directory."""
    root_path = Path(root)
    if not root_path.exists():
        return []

    stores = []
    # If the root itself is a store with index.parquet
    if (root_path / "index.parquet").exists():
        return [root_path]

    for candidate in sorted(root_path.glob("*.zarr")):
        if (candidate / "index.parquet").exists():
            stores.append(candidate)
    return stores


def scan_clean_pair_candidates(
    store_path: Path | str,
    target_maps: Optional[List[str]] = None,
    min_relief_delta: float = 0.0,
    max_count: Optional[int] = None,
) -> List[Dict[str, Any]]:
    """Scan store index and verify signal presence for clean pair export."""
    store = Path(store_path)
    index_file = store / "index.parquet"
    if not index_file.exists():
        return []

    tbl = pq.read_table(index_file)
    rows = tbl.to_pylist()

    g = zarr.open_group(str(store), mode="r")

    # Verify required arrays exist in the store
    if "minimap_rgb" not in g and "minimap_rgb_256" not in g:
        return []
    rgb_key = "minimap_rgb" if "minimap_rgb" in g else "minimap_rgb_256"

    # Need either terrain_shadow_256 or shadow_mask
    shadow_key = "terrain_shadow_256" if "terrain_shadow_256" in g else "shadow_mask" if "shadow_mask" in g else None
    if shadow_key is None:
        return []

    # Need height array
    height_key = "ground_intent_height_257" if "ground_intent_height_257" in g else "height_257" if "height_257" in g else None
    if height_key is None:
        return []

    has_wdl = "wdl_outer_17" in g and "wdl_inner_16" in g

    target_maps_set = {m.strip().lower() for m in target_maps} if target_maps else None

    candidates = []
    for idx, row in enumerate(rows):
        map_name = str(row.get("map", ""))
        if target_maps_set and map_name.strip().lower() not in target_maps_set:
            continue

        tile_x = int(row.get("tile_x", -1))
        tile_y = int(row.get("tile_y", -1))
        if tile_x < 0 or tile_y < 0:
            continue

        candidates.append(
            {
                "store_path": str(store.resolve()),
                "row_index": idx,
                "map": map_name,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "build_id": str(row.get("build_id", "unknown")),
                "rgb_key": rgb_key,
                "shadow_key": shadow_key,
                "height_key": height_key,
                "has_wdl": has_wdl,
            }
        )

        if max_count and len(candidates) >= max_count:
            break

    return candidates


def export_clean_pairs_dataset(
    source_stores: List[Path],
    output_path: Path | str,
    target_maps: Optional[List[str]] = None,
    validation_map: Optional[str] = None,
    val_ratio: float = 0.15,
    min_relief_delta: float = 1.0,
    limit: Optional[int] = None,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Execute batch export across Zarr map stores into a clean paired dataset."""
    all_candidates: List[Dict[str, Any]] = []
    for s in source_stores:
        cands = scan_clean_pair_candidates(s, target_maps=target_maps, max_count=limit)
        all_candidates.extend(cands)
        if limit and len(all_candidates) >= limit:
            all_candidates = all_candidates[:limit]
            break

    summary = {
        "schema": CLEAN_PAIR_SCHEMA,
        "source_stores_count": len(source_stores),
        "total_candidate_tiles": len(all_candidates),
        "dry_run": dry_run,
        "output_path": str(output_path),
    }

    if dry_run or len(all_candidates) == 0:
        return summary

    out_p = Path(output_path)
    out_p.mkdir(parents=True, exist_ok=True)

    # Initialize output Zarr group
    out_group = zarr.open_group(str(out_p), mode="w")
    count = len(all_candidates)

    out_group.attrs["schema"] = CLEAN_PAIR_SCHEMA
    out_group.attrs["tile_count"] = count

    # Pre-allocate arrays
    real_rgb_arr = out_group.create_array(
        "real_minimap_rgb",
        shape=(count, 256, 256, 3),
        chunks=(16, 256, 256, 3),
        dtype=np.uint8,
    )
    synth_dxt1_arr = out_group.create_array(
        "synthetic_hillshade_dxt1",
        shape=(count, 256, 256, 3),
        chunks=(16, 256, 256, 3),
        dtype=np.uint8,
    )
    wdl_lattice_arr = out_group.create_array(
        "wdl_lattice_33",
        shape=(count, 33, 33),
        chunks=(64, 33, 33),
        dtype=np.float32,
    )
    height_arr = out_group.create_array(
        "height_257",
        shape=(count, 257, 257),
        chunks=(16, 257, 257),
        dtype=np.float32,
    )

    # Cache open source stores to avoid re-opening
    open_groups: Dict[str, zarr.Group] = {}
    exported_rows: List[Dict[str, Any]] = []

    for out_idx, cand in enumerate(all_candidates):
        store_p = cand["store_path"]
        if store_p not in open_groups:
            open_groups[store_p] = zarr.open_group(store_p, mode="r")
        sg = open_groups[store_p]
        row_i = cand["row_index"]

        # 1. Real textured minimap
        rgb = np.asarray(sg[cand["rgb_key"]][row_i], dtype=np.uint8)
        if rgb.ndim == 2:
            rgb = np.stack([rgb, rgb, rgb], axis=-1)
        elif rgb.shape[-1] == 4:
            rgb = rgb[..., :3]

        # 2. Synthetic hillshade + DXT1 parity cycle
        shadow_raw = np.asarray(sg[cand["shadow_key"]][row_i], dtype=np.float32)
        if shadow_raw.ndim == 3 and shadow_raw.shape[-1] in (3, 4):
            shadow_raw = shadow_raw[..., 0]
        # Encode/decode through DXT1
        hillshade_dxt1 = dxt1_encode_decode_rgb(shadow_raw)

        # 3. WDL lattice
        if cand["has_wdl"]:
            outer = np.asarray(sg["wdl_outer_17"][row_i], dtype=np.float32)
            inner = np.asarray(sg["wdl_inner_16"][row_i], dtype=np.float32)
            wdl_lat = combine_wdl_lattice(outer, inner)
        else:
            wdl_lat = np.zeros((33, 33), dtype=np.float32)

        # 4. ADT elevation
        h257 = np.asarray(sg[cand["height_key"]][row_i], dtype=np.float32)
        min_z = float(np.min(h257))
        max_z = float(np.max(h257))
        delta_z = max_z - min_z

        # Assign split
        map_name = cand["map"]
        if validation_map:
            split = "val" if map_name.strip().lower() == validation_map.strip().lower() else "train"
        else:
            split = "val" if (out_idx % int(1.0 / val_ratio) == 0) else "train"

        # Write to Zarr
        real_rgb_arr[out_idx] = rgb
        synth_dxt1_arr[out_idx] = hillshade_dxt1
        wdl_lattice_arr[out_idx] = wdl_lat
        height_arr[out_idx] = h257

        exported_rows.append(
            {
                "index": out_idx,
                "map": map_name,
                "tile_x": cand["tile_x"],
                "tile_y": cand["tile_y"],
                "build_id": cand["build_id"],
                "min_z": min_z,
                "max_z": max_z,
                "relief_delta_z": delta_z,
                "split": split,
            }
        )

    # Write index.parquet
    tbl = pa.Table.from_pylist(exported_rows)
    pq.write_table(tbl, out_p / "index.parquet")

    summary["exported_count"] = count
    summary["train_count"] = sum(1 for r in exported_rows if r["split"] == "train")
    summary["val_count"] = sum(1 for r in exported_rows if r["split"] == "val")
    return summary
