#!/usr/bin/env python3
"""Run archaeology (tile inventory + weak signal analysis) directly from NPZ shards.

Skips the V50 Zarr store format. Reads NPZ shards from the harvest tool output
and produces the tile inventory CSV + synthesis sheets directly.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/v50_archaeology_from_npz.py \\
        --npz-dir ../output/archaeology/2_0_0_5610/npz/Expansion01 \\
        --output ../output/archaeology/2_0_0_5610/archaeo
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import math
from pathlib import Path

import numpy as np

# Constants matching v50_tile_inventory.py
WEAK_MIN_RANGE = 5.0
WEAK_NEAR_ZERO_BAND = 50.0
FLAT_NORMAL_Z = 0.9999


def classify_information(levels: int) -> str:
    if levels <= 1:
        return "bit_exact_flat"
    if levels <= 8:
        return "trace"
    if levels <= 64:
        return "coarse_terrain"
    return "rich_terrain"


def mcnr_tilted_fraction(normal_xyz: np.ndarray | None) -> float:
    if normal_xyz is None or normal_xyz.size == 0:
        return 0.0
    mask = ~np.all(np.abs(normal_xyz) < 1e-6, axis=-1)
    if not mask.any():
        return 0.0
    return float(np.mean(np.abs(normal_xyz[mask][:, 2]) < FLAT_NORMAL_Z))


def load_npz(path: Path) -> dict:
    """Load NPZ and return a dict with height, minimap, normals, etc."""
    data = np.load(path)
    result = {}
    for key in data.files:
        result[key] = data[key]
    return result


def inventory_npz(npz_dir: Path, near_zero_band: float) -> list[dict]:
    """Build tile inventory from NPZ shards."""
    rows = []
    npz_files = sorted(npz_dir.glob("*.npz"))
    
    for npz_path in npz_files:
        try:
            data = load_npz(npz_path)
        except Exception as e:
            print(f"  WARNING: failed to load {npz_path.name}: {e}", flush=True)
            continue
        
        # Parse tile coordinates from filename (e.g. Azeroth_26_34_harvest.npz)
        stem = npz_path.stem
        parts = stem.replace("_harvest", "").split("_")
        if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
            tile_x = int(parts[-2])
            tile_y = int(parts[-1])
            map_name = "_".join(parts[:-2])
        else:
            map_name = "unknown"
            tile_x = 0
            tile_y = 0
        
        has_height = "height_257" in data
        has_minimap = "minimap_rgb" in data
        has_normals = "normal_xyz" in data
        
        height = data.get("height_257")
        minimap = data.get("minimap_rgb")
        normals = data.get("normal_xyz")
        
        height_min = float(np.min(height)) if height is not None else 0.0
        height_max = float(np.max(height)) if height is not None else 0.0
        height_range = height_max - height_min
        
        # Surviving height levels
        if height is not None:
            uniq = np.unique(height)
            surviving_levels = len(uniq)
        else:
            surviving_levels = 0
        
        # Weak signal detection
        is_weak = height_range < WEAK_MIN_RANGE if has_height else False
        is_compressed = abs(np.mean(height)) < near_zero_band if (has_height and near_zero_band != float("inf")) else is_weak
        
        # MCNR tilt
        tilted = mcnr_tilted_fraction(normals)
        
        row = {
            "tile_key": f"{map_name}_{tile_x}_{tile_y}",
            "map": map_name,
            "tile_x": tile_x,
            "tile_y": tile_y,
            "has_height_257": has_height,
            "has_minimap_rgb": has_minimap,
            "has_normal_xyz": has_normals,
            "height_min": height_min,
            "height_max": height_max,
            "height_range": height_range,
            "is_weak_signal": is_weak,
            "is_compressed_range": is_compressed,
            "surviving_height_levels": surviving_levels,
            "information_class": classify_information(surviving_levels),
            "mcnr_tilted_fraction": tilted,
        }
        rows.append(row)
    
    return rows


def write_inventory(rows: list[dict], output: Path) -> None:
    """Write tile inventory CSV and JSON."""
    output.mkdir(parents=True, exist_ok=True)
    
    csv_path = output / "tiles.csv"
    if rows:
        fieldnames = list(rows[0].keys())
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
    
    json_path = output / "tiles.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"tiles": rows}, f, indent=2)
    
    # Summary
    weak = [r for r in rows if r["is_weak_signal"]]
    compressed = [r for r in rows if r["is_compressed_range"]]
    flat = [r for r in rows if r["information_class"] == "bit_exact_flat"]
    rich = [r for r in rows if r["information_class"] == "rich_terrain"]
    
    summary = {
        "total_tiles": len(rows),
        "weak_signal": len(weak),
        "compressed_range": len(compressed),
        "bit_exact_flat": len(flat),
        "rich_terrain": len(rich),
        "weak_tiles": [r["tile_key"] for r in weak],
        "flat_tiles": [r["tile_key"] for r in flat],
    }
    with open(output / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Wrote {len(rows)} tile records to {csv_path}")
    print(f"  Summary: {len(rows)} tiles, {len(weak)} weak, {len(flat)} flat, {len(rich)} rich")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Archaeology inventory from NPZ shards (no V50 Zarr store needed)"
    )
    parser.add_argument("--npz-dir", required=True, type=Path, help="Directory of NPZ shards")
    parser.add_argument("--output", required=True, type=Path, help="Output directory")
    parser.add_argument("--near-zero-band", type=float, default=WEAK_NEAR_ZERO_BAND,
                        help="Weak-signal |Z| band. Pass 'inf' for non-alpha clients.")
    args = parser.parse_args()
    
    if not args.npz_dir.exists():
        print(f"ERROR: NPZ directory not found: {args.npz_dir}")
        return 1
    
    print(f"Scanning {args.npz_dir} for NPZ shards...", flush=True)
    rows = inventory_npz(args.npz_dir, args.near_zero_band)
    
    if not rows:
        print("No NPZ shards found or none could be loaded.")
        return 1
    
    write_inventory(rows, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())