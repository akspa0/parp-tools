#!/usr/bin/env python3
"""
Regenerate heightmaps with GLOBAL normalization across the entire map.

This fixes the per-tile normalization issue where adjacent tiles have different
brightness scales, causing visible seams when stitched.

Usage:
    python regenerate_heightmaps_global.py <dataset_dir> [--stitch] [--max-size 16384]
"""

import argparse
import json
import re
from pathlib import Path
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
import threading

def load_tile_heights(json_path: Path) -> tuple[str, dict, float, float] | None:
    """Load height data from a tile JSON file."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        # Handle nested structure: terrain_data.heights[] or direct chunks[]
        terrain_data = data.get('terrain_data', data)
        tile_name = terrain_data.get('adt_tile', terrain_data.get('tileName', json_path.stem))
        
        # Try different possible locations for heights
        heights_list = terrain_data.get('heights', data.get('chunks', []))
        height_min = terrain_data.get('heightMin', data.get('heightMin', 0))
        height_max = terrain_data.get('heightMax', data.get('heightMax', 0))
        
        # Extract chunk heights
        chunk_heights = {}
        for chunk in heights_list:
            # idx or chunkIndex
            idx = chunk.get('idx', chunk.get('chunkIndex', 0))
            heights = chunk.get('h', chunk.get('heights', []))
            if len(heights) == 145:
                chunk_heights[idx] = heights
        
        return tile_name, chunk_heights, height_min, height_max
    except Exception as e:
        print(f"  Warning: Failed to load {json_path.name}: {e}")
        return None

def generate_heightmap(chunk_heights: dict, global_min: float, global_max: float) -> np.ndarray:
    """Generate a 256x256 heightmap using global normalization."""
    SIZE = 256
    heightmap = np.zeros((SIZE, SIZE), dtype=np.float32)
    
    height_range = global_max - global_min
    if height_range <= 0:
        height_range = 1.0
    
    # Process each chunk using Alpha MCVT format: 81 outer then 64 inner
    for chunk_idx, h_data in chunk_heights.items():
        if len(h_data) < 145:
            continue
            
        chunk_y = chunk_idx // 16
        chunk_x = chunk_idx % 16
        base_x = chunk_x * 16
        base_y = chunk_y * 16
        
        # Place 9x9 outer vertices at even positions
        for oy in range(9):
            for ox in range(9):
                px = base_x + ox * 2
                py = base_y + oy * 2
                if px < SIZE and py < SIZE:
                    z = h_data[oy * 9 + ox]
                    heightmap[py, px] = (z - global_min) / height_range
        
        # Place 8x8 inner vertices at odd positions
        for iy in range(8):
            for ix in range(8):
                px = base_x + ix * 2 + 1
                py = base_y + iy * 2 + 1
                if px < SIZE and py < SIZE:
                    z = h_data[81 + iy * 8 + ix]
                    heightmap[py, px] = (z - global_min) / height_range
    
    # Fill gaps with nearest neighbor
    for y in range(SIZE):
        for x in range(SIZE):
            if heightmap[y, x] == 0:
                # Check if this is truly zero or just unfilled
                # Look at neighbors
                nearest_val = 0
                min_dist = float('inf')
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < SIZE and 0 <= nx < SIZE:
                            val = heightmap[ny, nx]
                            if val > 0:
                                dist = dy*dy + dx*dx
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_val = val
                if nearest_val > 0:
                    heightmap[y, x] = nearest_val
    
    return np.clip(heightmap, 0, 1)

def generate_heightmap_pertile(chunk_heights: dict) -> np.ndarray:
    """Generate a 256x256 heightmap using PER-TILE normalization (shows hidden details)."""
    SIZE = 256
    heightmap = np.zeros((SIZE, SIZE), dtype=np.float32)
    
    # Find per-tile min/max
    all_heights = []
    for heights in chunk_heights.values():
        all_heights.extend(h for h in heights if not (np.isnan(h) or np.isinf(h)))
    
    if not all_heights:
        return heightmap
    
    tile_min = min(all_heights)
    tile_max = max(all_heights)
    height_range = tile_max - tile_min
    if height_range <= 0:
        height_range = 1.0
    
    # Process each chunk using Alpha MCVT format: 81 outer then 64 inner
    for chunk_idx, h_data in chunk_heights.items():
        if len(h_data) < 145:
            continue
            
        chunk_y = chunk_idx // 16
        chunk_x = chunk_idx % 16
        base_x = chunk_x * 16
        base_y = chunk_y * 16
        
        # Place 9x9 outer vertices at even positions
        for oy in range(9):
            for ox in range(9):
                px = base_x + ox * 2
                py = base_y + oy * 2
                if px < SIZE and py < SIZE:
                    z = h_data[oy * 9 + ox]
                    heightmap[py, px] = (z - tile_min) / height_range
        
        # Place 8x8 inner vertices at odd positions
        for iy in range(8):
            for ix in range(8):
                px = base_x + ix * 2 + 1
                py = base_y + iy * 2 + 1
                if px < SIZE and py < SIZE:
                    z = h_data[81 + iy * 8 + ix]
                    heightmap[py, px] = (z - tile_min) / height_range
    
    # Fill gaps with nearest neighbor
    for y in range(SIZE):
        for x in range(SIZE):
            if heightmap[y, x] == 0:
                nearest_val = 0
                min_dist = float('inf')
                for dy in range(-2, 3):
                    for dx in range(-2, 3):
                        if dy == 0 and dx == 0:
                            continue
                        ny, nx = y + dy, x + dx
                        if 0 <= ny < SIZE and 0 <= nx < SIZE:
                            val = heightmap[ny, nx]
                            if val > 0:
                                dist = dy*dy + dx*dx
                                if dist < min_dist:
                                    min_dist = dist
                                    nearest_val = val
                if nearest_val > 0:
                    heightmap[y, x] = nearest_val
    
    return np.clip(heightmap, 0, 1)

def regenerate_heightmaps(dataset_dir: Path, do_stitch: bool = True, max_size: int = 16384):
    """Regenerate all heightmaps with BOTH global and per-tile normalization."""
    
    # Find all JSON files - check dataset/ subdirectory first
    json_dir = dataset_dir / "dataset"
    if not json_dir.exists():
        json_dir = dataset_dir
    
    json_files = list(json_dir.glob("*.json"))
    # Filter out non-tile files
    json_files = [f for f in json_files if not f.name.startswith("texture_")]
    
    if not json_files:
        print(f"No tile JSON files found in {dataset_dir}")
        return
    
    print(f"Found {len(json_files)} tile JSON files")
    
    # First pass: find global height bounds
    print("Pass 1: Finding global height bounds...")
    global_min = float('inf')
    global_max = float('-inf')
    tile_data = {}
    
    for i, json_path in enumerate(json_files):
        result = load_tile_heights(json_path)
        if result:
            tile_name, chunk_heights, h_min, h_max = result
            tile_data[tile_name] = (chunk_heights, json_path)
            
            # Update global bounds from actual height values
            for heights in chunk_heights.values():
                for h in heights:
                    if not (np.isnan(h) or np.isinf(h)):
                        global_min = min(global_min, h)
                        global_max = max(global_max, h)
        
        if (i + 1) % 100 == 0:
            print(f"  Scanned {i + 1}/{len(json_files)} tiles...")
    
    if global_min >= global_max:
        global_min = 0
        global_max = 1
    
    print(f"Global height range: {global_min:.2f} to {global_max:.2f}")
    print(f"Total range: {global_max - global_min:.2f} units")
    
    # Second pass: regenerate heightmaps (BOTH global and per-tile)
    print(f"\nPass 2: Regenerating {len(tile_data)} heightmaps (global + per-tile)...")
    images_dir = dataset_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    processed = 0
    lock = threading.Lock()
    
    def process_tile(item):
        nonlocal processed
        tile_name, (chunk_heights, json_path) = item
        
        try:
            # Generate GLOBAL normalized heightmap (for training)
            heightmap_global = generate_heightmap(chunk_heights, global_min, global_max)
            heightmap_global_16 = (heightmap_global * 65535).astype(np.uint16)
            out_path_global = images_dir / f"{tile_name}_heightmap.png"
            Image.fromarray(heightmap_global_16).save(out_path_global)
            
            # Generate PER-TILE normalized heightmap (shows hidden details)
            heightmap_pertile = generate_heightmap_pertile(chunk_heights)
            heightmap_pertile_16 = (heightmap_pertile * 65535).astype(np.uint16)
            out_path_pertile = images_dir / f"{tile_name}_heightmap_local.png"
            Image.fromarray(heightmap_pertile_16).save(out_path_pertile)
            
            with lock:
                processed += 1
                if processed % 100 == 0:
                    print(f"  Generated {processed}/{len(tile_data)} heightmaps...")
            
            return tile_name, out_path_global, out_path_pertile
        except Exception as e:
            print(f"  Error processing {tile_name}: {e}")
            return None
    
    # Process in parallel
    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(process_tile, tile_data.items()))
    
    valid_results = [r for r in results if r]
    print(f"Generated {len(valid_results)} tile sets (global + per-tile each)")
    
    # Save global bounds for reference
    bounds_path = dataset_dir / "global_height_bounds.json"
    with open(bounds_path, 'w') as f:
        json.dump({
            "global_min": global_min,
            "global_max": global_max,
            "range": global_max - global_min,
            "tiles_processed": len(valid_results)
        }, f, indent=2)
    print(f"Saved bounds to {bounds_path}")
    
    # Stitch if requested
    if do_stitch:
        print("\nStitching global heightmaps...")
        stitch_heightmaps(dataset_dir, max_size, suffix="_heightmap.png", output_suffix="_full_heightmap_global.png")
        print("\nStitching per-tile (local) heightmaps...")
        stitch_heightmaps(dataset_dir, max_size, suffix="_heightmap_local.png", output_suffix="_full_heightmap_local.png")

def stitch_heightmaps(dataset_dir: Path, max_size: int = 16384, suffix: str = "_heightmap.png", output_suffix: str = "_full_heightmap.png"):
    """Stitch all heightmap tiles into a single PNG."""
    images_dir = dataset_dir / "images"
    
    # Find all heightmap files matching suffix
    heightmap_files = list(images_dir.glob(f"*{suffix}"))
    if not heightmap_files:
        print(f"No heightmap files found with suffix {suffix}")
        return
    
    # Parse coordinates - handle suffix properly
    suffix_escaped = re.escape(suffix)
    pattern = re.compile(rf"([A-Za-z]+)_(\d+)_(\d+){suffix_escaped}$")
    tiles = []
    map_name = None
    
    for f in heightmap_files:
        match = pattern.match(f.name)
        if match:
            map_name = match.group(1)
            x, y = int(match.group(2)), int(match.group(3))
            tiles.append((x, y, f))
    
    if not tiles:
        print("Could not parse tile coordinates")
        return
    
    print(f"Stitching {len(tiles)} tiles for {map_name}")
    
    # Calculate bounds
    min_x = min(t[0] for t in tiles)
    max_x = max(t[0] for t in tiles)
    min_y = min(t[1] for t in tiles)
    max_y = max(t[1] for t in tiles)
    
    tiles_wide = max_x - min_x + 1
    tiles_high = max_y - min_y + 1
    
    tile_size = 256
    full_width = tiles_wide * tile_size
    full_height = tiles_high * tile_size
    
    print(f"Full size: {full_width} x {full_height}")
    
    # Calculate scale
    scale = 1.0
    if full_width > max_size or full_height > max_size:
        scale = min(max_size / full_width, max_size / full_height)
        print(f"Scaling to {scale:.4f}")
    
    output_width = int(full_width * scale)
    output_height = int(full_height * scale)
    scaled_tile = int(tile_size * scale)
    
    print(f"Output size: {output_width} x {output_height}")
    
    # Create canvas
    canvas = np.zeros((output_height, output_width), dtype=np.uint16)
    
    # Load and place tiles
    for i, (x, y, path) in enumerate(tiles):
        try:
            img = Image.open(path)
            if scale < 1.0:
                img = img.resize((scaled_tile, scaled_tile), Image.LANCZOS)
            
            tile_data = np.array(img, dtype=np.uint16)
            
            dest_x = int((x - min_x) * scaled_tile)
            dest_y = int((y - min_y) * scaled_tile)
            
            h, w = tile_data.shape[:2]
            end_y = min(dest_y + h, output_height)
            end_x = min(dest_x + w, output_width)
            
            canvas[dest_y:end_y, dest_x:end_x] = tile_data[:end_y-dest_y, :end_x-dest_x]
            
        except Exception as e:
            print(f"  Warning: Failed to load {path.name}: {e}")
        
        if (i + 1) % 100 == 0:
            print(f"  Placed {i + 1}/{len(tiles)} tiles...")
    
    # Save
    stitched_dir = dataset_dir / "stitched"
    stitched_dir.mkdir(exist_ok=True)
    output_path = stitched_dir / f"{map_name}{output_suffix}"
    
    output_img = Image.fromarray(canvas)
    output_img.save(output_path)
    
    print(f"Saved: {output_path}")
    print(f"Final size: {output_width} x {output_height}")

def main():
    parser = argparse.ArgumentParser(
        description="Regenerate heightmaps with global normalization"
    )
    parser.add_argument("dataset_dir", type=Path, help="Path to VLM dataset directory")
    parser.add_argument("--no-stitch", action="store_true", help="Skip stitching")
    parser.add_argument("--max-size", type=int, default=16384,
                        help="Maximum dimension for stitched output (default: 16384)")
    
    args = parser.parse_args()
    
    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return 1
    
    regenerate_heightmaps(args.dataset_dir, not args.no_stitch, args.max_size)
    return 0

if __name__ == "__main__":
    exit(main())
