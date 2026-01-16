#!/usr/bin/env python3
"""
Render high-resolution heightmaps from VLM dataset JSON files.

Creates 256x256 heightmaps using ALL 145 MCVT vertices per chunk:
- Outer vertices (9 per row) at even pixel positions
- Inner vertices (8 per row) at odd pixel positions
- 16 pixels per chunk = 256 pixels per tile
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import argparse


# Maximum native resolution using all MCVT vertices
GRID_SIZE = 256  # 16 chunks * 16 pixels per chunk


def render_heightmap(json_path: Path, output_path: Path):
    """Render a 256x256 heightmap from JSON terrain data using all vertices."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    td = data.get("terrain_data", {})
    heights_data = td.get("heights", [])
    
    if not heights_data:
        print(f"  No heights in {json_path.name}")
        return False
    
    # Build 256x256 grid (16 chunks * 16 pixels per chunk)
    heightmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    weight_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
    for chunk_data in heights_data:
        idx = chunk_data.get("idx", -1)
        h = chunk_data.get("h", None)
        
        if h is None or idx < 0 or idx >= 256:
            continue
        
        chunk_y = idx // 16
        chunk_x = idx % 16
        # Each chunk is 16 pixels (2 pixels per unit square)
        base_x = chunk_x * 16
        base_y = chunk_y * 16
        
        # ALPHA ADT MCVT FORMAT: 81 outer (9x9) THEN 64 inner (8x8) - NOT interleaved!
        if len(h) < 145:
            continue
        
        # Build 17x17 grid from Alpha format
        grid17 = np.zeros((17, 17), dtype=np.float32)
        
        # Place 9x9 outer at even positions
        for oy in range(9):
            for ox in range(9):
                grid17[oy * 2, ox * 2] = h[oy * 9 + ox]
        
        # Place 8x8 inner at odd positions
        for iy in range(8):
            for ix in range(8):
                grid17[iy * 2 + 1, ix * 2 + 1] = h[81 + iy * 8 + ix]
        
        # Interpolate edges: y even, x odd
        for y in range(0, 17, 2):
            for x in range(1, 16, 2):
                grid17[y, x] = (grid17[y, x-1] + grid17[y, x+1]) / 2
        
        # Interpolate edges: y odd, x even
        for y in range(1, 16, 2):
            for x in range(0, 17, 2):
                grid17[y, x] = (grid17[y-1, x] + grid17[y+1, x]) / 2
        
        # Sample 16x16 output pixels directly from grid17
        for py in range(16):
            for px in range(16):
                out_x = base_x + px
                out_y = base_y + py
                if out_x < GRID_SIZE and out_y < GRID_SIZE:
                    heightmap[out_y, out_x] = grid17[py, px]
                    weight_map[out_y, out_x] = 1.0
    
    # Average overlapping vertices (chunk edges)
    weight_map[weight_map == 0] = 1.0
    heightmap /= weight_map
    
    # Fill gaps using bilinear interpolation (8-neighbor weighted)
    # Create mask of valid pixels
    valid_mask = (weight_map >= 0.5).astype(np.float32)
    
    # Use distance-weighted interpolation for gaps
    # First, dilate the valid data to fill gaps smoothly
    filled = heightmap.copy()
    
    # Multiple passes of neighbor averaging for smooth interpolation
    for _ in range(3):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if valid_mask[y, x] < 0.5:
                    # Bilinear: use diagonal neighbors too
                    neighbors = []
                    weights = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                                if valid_mask[ny, nx] >= 0.5 or filled[ny, nx] != 0:
                                    # Weight by inverse distance
                                    dist = np.sqrt(dy*dy + dx*dx)
                                    neighbors.append(filled[ny, nx])
                                    weights.append(1.0 / dist)
                    if neighbors:
                        filled[y, x] = np.average(neighbors, weights=weights)
        # Update heightmap for next pass
        heightmap = filled.copy()
    
    heightmap = filled
    
    # Normalize to [0, 65535] for 16-bit PNG
    h_min, h_max = heightmap.min(), heightmap.max()
    h_range = h_max - h_min
    
    if h_range < 1e-6:
        # Flat terrain
        normalized = np.full_like(heightmap, 0.5)
    else:
        normalized = (heightmap - h_min) / h_range
    
    heightmap_16bit = (normalized * 65535).astype(np.uint16)
    
    # Save as 16-bit grayscale
    img = Image.fromarray(heightmap_16bit, mode='I;16')
    img.save(output_path)
    
    # Also save 8-bit version for preview
    preview_path = output_path.with_name(output_path.stem + "_preview.png")
    heightmap_8bit = (normalized * 255).astype(np.uint8)
    img_preview = Image.fromarray(heightmap_8bit, mode='L')
    img_preview.save(preview_path)
    
    return True


def process_dataset(dataset_root: Path):
    """Process all JSONs in a dataset folder."""
    dataset_dir = dataset_root / "dataset"
    images_dir = dataset_root / "images"
    
    if not dataset_dir.exists():
        print(f"No dataset folder in {dataset_root}")
        return
    
    json_files = list(dataset_dir.glob("*.json"))
    print(f"Processing {len(json_files)} tiles in {dataset_root.name}...")
    
    rendered = 0
    for json_path in json_files:
        tile_name = json_path.stem
        output_path = images_dir / f"{tile_name}_heightmap_v2.png"
        
        if render_heightmap(json_path, output_path):
            rendered += 1
    
    print(f"  Rendered {rendered}/{len(json_files)} heightmaps")


def main():
    parser = argparse.ArgumentParser(description="Render heightmaps from VLM dataset JSONs")
    parser.add_argument("--dataset", type=str, help="Specific dataset folder to process")
    args = parser.parse_args()
    
    if args.dataset:
        process_dataset(Path(args.dataset))
    else:
        # Process all VLM datasets
        vlm_roots = [
            Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_shadowfang_v1"),
            Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_azeroth_v7"),
            Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_DeadminesInstance_v2"),
            Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_kalimdor_v4"),
        ]
        
        for root in vlm_roots:
            if root.exists():
                process_dataset(root)


if __name__ == "__main__":
    main()
