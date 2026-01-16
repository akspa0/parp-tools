#!/usr/bin/env python3
"""
Render high-resolution normal maps from VLM dataset JSON files.

Creates 256x256 normalmaps using ALL 145 MCNR normals per chunk:
- Outer normals (9 per row) at even pixel positions
- Inner normals (8 per row) at odd pixel positions
- 16 pixels per chunk = 256 pixels per tile
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import argparse


# Maximum native resolution using all MCNR normals
GRID_SIZE = 256  # 16 chunks * 16 pixels per chunk


def render_normalmap(json_path: Path, output_path: Path):
    """Render a 256x256 normalmap from JSON terrain data using all normals."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    td = data.get("terrain_data", {})
    chunk_layers = td.get("chunk_layers", [])
    
    if not chunk_layers:
        print(f"  No chunk_layers in {json_path.name}")
        return False
    
    # Build 256x256 grid
    normalmap = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
    weight_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
    for layer in chunk_layers:
        idx = layer.get("idx", -1)
        normals_raw = layer.get("normals", None)
        
        if normals_raw is None or idx < 0 or idx >= 256:
            continue
        
        chunk_y = idx // 16
        chunk_x = idx % 16
        # Each chunk is 16 pixels (2 pixels per unit square)
        base_x = chunk_x * 16
        base_y = chunk_y * 16
        
        # ALPHA ADT MCNR FORMAT: 81 outer (9x9) THEN 64 inner (8x8) - NOT interleaved!
        # This matches the Alpha MCVT format.
        # NOTE: Some exports have 448 bytes (with padding), some have 435 (145*3)
        if len(normals_raw) < 435:  # 145 * 3
            continue
        normals_raw = normals_raw[:435]  # Truncate padding if present
        
        # First 81 normals: 9x9 outer grid (row-major)
        for oy in range(9):
            for ox in range(9):
                px = base_x + ox * 2
                py = base_y + oy * 2
                n_idx = oy * 9 + ox
                
                if px < GRID_SIZE and py < GRID_SIZE:
                    nx = normals_raw[n_idx * 3 + 0] / 127.0
                    ny = normals_raw[n_idx * 3 + 1] / 127.0
                    nz = normals_raw[n_idx * 3 + 2] / 127.0
                    
                    normalmap[py, px, 0] += nx
                    normalmap[py, px, 1] += ny
                    normalmap[py, px, 2] += nz
                    weight_map[py, px] += 1.0
        
        # Next 64 normals: 8x8 inner grid (row-major)
        for iy in range(8):
            for ix in range(8):
                px = base_x + ix * 2 + 1
                py = base_y + iy * 2 + 1
                n_idx = 81 + iy * 8 + ix
                
                if px < GRID_SIZE and py < GRID_SIZE:
                    nx = normals_raw[n_idx * 3 + 0] / 127.0
                    ny = normals_raw[n_idx * 3 + 1] / 127.0
                    nz = normals_raw[n_idx * 3 + 2] / 127.0
                    
                    normalmap[py, px, 0] += nx
                    normalmap[py, px, 1] += ny
                    normalmap[py, px, 2] += nz
                    weight_map[py, px] += 1.0
    
    # Average overlapping normals (chunk edges)
    weight_map[weight_map == 0] = 1.0
    for c in range(3):
        normalmap[:, :, c] /= weight_map
    
    # Fill gaps using bilinear interpolation (8-neighbor weighted)
    valid_mask = (weight_map >= 0.5).astype(np.float32)
    filled = normalmap.copy()
    
    for _ in range(3):
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if valid_mask[y, x] < 0.5:
                    neighbors = []
                    weights = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                                if valid_mask[ny, nx] >= 0.5 or np.any(filled[ny, nx] != 0):
                                    dist = np.sqrt(dy*dy + dx*dx)
                                    neighbors.append(filled[ny, nx])
                                    weights.append(1.0 / dist)
                    if neighbors:
                        weights = np.array(weights)
                        weights /= weights.sum()
                        filled[y, x] = np.sum([n * w for n, w in zip(neighbors, weights)], axis=0)
        normalmap = filled.copy()
    
    normalmap = filled
    
    # Normalize vectors
    lengths = np.sqrt(np.sum(normalmap ** 2, axis=2, keepdims=True))
    lengths[lengths == 0] = 1.0
    normalmap = normalmap / lengths
    
    # Convert to RGB [0, 255]: map [-1, 1] -> [0, 255]
    normalmap_rgb = ((normalmap + 1) / 2 * 255).astype(np.uint8)
    
    # Save
    img = Image.fromarray(normalmap_rgb, mode='RGB')
    img.save(output_path)
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
        output_path = images_dir / f"{tile_name}_normalmap.png"
        
        if render_normalmap(json_path, output_path):
            rendered += 1
    
    print(f"  Rendered {rendered}/{len(json_files)} normal maps")


def main():
    parser = argparse.ArgumentParser(description="Render normal maps from VLM dataset JSONs")
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
