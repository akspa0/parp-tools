#!/usr/bin/env python3
"""
Render proper heightmaps from VLM dataset JSON files.

Creates smooth 129x129 heightmaps by properly stitching chunk vertices.
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def render_heightmap(json_path: Path, output_path: Path):
    """Render a heightmap from JSON terrain data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    td = data.get("terrain_data", {})
    heights_data = td.get("heights", [])
    
    if not heights_data:
        print(f"  No heights in {json_path.name}")
        return False
    
    # Build 129x129 grid (16 chunks * 8 vertices + 1 edge)
    grid_size = 129
    heightmap = np.zeros((grid_size, grid_size), dtype=np.float32)
    weight_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    for chunk_data in heights_data:
        idx = chunk_data.get("idx", -1)
        h = chunk_data.get("h", None)
        
        if h is None or idx < 0 or idx >= 256:
            continue
        
        chunk_y = idx // 16
        chunk_x = idx % 16
        base_x = chunk_x * 8
        base_y = chunk_y * 8
        
        # Each chunk has 145 vertices: 9x9 outer (81) + 8x8 inner (64)
        # We only use the 9x9 outer grid for the continuous map
        if len(h) < 81:
            continue
        
        for i in range(81):
            row = i // 9
            col = i % 9
            gx = base_x + col
            gy = base_y + row
            
            if gx < grid_size and gy < grid_size:
                heightmap[gy, gx] += h[i]
                weight_map[gy, gx] += 1.0
    
    # Average overlapping vertices
    weight_map[weight_map == 0] = 1.0
    heightmap /= weight_map
    
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
