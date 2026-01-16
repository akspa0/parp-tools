#!/usr/bin/env python3
"""
Render proper normal maps from VLM dataset JSON files.

The JSON contains per-chunk normals data that we need to reconstruct
into a continuous 129x129 grid (16 chunks * 8 + 1 edge).
"""

import json
import numpy as np
from pathlib import Path
from PIL import Image
import argparse


def render_normalmap(json_path: Path, output_path: Path):
    """Render a normal map from JSON terrain data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    td = data.get("terrain_data", {})
    chunk_layers = td.get("chunk_layers", [])
    
    if not chunk_layers:
        print(f"  No chunk_layers in {json_path.name}")
        return False
    
    # Build 129x129 grid (16 chunks * 8 vertices + 1 edge)
    grid_size = 129
    normalmap = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    weight_map = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    for layer in chunk_layers:
        idx = layer.get("idx", -1)
        normals_raw = layer.get("normals", None)
        
        if normals_raw is None or idx < 0 or idx >= 256:
            continue
        
        chunk_y = idx // 16
        chunk_x = idx % 16
        base_x = chunk_x * 8
        base_y = chunk_y * 8
        
        # Each chunk has 145 vertices: 9x9 outer (81) + 8x8 inner (64)
        # We only use the 9x9 outer grid for the continuous map
        if len(normals_raw) < 243:  # 81 * 3
            continue
        
        for i in range(81):
            row = i // 9
            col = i % 9
            gx = base_x + col
            gy = base_y + row
            
            if gx < grid_size and gy < grid_size:
                # Normals are stored as signed bytes [-127, 127], normalize to [-1, 1]
                nx = normals_raw[i * 3 + 0] / 127.0
                ny = normals_raw[i * 3 + 1] / 127.0
                nz = normals_raw[i * 3 + 2] / 127.0
                
                normalmap[gy, gx, 0] += nx
                normalmap[gy, gx, 1] += ny
                normalmap[gy, gx, 2] += nz
                weight_map[gy, gx] += 1.0
    
    # Average overlapping vertices
    weight_map[weight_map == 0] = 1.0
    for c in range(3):
        normalmap[:, :, c] /= weight_map
    
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
