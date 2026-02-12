#!/usr/bin/env python3
"""
WoW Minimap to Mesh V4-DPT - Inference for DPT-based models

Usage:
    python img2mesh_v4_dpt.py <minimap_image.png> [--output <dir>] [--model <path>]
"""

import os
import sys
import json
import argparse
import shutil
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Import DPT model
from train_height_regressor_v4_dpt import TerrainDPT

# Default paths
MODEL_PATH = Path(r"J:\vlm_output\wow_height_regressor_v4_dpt")
DEFAULT_OUTPUT_ROOT = Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_output\model_outputs")


def load_model(model_dir: Path, device: torch.device):
    """Load trained DPT model and normalization stats"""
    stats_path = model_dir / "normalization_stats.json"
    model_path = model_dir / "best_model.pt"
    
    if not model_path.exists():
        model_path = model_dir / "final_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    # Load stats
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Get DPT model name
    dpt_model = stats.get("dpt_model", "Intel/dpt-large")
    
    # Create model
    model = TerrainDPT(model_name=dpt_model, predict_normals=stats.get("predict_normals", True))
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, stats


def predict(model, image_path: Path, device: torch.device, image_size=384):
    """Run inference on a minimap image"""
    # Load image
    img = Image.open(image_path).convert("RGB")
    img = img.resize((image_size, image_size), Image.BILINEAR)
    
    # Standard ImageNet normalization for DPT
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    img_tensor = transforms.ToTensor()(img)
    img_tensor = normalize(img_tensor)
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        pred_heights, pred_normals = model(img_tensor)
    
    heights = pred_heights.squeeze(0).cpu().numpy()  # [256, 145]
    normals = pred_normals.squeeze(0).cpu().numpy() if pred_normals is not None else None
    
    return heights, normals


def heights_to_vlm_json(heights, normals, tile_name, stats):
    """Convert predicted heights to VLM dataset JSON format"""
    chunks = []
    
    height_range = stats.get("global_max", 500) - stats.get("global_min", 0)
    height_offset = stats.get("global_min", 0)
    
    for chunk_idx in range(256):
        chunk_y = chunk_idx // 16
        chunk_x = chunk_idx % 16
        
        chunk_heights = heights[chunk_idx].tolist()
        scaled_heights = [h * height_range + height_offset for h in chunk_heights]
        
        chunk_data = {
            "chunk_index": chunk_idx,
            "chunk_x": chunk_x,
            "chunk_y": chunk_y,
            "heights": scaled_heights,
            "is_hole": False,
            "chunk_position": [chunk_x * 33.333, chunk_y * 33.333, 0],
        }
        
        if normals is not None:
            chunk_data["normals"] = normals[chunk_idx].tolist()
        
        chunks.append(chunk_data)
    
    return {
        "tile_name": tile_name,
        "chunks": chunks,
        "prediction_stats": {
            "height_min": float(heights.min()),
            "height_max": float(heights.max()),
            "height_mean": float(heights.mean()),
            "height_std": float(heights.std()),
            "model_version": "v4_dpt",
        }
    }


def save_heightmap_image(heights, output_path):
    """Save heights as 16-bit grayscale PNG on a proper square grid.
    
    Heights are [256, 145] where 256 = 16x16 chunks, each with 145 vertices.
    The 145 vertices are: 9x9 outer grid (81) + 8x8 inner grid (64).
    We reconstruct this onto a 129x129 grid (16 chunks * 8 + 1 edge vertices).
    """
    grid_size = 129  # 16 chunks * 8 vertices per chunk + 1 for edge
    heightmap_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    weight_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    for chunk_idx in range(256):
        chunk_y = chunk_idx // 16
        chunk_x = chunk_idx % 16
        chunk_heights = heights[chunk_idx]
        
        base_x = chunk_x * 8
        base_y = chunk_y * 8
        
        for i in range(81):
            row, col = i // 9, i % 9
            gx = base_x + col
            gy = base_y + row
            if gx < grid_size and gy < grid_size:
                heightmap_grid[gy, gx] += chunk_heights[i]
                weight_grid[gy, gx] += 1.0
    
    weight_grid[weight_grid == 0] = 1.0
    heightmap_grid /= weight_grid
    
    h_min, h_max = heightmap_grid.min(), heightmap_grid.max()
    if h_max - h_min < 1e-6:
        normalized = np.full_like(heightmap_grid, 0.5)
    else:
        normalized = (heightmap_grid - h_min) / (h_max - h_min)
    
    heightmap_16bit = (normalized * 65535).astype(np.uint16)
    
    img = Image.fromarray(heightmap_16bit, mode='I;16')
    img.save(output_path)


def save_normalmap_image(normals, output_path):
    """Save normals as RGB image on a proper square grid."""
    grid_size = 129
    normalmap_grid = np.zeros((grid_size, grid_size, 3), dtype=np.float32)
    weight_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    for chunk_idx in range(256):
        chunk_y = chunk_idx // 16
        chunk_x = chunk_idx % 16
        chunk_normals = normals[chunk_idx]
        
        base_x = chunk_x * 8
        base_y = chunk_y * 8
        
        for i in range(81):
            row, col = i // 9, i % 9
            gx = base_x + col
            gy = base_y + row
            if gx < grid_size and gy < grid_size:
                normalmap_grid[gy, gx] += chunk_normals[i]
                weight_grid[gy, gx] += 1.0
    
    weight_grid[weight_grid == 0] = 1.0
    for c in range(3):
        normalmap_grid[:, :, c] /= weight_grid
    
    normals_rgb = ((normalmap_grid + 1) / 2 * 255).astype(np.uint8)
    
    img = Image.fromarray(normals_rgb, mode='RGB')
    img.save(output_path)


def generate_obj_mesh(heights, output_path, scale=1.0):
    """Generate OBJ mesh from heights"""
    vertices = []
    faces = []
    
    height_range = heights.max() - heights.min()
    if height_range < 1e-6:
        height_range = 1.0
    
    for chunk_idx in range(256):
        chunk_y = chunk_idx // 16
        chunk_x = chunk_idx % 16
        
        chunk_heights = heights[chunk_idx]
        chunk_size = 33.333 * scale
        base_x = chunk_x * chunk_size
        base_y = chunk_y * chunk_size
        
        for i in range(145):
            if i < 81:
                row, col = i // 9, i % 9
                x = base_x + col * (chunk_size / 8)
                y = base_y + row * (chunk_size / 8)
            else:
                inner_i = i - 81
                row, col = inner_i // 8, inner_i % 8
                x = base_x + (col + 0.5) * (chunk_size / 8)
                y = base_y + (row + 0.5) * (chunk_size / 8)
            
            z = chunk_heights[i] * height_range * scale * 100  # Exaggerate for visibility
            vertices.append((x, y, z))
    
    for chunk_idx in range(256):
        base_v = chunk_idx * 145 + 1
        
        for row in range(8):
            for col in range(8):
                v0 = base_v + row * 9 + col
                v1 = v0 + 1
                v2 = v0 + 9
                v3 = v2 + 1
                
                faces.append((v0, v2, v1))
                faces.append((v1, v2, v3))
    
    with open(output_path, 'w') as f:
        f.write(f"# WoW Terrain Mesh (V4-DPT model)\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    return len(vertices), len(faces)


def main():
    parser = argparse.ArgumentParser(description="WoW Minimap to Mesh V4-DPT")
    parser.add_argument("input", help="Input minimap image")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--model", "-m", default=str(MODEL_PATH), help="Model directory")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)
    
    tile_name = input_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = DEFAULT_OUTPUT_ROOT / f"{timestamp}_{tile_name}_dpt"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("WoW Minimap to Mesh V4-DPT")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_dir}")
    
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = load_model(model_dir, device)
    
    print("Predicting heights and normals...")
    heights, normals = predict(model, input_path, device)
    
    print(f"Height range: [{heights.min():.4f}, {heights.max():.4f}]")
    print(f"Height std: {heights.std():.4f}")
    if normals is not None:
        print(f"Normals shape: {normals.shape}")
    
    shutil.copy(input_path, output_dir / f"{tile_name}_minimap.png")
    
    print("\nGenerating VLM dataset JSON...")
    vlm_data = heights_to_vlm_json(heights, normals, tile_name, stats)
    json_path = output_dir / f"{tile_name}.json"
    with open(json_path, 'w') as f:
        json.dump(vlm_data, f, indent=2)
    print(f"Wrote: {json_path}")
    
    print("Generating heightmap image...")
    save_heightmap_image(heights, output_dir / f"{tile_name}_heightmap.png")
    
    if normals is not None:
        print("Generating normal map image...")
        save_normalmap_image(normals, output_dir / f"{tile_name}_normalmap.png")
    
    print("\nGenerating OBJ mesh...")
    n_verts, n_faces = generate_obj_mesh(heights, output_dir / f"{tile_name}.obj")
    print(f"Wrote {n_verts} vertices, {n_faces} faces")
    
    print("\n" + "=" * 60)
    print(f"Output files in {output_dir}:")
    print(f"  - {tile_name}.json")
    print(f"  - {tile_name}_heightmap.png")
    if normals is not None:
        print(f"  - {tile_name}_normalmap.png")
    print(f"  - {tile_name}.obj")
    print("=" * 60)


if __name__ == "__main__":
    main()
