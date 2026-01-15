#!/usr/bin/env python3
"""
WoW Minimap to Mesh V4 - Inference for V4 models

Usage:
    python img2mesh_v4.py <minimap_image.png> [--output <dir>] [--model <path>]
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

# Import V4 model
from train_height_regressor_v4 import TerrainRegressorV4

# Default paths
MODEL_PATH = Path(r"J:\vlm_output\wow_height_regressor_v4")
DEFAULT_OUTPUT_ROOT = Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_output\model_outputs")


def load_model(model_dir: Path, device: torch.device):
    """Load trained V4 model and normalization stats"""
    stats_path = model_dir / "normalization_stats.json"
    model_path = model_dir / "best_model.pt"
    
    if not model_path.exists():
        model_path = model_dir / "final_model.pt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"No model found in {model_dir}")
    
    # Load stats
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    # Determine backbone
    backbone = stats.get("backbone", "efficientnet_b0")
    
    # Create model
    model = TerrainRegressorV4(
        in_channels=stats.get("in_channels", 5),
        predict_normals=stats.get("predict_normals", True),
        backbone=backbone
    )
    
    # Load weights
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model = model.to(device)
    model.eval()
    
    return model, stats


def predict(model, image_path: Path, shadow_path: Path = None, alpha_path: Path = None, device: torch.device = None):
    """Run inference on a minimap image"""
    # Load image
    img = Image.open(image_path).convert("RGB")
    img_tensor = transforms.ToTensor()(img)
    
    # Load or create shadow
    if shadow_path and shadow_path.exists():
        shadow = Image.open(shadow_path).convert("L")
        shadow_tensor = transforms.ToTensor()(shadow)
    else:
        shadow_tensor = torch.zeros(1, img_tensor.shape[1], img_tensor.shape[2])
    
    # Load or create alpha
    if alpha_path and alpha_path.exists():
        alpha = Image.open(alpha_path).convert("L")
        alpha_tensor = transforms.ToTensor()(alpha)
    else:
        alpha_tensor = torch.ones(1, img_tensor.shape[1], img_tensor.shape[2])
    
    # Resize to 256x256
    target_size = (256, 256)
    img_tensor = F.interpolate(img_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    shadow_tensor = F.interpolate(shadow_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    alpha_tensor = F.interpolate(alpha_tensor.unsqueeze(0), size=target_size, mode='bilinear', align_corners=False).squeeze(0)
    
    # Stack: RGB + Shadow + Alpha = 5 channels
    pixel_values = torch.cat([img_tensor, shadow_tensor, alpha_tensor], dim=0)
    pixel_values = pixel_values.unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        pred_heights, pred_normals = model(pixel_values)
    
    heights = pred_heights.squeeze(0).cpu().numpy()  # [256, 145]
    normals = pred_normals.squeeze(0).cpu().numpy() if pred_normals is not None else None  # [256, 145, 3]
    
    return heights, normals


def heights_to_vlm_json(heights, normals, tile_name, stats):
    """Convert predicted heights to VLM dataset JSON format"""
    chunks = []
    
    # Per-tile normalization means heights are in [0, 1]
    # Scale to a reasonable world height range for visualization
    height_range = stats.get("global_max", 500) - stats.get("global_min", 0)
    height_offset = stats.get("global_min", 0)
    
    for chunk_idx in range(256):
        chunk_y = chunk_idx // 16
        chunk_x = chunk_idx % 16
        
        chunk_heights = heights[chunk_idx].tolist()
        
        # Scale heights to world coordinates
        scaled_heights = [h * height_range + height_offset for h in chunk_heights]
        
        chunk_data = {
            "chunk_index": chunk_idx,
            "chunk_x": chunk_x,
            "chunk_y": chunk_y,
            "heights": scaled_heights,
            "is_hole": False,
        }
        
        if normals is not None:
            chunk_normals = normals[chunk_idx].tolist()
            chunk_data["normals"] = chunk_normals
        
        # Add placeholder chunk position
        chunk_data["chunk_position"] = [chunk_x * 33.333, chunk_y * 33.333, 0]
        
        chunks.append(chunk_data)
    
    return {
        "tile_name": tile_name,
        "chunks": chunks,
        "prediction_stats": {
            "height_min": float(heights.min()),
            "height_max": float(heights.max()),
            "height_mean": float(heights.mean()),
            "model_version": "v4",
        }
    }


def save_heightmap_image(heights, output_path):
    """Save heights as 16-bit grayscale PNG"""
    h_min, h_max = heights.min(), heights.max()
    if h_max - h_min < 1e-6:
        normalized = np.zeros_like(heights)
    else:
        normalized = (heights - h_min) / (h_max - h_min)
    
    # Reshape to 2D image (16x16 chunks, each with 145 heights -> approximate as image)
    # For visualization, create a 256x145 image
    heightmap_2d = heights.reshape(256, 145)
    heightmap_16bit = (heightmap_2d * 65535).astype(np.uint16)
    
    img = Image.fromarray(heightmap_16bit, mode='I;16')
    img.save(output_path)


def save_normalmap_image(normals, output_path):
    """Save normals as RGB image"""
    # Normals are in [-1, 1], convert to [0, 255]
    normals_2d = normals.reshape(256, 145, 3)
    normals_rgb = ((normals_2d + 1) / 2 * 255).astype(np.uint8)
    
    img = Image.fromarray(normals_rgb, mode='RGB')
    img.save(output_path)


def generate_obj_mesh(heights, output_path, scale=1.0):
    """Generate OBJ mesh from heights"""
    # heights: [256, 145] - 256 chunks, 145 heights per chunk
    # WoW chunk layout: 9x9 outer + 8x8 inner = 145 heights
    
    vertices = []
    faces = []
    
    height_range = heights.max() - heights.min()
    if height_range < 1e-6:
        height_range = 1.0
    
    # Generate vertices for each chunk
    for chunk_idx in range(256):
        chunk_y = chunk_idx // 16
        chunk_x = chunk_idx % 16
        
        chunk_heights = heights[chunk_idx]
        
        # WoW chunk is 33.333 units
        chunk_size = 33.333 * scale
        base_x = chunk_x * chunk_size
        base_y = chunk_y * chunk_size
        
        # Generate 9x9 outer grid + 8x8 inner grid = 145 vertices
        for i in range(145):
            if i < 81:  # Outer 9x9
                row = i // 9
                col = i % 9
                x = base_x + col * (chunk_size / 8)
                y = base_y + row * (chunk_size / 8)
            else:  # Inner 8x8
                inner_i = i - 81
                row = inner_i // 8
                col = inner_i % 8
                x = base_x + (col + 0.5) * (chunk_size / 8)
                y = base_y + (row + 0.5) * (chunk_size / 8)
            
            z = chunk_heights[i] * height_range * scale
            vertices.append((x, y, z))
    
    # Generate faces (simplified - just connect outer grid)
    for chunk_idx in range(256):
        base_v = chunk_idx * 145 + 1  # OBJ is 1-indexed
        
        # Connect outer 9x9 grid
        for row in range(8):
            for col in range(8):
                v0 = base_v + row * 9 + col
                v1 = v0 + 1
                v2 = v0 + 9
                v3 = v2 + 1
                
                faces.append((v0, v2, v1))
                faces.append((v1, v2, v3))
    
    # Write OBJ
    with open(output_path, 'w') as f:
        f.write(f"# WoW Terrain Mesh (V4 model)\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    return len(vertices), len(faces)


def main():
    parser = argparse.ArgumentParser(description="WoW Minimap to Mesh V4")
    parser.add_argument("input", help="Input minimap image")
    parser.add_argument("--output", "-o", help="Output directory")
    parser.add_argument("--model", "-m", default=str(MODEL_PATH), help="Model directory")
    parser.add_argument("--shadow", "-s", help="Shadow map image")
    parser.add_argument("--alpha", "-a", help="Alpha map image")
    
    args = parser.parse_args()
    
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        sys.exit(1)
    
    model_dir = Path(args.model)
    if not model_dir.exists():
        print(f"Error: Model directory not found: {model_dir}")
        sys.exit(1)
    
    # Create output directory
    tile_name = input_path.stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = DEFAULT_OUTPUT_ROOT / f"{timestamp}_{tile_name}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("WoW Minimap to Mesh V4")
    print("=" * 60)
    print(f"Input: {input_path}")
    print(f"Output: {output_dir}")
    print(f"Model: {model_dir}")
    
    # Load model
    print("\nLoading model...")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, stats = load_model(model_dir, device)
    
    # Run inference
    print("Predicting heights and normals...")
    shadow_path = Path(args.shadow) if args.shadow else None
    alpha_path = Path(args.alpha) if args.alpha else None
    
    heights, normals = predict(model, input_path, shadow_path, alpha_path, device)
    
    print(f"Height range: [{heights.min():.4f}, {heights.max():.4f}]")
    print(f"Height std: {heights.std():.4f}")
    if normals is not None:
        print(f"Normals shape: {normals.shape}")
    
    # Copy input
    shutil.copy(input_path, output_dir / f"{tile_name}_minimap.png")
    print(f"Copied input minimap: {output_dir / f'{tile_name}_minimap.png'}")
    
    # Generate VLM JSON
    print("\nGenerating VLM dataset JSON...")
    vlm_data = heights_to_vlm_json(heights, normals, tile_name, stats)
    json_path = output_dir / f"{tile_name}.json"
    with open(json_path, 'w') as f:
        json.dump(vlm_data, f, indent=2)
    print(f"Wrote VLM dataset JSON: {json_path}")
    
    # Generate heightmap image
    print("Generating heightmap image...")
    heightmap_path = output_dir / f"{tile_name}_heightmap.png"
    save_heightmap_image(heights, heightmap_path)
    print(f"Wrote heightmap image: {heightmap_path}")
    
    # Generate normal map
    if normals is not None:
        print("Generating normal map image...")
        normalmap_path = output_dir / f"{tile_name}_normalmap.png"
        save_normalmap_image(normals, normalmap_path)
        print(f"Wrote normal map image: {normalmap_path}")
    
    # Generate OBJ mesh
    print("\nGenerating OBJ mesh...")
    obj_path = output_dir / f"{tile_name}.obj"
    n_verts, n_faces = generate_obj_mesh(heights, obj_path)
    print(f"Wrote {n_verts} vertices, {n_faces} faces to {obj_path}")
    
    print("\n" + "=" * 60)
    print(f"Output files in {output_dir}:")
    print(f"  - {tile_name}.json          (VLM dataset)")
    print(f"  - {tile_name}_minimap.png   (input copy)")
    print(f"  - {tile_name}_heightmap.png (16-bit heightmap)")
    if normals is not None:
        print(f"  - {tile_name}_normalmap.png (RGB normal map)")
    print(f"  - {tile_name}.obj           (3D mesh)")
    print("=" * 60)


if __name__ == "__main__":
    main()
