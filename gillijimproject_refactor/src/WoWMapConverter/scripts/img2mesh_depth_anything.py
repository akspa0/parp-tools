#!/usr/bin/env python3
"""
Depth Anything WoW - Inference Script

Uses fine-tuned Depth Anything V3 model to predict heightmaps from minimaps only.
No normalmap required - this is the minimap-only solution.

Usage:
    python img2mesh_depth_anything.py /path/to/minimap.png
    python img2mesh_depth_anything.py /path/to/minimaps --batch
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
import re

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Add DA3 to path
DA3_PATH = Path(__file__).parent.parent / "WoWMapConverter.Core" / "VLM" / "DepthAnything3" / "Depth-Anything-3" / "src"
sys.path.insert(0, str(DA3_PATH))

# Import model class
from train_depth_anything_finetune import DepthAnythingWoW, MODEL_NAME, INPUT_SIZE, OUTPUT_SIZE

# Default paths
DEFAULT_MODEL = Path(r"J:\vlm_output\depth_anything_wow_finetune\best_model.pt")
OUTPUT_ROOT = Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_output\model_outputs")


def load_model(model_path: Path, device: torch.device):
    """Load fine-tuned Depth Anything model."""
    model = DepthAnythingWoW(MODEL_NAME, OUTPUT_SIZE).to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"  Best val_loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model


def predict(model, image_path: Path, device: torch.device) -> np.ndarray:
    """Run inference on a single minimap image."""
    # Load and preprocess image
    image = Image.open(image_path).convert("RGB")
    image = image.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    
    # Convert to tensor and normalize
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    image_tensor = normalize(to_tensor(image)).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        depth = model(image_tensor)
    
    # Convert to numpy [OUTPUT_SIZE, OUTPUT_SIZE]
    heightmap = depth.squeeze().cpu().numpy()
    
    return heightmap


def save_heightmap_image(heightmap: np.ndarray, output_path: Path):
    """Save heightmap as 16-bit grayscale PNG."""
    h_min, h_max = heightmap.min(), heightmap.max()
    h_range = h_max - h_min
    
    if h_range < 1e-6:
        normalized = np.full_like(heightmap, 0.5)
    else:
        normalized = (heightmap - h_min) / h_range
    
    # 16-bit
    heightmap_16bit = (normalized * 65535).astype(np.uint16)
    img = Image.fromarray(heightmap_16bit, mode='I;16')
    img.save(output_path)
    
    # 8-bit preview
    preview_path = output_path.with_name(output_path.stem + "_preview.png")
    heightmap_8bit = (normalized * 255).astype(np.uint8)
    img_preview = Image.fromarray(heightmap_8bit, mode='L')
    img_preview.save(preview_path)
    
    return normalized


def generate_obj_mesh(heightmap: np.ndarray, output_path: Path, texture_path: Path = None, scale: float = 1.0):
    """Generate OBJ mesh from heightmap with optional texture."""
    h, w = heightmap.shape
    vertices = []
    uvs = []
    faces = []
    
    # Generate vertices and UVs
    for y in range(h):
        for x in range(w):
            wx = (x - w/2) * scale
            wy = (y - h/2) * scale
            wz = heightmap[y, x] * scale * 20
            vertices.append((wx, wz, wy))
            
            # UV with half-pixel inset
            half_pixel = 0.5 / 256
            u = half_pixel + (x / (w - 1)) * (1.0 - 2 * half_pixel)
            v = half_pixel + (1.0 - (y / (h - 1))) * (1.0 - 2 * half_pixel)
            uvs.append((u, v))
    
    # Generate faces - correct winding order
    for y in range(h - 1):
        for x in range(w - 1):
            v0 = y * w + x + 1
            v1 = y * w + (x + 1) + 1
            v2 = (y + 1) * w + (x + 1) + 1
            v3 = (y + 1) * w + x + 1
            
            faces.append((v0, v3, v2))
            faces.append((v0, v2, v1))
    
    # Write MTL if texture provided
    mtl_name = None
    if texture_path and texture_path.exists():
        mtl_path = output_path.with_suffix('.mtl')
        mtl_name = output_path.stem + "_mat"
        
        tex_dest = output_path.parent / texture_path.name
        if not tex_dest.exists():
            Image.open(texture_path).save(tex_dest)
        
        with open(mtl_path, 'w') as f:
            f.write(f"# Depth Anything WoW Material\n")
            f.write(f"newmtl {mtl_name}\n")
            f.write(f"Ka 1.0 1.0 1.0\n")
            f.write(f"Kd 1.0 1.0 1.0\n")
            f.write(f"Ks 0.0 0.0 0.0\n")
            f.write(f"d 1.0\n")
            f.write(f"illum 1\n")
            f.write(f"map_Kd {texture_path.name}\n")
        
        print(f"  Saved material: {mtl_path.name}")
    
    # Write OBJ
    with open(output_path, 'w') as f:
        f.write(f"# Depth Anything WoW Output\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        if mtl_name:
            f.write(f"mtllib {output_path.stem}.mtl\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        for uv in uvs:
            f.write(f"vt {uv[0]:.6f} {uv[1]:.6f}\n")
        
        f.write("\n")
        
        if mtl_name:
            f.write(f"usemtl {mtl_name}\n")
        
        for face in faces:
            f.write(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n")
    
    print(f"  Saved mesh: {len(vertices)} vertices, {len(faces)} faces")


def find_minimap_tiles(input_dir: Path) -> list:
    """Find all minimap tiles in directory."""
    tiles = []
    tile_pattern = re.compile(r'^(.+?)_(\d+)_(\d+)\.png$', re.IGNORECASE)
    
    for f in input_dir.glob("*.png"):
        if any(suffix in f.stem.lower() for suffix in ['_normalmap', '_heightmap', '_preview', '_mask']):
            continue
        
        match = tile_pattern.match(f.name)
        if match:
            map_name, x, y = match.groups()
            tiles.append({
                'path': f,
                'map_name': map_name,
                'x': int(x),
                'y': int(y),
                'name': f.stem
            })
    
    return sorted(tiles, key=lambda t: (t['map_name'], t['x'], t['y']))


def process_single(model, image_path: Path, output_dir: Path, device: torch.device, scale: float):
    """Process a single image."""
    tile_name = image_path.stem
    
    print(f"\nProcessing: {image_path.name}")
    
    # Predict heightmap
    heightmap = predict(model, image_path, device)
    print(f"  Output shape: {heightmap.shape}")
    print(f"  Height range: [{heightmap.min():.4f}, {heightmap.max():.4f}]")
    
    # Create output directory
    tile_output_dir = output_dir / tile_name
    tile_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save heightmap
    heightmap_path = tile_output_dir / f"{tile_name}_heightmap.png"
    save_heightmap_image(heightmap, heightmap_path)
    print(f"  Saved heightmap: {heightmap_path}")
    
    # Save OBJ mesh
    obj_path = tile_output_dir / f"{tile_name}.obj"
    generate_obj_mesh(heightmap, obj_path, texture_path=image_path, scale=scale)
    
    # Copy minimap
    minimap_copy = tile_output_dir / f"{tile_name}.png"
    Image.open(image_path).save(minimap_copy)
    
    # Metadata
    metadata = {
        "tile_name": tile_name,
        "input": str(image_path),
        "output_dir": str(tile_output_dir),
        "heightmap_shape": list(heightmap.shape),
        "height_min": float(heightmap.min()),
        "height_max": float(heightmap.max()),
        "model_version": "depth_anything_wow",
    }
    
    with open(tile_output_dir / f"{tile_name}.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return True


def main():
    parser = argparse.ArgumentParser(description="Depth Anything WoW Inference")
    parser.add_argument("input", type=str, help="Minimap image or directory")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Model checkpoint")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--scale", type=float, default=1.0, help="Mesh scale factor")
    parser.add_argument("--batch", action="store_true", help="Batch process directory")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        print("  Please train the model first using train_depth_anything_finetune.py")
        return
    
    model = load_model(model_path, device)
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = OUTPUT_ROOT / f"{timestamp}_depth_anything"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if input_path.is_file():
        # Single file
        process_single(model, input_path, output_dir, device, args.scale)
    elif args.batch or input_path.is_dir():
        # Batch process
        tiles = find_minimap_tiles(input_path)
        if not tiles:
            print(f"ERROR: No minimap tiles found in {input_path}")
            return
        
        print(f"Found {len(tiles)} tiles")
        
        success = 0
        for i, tile in enumerate(tiles):
            print(f"\n[{i+1}/{len(tiles)}]", end="")
            try:
                if process_single(model, tile['path'], output_dir, device, args.scale):
                    success += 1
            except Exception as e:
                print(f"  ERROR: {e}")
        
        print(f"\n{'='*50}")
        print(f"Batch complete: {success}/{len(tiles)} tiles")
    else:
        print(f"ERROR: Input not found: {input_path}")
    
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
