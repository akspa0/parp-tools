#!/usr/bin/env python3
"""
WoW Height Regressor V5 - Inference Script

Takes minimap + normalmap as input, outputs heightmap and OBJ mesh.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Import model from training script
from train_height_regressor_v5_multichannel import MultiChannelUNet, INPUT_SIZE, OUTPUT_SIZE

# Default model path
DEFAULT_MODEL = Path(r"J:\vlm_output\wow_height_regressor_v5_multichannel\best_model.pt")
OUTPUT_ROOT = Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_output\model_outputs")


def load_model(model_path: Path, device: torch.device):
    """Load trained model."""
    model = MultiChannelUNet().to(device)
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from {model_path}")
    print(f"  Best val_loss: {checkpoint.get('best_val_loss', 'N/A')}")
    print(f"  Epoch: {checkpoint.get('epoch', 'N/A')}")
    
    return model


def predict(model, minimap_path: Path, normalmap_path: Path, device: torch.device):
    """Run inference on minimap + normalmap."""
    # Load images
    minimap = Image.open(minimap_path).convert("RGB")
    normalmap = Image.open(normalmap_path).convert("RGB")
    
    # Resize to input size
    minimap = minimap.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    normalmap = normalmap.resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
    
    # Convert to tensors
    to_tensor = transforms.ToTensor()
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    
    minimap_t = normalize(to_tensor(minimap))
    normalmap_t = normalize(to_tensor(normalmap))
    
    # Concatenate to 6 channels
    input_tensor = torch.cat([minimap_t, normalmap_t], dim=0).unsqueeze(0).to(device)
    
    # Inference
    with torch.no_grad():
        output = model(input_tensor)
    
    # Convert to numpy [129, 129]
    heightmap = output.squeeze().cpu().numpy()
    
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


def generate_obj_mesh(heightmap: np.ndarray, output_path: Path, scale: float = 1.0):
    """Generate OBJ mesh from 129x129 heightmap."""
    h, w = heightmap.shape
    vertices = []
    faces = []
    
    # Generate vertices
    for y in range(h):
        for x in range(w):
            # Scale to WoW coordinates (roughly)
            wx = (x - w/2) * scale
            wy = (y - h/2) * scale
            wz = heightmap[y, x] * scale * 20  # Scale height (reduced from 100)
            vertices.append((wx, wz, wy))  # Y-up in OBJ
    
    # Generate faces (quads as triangles)
    for y in range(h - 1):
        for x in range(w - 1):
            # Vertex indices (1-based for OBJ)
            v0 = y * w + x + 1
            v1 = y * w + (x + 1) + 1
            v2 = (y + 1) * w + (x + 1) + 1
            v3 = (y + 1) * w + x + 1
            
            # Two triangles per quad
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))
    
    # Write OBJ
    with open(output_path, 'w') as f:
        f.write(f"# WoW Height Regressor V5 Output\n")
        f.write(f"# Vertices: {len(vertices)}\n")
        f.write(f"# Faces: {len(faces)}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        
        f.write("\n")
        
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    print(f"  Saved mesh: {len(vertices)} vertices, {len(faces)} faces")


def main():
    parser = argparse.ArgumentParser(description="WoW Height Regressor V5 Inference")
    parser.add_argument("minimap", type=str, help="Path to minimap image")
    parser.add_argument("--normalmap", type=str, help="Path to normalmap image (default: auto-detect)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to model checkpoint")
    parser.add_argument("--output", type=str, help="Output directory")
    parser.add_argument("--scale", type=float, default=1.0, help="Mesh scale factor")
    args = parser.parse_args()
    
    minimap_path = Path(args.minimap)
    if not minimap_path.exists():
        print(f"ERROR: Minimap not found: {minimap_path}")
        return
    
    # Auto-detect normalmap
    if args.normalmap:
        normalmap_path = Path(args.normalmap)
    else:
        # Try to find normalmap in same directory
        tile_name = minimap_path.stem
        normalmap_path = minimap_path.parent / f"{tile_name}_normalmap.png"
    
    if not normalmap_path.exists():
        print(f"ERROR: Normalmap not found: {normalmap_path}")
        print("  Use --normalmap to specify path")
        return
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        tile_name = minimap_path.stem
        output_dir = OUTPUT_ROOT / f"{timestamp}_{tile_name}_v5"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return
    
    model = load_model(model_path, device)
    
    # Run inference
    print(f"\nProcessing: {minimap_path.name}")
    print(f"  Normalmap: {normalmap_path.name}")
    
    heightmap = predict(model, minimap_path, normalmap_path, device)
    print(f"  Output shape: {heightmap.shape}")
    print(f"  Height range: [{heightmap.min():.4f}, {heightmap.max():.4f}]")
    
    # Save outputs
    tile_name = minimap_path.stem
    
    # Heightmap
    heightmap_path = output_dir / f"{tile_name}_heightmap.png"
    save_heightmap_image(heightmap, heightmap_path)
    print(f"  Saved heightmap: {heightmap_path}")
    
    # OBJ mesh
    obj_path = output_dir / f"{tile_name}.obj"
    generate_obj_mesh(heightmap, obj_path, scale=args.scale)
    
    # Copy inputs for reference
    minimap_copy = output_dir / f"{tile_name}_minimap.png"
    normalmap_copy = output_dir / f"{tile_name}_normalmap.png"
    Image.open(minimap_path).save(minimap_copy)
    Image.open(normalmap_path).save(normalmap_copy)
    
    # Save metadata
    metadata = {
        "tile_name": tile_name,
        "minimap": str(minimap_path),
        "normalmap": str(normalmap_path),
        "model": str(model_path),
        "output_dir": str(output_dir),
        "heightmap_shape": list(heightmap.shape),
        "height_min": float(heightmap.min()),
        "height_max": float(heightmap.max()),
        "height_mean": float(heightmap.mean()),
        "height_std": float(heightmap.std()),
        "model_version": "v5_multichannel",
    }
    
    metadata_path = output_dir / f"{tile_name}.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"\nOutput saved to: {output_dir}")


if __name__ == "__main__":
    main()
