#!/usr/bin/env python3
"""
Batch Terrain Restoration Tool

Processes folders of minimap tiles to generate heightmaps and OBJ meshes.
Works with raw minimap folders - no dataset JSON required.

For tiles without normalmaps, generates a flat normalmap automatically.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import re

import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np

# Import model from training script
from train_height_regressor_v5_multichannel import MultiChannelUNet, INPUT_SIZE, OUTPUT_SIZE

# Default model path
DEFAULT_MODEL = Path(r"J:\vlm_output\wow_height_regressor_v5_multichannel\best_model.pt")


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


def create_flat_normalmap(size: int = 256) -> Image.Image:
    """Create a flat normalmap (pointing straight up) for tiles without one."""
    # Normal pointing up: (0, 0, 1) -> RGB (128, 128, 255)
    normal_data = np.zeros((size, size, 3), dtype=np.uint8)
    normal_data[:, :, 0] = 128  # X = 0
    normal_data[:, :, 1] = 128  # Y = 0
    normal_data[:, :, 2] = 255  # Z = 1
    return Image.fromarray(normal_data, mode='RGB')


def predict(model, minimap: Image.Image, normalmap: Image.Image, device: torch.device) -> np.ndarray:
    """Run inference on minimap + normalmap images."""
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


def generate_obj_mesh(heightmap: np.ndarray, output_path: Path, texture_path: Path = None, scale: float = 1.0):
    """Generate OBJ mesh from 129x129 heightmap with optional texture."""
    h, w = heightmap.shape
    vertices = []
    uvs = []
    faces = []
    
    # Generate vertices and UVs
    for y in range(h):
        for x in range(w):
            # Scale to WoW coordinates (roughly)
            wx = (x - w/2) * scale
            wy = (y - h/2) * scale
            wz = heightmap[y, x] * scale * 20  # Scale height
            vertices.append((wx, wz, wy))  # Y-up in OBJ
            
            # UV coordinates with half-pixel inset to prevent edge sampling artifacts
            half_pixel = 0.5 / 256  # Assuming 256x256 texture
            u = half_pixel + (x / (w - 1)) * (1.0 - 2 * half_pixel)
            v = half_pixel + (1.0 - (y / (h - 1))) * (1.0 - 2 * half_pixel)  # Flip V
            uvs.append((u, v))
    
    # Generate faces (quads as triangles) - reversed winding for correct normals
    for y in range(h - 1):
        for x in range(w - 1):
            # Vertex indices (1-based for OBJ)
            v0 = y * w + x + 1
            v1 = y * w + (x + 1) + 1
            v2 = (y + 1) * w + (x + 1) + 1
            v3 = (y + 1) * w + x + 1
            
            # Two triangles per quad - reversed winding order (CCW -> CW)
            faces.append((v0, v3, v2))
            faces.append((v0, v2, v1))
    
    # Write MTL file if texture provided
    mtl_name = None
    if texture_path and texture_path.exists():
        mtl_path = output_path.with_suffix('.mtl')
        mtl_name = output_path.stem + "_mat"
        
        # Copy texture to output directory
        tex_dest = output_path.parent / texture_path.name
        if not tex_dest.exists():
            Image.open(texture_path).save(tex_dest)
        
        with open(mtl_path, 'w') as f:
            f.write(f"# WoW Terrain Restoration Material\n")
            f.write(f"newmtl {mtl_name}\n")
            f.write(f"Ka 1.0 1.0 1.0\n")
            f.write(f"Kd 1.0 1.0 1.0\n")
            f.write(f"Ks 0.0 0.0 0.0\n")
            f.write(f"d 1.0\n")
            f.write(f"illum 1\n")
            f.write(f"map_Kd {texture_path.name}\n")
    
    # Write OBJ
    with open(output_path, 'w') as f:
        f.write(f"# WoW Terrain Restoration Output\n")
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


def find_minimap_tiles(input_dir: Path) -> list:
    """Find all minimap tiles in directory, excluding normalmaps/heightmaps."""
    tiles = []
    
    # Pattern to match tile names like Map_X_Y.png or MapName_X_Y.png
    tile_pattern = re.compile(r'^(.+?)_(\d+)_(\d+)\.png$', re.IGNORECASE)
    
    for f in input_dir.glob("*.png"):
        # Skip derived files
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


def process_tile(tile_info: dict, model, device: torch.device, output_dir: Path, 
                 scale: float, generate_obj: bool, generate_dataset: bool,
                 flat_normalmap: Image.Image, height_scale: float = 100.0):
    """Process a single tile. All outputs go to single output_dir (no subfolders)."""
    tile_path = tile_info['path']
    tile_name = tile_info['name']
    
    try:
        # Load minimap
        minimap = Image.open(tile_path).convert("RGB")
        
        # Try to find normalmap
        normalmap_path = tile_path.parent / f"{tile_name}_normalmap.png"
        if normalmap_path.exists():
            normalmap = Image.open(normalmap_path).convert("RGB")
        else:
            # Use flat normalmap
            normalmap = flat_normalmap
        
        # Run inference
        heightmap = predict(model, minimap, normalmap, device)
        
        # All files go directly in output_dir (no subfolders)
        # Save heightmap
        heightmap_path = output_dir / f"{tile_name}_heightmap.png"
        save_heightmap_image(heightmap, heightmap_path)
        
        # Save OBJ if requested
        if generate_obj:
            obj_path = output_dir / f"{tile_name}.obj"
            generate_obj_mesh(heightmap, obj_path, texture_path=tile_path, scale=scale)
        
        # Save dataset JSON if requested
        if generate_dataset:
            from heightmap_to_dataset import heightmap_to_dataset_json
            dataset_json = heightmap_to_dataset_json(
                heightmap, tile_name, 
                minimap_path=tile_path,
                height_scale=height_scale
            )
            json_path = output_dir / f"{tile_name}.json"
            with open(json_path, 'w') as f:
                json.dump(dataset_json, f, indent=2)
        
        # Copy minimap for reference
        minimap_copy = output_dir / f"{tile_name}_minimap.png"
        minimap.save(minimap_copy)
        
        return {'name': tile_name, 'success': True}
        
    except Exception as e:
        return {'name': tile_name, 'success': False, 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Batch Terrain Restoration - Process folders of minimap tiles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Process all tiles in a folder
  python batch_terrain_restore.py /path/to/minimaps --output /path/to/output
  
  # Process with custom scale
  python batch_terrain_restore.py /path/to/minimaps --scale 2.0
  
  # Heightmaps only (no OBJ meshes)
  python batch_terrain_restore.py /path/to/minimaps --no-obj
  
  # Generate dataset JSON for ADT conversion
  python batch_terrain_restore.py /path/to/minimaps --dataset
  
  # Use specific model
  python batch_terrain_restore.py /path/to/minimaps --model /path/to/model.pt
"""
    )
    parser.add_argument("input_dir", type=str, help="Directory containing minimap tiles")
    parser.add_argument("--output", type=str, help="Output directory (default: input_dir/restored)")
    parser.add_argument("--model", type=str, default=str(DEFAULT_MODEL), help="Path to model checkpoint")
    parser.add_argument("--scale", type=float, default=1.0, help="Mesh scale factor")
    parser.add_argument("--height-scale", type=float, default=100.0, help="Height scale for dataset JSON")
    parser.add_argument("--no-obj", action="store_true", help="Skip OBJ mesh generation")
    parser.add_argument("--dataset", action="store_true", help="Generate dataset JSON for ADT conversion")
    args = parser.parse_args()
    
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        print(f"ERROR: Input directory not found: {input_dir}")
        return
    
    # Output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = input_dir / f"restored_{timestamp}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Find tiles
    tiles = find_minimap_tiles(input_dir)
    if not tiles:
        print(f"ERROR: No minimap tiles found in {input_dir}")
        print("  Expected format: MapName_X_Y.png")
        return
    
    print(f"Found {len(tiles)} minimap tiles")
    
    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"ERROR: Model not found: {model_path}")
        return
    
    model = load_model(model_path, device)
    
    # Create flat normalmap for tiles without one
    flat_normalmap = create_flat_normalmap()
    
    # Process tiles
    print(f"\nProcessing {len(tiles)} tiles...")
    print(f"Output: {output_dir}")
    print(f"Generate OBJ: {not args.no_obj}")
    print(f"Generate Dataset JSON: {args.dataset}")
    print()
    
    results = {'success': 0, 'failed': 0, 'errors': []}
    
    for i, tile in enumerate(tiles):
        print(f"[{i+1}/{len(tiles)}] {tile['name']}...", end=" ", flush=True)
        
        result = process_tile(
            tile, model, device, output_dir,
            scale=args.scale,
            generate_obj=not args.no_obj,
            generate_dataset=args.dataset,
            flat_normalmap=flat_normalmap,
            height_scale=args.height_scale
        )
        
        if result['success']:
            print("OK")
            results['success'] += 1
        else:
            print(f"FAILED: {result.get('error', 'Unknown error')}")
            results['failed'] += 1
            results['errors'].append(result)
    
    # Summary
    print(f"\n{'='*50}")
    print(f"Batch processing complete!")
    print(f"  Success: {results['success']}/{len(tiles)}")
    print(f"  Failed:  {results['failed']}/{len(tiles)}")
    print(f"  Output:  {output_dir}")
    
    # Save summary
    summary = {
        'input_dir': str(input_dir),
        'output_dir': str(output_dir),
        'model': str(model_path),
        'total_tiles': len(tiles),
        'success': results['success'],
        'failed': results['failed'],
        'errors': results['errors'],
        'tiles': [t['name'] for t in tiles],
    }
    
    summary_path = output_dir / "batch_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"  Summary: {summary_path}")


if __name__ == "__main__":
    main()
