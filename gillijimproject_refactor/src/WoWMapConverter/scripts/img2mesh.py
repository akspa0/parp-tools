"""
WoW Image-to-3D Inference (Tiny ViT)
====================================
Uses the trained Tiny ViT Regressor to generate a 3D terrain mesh from an image.
Includes post-processing smoothing to reduce noise in AI-generated heightmaps.

Usage:
    python img2mesh.py <image_path> [output_obj_path] [options]

Examples:
    # Basic usage (outputs to test_output/rebaked_minimaps/)
    python img2mesh.py minimap_chunk.png
    
    # With smoothing (recommended for AI outputs)
    python img2mesh.py minimap_chunk.png --smooth 2 --sigma 1.0
    
    # Custom output directory
    python img2mesh.py minimap_chunk.png -o J:\\custom\\output
    
    # Laplacian smoothing (preserves terrain detail)
    python img2mesh.py minimap_chunk.png --smooth 3 --smooth-method laplacian

Options:
    --smooth N          Apply N iterations of smoothing (default: 0)
    --smooth-method M   Smoothing method: 'gaussian', 'laplacian', 'bilateral', 'median'
    --sigma S           Gaussian/bilateral sigma (default: 1.0)
    --output-dir, -o    Output directory (default: test_output/rebaked_minimaps)
    --model             Custom model path

Output:
    <input_name>.obj    3D mesh with WoW chunk vertex layout (145 vertices)
    <input_name>.mtl    Material file referencing input image as texture

"""

import sys
import json
import torch
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_PATH = r"j:\vlm_output\wow_tiny_vit_regressor"


def smooth_heightmap(heights_1d, method='gaussian', iterations=1, sigma=1.0):
    """
    Smooth a 145-element heightmap array.
    
    The heightmap is a special WoW format with 9 outer rows and 8 inner rows:
    - Outer rows: 9 vertices each (rows 0, 2, 4, 6, 8, 10, 12, 14, 16)
    - Inner rows: 8 vertices each (rows 1, 3, 5, 7, 9, 11, 13, 15)
    
    We convert to a 17x9 grid for smoothing, then back.
    """
    if iterations <= 0:
        return heights_1d
    
    # Reshape to 17x9 grid (outer and inner rows interleaved)
    # Row 0: 9 outer verts (indices 0-8)
    # Row 1: 8 inner verts (indices 9-16) + 1 padding
    # Row 2: 9 outer verts (indices 17-25)
    # etc.
    
    # Create a regular 17x9 grid by interpolating
    grid = np.zeros((17, 9), dtype=np.float32)
    
    idx = 0
    for row in range(9):
        # Outer row (9 verts)
        for col in range(9):
            grid[row * 2, col] = heights_1d[idx]
            idx += 1
        
        if row < 8:
            # Inner row (8 verts) - place at half positions
            for col in range(8):
                # Inner verts are between outer verts
                grid[row * 2 + 1, col] = heights_1d[idx]
                idx += 1
            # Extrapolate last column
            grid[row * 2 + 1, 8] = grid[row * 2 + 1, 7]
    
    # Apply smoothing
    try:
        from scipy import ndimage
        from scipy.ndimage import gaussian_filter, uniform_filter
    except ImportError:
        print("Warning: scipy not available, skipping smoothing")
        return heights_1d
    
    for _ in range(iterations):
        if method == 'gaussian':
            grid = gaussian_filter(grid, sigma=sigma)
        elif method == 'laplacian':
            # Laplacian smoothing: average of neighbors
            kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0
            smoothed = ndimage.convolve(grid, kernel, mode='reflect')
            # Blend with original (0.5 = half smooth)
            grid = grid * 0.5 + smoothed * 0.5
        elif method == 'bilateral':
            try:
                from skimage.restoration import denoise_bilateral
                grid = denoise_bilateral(grid, sigma_spatial=sigma, sigma_color=0.1)
            except ImportError:
                print("Warning: skimage not available, falling back to gaussian")
                grid = gaussian_filter(grid, sigma=sigma)
        elif method == 'median':
            from scipy.ndimage import median_filter
            grid = median_filter(grid, size=3)
        else:
            grid = gaussian_filter(grid, sigma=sigma)
    
    # Convert back to 145-element array
    result = []
    for row in range(9):
        # Outer row
        for col in range(9):
            result.append(float(grid[row * 2, col]))
        
        if row < 8:
            # Inner row
            for col in range(8):
                result.append(float(grid[row * 2 + 1, col]))
    
    return result

def generate_obj(heights, out_path, texture_path=None):
    # Heights: 145 float values.
    # Grid size: 33.3333 yards (Chunk size).
    UNIT_SIZE = 33.3333 / 8.0
    
    verts = []
    uvs = []
    
    # We will store (x, y, z) for OBJ.
    # WoW Coords: X (North), Y (West), Z (Up).
    
    idx = 0
    
    # Generating vertices & UVs
    for row in range(9):
        # Outer Row (9 verts)
        y = row * UNIT_SIZE
        # UV V coord: Row 0 is Top (V=1), Row 8 is Bottom (V=0)
        v_coord = 1.0 - (row / 8.0)
        
        for col in range(9):
            x = col * UNIT_SIZE
            z = heights[idx]
            idx += 1
            verts.append((x, y, z)) 
            
            # UV U coord: Col 0 is Left (U=0), Col 8 is Right (U=1)
            u_coord = col / 8.0
            uvs.append((u_coord, v_coord))
            
        if row < 8:
            # Inner Row (8 verts)
            y_inner = y + (UNIT_SIZE / 2.0)
            v_coord_inner = 1.0 - ((row + 0.5) / 8.0)
            
            for col in range(8):
                x_inner = (col * UNIT_SIZE) + (UNIT_SIZE / 2.0)
                z = heights[idx]
                idx += 1
                verts.append((x_inner, y_inner, z))
                
                u_coord_inner = (col + 0.5) / 8.0
                uvs.append((u_coord_inner, v_coord_inner))

    faces = []
    
    # Indices calculation:
    # Row R Outer Start: R * (9+8) = 17*R
    # Row R Inner Start: 17*R + 9
    
    for r in range(8):
        row_start = r * 17
        next_row_start = (r+1) * 17
        inner_start = row_start + 9
        
        for c in range(8):
            # Indices (1-based for OBJ loop later)
            tl = row_start + c + 1 
            tr = row_start + c + 1 + 1
            bl = next_row_start + c + 1
            br = next_row_start + c + 1 + 1
            ct = inner_start + c + 1
            
            # 4 Triangles per square
            faces.append((tl, tr, ct))
            faces.append((tr, br, ct))
            faces.append((br, bl, ct))
            faces.append((bl, tl, ct))

    # MTL Generation
    mtl_filename = None
    if texture_path:
        mtl_path = Path(out_path).with_suffix(".mtl")
        mtl_filename = mtl_path.name
        
        # Ensure texture path is relative or absolute? 
        # MTL usually likes relative paths or just filenames if in same dir.
        # For simplicity, we assume user keeps texture relative or we write absolute.
        # Absolute paths in MTL can be finicky. 
        # Let's try to use the name and assume it's next to the obj? 
        # Or just write absolute path.
        tex_abs = Path(texture_path).resolve()
        
        with open(mtl_path, 'w') as f:
            f.write(f"newmtl TerrainMat\n")
            f.write(f"Ka 1.0 1.0 1.0\n")
            f.write(f"Kd 1.0 1.0 1.0\n")
            f.write(f"d 1.0\n")
            f.write(f"illum 0\n")
            f.write(f"map_Kd {tex_abs}\n")
            
        print(f"Saved MTL to {mtl_path}")

    # Write OBJ
    with open(out_path, 'w') as f:
        f.write(f"# WoW Terrain Chunk (AI Generated)\n")
        if mtl_filename:
            f.write(f"mtllib {mtl_filename}\n")
            f.write(f"usemtl TerrainMat\n")
            
        for i, v in enumerate(verts):
            f.write(f"v {v[0]:.4f} {v[2]:.4f} {-v[1]:.4f}\n") # Y-Up flip
        
        for u in uvs:
            f.write(f"vt {u[0]:.4f} {u[1]:.4f}\n")
            
        for p in faces:
            # f v1/vt1 v2/vt2 v3/vt3
            f.write(f"f {p[0]}/{p[0]} {p[1]}/{p[1]} {p[2]}/{p[2]}\n")
            
    print(f"Saved OBJ to {out_path}")

DEFAULT_OUTPUT_DIR = r"j:\wowDev\parp-tools\gillijimproject_refactor\test_output\rebaked_minimaps"


def main():
    parser = argparse.ArgumentParser(
        description="WoW Image-to-3D Inference - Generate terrain mesh from image using Tiny ViT"
    )
    parser.add_argument("image", help="Input image path")
    parser.add_argument("output", nargs="?", help="Output OBJ path (default: output_dir/<input_name>.obj)")
    parser.add_argument("--output-dir", "-o", default=DEFAULT_OUTPUT_DIR,
                        help=f"Output directory (default: {DEFAULT_OUTPUT_DIR})")
    parser.add_argument("--smooth", type=int, default=0, 
                        help="Number of smoothing iterations (default: 0)")
    parser.add_argument("--smooth-method", choices=['gaussian', 'laplacian', 'bilateral', 'median'],
                        default='gaussian', help="Smoothing method (default: gaussian)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Gaussian/bilateral sigma (default: 1.0)")
    parser.add_argument("--model", default=MODEL_PATH,
                        help=f"Model path (default: {MODEL_PATH})")
    
    args = parser.parse_args()
    
    img_path = args.image
    
    # Determine output path
    if args.output:
        out_path = Path(args.output)
    else:
        # Use output directory + input filename with .obj extension
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(img_path).stem + ".obj")
    
    model_path = args.model
    
    print(f"Loading model from {model_path}...")
    try:
        model = ViTForImageClassification.from_pretrained(model_path)
        processor = ViTImageProcessor.from_pretrained(model_path)
        
        # Load Stats
        stats_path = Path(model_path) / "normalization_stats.json"
        with open(stats_path, 'r') as f:
            stats = json.load(f)
            mean = torch.tensor(stats["mean"])
            std = torch.tensor(stats["std"])
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return

    print("Processing Image...")
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    print("Running Inference...")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits # Regression output
        
    # De-normalize
    # y = (x - mean) / std  ->  x = y * std + mean
    pred_heights = (logits[0] * std) + mean
    pred_heights = pred_heights.tolist()
    
    # Apply smoothing if requested
    if args.smooth > 0:
        print(f"Applying {args.smooth}x {args.smooth_method} smoothing (sigma={args.sigma})...")
        pred_heights = smooth_heightmap(
            pred_heights, 
            method=args.smooth_method, 
            iterations=args.smooth, 
            sigma=args.sigma
        )
    
    print("Generating Mesh...")
    # Pass img_path as texture_path
    generate_obj(pred_heights, out_path, texture_path=img_path)
    
    print(f"Done! Output: {out_path}")

if __name__ == "__main__":
    main()
