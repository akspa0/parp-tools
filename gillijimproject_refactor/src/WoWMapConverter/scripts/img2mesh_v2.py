"""
WoW Image-to-3D Inference V2
============================
Uses the improved Height Regressor V2 model with global normalization.

Usage:
    python img2mesh_v2.py <image_path> [options]

Options:
    --smooth N          Apply N iterations of smoothing (default: 0)
    --smooth-method M   Smoothing method: 'gaussian', 'laplacian', 'bilateral', 'median'
    --sigma S           Gaussian/bilateral sigma (default: 1.0)
    --output-dir, -o    Output directory (default: test_output/rebaked_minimaps)
    --model             Custom model path
"""

import sys
import json
import torch
import torch.nn as nn
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import ViTModel, ViTImageProcessor

MODEL_PATH = Path(r"j:\vlm_output\wow_height_regressor_v2")
DEFAULT_OUTPUT_DIR = Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_output\rebaked_minimaps")

NUM_HEIGHTS = 145


class HeightRegressionHead(nn.Module):
    def __init__(self, hidden_size=768):
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, NUM_HEIGHTS)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, x):
        x = self.dropout(self.relu(self.fc1(x)))
        x = self.dropout(self.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


class HeightRegressorModel(nn.Module):
    def __init__(self, model_name="google/vit-base-patch16-224"):
        super().__init__()
        self.vit = ViTModel.from_pretrained(model_name)
        self.head = HeightRegressionHead(self.vit.config.hidden_size)
        
    def forward(self, pixel_values):
        outputs = self.vit(pixel_values=pixel_values)
        cls_output = outputs.last_hidden_state[:, 0]
        heights = self.head(cls_output)
        return heights


def smooth_heightmap(heights_1d, method='gaussian', iterations=1, sigma=1.0):
    """Smooth a 145-element heightmap array."""
    if iterations <= 0:
        return heights_1d
    
    # Create a 17x9 grid for smoothing
    grid = np.zeros((17, 9), dtype=np.float32)
    
    idx = 0
    for row in range(9):
        for col in range(9):
            grid[row * 2, col] = heights_1d[idx]
            idx += 1
        
        if row < 8:
            for col in range(8):
                grid[row * 2 + 1, col] = heights_1d[idx]
                idx += 1
            grid[row * 2 + 1, 8] = grid[row * 2 + 1, 7]
    
    try:
        from scipy import ndimage
        from scipy.ndimage import gaussian_filter
    except ImportError:
        print("Warning: scipy not available, skipping smoothing")
        return heights_1d
    
    for _ in range(iterations):
        if method == 'gaussian':
            grid = gaussian_filter(grid, sigma=sigma)
        elif method == 'laplacian':
            kernel = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]]) / 4.0
            smoothed = ndimage.convolve(grid, kernel, mode='reflect')
            grid = grid * 0.5 + smoothed * 0.5
        elif method == 'median':
            from scipy.ndimage import median_filter
            grid = median_filter(grid, size=3)
        else:
            grid = gaussian_filter(grid, sigma=sigma)
    
    # Convert back to 145-element array
    result = []
    for row in range(9):
        for col in range(9):
            result.append(float(grid[row * 2, col]))
        if row < 8:
            for col in range(8):
                result.append(float(grid[row * 2 + 1, col]))
    
    return result


def generate_obj(heights, out_path, texture_path=None):
    """Generate OBJ mesh from 145 height values."""
    UNIT_SIZE = 33.3333 / 8.0
    
    verts = []
    uvs = []
    
    idx = 0
    for row in range(9):
        y = row * UNIT_SIZE
        v_coord = 1.0 - (row / 8.0)
        
        for col in range(9):
            x = col * UNIT_SIZE
            z = heights[idx]
            idx += 1
            verts.append((x, y, z))
            u_coord = col / 8.0
            uvs.append((u_coord, v_coord))
            
        if row < 8:
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
    for r in range(8):
        row_start = r * 17
        next_row_start = (r+1) * 17
        inner_start = row_start + 9
        
        for c in range(8):
            tl = row_start + c + 1
            tr = row_start + c + 2
            bl = next_row_start + c + 1
            br = next_row_start + c + 2
            ct = inner_start + c + 1
            
            faces.append((tl, tr, ct))
            faces.append((tr, br, ct))
            faces.append((br, bl, ct))
            faces.append((bl, tl, ct))

    # MTL
    mtl_filename = None
    if texture_path:
        mtl_path = Path(out_path).with_suffix(".mtl")
        mtl_filename = mtl_path.name
        tex_abs = Path(texture_path).resolve()
        
        with open(mtl_path, 'w') as f:
            f.write(f"newmtl TerrainMat\n")
            f.write(f"Ka 1.0 1.0 1.0\n")
            f.write(f"Kd 1.0 1.0 1.0\n")
            f.write(f"d 1.0\n")
            f.write(f"illum 0\n")
            f.write(f"map_Kd {tex_abs}\n")

    # OBJ
    with open(out_path, 'w') as f:
        f.write(f"# WoW Terrain Chunk (AI Generated V2)\n")
        if mtl_filename:
            f.write(f"mtllib {mtl_filename}\n")
            f.write(f"usemtl TerrainMat\n")
            
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[2]:.4f} {-v[1]:.4f}\n")
        
        for u in uvs:
            f.write(f"vt {u[0]:.4f} {u[1]:.4f}\n")
            
        for p in faces:
            f.write(f"f {p[0]}/{p[0]} {p[1]}/{p[1]} {p[2]}/{p[2]}\n")
            
    print(f"Saved OBJ to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="WoW Image-to-3D V2 - Improved height prediction")
    parser.add_argument("image", help="Input image path")
    parser.add_argument("output", nargs="?", help="Output OBJ path")
    parser.add_argument("--output-dir", "-o", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--smooth", type=int, default=1, help="Smoothing iterations (default: 1)")
    parser.add_argument("--smooth-method", choices=['gaussian', 'laplacian', 'median'], default='gaussian')
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--model", default=str(MODEL_PATH))
    
    args = parser.parse_args()
    
    img_path = args.image
    model_path = Path(args.model)
    
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / (Path(img_path).stem + ".obj")
    
    # Load model
    print(f"Loading model from {model_path}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load stats
    stats_path = model_path / "normalization_stats.json"
    if not stats_path.exists():
        print(f"Error: Stats file not found at {stats_path}")
        print("Make sure you've trained the V2 model first!")
        return
    
    with open(stats_path, 'r') as f:
        stats = json.load(f)
    
    global_min = stats["global_min"]
    global_max = stats["global_max"]
    
    # Load model
    model = HeightRegressorModel()
    
    checkpoint_path = model_path / "best_model.pt"
    if not checkpoint_path.exists():
        checkpoint_path = model_path / "final_model.pt"
    
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint)
    else:
        print(f"Error: No model checkpoint found at {model_path}")
        return
    
    model = model.to(device)
    model.eval()
    
    # Load processor
    processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")
    
    # Process image
    print("Processing image...")
    image = Image.open(img_path).convert("RGB")
    inputs = processor(images=image, return_tensors="pt")
    
    # Inference
    print("Running inference...")
    with torch.no_grad():
        outputs = model(inputs["pixel_values"].to(device))
        pred_norm = outputs[0].cpu().numpy()
    
    # Denormalize: from [-1, 1] to world units
    pred_heights = (pred_norm + 1.0) / 2.0 * (global_max - global_min) + global_min
    pred_heights = pred_heights.tolist()
    
    # Apply smoothing
    if args.smooth > 0:
        print(f"Applying {args.smooth}x {args.smooth_method} smoothing...")
        pred_heights = smooth_heightmap(pred_heights, args.smooth_method, args.smooth, args.sigma)
    
    # Generate mesh
    print("Generating mesh...")
    generate_obj(pred_heights, out_path, texture_path=img_path)
    
    print(f"Done! Output: {out_path}")


if __name__ == "__main__":
    main()
