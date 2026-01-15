"""
WoW Image-to-3D Inference (Tiny ViT)
====================================
Uses the trained Tiny ViT Regressor to generate a 3D terrain mesh from an image.

Usage:
    python img2mesh.py <image_path> [output_obj_path]

"""

import sys
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

MODEL_PATH = r"j:\vlm_output\wow_tiny_vit_regressor"

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

def main():
    if len(sys.argv) < 2:
        print("Usage: python img2mesh.py <image_path>")
        return
        
    img_path = sys.argv[1]
    out_path = sys.argv[2] if len(sys.argv) > 2 else Path(img_path).with_suffix(".obj")
    
    print(f"Loading model from {MODEL_PATH}...")
    try:
        model = ViTForImageClassification.from_pretrained(MODEL_PATH)
        processor = ViTImageProcessor.from_pretrained(MODEL_PATH)
        
        # Load Stats
        stats_path = Path(MODEL_PATH) / "normalization_stats.json"
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
    
    print("Generating Mesh...")
    # Pass img_path as texture_path
    generate_obj(pred_heights, out_path, texture_path=img_path)
    
    # Also dump JSON if needed
    # with open(str(out_path) + ".json", 'w') as f:
    #    json.dump(pred_heights, f)

if __name__ == "__main__":
    main()
