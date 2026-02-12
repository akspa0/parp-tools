
import os
import torch
import torchvision.transforms.functional as TF
import numpy as np
from pathlib import Path
from PIL import Image
import re
from train_v7_6 import MultiHeadUNet  # Revert to V7 (ResNet34)

# --- Configuration ---
MODEL_PATH = Path("output_v7_6/checkpoints/latest.pth") # V7 Checkpoint
INPUT_DIR = Path("inference_input")
OUTPUT_DIR = Path("stitched_output_v7_restore") # New clean output
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TILE_SIZE = 512
GRID_SIZE = 64
FULL_SIZE = TILE_SIZE * GRID_SIZE  # 32768

# Regex to parse MapName_X_Y.png or MapName_X_Y_vcol.png
PATTERN = re.compile(r"(.+)_(\d{1,2})_(\d{1,2})(?:_vcol)?\.png$")

def stitch_maps():
    print(f"Loading Model from {MODEL_PATH}...")
    model = MultiHeadUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()

    # Scan Files
    files = list(INPUT_DIR.glob("*.png"))
    map_groups = {}
    
    for p in files:
        m = PATTERN.match(p.name)
        if m:
            map_name, x, y = m.groups()
            x, y = int(x), int(y)
            if map_name not in map_groups:
                map_groups[map_name] = []
            map_groups[map_name].append((x, y, p))
            
    if not map_groups:
        print("No valid MapName_X_Y.png files found in inference_input/")
        return

    for map_name, tiles in map_groups.items():
        print(f"Stitching Map: {map_name} ({len(tiles)} tiles)...")
        
        print("Allocating Albedo Canvas (32k x 32k RGB)...")
        full_albedo = Image.new("RGB", (FULL_SIZE, FULL_SIZE), (0, 0, 0))
        
        print("Allocating Height Canvas (32k x 32k I16)...")
        full_height = Image.new("I;16", (FULL_SIZE, FULL_SIZE), 0) # 0 = Base Level

        # Process Tiles
        for x, y, p in tiles:
            print(f"  Processing tile {x}, {y}...")
            
            # Load & Inference
            img = Image.open(p).convert("RGB")
            if img.size != (512, 512):
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                
            input_tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE)
            
            with torch.no_grad(), torch.amp.autocast('cuda'):
                pred_h, pred_a = model(input_tensor)
                
            # --- Post Process ---
            
            # Height: Apply Smoothing (v7.6 Standard)
            pred_h = TF.gaussian_blur(pred_h, kernel_size=[5, 5], sigma=[1.0, 1.0])
            
            h_np = pred_h.squeeze().float().cpu().numpy()
            h_16bit = (h_np * 65535).astype(np.uint16)
            tile_h_img = Image.fromarray(h_16bit, mode="I;16")
            
            # Albedo
            tile_a_img = TF.to_pil_image(pred_a.squeeze().cpu())
            
            # Paste into Quilt
            px = x * TILE_SIZE
            py = y * TILE_SIZE
            
            full_albedo.paste(tile_a_img, (px, py))
            full_height.paste(tile_h_img, (px, py))
            
            # --- Per-Tile OBJ Export (User Requested) ---
            objs_dir = OUTPUT_DIR / "objs"
            objs_dir.mkdir(exist_ok=True)
            
            tile_base = f"{map_name}_{x}_{y}"
            
            # Save Texture for MTL
            tex_path = objs_dir / f"{tile_base}.png"
            tile_a_img.save(tex_path)
            
            # Generate OBJ (512x512)
            generate_obj(
                height_map=h_16bit,
                albedo_filename=tex_path.name,
                output_path=objs_dir / f"{tile_base}.obj",
                mat_path=objs_dir / f"{tile_base}.mtl",
                stride_scale=1 # Full Res
            )

        # Save Big Files (Reference)
        print(f"Saving {map_name}_Full_Albedo.png...")
        full_albedo.save(OUTPUT_DIR / f"{map_name}_Full_Albedo.png")
        
        print(f"Saving {map_name}_Full_Height.png...")
        full_height.save(OUTPUT_DIR / f"{map_name}_Full_Height.png")
        
        print(f"Done: {map_name}")

def generate_obj(height_map, albedo_filename, output_path, mat_path, stride_scale=1):
    h, w = height_map.shape
    PIXEL_SCALE = 533.3333 / 512.0
    scale_xz = PIXEL_SCALE * stride_scale
    MAX_HEIGHT = 1200.0
    scale_y = MAX_HEIGHT / 65535.0
    
    with open(output_path, 'w') as f:
        f.write(f"mtllib {mat_path.name}\n")
        
        for y in range(h):
            for x in range(w):
                px = x * scale_xz
                pz = -y * scale_xz 
                py = height_map[y, x] * scale_y
                f.write(f"v {px:.4f} {py:.4f} {pz:.4f}\n")
                
                tu = x / (w - 1)
                tv = 1 - (y / (h - 1))
                f.write(f"vt {tu:.4f} {tv:.4f}\n")
                
        f.write("usemtl TerrainMat\n")
        
        for y in range(h - 1):
            for x in range(w - 1):
                i0 = y * w + x + 1
                i1 = i0 + 1
                i2 = (y + 1) * w + x + 2
                i3 = (y + 1) * w + x + 1
                f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2} {i3}/{i3}\n")

    with open(mat_path, 'w') as m:
        m.write("newmtl TerrainMat\nKa 1 1 1\nKd 1 1 1\nKs 0 0 0\n")
        m.write(f"map_Kd {albedo_filename}\n")

if __name__ == "__main__":
    stitch_maps()
