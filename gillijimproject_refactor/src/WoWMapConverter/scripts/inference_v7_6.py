
import os
import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from pathlib import Path
from PIL import Image
import numpy as np

# --- Configuration ---
MODEL_PATH = Path("output_v7_6/checkpoints/latest.pth")
INPUT_DIR = Path("inference_input")
OUTPUT_DIR = Path("inference_output")
INPUT_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Model Definition (Must match correct V7.6 Architecture) ---
# Copying the corrected MultiHeadUNet class to ensure compatibility
# Ideally, this should be imported, but for a standalone script, I'll redefine or import if possible.
# To avoid duplication drift, I will try to import from train_v7_6 if it's in the same folder.
# But `train_v7_6.py` has code that runs on import (if not guarded). The previous write had `if __name__ == "__main__":`.
# So import is safe.

try:
    from train_v7_6 import MultiHeadUNet
except ImportError:
    print("Could not import MultiHeadUNet from train_v7_6.py. Please ensure it is in the same directory.")
    exit(1)

def run_inference():
    print(f"Loading model from {MODEL_PATH}...")
    model = MultiHeadUNet().to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    model.eval()
    
    input_files = list(INPUT_DIR.glob("*.png"))
    print(f"Found {len(input_files)} images in {INPUT_DIR}")
    
    if not input_files:
        print("No images found! Please copy some 512x512 minimaps (.png) into 'inference_input/'.")
        return

    for img_path in input_files:
        try:
            # Load and Preprocess
            img = Image.open(img_path).convert("RGB") # Force RGB (3 channel)
            
            # Smart Resize if needed (User wants to test "any" tile)
            if img.size != (512, 512):
                print(f"Resizing {img_path.name} from {img.size} to (512, 512)")
                img = img.resize((512, 512), Image.Resampling.LANCZOS)
                
            input_tensor = TF.to_tensor(img).unsqueeze(0).to(DEVICE) # (1, 3, 512, 512)
            
            # Predict
            with torch.no_grad(), torch.amp.autocast('cuda'):
                pred_h, pred_a = model(input_tensor)
            
            base_name = img_path.stem

            # Post-process Height (1 Channel) -> 16-bit Uint
            # Apply Gaussian Smoothing (User Request)
            # Kernel 5x5, Sigma 1.0 provides nice noise reduction without destroying mountains
            pred_h = TF.gaussian_blur(pred_h, kernel_size=[5, 5], sigma=[1.0, 1.0])

            # Model output is 0-1 (Sigmoid)
            # Global Heightmap Standard: 0-65535
            # Force float32 to avoid float16 overflow when multiplying by 65535
            h_np = pred_h.squeeze().float().cpu().numpy()
            h_16bit = (h_np * 65535).astype(np.uint16)
            
            h_img = Image.fromarray(h_16bit, mode="I;16")
            
            # Post-process Albedo (3 Channels) -> 8-bit RGB
            a_tensor = pred_a.squeeze().cpu()
            a_img = TF.to_pil_image(a_tensor)
            
            # Save Images
            base_name = img_path.stem
            h_png_path = OUTPUT_DIR / f"{base_name}_height_pred.png"
            a_png_path = OUTPUT_DIR / f"{base_name}_albedo_pred.png"
            
            h_img.save(h_png_path)
            a_img.save(a_png_path)
            
            # Generate OBJ
            print(f"Generating OBJ for {base_name}...")
            generate_obj(
                height_map=h_16bit, 
                albedo_path=a_png_path.name, # Mtl references filename
                output_name=OUTPUT_DIR / f"{base_name}.obj",
                mat_name=OUTPUT_DIR / f"{base_name}.mtl"
            )
            
            print(f"Processed: {base_name} -> .obj")
            
        except Exception as e:
            print(f"Error validating {img_path.name}: {e}")

def generate_obj(height_map, albedo_path, output_name, mat_name):
    """
    Converts heightmap (H,W) array to a grid mesh OBJ.
    """
    h, w = height_map.shape
    
    # Vertices & UVs
    # Grid centered at 0,0? Or 0 to 533?
    # WoW Tile is ~533.3333 yards.
    TILE_SIZE = 533.3333
    scale_y = (TILE_SIZE / h)
    scale_x = (TILE_SIZE / w)
    
    # Height Scaling:
    # 0-65535 maps to what physical height?
    # WoW max height is ~2000? Let's assume a reasonable scale factor or configurable
    # For now, we Normalize to 0-1 then multiply by Max Height (e.g. 1000 yards usually covers it)
    MAX_HEIGHT = 1200.0
    scale_z = MAX_HEIGHT / 65535.0
    
    with open(output_name, 'w') as f:
        f.write(f"mtllib {mat_name.name}\n")
        
        # 1. Vertices
        for y in range(h):
            for x in range(w):
                # X, Z, Y (Y is up in WoW/OBJ usually, or Z up?)
                # OBJ Y is usually Up. WoW Z is Up.
                # Let's write X (Right), Z (Up), -Y (Forward) to match standard view
                
                px = x * scale_x
                py = height_map[y, x] * scale_z # Height -> Y
                pz = -y * scale_y
                
                # Write Vert: v x z y (Z-Up)
                f.write(f"v {px:.4f} {pz:.4f} {py:.4f}\n")
                
                # Write UV: vt u v
                # V is flipped in images usually (0 bottom) vs image (0 top)
                f.write(f"vt {x/(w-1):.4f} {(1 - y/(h-1)):.4f}\n")
                
        # 2. Material Group
        f.write(f"usemtl TerrainMat\n")
        
        # 3. Faces (Quads)
        # 0 1
        # 3 2
        for y in range(h - 1):
            for x in range(w - 1):
                # Indices 1-based
                # Current Row: i = y*w + x + 1
                # Next Row:    j = (y+1)*w + x + 1
                
                i0 = y * w + x + 1
                i1 = i0 + 1
                i2 = (y + 1) * w + x + 1 + 1
                i3 = (y + 1) * w + x + 1
                
                # f v/vt v/vt v/vt v/vt
                f.write(f"f {i0}/{i0} {i1}/{i1} {i2}/{i2} {i3}/{i3}\n")

    # Write MTL
    with open(mat_name, 'w') as m:
        m.write("newmtl TerrainMat\n")
        m.write("Ka 1.0 1.0 1.0\n")
        m.write("Kd 1.0 1.0 1.0\n")
        m.write(f"map_Kd {albedo_path}\n") # Relative path to texture in same folder

if __name__ == "__main__":
    run_inference()
