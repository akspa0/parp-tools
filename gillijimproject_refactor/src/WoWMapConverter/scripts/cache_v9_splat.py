
import os
import json
import torch
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
import torchvision.transforms.functional as TF

# --- Configuration ---
ROOT_DIR = Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Azeroth_v30")
OUTPUT_DIR = Path("cached_v7_6") # Save into same cache folder
OUTPUT_DIR.mkdir(exist_ok=True)

TARGET_RES = 512
CHUNK_RES = 32

def load_image_tensor(path, size=None, grayscale=False):
    if not path.exists():
        return None
    try:
        img = Image.open(path)
        if size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        tensor = TF.to_tensor(img)
        if grayscale:
            if tensor.shape[0] > 1: tensor = tensor[0:1]
        return tensor
    except Exception:
        return None

def extract_splat_map(tile_json_path, masks_dir):
    """
    Extracts 4-channel Splat Map (Layers 1-4 Alphas).
    """
    try:
        with open(tile_json_path, 'r') as f:
            data = json.load(f)
    except Exception:
        return None

    # (4, 512, 512) for up to 4 overlay layers
    full_splat = torch.zeros((4, TARGET_RES, TARGET_RES), dtype=torch.float32)
    
    chunk_layers_map = {item['idx']: item['layers'] for item in data.get('terrain_data', {}).get('chunk_layers', [])}
    
    for r in range(16):
        for c in range(16):
            chunk_idx = r * 16 + c
            y_start, y_end = r * CHUNK_RES, (r + 1) * CHUNK_RES
            x_start, x_end = c * CHUNK_RES, (c + 1) * CHUNK_RES
            
            chunk_splat = torch.zeros((4, CHUNK_RES, CHUNK_RES), dtype=torch.float32)
            
            layers = chunk_layers_map.get(chunk_idx, [])
            if not layers: continue

            # Iterate layers. 
            # Layer 0 = Base (Ignored in Splat Map usually, or implicit).
            # We want to capture the "Brush Patterns" of overlays.
            for i, layer_info in enumerate(layers):
                if i == 0: continue # Skip base
                if i > 4: break     # Max 4 overlays supported for now (Splat limit)
                
                channel_idx = i - 1 # Layer 1 -> Ch 0
                
                alpha_rel_path = layer_info.get('alpha_path')
                if alpha_rel_path:
                    mask_path = ROOT_DIR / alpha_rel_path
                    mask_tensor = load_image_tensor(mask_path, size=(CHUNK_RES, CHUNK_RES), grayscale=True)
                    
                    if mask_tensor is not None:
                        # mask_tensor is (1, 32, 32)
                        chunk_splat[channel_idx] = mask_tensor[0]
            
            full_splat[:, y_start:y_end, x_start:x_end] = chunk_splat

    return full_splat

def process_alphas():
    dataset_dir = ROOT_DIR / "dataset"
    json_files = list(dataset_dir.glob("Azeroth_*.json"))
    
    print(f"Extracting Alpha Splats for {len(json_files)} tiles...")
    
    for json_file in tqdm(json_files):
        try:
            parts = json_file.stem.split('_')
            x, y = parts[1], parts[2]
            
            # Generate Target
            splat = extract_splat_map(json_file, ROOT_DIR / "masks")
            
            if splat is not None:
                # Save
                torch.save(splat.half(), OUTPUT_DIR / f"target_splat_{x}_{y}.pt")
                
        except Exception as e:
            print(f"Error {json_file.name}: {e}")

if __name__ == "__main__":
    process_alphas()
