
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
OUTPUT_DIR = Path("cached_v7_6")
OUTPUT_DIR.mkdir(exist_ok=True)

# Target Resolution (V7++ Native)
TARGET_RES = 512
CHUNK_RES = 32  # 512 / 16 chunks = 32 pixels per chunk

def load_image_tensor(path, size=None, grayscale=False):
    if not path.exists():
        return None
    try:
        img = Image.open(path)
        if size:
            img = img.resize(size, Image.Resampling.LANCZOS)
        
        # Convert to Tensor (C, H, W) normalized 0-1
        tensor = TF.to_tensor(img) 
        
        if grayscale:
             # Force 1 channel
            if tensor.shape[0] > 1:
                tensor = tensor[0:1]
        elif tensor.shape[0] == 4:
            # Drop Alpha for regular images (Minimap/Texture) -> Force RGB
            tensor = tensor[:3]
            
        return tensor
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None

def synthesize_albedo(tile_json_path, masks_dir, tilesets_dir):
    """
    Synthesizes a 512x512 Albedo map from the JSON layer data.
    """
    try:
        with open(tile_json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Failed to load JSON {tile_json_path}: {e}")
        return None

    # Blank Canvas (3, 512, 512)
    full_albedo = torch.zeros((3, TARGET_RES, TARGET_RES), dtype=torch.float32)
    
    # Iterate over chunks (0 to 255)
    # The JSON structure for layers: data['terrain_data']['chunk_layers'] -> list of dicts with 'idx' and 'layers'
    
    chunk_layers_map = {item['idx']: item['layers'] for item in data.get('terrain_data', {}).get('chunk_layers', [])}
    
    tile_name = tile_json_path.stem # e.g. Azeroth_32_48

    for r in range(16):
        for c in range(16):
            chunk_idx = r * 16 + c
            
            # Define pixel window for this chunk in the global 512 map
            y_start, y_end = r * CHUNK_RES, (r + 1) * CHUNK_RES
            x_start, x_end = c * CHUNK_RES, (c + 1) * CHUNK_RES
            
            # Start with Black (or maybe a default texture if layer 0 is missing?)
            # Usually Layer 0 is the base.
            chunk_albedo = torch.zeros((3, CHUNK_RES, CHUNK_RES), dtype=torch.float32)
            
            layers = chunk_layers_map.get(chunk_idx, [])
            
            if not layers:
                # Fill with black or debug color if no layers
                continue

            for i, layer_info in enumerate(layers):
                tex_path_raw = layer_info.get('texture_path', '')
                # Clean path: "Tileset\\Foo\\Bar.blp" -> "Bar.png" check in tilesets dir
                tex_name_blp = Path(tex_path_raw).name
                tex_name_png = tex_name_blp.replace('.blp', '.png').replace('.BLP', '.png')
                
                tex_file = tilesets_dir / tex_name_png
                
                # Load Texture and resize to CHUNK_RES (32x32)
                # We simply load it once. Optimization: Cache loaded textures in memory if they repeat often?
                # For now, just load.
                texture_tensor = load_image_tensor(tex_file, size=(CHUNK_RES, CHUNK_RES))
                
                if texture_tensor is None:
                    # Missing texture, skip or use pink placeholder?
                    # Let's skip to avoid crashing, maybe log
                    continue
                
                # Handling Alpha
                # Layer 0 usually has no alpha (it's base), effectively alpha=1
                # Subsequent layers have alpha masks
                
                if i == 0:
                    # Base layer, fully opaque override (usually)
                    # But blending logic in WoW is: Lerp(Current, New, Alpha)
                    chunk_albedo = texture_tensor
                else:
                    # Load Alpha Mask
                    # Masks are named: Azeroth_32_48_c{chunk_idx}_l{layer_idx_in_list??? or i?}
                    # Looking at JSON sample: "alpha_path": "masks/Azeroth_32_48_c0_l1.png"
                    # It seems 'alpha_path' key exists if there is an alpha.
                    alpha_rel_path = layer_info.get('alpha_path')
                    
                    if alpha_rel_path:
                        mask_path = ROOT_DIR / alpha_rel_path
                        # Mask on disk is 64x64, we need 32x32 to match our target chunk size
                        mask_tensor = load_image_tensor(mask_path, size=(CHUNK_RES, CHUNK_RES), grayscale=True)
                        
                        if mask_tensor is not None:
                            # Composite: Result = Result * (1-Alpha) + New * Alpha
                            chunk_albedo = chunk_albedo * (1 - mask_tensor) + texture_tensor * mask_tensor
                    else:
                        # If no alpha path but it's not layer 0? 
                        # Check flags? "flags": 256 often means compressed alpha or something.
                        # For now, if no alpha specified for >0, assume additive or skipped?
                        # Let's assume skipped if no mask for non-base layer to prevent full overwrite
                        pass
            
            # Place chunk into full map
            full_albedo[:, y_start:y_end, x_start:x_end] = chunk_albedo

    return full_albedo

def process_dataset():
    # Find all JSONs in dataset folder
    dataset_dir = ROOT_DIR / "dataset"
    json_files = list(dataset_dir.glob("Azeroth_*.json"))
    
    print(f"Found {len(json_files)} tiles to process.")
    
    for json_file in tqdm(json_files):
        try:
            # Parse coordinates from filename: Azeroth_32_48.json
            parts = json_file.stem.split('_')
            x, y = parts[1], parts[2]
            
            # Paths
            minimap_path = ROOT_DIR / "images" / f"Azeroth_{x}_{y}.png"
            height_path = ROOT_DIR / "images" / f"Azeroth_{x}_{y}_heightmap_global.png"
            
            # 1. Load Inputs/Targets
            minimap = load_image_tensor(minimap_path) # Should be 512x512
            height = load_image_tensor(height_path, grayscale=True) # Should be 512x512
            
            if minimap is None or height is None:
                continue
                
            # Verify Dimensions (V7++ Strictness)
            if minimap.shape[1:] != (TARGET_RES, TARGET_RES) or height.shape[1:] != (TARGET_RES, TARGET_RES):
                 # Resize if strictly necessary but warn? 
                 # User said "No downsampling", implies trusting source.
                 # If source is 256, we might need to upscale or skip?
                 # Let's Resize to Target to ensure tensors stack, but warn.
                 minimap = TF.resize(minimap, [TARGET_RES, TARGET_RES])
                 height = TF.resize(height, [TARGET_RES, TARGET_RES])

            # 2. Synthesize Albedo
            albedo = synthesize_albedo(
                json_file, 
                ROOT_DIR / "masks", 
                ROOT_DIR / "tilesets"
            )
            
            if albedo is None:
                continue

            # 3. Cache
            # Save as Float16 to save space
            torch.save(minimap.half(), OUTPUT_DIR / f"input_{x}_{y}.pt")
            torch.save(height.half(), OUTPUT_DIR / f"target_height_{x}_{y}.pt")
            torch.save(albedo.half(), OUTPUT_DIR / f"target_albedo_{x}_{y}.pt")
            
        except Exception as e:
            print(f"Failed to process {json_file.name}: {e}")

if __name__ == "__main__":
    process_dataset()
