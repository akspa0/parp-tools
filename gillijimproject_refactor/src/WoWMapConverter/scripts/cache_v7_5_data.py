#!/usr/bin/env python3
"""
WoW V7.5 Lite - Data Cacher
Pre-processes dataset into tensors for rapid CPU training.
Features:
- Resolution: 256x256 (Downscaled from 512)
- Rotates Inputs -90 Degrees (Fix)
- Ignores Water Mask
- Computes Synthetic WDL (16x16) for Residual Learning
"""

import os
import sys
import json
import torch
from pathlib import Path
from PIL import Image
import numpy as np
from tqdm import tqdm
from torchvision import transforms

# Config
DATASET_ROOTS = [
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Azeroth_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Kalimdor_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Kalidar_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_DeadminesInstance_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Shadowfang_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_PVPZone01_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_PVPZone02_v30"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_RazorfenKraulInstance_v30"),
]
OUTPUT_DIR = Path(r"./vlm_output/v7_5_lite")
CACHE_FILE_TRAIN = OUTPUT_DIR / "train_cache.pt"
CACHE_FILE_VAL = OUTPUT_DIR / "val_cache.pt"

INPUT_SIZE = 256
OUTPUT_SIZE = 256
VAL_SPLIT = 0.1

# Global Range used for normalization
HEIGHT_GLOBAL_MIN = -1000.0
HEIGHT_GLOBAL_MAX = 3000.0
RANGE = HEIGHT_GLOBAL_MAX - HEIGHT_GLOBAL_MIN

def load_heightmap(path, size):
    img = Image.open(path)
    if img.mode == 'I;16':
        arr = np.array(img, dtype=np.float32) / 65535.0
    elif img.mode == 'I':
        arr = np.array(img, dtype=np.float32)
        arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-8)
    else:
        img = img.convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
    
    # Resize if needed
    if arr.shape[0] != size:
        from scipy.ndimage import zoom
        scale = size / arr.shape[0]
        arr = zoom(arr, scale, order=1)
    return torch.from_numpy(arr).unsqueeze(0).float()

def process_dataset():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    samples = []
    print("Scanning datasets...")
    for root in DATASET_ROOTS:
        if not root.exists(): continue
        ds_dir = root / "dataset"
        for json_path in ds_dir.glob("*.json"):
            try:
                with open(json_path, 'r') as f: data = json.load(f)
                td = data.get("terrain_data", {})
                
                # Validation: Must have Minimap, Normal, Global Height, Local Height
                hm_g = td.get("heightmap_global") or td.get("heightmap")
                hm_l = td.get("heightmap_local") or td.get("heightmap")
                nm = td.get("normalmap")
                
                if not (hm_g and hm_l and nm): continue
                
                # Check files exist
                hm_g_path = root / hm_g
                hm_l_path = root / hm_l
                nm_path = root / nm
                mm_path = root / "images" / f"{json_path.stem}.png"
                
                if not (hm_g_path.exists() and hm_l_path.exists() and nm_path.exists() and mm_path.exists()):
                    continue
                
                samples.append({
                    "json": json_path, "mm": mm_path, "nm": nm_path, 
                    "hm_g": hm_g_path, "hm_l": hm_l_path, "td": td
                })
            except: continue
            
    print(f"Found {len(samples)} valid samples.")
    
    # Shuffling
    np.random.shuffle(samples)
    split_idx = int(len(samples) * (1 - VAL_SPLIT))
    train_samples = samples[:split_idx]
    val_samples = samples[split_idx:]
    
    # Process
    process_batch(train_samples, CACHE_FILE_TRAIN)
    process_batch(val_samples, CACHE_FILE_VAL)

def process_batch(sample_list, output_path):
    print(f"Processing {len(sample_list)} samples to {output_path}...")
    
    data_list = []
    to_tensor = transforms.ToTensor()
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    for s in tqdm(sample_list):
        try:
            # 1. Inputs (Minimap, Normal)
            mm = Image.open(s["mm"]).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            nm = Image.open(s["nm"]).convert("RGB").resize((INPUT_SIZE, INPUT_SIZE), Image.BILINEAR)
            
            # FIX: Rotate Inputs -90
            mm = mm.rotate(-90)
            nm = nm.rotate(-90)
            
            mm_t = norm(to_tensor(mm))
            nm_t = norm(to_tensor(nm))
            
            # 2. Targets (Height Global, Local)
            hm_g = load_heightmap(s["hm_g"], OUTPUT_SIZE)
            hm_l = load_heightmap(s["hm_l"], OUTPUT_SIZE)
            
            # 3. Synthetic WDL (Downsample Global Height to 16x16 then Upsample)
            # This creates a "low frequency" version of the terrain
            wdl_low = torch.nn.functional.interpolate(hm_g.unsqueeze(0), size=(16, 16), mode='bilinear')
            wdl_upscaled = torch.nn.functional.interpolate(wdl_low, size=(OUTPUT_SIZE, OUTPUT_SIZE), mode='bilinear').squeeze(0)
            
            # RESIDUAL: Target = Height - WDL
            # We predict the detailing required to go from WDL -> Height
            target_residual = hm_g - wdl_upscaled
            
            # 4. Metadata Channels (H_Min, H_Max)
            g_min = s["td"].get("height_global_min", HEIGHT_GLOBAL_MIN)
            g_max = s["td"].get("height_global_max", HEIGHT_GLOBAL_MAX)
            # Normalize to 0-1 relative to global range
            min_n = (g_min - HEIGHT_GLOBAL_MIN) / RANGE
            max_n = (g_max - HEIGHT_GLOBAL_MIN) / RANGE
            
            h_min_mask = torch.full((1, INPUT_SIZE, INPUT_SIZE), min_n, dtype=torch.float32)
            h_max_mask = torch.full((1, INPUT_SIZE, INPUT_SIZE), max_n, dtype=torch.float32)
            
            # 5. Object Mask (Simple)
            obj_mask = torch.zeros((1, INPUT_SIZE, INPUT_SIZE), dtype=torch.float32)
            objects = s["td"].get("objects")
            if objects:
                # Simplified rendering for caching speed
                scale_factor = INPUT_SIZE / 533.333
                for obj in objects:
                    px, py = obj.get("pos_x", 0), obj.get("pos_y", 0)
                    nx = int((px / 533.333) * INPUT_SIZE) % INPUT_SIZE
                    ny = int((py / 533.333) * INPUT_SIZE) % INPUT_SIZE
                    # Draw 3x3 dot
                    y1, y2 = max(0, ny-1), min(INPUT_SIZE, ny+2)
                    x1, x2 = max(0, nx-1), min(INPUT_SIZE, nx+2)
                    obj_mask[0, y1:y2, x1:x2] = 1.0
            
            # Pack Input: [MM(3), NM(3), WDL(1), Min(1), Max(1), Obj(1)] = 10 Channels (No Water)
            input_t = torch.cat([mm_t, nm_t, wdl_upscaled, h_min_mask, h_max_mask, obj_mask], dim=0)
            
            # Pack Target: [Height_Global(1), Height_Local(1), Residual(1)]
            # We keep Global/Local for reference, but train on Residual
            target_t = torch.cat([hm_g, hm_l, target_residual], dim=0)
            
            # Store compressed (fp16 to save RAM/Disk)
            data_list.append((input_t.half(), target_t.half()))
            
        except Exception as e:
            # print(f"Error processing {s['json']}: {e}")
            continue

    # Save
    print(f"Saving {len(data_list)} tensors...")
    torch.save(data_list, output_path)
    print("Done.")

if __name__ == "__main__":
    process_dataset()
