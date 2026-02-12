"""
Generate Visual Similarity Dataset (Contrastive Learning)
=========================================================
Uses algorithmic prefab matches (from library.json) to generate 
Positive (Match) and Negative (No-Match) image pairs for VLM training.

Prompt Strategy:
- Input: Two images (A, B)
- Question: "Are these two terrain chunks visually identical prefabs?"
- Answer: "YES" / "NO"
"""

import json
import random
from pathlib import Path
from collections import defaultdict

LIBRARY_PATH = r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Kalidar\prefab_library.json"
DATASET_ROOT = r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Kalidar"
OUTPUT_JSONL = r"j:\wowDev\parp-tools\gillijimproject_refactor\visual_similarity.jsonl"

def main():
    print("Loading Prefab Library...")
    with open(LIBRARY_PATH, 'r', encoding='utf-8') as f:
        lib = json.load(f)
        
    prefabs = lib.get("prefabs", [])
    print(f"Loaded {len(prefabs)} unique prefab signatures.")
    
    # Filter for prefabs with > 1 instance (for positive pairs)
    multi_instance = [p for p in prefabs if p["count"] > 1]
    print(f"Found {len(multi_instance)} multi-instance prefabs (Candidates for Positive Pairs).")
    
    dataset = []
    
    # Load Instance Map
    instance_map_path = Path(OUTPUT_JSONL).parent / "test_data/vlm-datasets/053_Kalidar/macro_gallery_v2/prefab_instances.json"
    
    if not instance_map_path.exists():
        print(f"Error: Instance map not found at {instance_map_path}")
        print("Run terrain_librarian.py first.")
        return
        
    print(f"Loading Instance Map: {instance_map_path}")
    with open(instance_map_path, 'r') as f:
        instance_map = json.load(f) # Hash -> List["Tile.json:cIdx"]
        
    # Build a lookup for "Tile.json" -> "Full/Path/To/Image.png"
    # We need to scan the dataset dir to find the images corresponding to the json files
    # Heuristic: Tile.json exists in dataset/, Image exists in images/Tile.png
    
    # Generate Training Pairs
    pos_pairs = []
    neg_pairs = []
    
    valid_hashes = [h for h, locs in instance_map.items() if len(locs) > 1]
    print(f"Hashes with multiple instances: {len(valid_hashes)}")
    
    # 1. Positive Pairs (Same Hash, Different Loc)
    # Limit to prevent explosion. 1000 pairs?
    TARGET_PAIRS = 500
    
    print(f"Generating {TARGET_PAIRS} Positive Pairs...")
    attempts = 0
    while len(pos_pairs) < TARGET_PAIRS and attempts < TARGET_PAIRS * 10:
        attempts += 1
        h = random.choice(valid_hashes)
        locs = instance_map[h]
        if len(locs) < 2: continue
        
        # Pick 2 distinct
        l1, l2 = random.sample(locs, 2)
        pos_pairs.append((l1, l2))
        
    # 2. Negative Pairs (Different Hash)
    print(f"Generating {TARGET_PAIRS} Negative Pairs...")
    all_hashes = list(instance_map.keys())
    attempts = 0
    while len(neg_pairs) < TARGET_PAIRS and attempts < TARGET_PAIRS * 10:
        attempts += 1
        h1, h2 = random.sample(all_hashes, 2)
        if h1 == h2: continue
        
        l1 = random.choice(instance_map[h1])
        l2 = random.choice(instance_map[h2])
        neg_pairs.append((l1, l2))
        
    print(f"Total: {len(pos_pairs)} Pos, {len(neg_pairs)} Neg")
    
    # Write JSONL
    # Format: User: "Are these identical?" + Image1 + Image2. Assistant: "YES" / "NO"
    # Challenge: Unsloth/Qwen usually takes 1 image per turn or interleaved.
    # We will simply interleave: Text, Image1, Text, Image2.
    
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as f:
        # Helper to resolve image path
        # loc format: "Kalidar_30_23.json:123"
        
        def loc_to_image_data(loc_str):
            json_name, c_idx = loc_str.split(':')
            # Image path: .../053_Kalidar/images/{json_name - .json}.png
            # Crop? No, we don't have cropping logic here easily.
            # Using the FULL 2048x2048 Minimap for a tiny chunk comparison is bad for the model resolution.
            # But the user wants "Visual Similarity".
            # IDEALLY we crop.
            # For this prototype, we will just point to the full image and textually describe the chunk?
            # "Look at chunk index {c_idx}..." -> Model can't do that easily without overlay.
            
            # CRITICAL: We need cropped images.
            # Does 'terrain_librarian' or 'VlmDatasetExporter' export crops? No.
            # Minimap is one big PNG.
            # We can use PIL here to dynamic crop!
            
            base_name = json_name.replace(".json", "")
            img_path = Path(DATASET_ROOT) / "images" / f"{base_name}.png"
            
            # Calculate crop
            c_idx = int(c_idx)
            row = c_idx // 16
            col = c_idx % 16
            
            # Minimap size? Usually 1024x1024 or 512x512.
            # WDT says? Assuming standard.
            # Let's assume standard behavior: 64x64 pixels per chunk?
            # if 1024x1024 total.
            
            # We will pass the crop coordinates in text for now if we can't crop, 
            # BUT Unsloth expects a file path.
            # We should probably generate temporary crops or use a "Center Crop" logic.
            # Let's assume we crop and save to a temp dir?
            # Or just pass the full image.
            
            # Decision: Pass Full Image. Prompt: "Look at the grid cell at row {row}, col {col}..."
            # This is hard for the model.
            
            # Pivot: Just write the crops to `dataset/crops/` on the fly?
            # Yes.
            return img_path, row, col

        # Ensure crop dir
        crop_dir = Path(DATASET_ROOT) / "crops"
        crop_dir.mkdir(exist_ok=True)
        
        from PIL import Image
        
        def get_crop_path(loc_str):
            # Optim: Cache crops?
            json_name, c_idx = loc_str.split(':')
            c_idx = int(c_idx)
            base_name = json_name.replace(".json", "")
            crop_name = f"{base_name}_c{c_idx}.png"
            out_path = crop_dir / crop_name
            
            if out_path.exists():
                return out_path
                
            # Generate
            src_img = Path(DATASET_ROOT) / "images" / f"{base_name}.png"
            if not src_img.exists(): return None
            
            try:
                base_img = Image.open(src_img)
                # 256x256 tiles? 
                # Chunk is 1/16th.
                w, h = base_img.size
                cw, ch = w // 16, h // 16
                row, col = c_idx // 16, c_idx % 16
                
                left = col * cw
                upper = row * ch
                crop = base_img.crop((left, upper, left+cw, upper+ch))
                crop.save(out_path)
                return out_path
            except:
                return None

        count = 0
        
        # Write Positive
        for l1, l2 in pos_pairs:
            p1 = get_crop_path(l1)
            p2 = get_crop_path(l2)
            if not p1 or not p2: continue
            
            entry = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "Compare these two terrain chunks. Are they visually identical prefabs?"},
                        {"type": "image", "image": str(p1)},
                        {"type": "image", "image": str(p2)}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "YES. These chunks contain identical texture and geometry patterns, indicating they are the same prefab."}
                    ]}
                ]
            }
            f.write(json.dumps(entry) + "\n")
            count += 1
            
        # Write Negative
        for l1, l2 in neg_pairs:
            p1 = get_crop_path(l1)
            p2 = get_crop_path(l2)
            if not p1 or not p2: continue
            
            entry = {
                "messages": [
                    {"role": "user", "content": [
                        {"type": "text", "text": "Compare these two terrain chunks. Are they visually identical prefabs?"},
                        {"type": "image", "image": str(p1)},
                        {"type": "image", "image": str(p2)}
                    ]},
                    {"role": "assistant", "content": [
                        {"type": "text", "text": "NO. These chunks have overlapping features but distinct texture distributions or geometry, so they are not identical."}
                    ]}
                ]
            }
            f.write(json.dumps(entry) + "\n")
            count += 1
            
    print(f"Generated {count} training samples in {OUTPUT_JSONL}")
