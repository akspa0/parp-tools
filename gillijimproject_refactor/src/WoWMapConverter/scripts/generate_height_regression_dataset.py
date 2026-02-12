"""
Generate Height Regression Dataset (Generative 3D)
==================================================
Creates a VLM training dataset where:
- Input: Image Crop (Minimap Chunk)
- Output: JSON Array of Height values (145 floats).

Usage:
    python generate_height_regression_dataset.py
"""

import json
from pathlib import Path
from PIL import Image

# Hardcoded absolute paths that we know work
ROOT_DIR = Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_Kalidar")
DATASET_DIR = ROOT_DIR / "dataset"
OUTPUT_JSONL = Path(r"j:\wowDev\parp-tools\gillijimproject_refactor\height_regression.jsonl")
CROP_DIR = ROOT_DIR / "crops"

def main():
    print(f"Scanning dataset in {DATASET_DIR}...")
    
    CROP_DIR.mkdir(parents=True, exist_ok=True)
    
    # Explicitly grab from the known valid folder
    files = list(DATASET_DIR.glob("*.json"))
    print(f"Found {len(files)} JSON files in {DATASET_DIR}")
    
    if len(files) == 0:
        print("CRITICAL ERROR: No files found. Aborting.")
        return

    count = 0
    with open(OUTPUT_JSONL, 'w', encoding='utf-8') as out_f:
        for json_path in files:
            try:
                # Load JSON
                with open(json_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                # Check Image
                img_rel = data.get("image")
                
                # Image Resolution
                if img_rel:
                    img_path = ROOT_DIR / img_rel
                else:
                    # Fallback: Inference from filename
                    # Kalidar_30_23.json -> images/Kalidar_30_23.png
                    img_path = ROOT_DIR / "images" / f"{json_path.stem}.png"
                
                if not img_path.exists():
                     # Fallback 2: relative to json
                     if img_rel: img_path = json_path.parent / img_rel
                
                if not img_path.exists():
                    print(f"Skipping {json_path.name}: Image missing at {img_path}")
                    continue

                # Process
                with Image.open(img_path) as full_img:
                    full_img.load() # Force load
                    width, height = full_img.size
                    cw = width // 16
                    ch = height // 16
                    
                    td = data.get("terrain_data", {})
                    heights_list = td.get("heights", [])
                    # Key is 'h' in actual JSON, not 'heights'
                    chunk_h_map = {c["idx"]: c.get("h", c.get("heights", [])) for c in heights_list}
                    
                    for idx, h_vals in chunk_h_map.items():
                        if len(h_vals) != 145: continue
                        
                        row = idx // 16
                        col = idx % 16
                        left = col * cw
                        upper = row * ch
                        
                        crop = full_img.crop((left, upper, left+cw, upper+ch))
                        crop_name = f"{json_path.stem}_c{idx}.png"
                        crop_out = CROP_DIR / crop_name
                        crop.save(crop_out)
                        
                        # Generate Entry
                        h_clean = [round(x, 2) for x in h_vals]
                        entry = {
                            "messages": [
                                {"role": "user", "content": [
                                    {"type": "text", "text": "Extract heightmap"},
                                    {"type": "image", "image": str(crop_out)}
                                ]},
                                {"role": "assistant", "content": [
                                    {"type": "text", "text": json.dumps(h_clean)}
                                ]}
                            ]
                        }
                        out_f.write(json.dumps(entry) + "\n")
                        count += 1
                        
                        if count % 1000 == 0:
                            print(f"Generated {count} samples...")
                            
            except Exception as e:
                print(f"Error processing {json_path.name}: {e}")
                
    print(f"Done. Generated {count} samples in {OUTPUT_JSONL}")

if __name__ == "__main__":
    main()
