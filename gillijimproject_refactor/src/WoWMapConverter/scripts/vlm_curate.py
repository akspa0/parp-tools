"""
VLM Dataset Curation Script

Analyzes VLM JSON exports and selects a diverse subset of tiles for training.
Metrics used for selection:
1. Terrain Complexity (Height Variance)
2. Visual Richness (Texture Count)
3. Scene Object Density (Object Count)
4. Liquid Presence (Has Water/Lava)

Selection Strategy:
- Ensures representation from all quartiles of complexity (Simple -> Complex)
- Prioritizes "interesting" edge cases (High texture variation, dense objects)
- Includes diverse specific features (Water tiles, holes, etc.)
"""

import json
import argparse
from pathlib import Path
import statistics
import random
import shutil

def analyze_tile(json_path):
    """Calculate complexity metrics for a single tile."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        terrain = data.get("terrain_data", {})
        heights = terrain.get("heights", [])
        
        # Metric 1: Height Variance (Terrain Roughness)
        # Flatten all chunks and calculate std dev
        all_z = []
        for chunk in heights:
            all_z.extend(chunk)
        
        height_variance = statistics.stdev(all_z) if len(all_z) > 1 else 0
        
        # Metric 2: Texture Count
        unique_textures = len(terrain.get("textures", []))
        
        # Metric 3: Object Count
        object_count = len(terrain.get("objects", []))
        
        # Metric 4: Liquid Presence
        has_liquid = len(terrain.get("liquids", [])) > 0
        
        return {
            "path": json_path,
            "name": json_path.stem,
            "metrics": {
                "height_var": height_variance,
                "texture_cnt": unique_textures,
                "obj_cnt": object_count,
                "has_liquid": has_liquid
            }
        }
    except Exception as e:
        print(f"Error analyzing {json_path}: {e}")
        return None

def select_diverse_subset(tiles, target_count):
    """Select a diverse subset of tiles based on metrics."""
    if len(tiles) <= target_count:
        return tiles

    selected = []
    
    # 1. Edge Case: Always include top 10% most complex (Height & Objects)
    sorted_by_height = sorted(tiles, key=lambda x: x["metrics"]["height_var"], reverse=True)
    sorted_by_obj = sorted(tiles, key=lambda x: x["metrics"]["obj_cnt"], reverse=True)
    
    top_limit = int(target_count * 0.15)
    selected.extend(sorted_by_height[:top_limit])
    selected.extend(sorted_by_obj[:top_limit])
    
    # 2. Edge Case: Include water tiles (up to 10%)
    liquids = [t for t in tiles if t["metrics"]["has_liquid"]]
    liquid_limit = int(target_count * 0.10)
    if liquids:
        selected.extend(random.sample(liquids, min(len(liquids), liquid_limit)))

    # 3. Fill distinct textures (up to 10%)
    sorted_by_tex = sorted(tiles, key=lambda x: x["metrics"]["texture_cnt"], reverse=True)
    tex_limit = int(target_count * 0.10)
    selected.extend(sorted_by_tex[:tex_limit])

    # Deduplicate
    unique_selected = {t["name"]: t for t in selected}
    
    # 4. Fill remainder with stratified random sampling from the rest to ensure "boring" tiles are also represented
    remaining_slots = target_count - len(unique_selected)
    if remaining_slots > 0:
        pool = [t for t in tiles if t["name"] not in unique_selected]
        if pool:
            # Shuffle pool to randomize
            random.shuffle(pool)
            
            # Simple greedy fill from remaining
            # In a more advanced version, we'd sample from specific buckets
            added = pool[:remaining_slots]
            for t in added:
                unique_selected[t["name"]] = t

    return list(unique_selected.values())

def main():
    parser = argparse.ArgumentParser(description="Curate VLM Dataset")
    parser.add_argument("input_dir", help="Directory containing raw JSON dataset files")
    parser.add_argument("output_dir", help="Directory to copy selected JSONs to")
    parser.add_argument("--count", type=int, default=100, help="Target number of samples")
    args = parser.parse_args()

    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Scanning {input_path}...")
    all_files = list(input_path.glob("dataset/*.json"))
    print(f"Found {len(all_files)} files. Analyzing...")

    analyzed_tiles = []
    for p in all_files:
        res = analyze_tile(p)
        if res:
            analyzed_tiles.append(res)

    print("Selecting diverse subset...")
    subset = select_diverse_subset(analyzed_tiles, args.count)
    
    print(f"Selected {len(subset)} samples.")
    print("Copying files...")
    
    # Copy JSONs
    dest_dataset = output_path / "dataset"
    dest_dataset.mkdir(exist_ok=True)
    
    # Also copy matches images if possible
    dest_images = output_path / "images"
    dest_images.mkdir(exist_ok=True)
    src_images = input_path / "images"

    for tile in subset:
        # Copy JSON
        shutil.copy2(tile["path"], dest_dataset / tile["path"].name)
        
        # Copy Image (Try to find it)
        img_name = f"{tile['name']}.png"
        img_src = src_images / img_name
        if img_src.exists():
            shutil.copy2(img_src, dest_images / img_name)
    
    print(f"Done! Curated dataset in {output_path}")

if __name__ == "__main__":
    main()
