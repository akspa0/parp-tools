"""
VLM Dataset Curation Script - FULL DATA ANALYSIS
=================================================

Analyzes ALL VLM JSON data to select a diverse, high-quality training subset.

Metrics analyzed:
1. Height Complexity - Variance in terrain elevation
2. Texture Richness - Number of unique textures + layer count
3. Object Density - M2 models + WMO buildings
4. Liquid Coverage - Water/lava presence and area
5. Shadow Data - Shadow map presence
6. Chunk Layer Complexity - Alpha masks, normals, MCCV colors
7. WDL Data - Low-res heightmap presence
8. Hole Coverage - Terrain holes
"""

import json
import argparse
import math
import statistics
import random
import shutil
from pathlib import Path
from typing import Optional, Dict, Any, List


def analyze_tile(json_path: Path) -> Optional[Dict[str, Any]]:
    """
    Analyze ALL data in a tile and compute comprehensive metrics.
    """
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        terrain = data.get("terrain_data", {})
        
        # ====== 1. HEIGHT COMPLEXITY ======
        heights = terrain.get("heights") or []
        all_z = []
        for chunk in heights:
            vals = chunk.get("h", []) if isinstance(chunk, dict) else chunk
            all_z.extend([float(x) for x in vals if x is not None])
        
        height_var = statistics.stdev(all_z) if len(all_z) > 1 else 0
        height_range = (terrain.get("height_min", 0), terrain.get("height_max", 0))
        
        # ====== 2. TEXTURE RICHNESS ======
        textures = terrain.get("textures") or []
        tex_count = len(textures)
        
        chunk_layers = terrain.get("chunk_layers") or []
        total_layers = sum(len(cl.get("layers", [])) for cl in chunk_layers if isinstance(cl, dict))
        avg_layers_per_chunk = total_layers / len(chunk_layers) if chunk_layers else 0
        
        # ====== 3. ALPHA MASK COMPLEXITY ======
        alpha_masks = terrain.get("alpha_masks") or []
        alpha_count = len(alpha_masks)
        
        # Count chunks with alpha_bits in layers
        chunks_with_alpha = 0
        for cl in chunk_layers:
            if isinstance(cl, dict):
                for layer in cl.get("layers", []):
                    if isinstance(layer, dict) and layer.get("alpha_bits"):
                        chunks_with_alpha += 1
                        break
        
        # ====== 4. NORMALS DATA ======
        chunks_with_normals = sum(1 for cl in chunk_layers 
                                   if isinstance(cl, dict) and cl.get("normals"))
        
        # ====== 5. MCCV VERTEX COLORS ======
        chunks_with_mccv = sum(1 for cl in chunk_layers 
                               if isinstance(cl, dict) and cl.get("mccv_colors"))
        
        # ====== 6. SHADOW DATA ======
        shadow_bits = terrain.get("shadow_bits") or []
        shadow_maps = terrain.get("shadow_maps") or []
        shadow_count = len(shadow_bits) + len(shadow_maps)
        
        # ====== 7. LIQUID DATA ======
        liquids = terrain.get("liquids") or []
        liquid_count = len(liquids)
        
        liquid_types = set()
        liquid_height_values = 0
        for liq in liquids:
            if isinstance(liq, dict):
                liquid_types.add(liq.get("type", 0))
                lh = liq.get("heights") or []
                liquid_height_values += len(lh)
        
        has_liquid_mask = terrain.get("liquid_mask") is not None
        has_liquid_height = terrain.get("liquid_height") is not None
        
        # ====== 8. OBJECT PLACEMENTS ======
        objects = terrain.get("objects") or []
        obj_count = len(objects)
        
        m2_count = sum(1 for o in objects if isinstance(o, dict) and o.get("category") == "m2")
        wmo_count = sum(1 for o in objects if isinstance(o, dict) and o.get("category") == "wmo")
        
        # Object variety (unique names)
        obj_names = set(o.get("name", "") for o in objects if isinstance(o, dict))
        obj_variety = len(obj_names)
        
        # ====== 9. TERRAIN HOLES ======
        holes = terrain.get("holes") or []
        holes_count = sum(1 for h in holes if h != 0)  # Non-zero = has holes
        
        # ====== 10. CHUNK POSITIONS ======
        chunk_positions = terrain.get("chunk_positions") or []
        has_positions = len(chunk_positions) > 0
        
        # ====== 11. WDL LOW-RES DATA ======
        wdl = terrain.get("wdl_heights")
        has_wdl = wdl is not None
        
        # ====== COMPUTE COMPOSITE SCORE ======
        # Higher score = more interesting tile
        score = 0
        
        # Height complexity (0-30 points)
        score += min(30, height_var * 2)
        
        # Texture richness (0-20 points)
        score += min(10, tex_count)
        score += min(10, avg_layers_per_chunk * 3)
        
        # Alpha complexity (0-15 points)
        score += min(15, chunks_with_alpha / 256 * 15) if chunk_layers else 0
        
        # Object density (0-20 points)
        score += min(10, obj_count / 10)
        score += min(10, obj_variety)
        
        # Liquid presence (0-10 points)
        if liquid_count > 0:
            score += 5
            score += min(5, len(liquid_types) * 2)
        
        # Shadow data (0-5 points)
        if shadow_count > 0:
            score += 5
        
        # Holes make it interesting (0-5 points)
        if holes_count > 0:
            score += min(5, holes_count)
        
        return {
            "path": json_path,
            "name": json_path.stem,
            "score": score,
            "metrics": {
                # Heights
                "height_var": height_var,
                "height_min": height_range[0],
                "height_max": height_range[1],
                "chunk_height_count": len(heights),
                
                # Textures
                "texture_count": tex_count,
                "chunk_layer_count": len(chunk_layers),
                "total_layers": total_layers,
                "avg_layers": avg_layers_per_chunk,
                
                # Alpha/Normals/MCCV
                "alpha_mask_count": alpha_count,
                "chunks_with_alpha": chunks_with_alpha,
                "chunks_with_normals": chunks_with_normals,
                "chunks_with_mccv": chunks_with_mccv,
                
                # Shadows
                "shadow_count": shadow_count,
                
                # Liquids
                "liquid_chunk_count": liquid_count,
                "liquid_types": list(liquid_types),
                "liquid_height_values": liquid_height_values,
                "has_liquid_mask": has_liquid_mask,
                
                # Objects
                "object_count": obj_count,
                "m2_count": m2_count,
                "wmo_count": wmo_count,
                "object_variety": obj_variety,
                
                # Other
                "holes_count": holes_count,
                "has_chunk_positions": has_positions,
                "has_wdl": has_wdl,
            }
        }
        
    except Exception as e:
        print(f"Error analyzing {json_path}: {e}")
        return None


def select_diverse_subset(tiles: List[dict], target_count: int) -> List[dict]:
    """
    Select a diverse subset using score-based ranking.
    
    Strategy:
    - Top 40% by score (most interesting)
    - 30% random from middle tier
    - 20% with specific features (liquids, objects, holes)
    - 10% random "simple" tiles for baseline
    """
    if len(tiles) <= target_count:
        return tiles
    
    # Sort by score
    sorted_tiles = sorted(tiles, key=lambda x: x["score"], reverse=True)
    
    selected = {}
    
    # Top 40% - highest scores
    top_count = int(target_count * 0.4)
    for t in sorted_tiles[:top_count]:
        selected[t["name"]] = t
    
    # Feature-specific: Tiles with liquids (10%)
    liquid_tiles = [t for t in tiles if t["metrics"]["liquid_chunk_count"] > 0 
                    and t["name"] not in selected]
    liquid_count = int(target_count * 0.1)
    for t in liquid_tiles[:liquid_count]:
        selected[t["name"]] = t
    
    # Feature-specific: Tiles with high object density (10%)
    object_tiles = sorted([t for t in tiles if t["name"] not in selected],
                          key=lambda x: x["metrics"]["object_count"], reverse=True)
    obj_count = int(target_count * 0.1)
    for t in object_tiles[:obj_count]:
        selected[t["name"]] = t
    
    # Feature-specific: Tiles with terrain holes (5%)
    hole_tiles = [t for t in tiles if t["metrics"]["holes_count"] > 0 
                  and t["name"] not in selected]
    hole_count = int(target_count * 0.05)
    for t in hole_tiles[:hole_count]:
        selected[t["name"]] = t
    
    # Random from remaining (fill to target)
    remaining = [t for t in tiles if t["name"] not in selected]
    random.shuffle(remaining)
    fill_count = target_count - len(selected)
    for t in remaining[:fill_count]:
        selected[t["name"]] = t
    
    return list(selected.values())


def main():
    parser = argparse.ArgumentParser(description="Curate VLM Dataset (Full Data Analysis)")
    parser.add_argument("input_dir", help="Directory containing raw VLM export")
    parser.add_argument("output_dir", help="Directory for curated output")
    parser.add_argument("--count", type=int, default=300, help="Target sample count")
    parser.add_argument("--stats", action="store_true", help="Print detailed stats")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"Scanning {input_path}...")
    all_files = list((input_path / "dataset").glob("*.json"))
    print(f"Found {len(all_files)} files. Analyzing ALL data...")
    
    analyzed = []
    for i, p in enumerate(all_files):
        result = analyze_tile(p)
        if result:
            analyzed.append(result)
        if (i + 1) % 100 == 0:
            print(f"  Analyzed {i + 1}/{len(all_files)}...")
    
    print(f"Analyzed {len(analyzed)} tiles successfully.")
    
    # Print stats if requested
    if args.stats and analyzed:
        print("\n=== Dataset Statistics ===")
        scores = [t["score"] for t in analyzed]
        print(f"Score range: {min(scores):.1f} - {max(scores):.1f}")
        print(f"Average score: {statistics.mean(scores):.1f}")
        
        with_liquids = sum(1 for t in analyzed if t["metrics"]["liquid_chunk_count"] > 0)
        with_objects = sum(1 for t in analyzed if t["metrics"]["object_count"] > 0)
        with_holes = sum(1 for t in analyzed if t["metrics"]["holes_count"] > 0)
        with_wdl = sum(1 for t in analyzed if t["metrics"]["has_wdl"])
        
        print(f"Tiles with liquids: {with_liquids}")
        print(f"Tiles with objects: {with_objects}")
        print(f"Tiles with holes: {with_holes}")
        print(f"Tiles with WDL: {with_wdl}")
    
    # Select subset
    print(f"\nSelecting diverse subset of {args.count} tiles...")
    subset = select_diverse_subset(analyzed, args.count)
    print(f"Selected {len(subset)} tiles.")
    
    # Copy files
    print("Copying files...")
    dest_dataset = output_path / "dataset"
    dest_images = output_path / "images"
    dest_dataset.mkdir(exist_ok=True)
    dest_images.mkdir(exist_ok=True)
    
    # Also copy tilesets and other directories
    for subdir in ["tilesets", "shadows", "masks", "liquids", "stitched"]:
        src = input_path / subdir
        if src.exists():
            dest = output_path / subdir
            if not dest.exists():
                shutil.copytree(src, dest)
    
    for tile in subset:
        # Copy JSON
        shutil.copy2(tile["path"], dest_dataset / tile["path"].name)
        
        # Copy image
        img_src = input_path / "images" / f"{tile['name']}.png"
        if img_src.exists():
            shutil.copy2(img_src, dest_images / img_src.name)
    
    print(f"\nDone! Curated dataset in {output_path}")
    print(f"  - {len(subset)} JSON files in dataset/")
    print(f"  - Images in images/")
    print(f"  - Tilesets, shadows, masks copied")


if __name__ == "__main__":
    main()
