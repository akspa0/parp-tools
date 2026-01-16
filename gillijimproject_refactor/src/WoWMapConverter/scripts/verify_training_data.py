#!/usr/bin/env python3
"""
Verify Training Data Pipeline

Checks all components needed for V6 training:
1. Heightmaps rendered at 256x256
2. Normalmaps rendered at 256x256
3. JSON files have height_min/height_max
4. JSON files have WDL data
5. Minimap images exist
6. All data can be loaded correctly
"""

import json
import sys
from pathlib import Path
from PIL import Image
import numpy as np

DATASET_ROOTS = [
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_shadowfang_v1"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_azeroth_v7"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_DeadminesInstance_v2"),
    Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_kalimdor_v4"),
]


def check_dataset(root: Path):
    """Check a single dataset for completeness."""
    print(f"\n{'='*60}")
    print(f"Checking: {root.name}")
    print(f"{'='*60}")
    
    if not root.exists():
        print(f"  ❌ Dataset root does not exist!")
        return None
    
    images_dir = root / "images"
    dataset_dir = root / "dataset"
    
    if not images_dir.exists():
        print(f"  ❌ images/ directory missing")
        return None
    
    if not dataset_dir.exists():
        print(f"  ❌ dataset/ directory missing")
        return None
    
    # Find all JSON files
    json_files = list(dataset_dir.glob("*.json"))
    print(f"  JSON files: {len(json_files)}")
    
    stats = {
        "total_json": len(json_files),
        "has_minimap": 0,
        "has_normalmap": 0,
        "has_heightmap": 0,
        "has_height_bounds": 0,
        "has_wdl": 0,
        "complete": 0,
        "heightmap_sizes": [],
        "normalmap_sizes": [],
        "height_ranges": [],
    }
    
    issues = []
    
    for json_path in json_files[:5]:  # Check first 5 in detail
        tile_name = json_path.stem
        
        # Check image files
        minimap_path = images_dir / f"{tile_name}.png"
        normalmap_path = images_dir / f"{tile_name}_normalmap.png"
        heightmap_path = images_dir / f"{tile_name}_heightmap_v2_preview.png"
        
        has_minimap = minimap_path.exists()
        has_normalmap = normalmap_path.exists()
        has_heightmap = heightmap_path.exists()
        
        # Load JSON
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            terrain = data.get("terrain_data", {})
            
            height_min = terrain.get("height_min")
            height_max = terrain.get("height_max")
            wdl_data = terrain.get("wdl_heights")
            
            has_height_bounds = height_min is not None and height_max is not None
            has_wdl = wdl_data is not None and "outer_17" in (wdl_data or {})
            
            if has_height_bounds:
                stats["height_ranges"].append((height_min, height_max))
            
        except Exception as e:
            issues.append(f"{tile_name}: JSON parse error - {e}")
            has_height_bounds = False
            has_wdl = False
            height_min = height_max = None
        
        # Check image sizes
        if has_heightmap:
            try:
                img = Image.open(heightmap_path)
                stats["heightmap_sizes"].append(img.size)
            except Exception as e:
                issues.append(f"{tile_name}: Heightmap load error - {e}")
        
        if has_normalmap:
            try:
                img = Image.open(normalmap_path)
                stats["normalmap_sizes"].append(img.size)
            except Exception as e:
                issues.append(f"{tile_name}: Normalmap load error - {e}")
        
        # Print sample details
        print(f"\n  Sample: {tile_name}")
        print(f"    Minimap:      {'✅' if has_minimap else '❌'}")
        print(f"    Normalmap:    {'✅' if has_normalmap else '❌'}")
        print(f"    Heightmap:    {'✅' if has_heightmap else '❌'}")
        print(f"    Height bounds: {'✅' if has_height_bounds else '❌'} ", end="")
        if has_height_bounds:
            print(f"[{height_min:.2f}, {height_max:.2f}]")
        else:
            print("")
        print(f"    WDL data:     {'✅' if has_wdl else '❌'}")
    
    # Count all files
    for json_path in json_files:
        tile_name = json_path.stem
        
        if (images_dir / f"{tile_name}.png").exists():
            stats["has_minimap"] += 1
        if (images_dir / f"{tile_name}_normalmap.png").exists():
            stats["has_normalmap"] += 1
        if (images_dir / f"{tile_name}_heightmap_v2_preview.png").exists():
            stats["has_heightmap"] += 1
        
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            terrain = data.get("terrain_data", {})
            
            if terrain.get("height_min") is not None and terrain.get("height_max") is not None:
                stats["has_height_bounds"] += 1
            
            wdl = terrain.get("wdl_heights")
            if wdl and "outer_17" in wdl:
                stats["has_wdl"] += 1
            
            # Check if complete for V6 training (WDL optional - uses fallback)
            has_all = (
                (images_dir / f"{tile_name}.png").exists() and
                (images_dir / f"{tile_name}_normalmap.png").exists() and
                (images_dir / f"{tile_name}_heightmap_v2_preview.png").exists() and
                terrain.get("height_min") is not None
            )
            if has_all:
                stats["complete"] += 1
                
        except Exception:
            pass
    
    # Print summary
    print(f"\n  Summary:")
    print(f"    Total JSON files:     {stats['total_json']}")
    print(f"    With minimap:         {stats['has_minimap']}")
    print(f"    With normalmap:       {stats['has_normalmap']}")
    print(f"    With heightmap:       {stats['has_heightmap']}")
    print(f"    With height bounds:   {stats['has_height_bounds']}")
    print(f"    With WDL data:        {stats['has_wdl']}")
    print(f"    Complete for V6:      {stats['complete']}")
    
    # Check image sizes
    if stats["heightmap_sizes"]:
        unique_hm = set(stats["heightmap_sizes"])
        print(f"\n  Heightmap sizes: {unique_hm}")
        if (256, 256) not in unique_hm:
            print(f"    ⚠️ WARNING: Heightmaps not 256x256!")
    
    if stats["normalmap_sizes"]:
        unique_nm = set(stats["normalmap_sizes"])
        print(f"  Normalmap sizes: {unique_nm}")
        if (256, 256) not in unique_nm:
            print(f"    ⚠️ WARNING: Normalmaps not 256x256!")
    
    # Height range statistics
    if stats["height_ranges"]:
        mins = [r[0] for r in stats["height_ranges"]]
        maxs = [r[1] for r in stats["height_ranges"]]
        print(f"\n  Height range samples:")
        print(f"    Min heights: {min(mins):.2f} to {max(mins):.2f}")
        print(f"    Max heights: {min(maxs):.2f} to {max(maxs):.2f}")
    
    if issues:
        print(f"\n  Issues found:")
        for issue in issues[:5]:
            print(f"    - {issue}")
    
    return stats


def main():
    print("=" * 60)
    print("TRAINING DATA VERIFICATION")
    print("=" * 60)
    
    all_stats = []
    
    for root in DATASET_ROOTS:
        stats = check_dataset(root)
        if stats:
            all_stats.append(stats)
    
    # Overall summary
    print(f"\n{'='*60}")
    print("OVERALL SUMMARY")
    print(f"{'='*60}")
    
    total_json = sum(s["total_json"] for s in all_stats)
    total_complete = sum(s["complete"] for s in all_stats)
    
    print(f"Total tiles:          {total_json}")
    print(f"Complete for V6:      {total_complete}")
    print(f"Coverage:             {100*total_complete/total_json:.1f}%" if total_json > 0 else "0%")
    
    if total_complete == 0:
        print("\n❌ NO COMPLETE SAMPLES - Cannot train V6!")
        print("\nMissing data - check:")
        print("  - Were heightmaps re-rendered? (render_heightmaps.py)")
        print("  - Were normalmaps re-rendered? (render_normalmaps.py)")
        print("  - Do JSONs have height_min/max? (VlmDatasetExporter)")
        print("  - Do JSONs have wdl_heights? (VlmDatasetExporter)")
    elif total_complete < total_json:
        print(f"\n⚠️ {total_json - total_complete} tiles missing some data")
    else:
        print("\n✅ All tiles ready for V6 training!")


if __name__ == "__main__":
    main()
