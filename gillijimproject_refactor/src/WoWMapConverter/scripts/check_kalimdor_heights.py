#!/usr/bin/env python3
"""Quick check of Kalimdor height data."""
import json
from pathlib import Path

json_path = Path(r"J:\wowDev\parp-tools\gillijimproject_refactor\test_data\vlm-datasets\053_kalimdor_v4\dataset\Kalimdor_19_12.json")

with open(json_path, 'r') as f:
    data = json.load(f)

terrain = data.get("terrain_data", {})
print(f"height_min: {terrain.get('height_min')}")
print(f"height_max: {terrain.get('height_max')}")

heights = terrain.get("heights", [])
print(f"Number of chunks: {len(heights)}")

if heights:
    all_h = []
    for chunk in heights:
        h = chunk.get("h", [])
        all_h.extend(h)
    
    print(f"Total height values: {len(all_h)}")
    if all_h:
        print(f"Actual min: {min(all_h):.4f}")
        print(f"Actual max: {max(all_h):.4f}")
        print(f"Sample values: {all_h[:10]}")
        
        # Check if all values are the same
        unique = set(all_h)
        print(f"Unique values: {len(unique)}")
