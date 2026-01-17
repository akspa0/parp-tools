#!/usr/bin/env python3
"""Debug script to examine heightmap JSON data format."""
import json
from pathlib import Path

dataset_dir = Path("test_data/vlm-datasets/053_azeroth_v20/dataset")
json_files = list(dataset_dir.glob("*.json"))
print(f"Found {len(json_files)} JSON files")

if json_files:
    p = json_files[0]
    print(f"\nExamining: {p.name}")
    
    with open(p) as f:
        d = json.load(f)
    
    td = d.get("terrain_data", {})
    print(f"terrain_data keys: {list(td.keys())}")
    
    heights = td.get("heights", [])
    print(f"Number of height chunks: {len(heights)}")
    
    if heights:
        h = heights[0]
        print(f"\nChunk 0:")
        print(f"  Keys: {list(h.keys())}")
        print(f"  idx: {h.get('idx')}")
        
        h_data = h.get("h", [])
        print(f"  Height array length: {len(h_data)}")
        print(f"  First 10 heights: {h_data[:10]}")
        print(f"  Min height: {min(h_data):.2f}")
        print(f"  Max height: {max(h_data):.2f}")
        
        # Check multiple chunks for edge consistency
        print(f"\nChecking edge values between adjacent chunks...")
        chunk_dict = {c.get("idx"): c.get("h", []) for c in heights}
        
        # Check chunk 0 and chunk 1 (horizontal neighbors)
        if 0 in chunk_dict and 1 in chunk_dict:
            c0_h = chunk_dict[0]
            c1_h = chunk_dict[1]
            if len(c0_h) >= 81 and len(c1_h) >= 81:
                # Chunk 0 right edge vs chunk 1 left edge
                c0_right = [c0_h[i*9 + 8] for i in range(9)]  # Last column
                c1_left = [c1_h[i*9] for i in range(9)]        # First column
                print(f"  Chunk 0 right edge: {c0_right[:3]}...")
                print(f"  Chunk 1 left edge:  {c1_left[:3]}...")
                print(f"  Edge match: {c0_right == c1_left}")
