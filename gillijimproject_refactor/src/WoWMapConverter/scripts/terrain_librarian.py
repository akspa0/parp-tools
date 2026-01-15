"""
Terrain Librarian - Prefabricated Brush Detector
===============================================
Analyzes VLM Terrain Data to identifying recurring "Prefab" patterns 
in Heightmaps and Alpha Masks.

Core Logic:
1.  Ingest VLM JSON Dataset.
2.  Extract Chunk Fingerprints (Heights + Alpha).
3.  Cluster similar chunks using Nearest Neighbors or Hashing.
4.  Identify "Macro Patterns" (Spatial groups of prefabs).
5.  Export Universal Prefab Library.
"""

import json
import argparse
import base64
import numpy as np
import cv2
import hashlib
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from typing import List, Dict, Tuple, Any

def decode_alpha(base64_str: str, size=64) -> np.ndarray:
    """Decode base64 alpha mask to numpy array."""
    try:
        data = base64.b64decode(base64_str)
        # Assuming 8-bit alpha (64x64 = 4096 bytes) or compressed
        # If raw bytes match size^2, reshape.
        if len(data) == size * size:
            return np.frombuffer(data, dtype=np.uint8).reshape((size, size))
        
        # If compressed or minimal, we might need to rely on the PNG or skip
        # For now, simplistic check
        return np.zeros((size, size), dtype=np.uint8)
    except:
        return np.zeros((size, size), dtype=np.uint8)

def get_canonical_geometry_hash(heights: List[float]) -> str:
    """
    Compute a hash invariant to rotation and mirroring.
    145 floats = 9x9 Outer + 8x8 Inner.
    Structure in file is row-interleaved:
    Row 0: 9 Outer
    Row 0: 8 Inner
    Row 1: 9 Outer
    ...
    """
    if not heights or len(heights) != 145:
        return "invalid"

    # De-interlace
    outer = np.zeros((9, 9), dtype=np.float32)
    inner = np.zeros((8, 8), dtype=np.float32)
    
    idx = 0
    for r in range(9):
        # Outer row
        for c in range(9):
            outer[r, c] = heights[idx]
            idx += 1
        
        # Inner row (only 8 rows)
        if r < 8:
            for c in range(8):
                inner[r, c] = heights[idx]
                idx += 1
                
    # Normalize Heights (Relative to Min Z)
    min_z = min(outer.min(), inner.min())
    outer -= min_z
    inner -= min_z
    
    # Generate 8 Symmetries
    hashes = []
    
    # Operations: (Reflect, Rotate)
    # 0: I
    # 1: Rot90
    # 2: Rot180
    # 3: Rot270
    # 4: FlipLR
    # 5: FlipLR + Rot90
    # ...
    
    # Base images
    o_base = outer
    i_base = inner
    
    for flip_code in [None, 1]: # None=NoFlip, 1=FlipHorizontal
        o_flipped = o_base if flip_code is None else np.fliplr(o_base)
        i_flipped = i_base if flip_code is None else np.fliplr(i_base)
        
        for k in range(4): # 0, 1, 2, 3 rotations (90 deg)
            o_rot = np.rot90(o_flipped, k)
            i_rot = np.rot90(i_flipped, k)
            
            # Hash
            payload = o_rot.tobytes() + i_rot.tobytes()
            hashes.append(hashlib.sha256(payload).hexdigest())
            
    return min(hashes) # The canonical hash

import re

# Hash function: Geometry + Alpha (Stricter "Visual Prefab")
def get_prefab_hash(heights: List[float], layers: List[Dict]) -> str:
    """Hash Geometry + Alpha Patterns."""
    # 1. Geometry (Canonical or Raw? User said 'square patterns... in masks', so assume mirroring matters less? 
    # Use Canonical anyway to deduplicate mirrored stamps)
    geo_sig = get_canonical_geometry_hash(heights)
    
    # 2. Key Alpha Layers (e.g. Dirt, Grass, Road)
    # We hash the Base64 strings directly (assuming consistent encoding)
    alpha_sig = ""
    if layers:
        for l in layers:
            # Handle potential None values safely
            bits = l.get("alpha_bits")
            alpha_sig += (bits if bits else "")
            
    return hashlib.sha256((geo_sig + alpha_sig).encode('utf-8')).hexdigest()

def process_tile_full(json_path: Path) -> Dict[str, Any]:
    """Extract full chunk data for unique prefabs."""
    extracted = {} # hash -> chunk_data
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
            
        td = data.get("terrain_data", {})
        tile_name = td.get("adt_tile", json_path.stem)
        
        # Index data by chunk_idx
        c_heights = {x["idx"]: x for x in td.get("heights", [])}
        c_layers = {x["idx"]: x["layers"] for x in td.get("chunk_layers", [])}
        c_normals = {x["idx"]: x for x in td.get("normals", [])} if "normals" in td else {}
        c_mccv = {x["idx"]: x for x in td.get("mccv_colors", [])} if "mccv_colors" in td else {}
        
        # Iterate chunks
        for i in range(256):
            if i in c_heights:
                h_obj = c_heights[i]
                l_list = c_layers.get(i, [])
                
                # Compute Hash
                sig = get_prefab_hash(h_obj["h"], l_list)
                
                # Store full chunk object (only if needed by caller, but we return all for dedupe)
                extracted[sig] = {
                    "hash": sig,
                    "heights": h_obj,
                    "layers": l_list,
                    "normals": c_normals.get(i),
                    "mccv": c_mccv.get(i),
                    "source": f"{tile_name}_c{i}"
                }
                
    except Exception as e:
        print(f"Error {json_path}: {e}")
        
    return extracted

def main():
    parser = argparse.ArgumentParser(description="Terrain Prefab Librarian & Gallery Generator")
    parser.add_argument("input_dir", help="Folder containing VLM JSONs")
    parser.add_argument("output_dir", help="Output folder for Gallery JSONs")
    args = parser.parse_args()
    
    input_path = Path(args.input_dir)
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # 1. Scan Files
    json_files = list(input_path.glob("dataset/*.json"))
    if not json_files: json_files = list(input_path.glob("*.json"))
    
    if not json_files:
        print("No JSON files found.")
        return

    print(f"Scanning {len(json_files)} tiles for Unique Prefabs (Geometry + Textures)...")
    
    # 2. Collect Unique Prefabs
    unique_prefabs = {} # hash -> data
    
    # Serial for now to avoid memory explosion if picking defaults
    # (Or parallel then merge)
    with ProcessPoolExecutor() as exe:
        for res in exe.map(process_tile_full, json_files):
            for sig, data in res.items():
                if sig not in unique_prefabs:
                    unique_prefabs[sig] = data
                # We could track counts here if we returned counts separately
                
    print(f"Found {len(unique_prefabs)} unique prefabs.")
    
    # 3. Macro Prefab Detection (Seed-and-Grow)
    # Build Global Grid
    global_grid = {} # (x, y) -> hash
    chunk_db = {}    # hash -> chunk_data
    
    print("Building Global Hash Grid...")
    # Re-scan to build grid
    for p in json_files:
        try:
            with open(p, 'r', encoding='utf-8') as f:
                data = json.load(f)
                import re
                match = re.search(r'_(\d+)_(\d+)\.json', p.name)
                if match:
                    tx, ty = int(match.group(1)), int(match.group(2))
                    extracted = process_tile_full(p)
                    for sig, cdata in extracted.items():
                        c_idx = int(cdata["source"].split("_c")[-1])
                        gx = tx * 16 + (c_idx % 16)
                        gy = ty * 16 + (c_idx // 16)
                        
                        global_grid[(gx, gy)] = sig
                        if sig not in chunk_db:
                            chunk_db[sig] = cdata
        except Exception: 
            pass

    print(f"Global Grid Size: {len(global_grid)} chunks.")
    
    # Invert Index: Hash -> List of Coords
    # Coords are (global_x, global_y), we might want Tile+ChunkIdx for easier lookup
    # But (gx, gy) is fine if we can map back.
    # Actually, let's store detailed locations: {"tile": name, "chunk": cidx}
    
    # We need to rebuild this map with rich data or just dump the global grid.
    # Let's dump the global grid (sparse) to 'prefab_instances.json'
    # Format: "hash": ["x_y", "x_y"...] or something compact.
    
    instance_map_path = output_path / "prefab_instances.json"
    
    # Convert tuple keys to string for JSON
    # Map: Hash -> List of "gx,gy"
    hash_to_locs = defaultdict(list)
    for (gx, gy), h in global_grid.items():
        # We need the Tile Name to load the image later!
        # Calculating Tile Name from Gx/Gy is tricky without the map layout offset (we assumed 30_30? No, we parsed it).
        # We parsed it in the loop but didn't store the map.
        
        # We need to store (TileName, ChunkIdx) in the grid or db.
        # Let's update global_grid to store (Hash, TileName, ChunkIdx)
        # Wait, global_grid is (x,y)->hash.
        # Chunk DB is hash->data (which has 'source' = Tile_cIdx).
        # We can use Chunk DB to get *one* source, but not *all*.
        
        pass 
    
    # Re-scan for full instance map? Expensive.
    # We already have 'global_grid' = (gx, gy) -> hash.
    # We can perform the heuristic: TileX = gx // 16, TileY = gy // 16.
    # TileName = f"Kalidar_{TileX}_{TileY}" (Matches regex input).
    
    hash_to_rich_locs = defaultdict(list)
    for (gx, gy), h in global_grid.items():
        tx = gx // 16
        ty = gy // 16
        c_idx = (gy % 16) * 16 + (gx % 16)
        # Warning: This assumes the input file naming convention matches map coords!
        # The regex `_(\d+)_(\d+)` was used to populate gx/gy, so this reverse is valid.
        
        tile_name = f"Kalidar_{tx}_{ty}" # Hardcoded prefix? Or assume caller knows.
        # Actually better to just store "gx_gy": "hash" and let the loader figure it out.
        hash_to_rich_locs[h].append(f"{tile_name}.json:{c_idx}")

    with open(instance_map_path, 'w') as f:
        json.dump(hash_to_rich_locs, f, indent=2)
        
    print(f"Instance Map saved to {instance_map_path}")

    hash_to_coords = defaultdict(list)
    for coord, h in global_grid.items():
        hash_to_coords[h].append(coord)
        
    print("Finding Macro Prefabs (Variable Size)...")
    
    # Store found rectangles: Signature -> Data
    # Signature = Hash of the block hashes
    macro_prefabs = {} 
    visited_seeds = set() # Avoid processing same pair twice
    
    # Iterate all hashes that have >1 occurrence
    repeating_hashes = [h for h, coords in hash_to_coords.items() if len(coords) > 1]
    
    # Optimization: Sort by frequency? Or ignore common "flat" tiles?
    # Skipping the most common hash (flat terrain) might save massive time
    # But user might want to see large flat areas.
    # Let's limit processed seeds for speed if needed.
    
    for h_val in repeating_hashes:
        coords = hash_to_coords[h_val]
        # Compare every pair (expensive!)
        # If len(coords) is huge (e.g. 8000), this is O(N^2).
        # We MUST skip massive groups (likely flat/ocean).
        if len(coords) > 500: 
            continue 
            
        for i in range(len(coords)):
            for j in range(i + 1, len(coords)):
                p1 = coords[i]
                p2 = coords[j]
                
                if (p1, p2) in visited_seeds: continue
                
                # Check neighbors to grow rectangle
                # Grow Right (Width)
                w = 1
                while True:
                    next_p1 = (p1[0] + w, p1[1])
                    next_p2 = (p2[0] + w, p2[1])
                    if next_p1 in global_grid and next_p2 in global_grid:
                        if global_grid[next_p1] == global_grid[next_p2]:
                            w += 1
                            continue
                    break
                    
                # Grow Down (Height)
                h_dim = 1
                while True:
                    # Check entire row at h_dim
                    match_row = True
                    for k in range(w):
                        check_p1 = (p1[0] + k, p1[1] + h_dim)
                        check_p2 = (p2[0] + k, p2[1] + h_dim)
                        
                        if check_p1 not in global_grid or check_p2 not in global_grid:
                            match_row = False; break
                        if global_grid[check_p1] != global_grid[check_p2]:
                            match_row = False; break
                    
                    if match_row:
                        h_dim += 1
                    else:
                        break
                        
                # Filter small stuff (User said: avoid 1x1, 2x1, 2x2?)
                # "probably not 2x2" -> So > 4 chunks?
                # Or at least W>=2 AND H>=2 ?
                # Or Area >= 6?
                if w * h_dim < 6:
                    continue
                    
                # We found a macro match!
                # Record it
                # Extract signature (hashes of the block)
                block_hashes = []
                for r in range(h_dim):
                    for c in range(w):
                        block_hashes.append(global_grid[(p1[0]+c, p1[1]+r)])
                
                macro_sig = hashlib.sha256(("".join(block_hashes)).encode('utf-8')).hexdigest()
                
                if macro_sig not in macro_prefabs:
                    macro_prefabs[macro_sig] = {
                        "width": w,
                        "height": h_dim,
                        "hashes": block_hashes, # Row-major
                        "count": 0,
                        "example": p1 # Top-left coord of one instance
                    }
                macro_prefabs[macro_sig]["count"] += 1
                
                # Mark as visited (approximate optimization)
                visited_seeds.add((p1, p2))

    print(f"Found {len(macro_prefabs)} macro prefabs (Variable Size > 6 chunks).")

    # Pack into Gallery
    # Group by size? Mixed packing?
    # Simple: One macro per file if huge, or grid packing.
    # Grid packing variable sizes is hard (Bin Packing).
    # Simplification: Only pack top ones, or one per tile if lazy.
    # Beaker: "Just contains the collection, for all to see first-hand"
    
    # Let's pack them into 16x16 tiles.
    # Simple Shelf Packing algorithm.
    
    # Sort macros by height then width
    sorted_macros = sorted(macro_prefabs.values(), key=lambda m: (m["height"], m["width"]), reverse=True)
    
    current_tile_idx = 0
    current_tile_x = 0
    current_tile_y = 0
    current_row_h = 0
    
    # We output a list of (macro_data, tile_index, offset_x, offset_y)
    placements = []
    
    current_tx = 0
    current_ty = 0
    row_h = 0
    
    # Current tile capacity
    MAX_W = 16
    MAX_H = 16
    
    active_tile_cmds = [] # List of (macro, x, y)
    
    def flush_tile(cmds, t_idx):
        if not cmds: return
        
        # Tile Coords
        # Start at 50_50
        tx, ty = 50 + (t_idx % 32), 50 + (t_idx // 32)
        tile_name = f"Macro_Zoo_{tx}_{ty}"
        
        chunk_heights_out = []
        chunk_layers_out = []
        chunk_normals_out = []
        chunk_mccv_out = []
        
        for m_data, off_x, off_y in cmds:
            w, h = m_data["width"], m_data["height"]
            hashes = m_data["hashes"]
            
            # Place macro chunks
            for r in range(h):
                for c in range(w):
                    h_val = hashes[r*w + c]
                    c_data = chunk_db.get(h_val)
                    if not c_data: continue # Should not happen
                    
                    target_cx = off_x + c
                    target_cy = off_y + r
                    
                    if target_cx >= 16 or target_cy >= 16: continue
                    
                    target_cidx = target_cy * 16 + target_cx
                    
                    # Clone data
                    h_copy = c_data["heights"].copy()
                    h_copy["idx"] = target_cidx
                    chunk_heights_out.append(h_copy)
                    
                    if c_data["layers"]:
                        chunk_layers_out.append({"idx": target_cidx, "layers": c_data["layers"]})
                    if c_data["normals"]:
                        n_copy = c_data["normals"].copy()
                        n_copy["idx"] = target_cidx
                        chunk_normals_out.append(n_copy)
                    if c_data["mccv"]:
                        m_copy = c_data["mccv"].copy()
                        m_copy["idx"] = target_cidx
                        chunk_mccv_out.append(m_copy)
                        
        # Write JSON
        gallery_json = {
            "terrain_data": {
                "adt_tile": tile_name,
                "heights": chunk_heights_out,
                "chunk_layers": chunk_layers_out,
                "normals": chunk_normals_out,
                "mccv_colors": chunk_mccv_out,
                "liquids": [], "objects": [], "holes": [], "chunk_positions": []
            },
            "image": f"images/{tile_name}.png"
        }
        
        out_file = output_path / f"{tile_name}.json"
        with open(out_file, 'w') as f:
            json.dump(gallery_json, f, indent=2)

    # Shelf Packing
    # Reset
    cx, cy, rh = 0, 0, 0
    t_idx = 0
    
    for m in sorted_macros:
        mw, mh = m["width"], m["height"]
        
        if mw > 16 or mh > 16:
            print(f"Skipping macro too big for tile: {mw}x{mh}")
            continue
            
        # Check if fits in current row
        if cx + mw > 16:
            # New row
            cx = 0
            cy += rh
            rh = 0
            
        # Check if fits in tile (height)
        if cy + mh > 16:
            # New Tile
            flush_tile(active_tile_cmds, t_idx)
            active_tile_cmds = []
            t_idx += 1
            cx, cy, rh = 0, 0, 0
            
        # Place
        active_tile_cmds.append((m, cx, cy))
        cx += mw
        rh = max(rh, mh)
        
    # Flush last
    flush_tile(active_tile_cmds, t_idx)
            
    print(f"Macro Gallery Complete in {output_path}")

if __name__ == "__main__":
    main()
