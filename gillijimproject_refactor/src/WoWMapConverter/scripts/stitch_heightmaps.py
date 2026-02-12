#!/usr/bin/env python3
"""
Stitch individual tile heightmaps into a full-map PNG.

Usage:
    python stitch_heightmaps.py <dataset_dir> [--max-size 16384] [--output <path>]

Example:
    python stitch_heightmaps.py J:/path/to/053_azeroth_v10
    python stitch_heightmaps.py J:/path/to/053_kalimdor_v5 --max-size 8192
"""

import argparse
import re
from pathlib import Path
import numpy as np
from PIL import Image

def find_heightmap_tiles(images_dir: Path, map_name: str) -> list[tuple[int, int, Path]]:
    """Find all heightmap tiles and parse their coordinates."""
    pattern = re.compile(rf"{re.escape(map_name)}_(\d+)_(\d+)_heightmap\.png$")
    tiles = []
    
    for f in images_dir.glob(f"{map_name}_*_*_heightmap.png"):
        match = pattern.match(f.name)
        if match:
            x, y = int(match.group(1)), int(match.group(2))
            tiles.append((x, y, f))
    
    return tiles

def stitch_heightmaps(dataset_dir: Path, max_size: int = 16384, output_path: Path = None):
    """Stitch all heightmap tiles into a single PNG."""
    
    # Detect map name from directory
    map_name = dataset_dir.name.split('_')[1] if '_' in dataset_dir.name else None
    if not map_name:
        # Try to detect from files
        images_dir = dataset_dir / "images"
        if not images_dir.exists():
            print(f"Error: images/ directory not found in {dataset_dir}")
            return
        
        # Get map name from first heightmap file
        for f in images_dir.glob("*_heightmap.png"):
            parts = f.stem.replace("_heightmap", "").rsplit("_", 2)
            if len(parts) >= 3:
                map_name = parts[0]
                break
    
    if not map_name:
        print("Error: Could not detect map name")
        return
    
    # Capitalize first letter for matching file names
    map_name = map_name.capitalize()
    
    images_dir = dataset_dir / "images"
    tiles = find_heightmap_tiles(images_dir, map_name)
    
    if not tiles:
        print(f"No heightmap tiles found for {map_name} in {images_dir}")
        return
    
    print(f"Found {len(tiles)} heightmap tiles for {map_name}")
    
    # Calculate bounds
    min_x = min(t[0] for t in tiles)
    max_x = max(t[0] for t in tiles)
    min_y = min(t[1] for t in tiles)
    max_y = max(t[1] for t in tiles)
    
    tiles_wide = max_x - min_x + 1
    tiles_high = max_y - min_y + 1
    
    # Each tile is 256x256
    tile_size = 256
    full_width = tiles_wide * tile_size
    full_height = tiles_high * tile_size
    
    print(f"Map bounds: ({min_x},{min_y}) to ({max_x},{max_y})")
    print(f"Full size: {full_width} x {full_height} pixels")
    
    # Calculate scale to fit within max_size
    scale = 1.0
    if full_width > max_size or full_height > max_size:
        scale = min(max_size / full_width, max_size / full_height)
        print(f"Scaling to {scale:.4f} to fit within {max_size}x{max_size}")
    
    output_width = int(full_width * scale)
    output_height = int(full_height * scale)
    scaled_tile_size = int(tile_size * scale)
    
    print(f"Output size: {output_width} x {output_height} pixels")
    
    # Create canvas (16-bit grayscale)
    canvas = np.zeros((output_height, output_width), dtype=np.uint16)
    
    # Load and place each tile
    loaded = 0
    for x, y, path in tiles:
        try:
            img = Image.open(path)
            
            # Resize if needed
            if scale < 1.0:
                img = img.resize((scaled_tile_size, scaled_tile_size), Image.LANCZOS)
            
            # Convert to numpy
            tile_data = np.array(img, dtype=np.uint16)
            
            # Calculate position
            dest_x = int((x - min_x) * scaled_tile_size)
            dest_y = int((y - min_y) * scaled_tile_size)
            
            # Handle edge cases where tile might exceed bounds
            h, w = tile_data.shape[:2] if len(tile_data.shape) > 1 else (tile_data.shape[0], 1)
            if len(tile_data.shape) == 1:
                continue  # Skip malformed tiles
            
            end_y = min(dest_y + h, output_height)
            end_x = min(dest_x + w, output_width)
            
            canvas[dest_y:end_y, dest_x:end_x] = tile_data[:end_y-dest_y, :end_x-dest_x]
            loaded += 1
            
            if loaded % 100 == 0:
                print(f"  Loaded {loaded}/{len(tiles)} tiles...")
                
        except Exception as e:
            print(f"  Warning: Failed to load {path.name}: {e}")
    
    print(f"Loaded {loaded}/{len(tiles)} tiles")
    
    # Determine output path
    if output_path is None:
        stitched_dir = dataset_dir / "stitched"
        stitched_dir.mkdir(exist_ok=True)
        output_path = stitched_dir / f"{map_name}_full_heightmap.png"
    
    # Save as 16-bit PNG
    output_img = Image.fromarray(canvas.astype(np.uint16))
    output_img.save(output_path)
    
    print(f"Saved: {output_path}")
    print(f"Final size: {output_width} x {output_height}")
    
    return output_path

def main():
    parser = argparse.ArgumentParser(description="Stitch heightmap tiles into full-map PNG")
    parser.add_argument("dataset_dir", type=Path, help="Path to VLM dataset directory")
    parser.add_argument("--max-size", type=int, default=16384, 
                        help="Maximum dimension (default: 16384)")
    parser.add_argument("--output", "-o", type=Path, default=None,
                        help="Output path (default: <dataset>/stitched/<map>_full_heightmap.png)")
    
    args = parser.parse_args()
    
    if not args.dataset_dir.exists():
        print(f"Error: Dataset directory not found: {args.dataset_dir}")
        return 1
    
    stitch_heightmaps(args.dataset_dir, args.max_size, args.output)
    return 0

if __name__ == "__main__":
    exit(main())
