#!/usr/bin/env python3
"""
Heightmap to Dataset Converter

Converts predicted 129x129 heightmaps back to VLM dataset JSON format,
which can then be used to generate ADT files.

The 129x129 grid maps to 16x16 chunks, each with 145 vertices:
- 9x9 outer grid (81 vertices)
- 8x8 inner grid (64 vertices) interleaved

Vertex layout per chunk (145 total):
  Row 0: 9 outer vertices
  Row 1: 8 inner vertices  
  Row 2: 9 outer vertices
  ... alternating ...
  Row 16: 9 outer vertices (last row)
"""

import json
import argparse
import numpy as np
from pathlib import Path
from PIL import Image
from datetime import datetime


def load_heightmap(path: Path) -> np.ndarray:
    """Load heightmap from 16-bit or 8-bit PNG."""
    img = Image.open(path)
    
    if img.mode == 'I;16':
        # 16-bit grayscale
        arr = np.array(img, dtype=np.float32) / 65535.0
    elif img.mode == 'L':
        # 8-bit grayscale
        arr = np.array(img, dtype=np.float32) / 255.0
    else:
        # Convert to grayscale
        img = img.convert('L')
        arr = np.array(img, dtype=np.float32) / 255.0
    
    return arr


def extract_chunk_heights(heightmap: np.ndarray, chunk_idx: int) -> list:
    """
    Extract 145 height values for a single chunk from the 129x129 grid.
    
    The 129x129 grid has 8 vertices per chunk edge + 1 shared vertex.
    Each chunk covers an 8x8 region of the grid (plus shared edges).
    
    Chunk layout (16x16 chunks):
    - chunk_idx = row * 16 + col
    - Each chunk spans 8 vertices in each direction
    
    MCVT vertex order (145 total):
    - Alternating rows of 9 outer and 8 inner vertices
    - Outer vertices are on the chunk boundary
    - Inner vertices are at half-step positions
    """
    row = chunk_idx // 16
    col = chunk_idx % 16
    
    # Starting position in the 129x129 grid
    # Each chunk is 8 vertices wide/tall
    start_y = row * 8
    start_x = col * 8
    
    heights = []
    
    # MCVT has 17 rows alternating between 9 and 8 vertices
    for i in range(17):
        if i % 2 == 0:
            # Outer row: 9 vertices at integer positions
            outer_y = start_y + (i // 2)
            for j in range(9):
                x = start_x + j
                # Clamp to grid bounds
                x = min(x, 128)
                y = min(outer_y, 128)
                heights.append(float(heightmap[y, x]))
        else:
            # Inner row: 8 vertices at half-step positions
            # These are interpolated between outer vertices
            inner_y = start_y + (i // 2)
            for j in range(8):
                x = start_x + j
                # Inner vertices are at (x+0.5, y+0.5) - interpolate
                x0, x1 = min(x, 127), min(x + 1, 128)
                y0, y1 = min(inner_y, 127), min(inner_y + 1, 128)
                
                # Bilinear interpolation for inner vertex
                h = (heightmap[y0, x0] + heightmap[y0, x1] + 
                     heightmap[y1, x0] + heightmap[y1, x1]) / 4.0
                heights.append(float(h))
    
    return heights


def compute_chunk_normals(heightmap: np.ndarray, chunk_idx: int, scale: float = 1.0) -> list:
    """
    Compute normals for a chunk from the heightmap.
    Returns 145 * 3 = 435 signed bytes (x, y, z per vertex).
    """
    row = chunk_idx // 16
    col = chunk_idx % 16
    start_y = row * 8
    start_x = col * 8
    
    normals = []
    
    for i in range(17):
        if i % 2 == 0:
            # Outer row
            outer_y = start_y + (i // 2)
            for j in range(9):
                x = start_x + j
                x = min(x, 128)
                y = min(outer_y, 128)
                
                # Compute normal from height gradient
                nx, ny, nz = compute_normal_at(heightmap, x, y, scale)
                normals.extend([nx, ny, nz])
        else:
            # Inner row
            inner_y = start_y + (i // 2)
            for j in range(8):
                x = start_x + j
                x = min(x, 127)
                y = min(inner_y, 127)
                
                # Average normal for inner vertex
                nx, ny, nz = compute_normal_at(heightmap, x, y, scale)
                normals.extend([nx, ny, nz])
    
    return normals


def compute_normal_at(heightmap: np.ndarray, x: int, y: int, scale: float) -> tuple:
    """Compute normal at a point using central differences."""
    h, w = heightmap.shape
    
    # Get neighboring heights
    h_left = heightmap[y, max(0, x - 1)]
    h_right = heightmap[y, min(w - 1, x + 1)]
    h_up = heightmap[max(0, y - 1), x]
    h_down = heightmap[min(h - 1, y + 1), x]
    
    # Gradient
    dx = (h_right - h_left) * scale
    dy = (h_down - h_up) * scale
    
    # Normal vector (pointing up)
    nx = -dx
    ny = -dy
    nz = 1.0
    
    # Normalize
    length = np.sqrt(nx*nx + ny*ny + nz*nz)
    if length > 0:
        nx /= length
        ny /= length
        nz /= length
    
    # Convert to signed bytes (-127 to 127)
    nx_byte = int(np.clip(nx * 127, -127, 127))
    ny_byte = int(np.clip(ny * 127, -127, 127))
    nz_byte = int(np.clip(nz * 127, 0, 127))  # Z is always positive
    
    return nx_byte, ny_byte, nz_byte


def heightmap_to_dataset_json(
    heightmap: np.ndarray,
    tile_name: str,
    minimap_path: Path = None,
    height_scale: float = 100.0,
    base_height: float = 0.0
) -> dict:
    """
    Convert a 129x129 heightmap to VLM dataset JSON format.
    
    Args:
        heightmap: 129x129 normalized heightmap (0-1 range)
        tile_name: Name like "MapName_X_Y"
        minimap_path: Optional path to minimap image
        height_scale: Scale factor for height values
        base_height: Base height offset
    
    Returns:
        Dataset JSON dict compatible with training scripts
    """
    if heightmap.shape != (129, 129):
        raise ValueError(f"Expected 129x129 heightmap, got {heightmap.shape}")
    
    # Scale heights to world units
    scaled_heights = heightmap * height_scale + base_height
    
    # Extract per-chunk data
    heights_data = []
    chunk_layers_data = []
    
    for chunk_idx in range(256):
        # Heights
        chunk_heights = extract_chunk_heights(scaled_heights, chunk_idx)
        heights_data.append({
            "idx": chunk_idx,
            "h": chunk_heights
        })
        
        # Normals
        chunk_normals = compute_chunk_normals(heightmap, chunk_idx, height_scale)
        
        # Chunk layers (minimal - just base layer)
        chunk_layers_data.append({
            "idx": chunk_idx,
            "layers": [{
                "tex_id": 0,
                "texture_path": "Tileset\\Generic\\Black.blp",
                "flags": 0,
                "alpha_off": 0,
                "effect_id": 65535,
                "ground_effects": None,
                "alpha_bits": None,
                "alpha_path": None
            }],
            "shadow_path": None,
            "normals": chunk_normals
        })
    
    # Build dataset JSON
    dataset = {
        "image": f"images/{tile_name}.png" if minimap_path else None,
        "depth": None,
        "terrain_data": {
            "adt_tile": tile_name,
            "heights": heights_data,
            "holes": [0] * 256,  # No holes
            "textures": ["Tileset\\Generic\\Black.blp"],
            "chunk_layers": chunk_layers_data
        }
    }
    
    return dataset


def process_heightmap(
    heightmap_path: Path,
    output_dir: Path,
    tile_name: str = None,
    minimap_path: Path = None,
    height_scale: float = 100.0,
    base_height: float = 0.0
):
    """Process a single heightmap and save dataset JSON."""
    
    # Load heightmap
    heightmap = load_heightmap(heightmap_path)
    
    # Resize to 129x129 if needed
    if heightmap.shape != (129, 129):
        from PIL import Image
        img = Image.fromarray((heightmap * 255).astype(np.uint8))
        img = img.resize((129, 129), Image.BILINEAR)
        heightmap = np.array(img, dtype=np.float32) / 255.0
    
    # Determine tile name
    if tile_name is None:
        tile_name = heightmap_path.stem.replace('_heightmap', '').replace('_preview', '')
    
    # Create dataset JSON
    dataset = heightmap_to_dataset_json(
        heightmap, tile_name, minimap_path, height_scale, base_height
    )
    
    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    
    json_path = output_dir / f"{tile_name}.json"
    with open(json_path, 'w') as f:
        json.dump(dataset, f, indent=2)
    
    print(f"Saved: {json_path}")
    return json_path


def main():
    parser = argparse.ArgumentParser(
        description="Convert predicted heightmaps to VLM dataset JSON format"
    )
    parser.add_argument("input", type=str, help="Heightmap image or directory")
    parser.add_argument("--output", type=str, help="Output directory for JSON files")
    parser.add_argument("--height-scale", type=float, default=100.0, 
                        help="Height scale factor (default: 100)")
    parser.add_argument("--base-height", type=float, default=0.0,
                        help="Base height offset (default: 0)")
    parser.add_argument("--minimap-dir", type=str, 
                        help="Directory containing minimap images")
    args = parser.parse_args()
    
    input_path = Path(args.input)
    
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = input_path.parent / "dataset"
    
    if input_path.is_file():
        # Single file
        minimap_path = None
        if args.minimap_dir:
            tile_name = input_path.stem.replace('_heightmap', '').replace('_preview', '')
            minimap_path = Path(args.minimap_dir) / f"{tile_name}.png"
            if not minimap_path.exists():
                minimap_path = None
        
        process_heightmap(
            input_path, output_dir,
            minimap_path=minimap_path,
            height_scale=args.height_scale,
            base_height=args.base_height
        )
    else:
        # Directory - process all heightmaps
        heightmaps = list(input_path.glob("*_heightmap.png")) + \
                     list(input_path.glob("*_heightmap_preview.png"))
        
        if not heightmaps:
            # Try any PNG that's not a minimap/normalmap
            heightmaps = [p for p in input_path.glob("*.png") 
                         if not any(s in p.stem for s in ['minimap', 'normalmap', 'mask'])]
        
        print(f"Found {len(heightmaps)} heightmaps")
        
        for hm_path in heightmaps:
            tile_name = hm_path.stem.replace('_heightmap', '').replace('_preview', '')
            
            minimap_path = None
            if args.minimap_dir:
                minimap_path = Path(args.minimap_dir) / f"{tile_name}.png"
                if not minimap_path.exists():
                    minimap_path = None
            
            try:
                process_heightmap(
                    hm_path, output_dir,
                    tile_name=tile_name,
                    minimap_path=minimap_path,
                    height_scale=args.height_scale,
                    base_height=args.base_height
                )
            except Exception as e:
                print(f"Error processing {hm_path.name}: {e}")
    
    print(f"\nDataset JSON files saved to: {output_dir}")
    print("Use WoWMapConverter.Core to convert these to ADT files.")


if __name__ == "__main__":
    main()
