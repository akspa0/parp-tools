#!/usr/bin/env python3
"""
Heightmap Archaeology Tool
==========================
Explores sedimentary layers in WoW heightmap data to reveal hidden development history.

Features:
1. Generate relative heightmaps per tile (shows local detail)
2. Cluster tiles by height range to identify development timeframes
3. Visualize "eraser brush" patterns used to scrub terrain
4. Export 3D OBJ meshes with global height normalization
5. Export analysis as sorted/grouped PNGs

Usage:
    python heightmap_archaeology.py --dataset test_data/vlm-datasets/053_azeroth_v20 --output archaeology_output/ --export-mesh
"""

import argparse
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
from tqdm import tqdm

# Height bucket thresholds for clustering tiles by era/depth
HEIGHT_BUCKETS = [
    ("surface", -100, 500),
    ("shallow_buried", -300, -100),
    ("deep_buried", -600, -300),
    ("abyss", -1000, -600),
    ("void", -float('inf'), -1000),
]

# Global height range for mesh export (ancient terrain limits)
GLOBAL_HEIGHT_MIN = -1000.0
GLOBAL_HEIGHT_MAX = 1000.0

# WoW tile size in world units
TILE_SIZE = 533.33333  # ADT tile is 533.33 yards


def load_tile_json(json_path: Path) -> dict:
    """Load a tile JSON and extract height data."""
    with open(json_path, 'r') as f:
        data = json.load(f)
    
    td = data.get("terrain_data", {})
    return {
        "tile_name": json_path.stem,
        "height_min": td.get("height_min"),
        "height_max": td.get("height_max"),
        "height_global_min": td.get("height_global_min"),
        "height_global_max": td.get("height_global_max"),
        "heights": td.get("heights", []),
        "holes": td.get("holes", []),
    }


def build_height_grid(tile_data: dict, resolution: int = 145) -> np.ndarray:
    """
    Build a height grid from tile JSON data.
    Returns grid in world height units (not normalized).
    
    WoW MCVT format: 145 heights per chunk (9×9 outer + 8×8 inner interleaved)
    For simplicity, we use only the 9×9 outer vertices, giving us:
    - 16×16 chunks with 9×9 vertices each
    - Shared edges: chunk (0,0) right edge = chunk (1,0) left edge
    - Total unique vertices: (9-1)*16 + 1 = 129 per axis
    """
    heights = tile_data.get("heights", [])
    if not heights:
        return None
    
    # Build a 129×129 grid with proper edge sharing
    # Each chunk contributes 8 new columns/rows + shares 1 edge
    # 16 chunks * 8 + 1 shared = 129
    grid_size = 129
    full_grid = np.zeros((grid_size, grid_size), dtype=np.float32)
    
    for chunk_data in heights:
        if not isinstance(chunk_data, dict):
            continue
        idx = chunk_data.get("idx", 0)
        h = chunk_data.get("h", [])
        if len(h) < 81:
            continue
        
        chunk_x = idx % 16
        chunk_y = idx // 16
        
        # Extract outer 9×9 vertices
        outer = np.array(h[:81]).reshape(9, 9)
        
        # Place in grid with shared edges
        # Each chunk starts at offset (chunk_x * 8, chunk_y * 8)
        # and contributes a 9×9 region (overlapping edge with next chunk)
        gx = chunk_x * 8
        gy = chunk_y * 8
        
        # Place all 9×9 vertices (edge vertices overwrite previous - they should match)
        full_grid[gy:gy+9, gx:gx+9] = outer
    
    # Resize to requested resolution if needed
    if resolution != grid_size:
        from scipy.ndimage import zoom
        scale = resolution / grid_size
        full_grid = zoom(full_grid, scale, order=1)
    
    return full_grid


def export_tile_obj(tile_data: dict, output_path: Path, tile_x: int, tile_y: int,
                    global_min: float = GLOBAL_HEIGHT_MIN, 
                    global_max: float = GLOBAL_HEIGHT_MAX) -> bool:
    """
    Export a single tile as OBJ mesh with global height normalization.
    Tiles are positioned in world space based on their X,Y coordinates.
    """
    grid = build_height_grid(tile_data, resolution=65)  # Smaller for faster export
    if grid is None:
        return False
    
    res = grid.shape[0]
    vertices = []
    faces = []
    
    # Calculate world position offset for this tile
    world_x_offset = tile_x * TILE_SIZE
    world_z_offset = tile_y * TILE_SIZE
    
    # Normalize heights to global range for visualization
    # This maps -1000..1000 to a reasonable visual scale
    height_scale = 1.0  # Keep world units for now
    
    # Generate vertices
    for gy in range(res):
        for gx in range(res):
            # World position
            wx = world_x_offset + (gx / (res - 1)) * TILE_SIZE
            wz = world_z_offset + (gy / (res - 1)) * TILE_SIZE
            wy = grid[gy, gx] * height_scale  # Height in world units
            
            # Clamp to global range for visualization
            wy = np.clip(wy, global_min, global_max)
            
            vertices.append((wx, wy, wz))
    
    # Generate faces (two triangles per quad)
    for gy in range(res - 1):
        for gx in range(res - 1):
            # Vertex indices (1-indexed for OBJ)
            v0 = gy * res + gx + 1
            v1 = gy * res + (gx + 1) + 1
            v2 = (gy + 1) * res + (gx + 1) + 1
            v3 = (gy + 1) * res + gx + 1
            
            faces.append((v0, v1, v2))
            faces.append((v0, v2, v3))
    
    # Write OBJ file
    with open(output_path, 'w') as f:
        f.write(f"# Heightmap Archaeology Export\n")
        f.write(f"# Tile: {tile_data['tile_name']}\n")
        f.write(f"# Height range: {tile_data.get('height_min', 0):.1f} to {tile_data.get('height_max', 0):.1f}\n")
        f.write(f"# Global normalization: {global_min} to {global_max}\n\n")
        
        for v in vertices:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        
        f.write(f"\n# {len(faces)} faces\n")
        for face in faces:
            f.write(f"f {face[0]} {face[1]} {face[2]}\n")
    
    return True


def export_combined_obj(tiles: list, output_path: Path, era_filter: str = None,
                        global_min: float = GLOBAL_HEIGHT_MIN,
                        global_max: float = GLOBAL_HEIGHT_MAX,
                        height_scale: float = 1.0,
                        tile_min: float = None,
                        tile_max: float = None) -> int:
    """
    Export multiple tiles as a single combined OBJ mesh.
    Optionally filter by era classification or tile height range.
    """
    all_vertices = []
    all_faces = []
    vertex_offset = 0
    exported_count = 0
    
    # Find coordinate bounds for normalization
    tile_coords = []
    for tile in tiles:
        if tile.get("height_min") is None:
            continue
        
        # Apply tile height filter
        if tile_min is not None and tile.get("height_min", 0) < tile_min:
            continue
        if tile_max is not None and tile.get("height_max", 0) > tile_max:
            continue
            
        parts = tile["tile_name"].split("_")
        if len(parts) >= 3:
            try:
                tx = int(parts[-2])
                ty = int(parts[-1])
                tile_coords.append((tx, ty, tile))
            except ValueError:
                continue
    
    if not tile_coords:
        return 0, None
    
    # Find coordinate bounds
    min_tx = min(tc[0] for tc in tile_coords)
    min_ty = min(tc[1] for tc in tile_coords)
    max_tx = max(tc[0] for tc in tile_coords)
    max_ty = max(tc[1] for tc in tile_coords)
    
    # For texture atlas, we'll create UV coordinates spanning the entire tile range
    tile_range_x = max_tx - min_tx + 1
    tile_range_y = max_ty - min_ty + 1
    
    for tile_x, tile_y, tile in tqdm(tile_coords, desc="Building mesh"):
        # Optional era filter
        if era_filter:
            era = classify_tile_era(tile["height_min"], tile["height_max"])
            if era != era_filter:
                continue
        
        grid = build_height_grid(tile, resolution=33)
        if grid is None:
            continue
        
        res = grid.shape[0]
        
        # Calculate world position
        offset_x = (tile_x - min_tx) * TILE_SIZE
        offset_z = (tile_y - min_ty) * TILE_SIZE
        
        # UV coordinates: map each tile to its portion of the texture atlas
        # Each tile occupies a (1/tile_range_x, 1/tile_range_y) region
        uv_base_x = (tile_x - min_tx) / tile_range_x
        uv_base_y = (tile_y - min_ty) / tile_range_y
        uv_scale_x = 1.0 / tile_range_x
        uv_scale_y = 1.0 / tile_range_y
        
        step = TILE_SIZE / (res - 1)
        
        # Add vertices with UVs
        for gy in range(res):
            for gx in range(res):
                wx = offset_x + gx * step
                wz = offset_z + gy * step
                # Apply height scale before clipping
                raw_height = grid[gy, gx] * height_scale
                wy = np.clip(raw_height, global_min, global_max)
                
                # UV: local position within tile (0-1) scaled to atlas region
                u = uv_base_x + (gx / (res - 1)) * uv_scale_x
                v = uv_base_y + (gy / (res - 1)) * uv_scale_y
                
                all_vertices.append((wx, wy, wz, u, v))
        
        # Add faces
        for gy in range(res - 1):
            for gx in range(res - 1):
                v0 = vertex_offset + gy * res + gx + 1
                v1 = vertex_offset + gy * res + (gx + 1) + 1
                v2 = vertex_offset + (gy + 1) * res + (gx + 1) + 1
                v3 = vertex_offset + (gy + 1) * res + gx + 1
                all_faces.append((v0, v3, v2))
                all_faces.append((v0, v2, v1))
        
        vertex_offset += res * res
        exported_count += 1
    
    if not all_vertices:
        return 0
    
    # Write combined OBJ with UVs
    print(f"Writing {len(all_vertices)} vertices, {len(all_faces)} faces...")
    mtl_name = output_path.stem
    with open(output_path, 'w') as f:
        f.write(f"# Heightmap Archaeology - Combined Export\n")
        f.write(f"# Tiles: {exported_count}\n")
        f.write(f"# Global height range: {global_min} to {global_max}\n")
        f.write(f"mtllib {mtl_name}.mtl\n\n")
        
        # Write vertices
        for v in all_vertices:
            f.write(f"v {v[0]:.2f} {v[1]:.2f} {v[2]:.2f}\n")
        
        f.write(f"\n# Texture coordinates\n")
        for v in all_vertices:
            f.write(f"vt {v[3]:.6f} {v[4]:.6f}\n")
        
        f.write(f"\nusemtl terrain\n")
        for face in all_faces:
            # f v1/vt1 v2/vt2 v3/vt3
            f.write(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}\n")
    
    # Write MTL file with texture reference
    mtl_path = output_path.with_suffix('.mtl')
    texture_name = f"{mtl_name}_atlas.png"
    with open(mtl_path, 'w') as f:
        f.write("# Heightmap material with minimap texture\n")
        f.write("newmtl terrain\n")
        f.write("Ka 1.0 1.0 1.0\n")
        f.write("Kd 1.0 1.0 1.0\n")
        f.write("Ks 0.0 0.0 0.0\n")
        f.write(f"map_Kd {texture_name}\n")
    
    # Return info needed for atlas generation
    return exported_count, {
        "min_tx": min_tx, "min_ty": min_ty,
        "max_tx": max_tx, "max_ty": max_ty,
        "tile_range_x": tile_range_x, "tile_range_y": tile_range_y,
        "exported_tiles": [(tx, ty, t["tile_name"]) for tx, ty, t in tile_coords 
                          if not era_filter or classify_tile_era(t["height_min"], t["height_max"]) == era_filter],
        "texture_path": output_path.parent / texture_name
    }


def generate_texture_atlas(atlas_info: dict, images_dir: Path, tile_size: int = 256) -> bool:
    """
    Generate a texture atlas from minimap images.
    
    Args:
        atlas_info: Dictionary with tile bounds and list of tiles
        images_dir: Path to images directory containing minimap PNGs
        tile_size: Size of each tile in the atlas (default 256)
    """
    from PIL import Image
    
    if not atlas_info or not atlas_info.get("exported_tiles"):
        return False
    
    tile_range_x = atlas_info["tile_range_x"]
    tile_range_y = atlas_info["tile_range_y"]
    min_tx = atlas_info["min_tx"]
    min_ty = atlas_info["min_ty"]
    texture_path = atlas_info["texture_path"]
    
    # Create atlas image
    atlas_width = tile_range_x * tile_size
    atlas_height = tile_range_y * tile_size
    
    # Limit max size to avoid memory issues
    max_size = 8192
    if atlas_width > max_size or atlas_height > max_size:
        scale = min(max_size / atlas_width, max_size / atlas_height)
        tile_size = int(tile_size * scale)
        atlas_width = tile_range_x * tile_size
        atlas_height = tile_range_y * tile_size
        print(f"Atlas too large, scaling to {atlas_width}x{atlas_height} (tile_size={tile_size})")
    
    print(f"Creating {atlas_width}x{atlas_height} texture atlas...")
    atlas = Image.new('RGB', (atlas_width, atlas_height), color=(32, 32, 32))
    
    placed = 0
    for tx, ty, tile_name in tqdm(atlas_info["exported_tiles"], desc="Stitching atlas"):
        # Find minimap image
        minimap_path = images_dir / f"{tile_name}_minimap.png"
        if not minimap_path.exists():
            continue
        
        try:
            tile_img = Image.open(minimap_path).convert('RGB')
            tile_img = tile_img.resize((tile_size, tile_size), Image.LANCZOS)
            
            # Calculate position in atlas
            ax = (tx - min_tx) * tile_size
            ay = (ty - min_ty) * tile_size
            
            atlas.paste(tile_img, (ax, ay))
            placed += 1
        except Exception as e:
            print(f"  Error loading {minimap_path.name}: {e}")
    
    print(f"Placed {placed} tiles in atlas")
    atlas.save(texture_path, 'PNG')
    print(f"Atlas saved to {texture_path}")
    
    return True


def render_relative_heightmap(tile_data: dict, output_path: Path):
    """Render a per-tile relative heightmap that shows local detail."""
    heights = tile_data.get("heights", [])
    if not heights:
        return None
    
    grid = np.zeros((256, 256), dtype=np.float32)
    
    for chunk_data in heights:
        if not isinstance(chunk_data, dict):
            continue
        idx = chunk_data.get("idx", 0)
        h = chunk_data.get("h", [])
        if len(h) < 81:
            continue
        
        chunk_y = idx // 16
        chunk_x = idx % 16
        outer = np.array(h[:81]).reshape(9, 9)
        
        from scipy.ndimage import zoom
        upsampled = zoom(outer, 16/9, order=1)[:16, :16]
        
        py = chunk_y * 16
        px = chunk_x * 16
        grid[py:py+16, px:px+16] = upsampled
    
    h_min = grid.min()
    h_max = grid.max()
    if h_max - h_min < 0.01:
        normalized = np.zeros_like(grid)
    else:
        normalized = (grid - h_min) / (h_max - h_min)
    
    img = Image.fromarray((normalized * 255).astype(np.uint8), mode='L')
    img.save(output_path)
    
    return {"tile_min": h_min, "tile_max": h_max, "range": h_max - h_min}


def classify_tile_era(height_min: float, height_max: float) -> str:
    """Classify a tile into a developmental 'era' based on height range."""
    avg_height = (height_min + height_max) / 2
    for era_name, lo, hi in HEIGHT_BUCKETS:
        if lo <= avg_height < hi:
            return era_name
    return "unknown"


def generate_height_histogram(tiles: list, output_path: Path):
    """Generate a histogram of tile height distributions."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    
    mins = [t["height_min"] for t in tiles if t["height_min"] is not None]
    maxs = [t["height_max"] for t in tiles if t["height_max"] is not None]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    
    axes[0].hist(mins, bins=50, color='blue', alpha=0.7)
    axes[0].set_title('Distribution of Tile Minimum Heights')
    axes[0].set_xlabel('Height (world units)')
    axes[0].axvline(x=0, color='red', linestyle='--', label='Sea level')
    axes[0].legend()
    
    axes[1].hist(maxs, bins=50, color='green', alpha=0.7)
    axes[1].set_title('Distribution of Tile Maximum Heights')
    axes[1].set_xlabel('Height (world units)')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Heightmap Archaeology Tool")
    parser.add_argument("--dataset", type=Path, required=True, help="Path to VLM dataset")
    parser.add_argument("--output", type=Path, default=Path("archaeology_output"), help="Output directory")
    parser.add_argument("--render-relative", action="store_true", help="Render relative heightmaps per tile")
    parser.add_argument("--analyze-eras", action="store_true", help="Classify tiles into developmental eras")
    parser.add_argument("--find-outliers", action="store_true", help="Find tiles with unusual height ranges")
    parser.add_argument("--export-mesh", action="store_true", help="Export 3D OBJ meshes")
    parser.add_argument("--mesh-era", type=str, default=None, help="Filter mesh export by era (surface, shallow_buried, deep_buried, abyss)")
    parser.add_argument("--height-min", type=float, default=GLOBAL_HEIGHT_MIN, help="Global height minimum for mesh (default: -1000)")
    parser.add_argument("--height-max", type=float, default=GLOBAL_HEIGHT_MAX, help="Global height maximum for mesh (default: 1000)")
    parser.add_argument("--height-scale", type=float, default=1.0, help="Scale factor for heights (use 2-4 for buried terrain)")
    parser.add_argument("--tile-min", type=float, default=None, help="Only include tiles with min height >= this value")
    parser.add_argument("--tile-max", type=float, default=None, help="Only include tiles with max height <= this value")
    parser.add_argument("--all", action="store_true", help="Run all analyses (not mesh)")
    args = parser.parse_args()
    
    if args.all:
        args.render_relative = True
        args.analyze_eras = True
        args.find_outliers = True
    
    dataset_dir = args.dataset / "dataset"
    args.output.mkdir(parents=True, exist_ok=True)
    
    # Load all tile data
    print(f"Loading tiles from {dataset_dir}...")
    json_files = list(dataset_dir.glob("*.json"))
    tiles = []
    for jf in tqdm(json_files, desc="Loading JSONs"):
        try:
            tile_data = load_tile_json(jf)
            tile_data["json_path"] = jf
            tiles.append(tile_data)
        except Exception as e:
            print(f"  Error loading {jf.name}: {e}")
    
    print(f"Loaded {len(tiles)} tiles")
    
    # Export 3D meshes
    if args.export_mesh:
        print(f"\nExporting 3D meshes (height range: {args.height_min} to {args.height_max})...")
        mesh_dir = args.output / "meshes"
        mesh_dir.mkdir(exist_ok=True)
        
        # Export combined mesh
        era_suffix = f"_{args.mesh_era}" if args.mesh_era else ""
        combined_path = mesh_dir / f"combined{era_suffix}.obj"
        result = export_combined_obj(tiles, combined_path, 
                                    era_filter=args.mesh_era,
                                    global_min=args.height_min,
                                    global_max=args.height_max,
                                    height_scale=args.height_scale,
                                    tile_min=args.tile_min,
                                    tile_max=args.tile_max)
        count, atlas_info = result
        print(f"Exported {count} tiles to {combined_path}")
        
        # Generate texture atlas from minimaps
        images_dir = args.dataset / "images"
        if images_dir.exists() and atlas_info:
            generate_texture_atlas(atlas_info, images_dir)
        else:
            print(f"(Skipping atlas - images directory not found: {images_dir})")
        
        # Also export by era if not filtered (skip if custom height filter is used)
        if not args.mesh_era and args.tile_min is None and args.tile_max is None:
            for era_name, _, _ in HEIGHT_BUCKETS:
                era_path = mesh_dir / f"{era_name}.obj"
                result = export_combined_obj(tiles, era_path, era_filter=era_name,
                                                global_min=args.height_min,
                                                global_max=args.height_max,
                                                height_scale=args.height_scale)
                era_count, era_atlas_info = result
                if era_count > 0:
                    print(f"  {era_name}: {era_count} tiles → {era_path.name}")
                    if images_dir.exists() and era_atlas_info:
                        generate_texture_atlas(era_atlas_info, images_dir)
    
    # Render relative heightmaps
    if args.render_relative:
        print("\nRendering relative heightmaps...")
        relative_dir = args.output / "relative_heightmaps"
        relative_dir.mkdir(exist_ok=True)
        
        for tile in tqdm(tiles, desc="Rendering"):
            out_path = relative_dir / f"{tile['tile_name']}_relative.png"
            try:
                render_relative_heightmap(tile, out_path)
            except Exception as e:
                print(f"  Error rendering {tile['tile_name']}: {e}")
    
    # Analyze developmental eras
    if args.analyze_eras:
        print("\nAnalyzing developmental eras...")
        era_groups = defaultdict(list)
        
        for tile in tiles:
            if tile["height_min"] is None:
                continue
            era = classify_tile_era(tile["height_min"], tile["height_max"])
            era_groups[era].append(tile)
        
        eras_dir = args.output / "by_era"
        eras_dir.mkdir(exist_ok=True)
        
        with open(args.output / "era_summary.txt", "w") as f:
            f.write("Developmental Era Classification\n")
            f.write("=" * 40 + "\n\n")
            for era, tiles_in_era in sorted(era_groups.items()):
                f.write(f"{era}: {len(tiles_in_era)} tiles\n")
                
                era_subdir = eras_dir / era
                era_subdir.mkdir(exist_ok=True)
                
                with open(era_subdir / "tiles.txt", "w") as era_f:
                    for t in sorted(tiles_in_era, key=lambda x: x["height_min"] or 0):
                        era_f.write(f"{t['tile_name']}: [{t['height_min']:.1f}, {t['height_max']:.1f}]\n")
        
        print(f"Era classification saved to {args.output / 'era_summary.txt'}")
    
    # Find outliers
    if args.find_outliers:
        print("\nFinding outlier tiles...")
        valid_tiles = [t for t in tiles if t["height_min"] is not None]
        if valid_tiles:
            ranges = [(t["height_max"] - t["height_min"], t) for t in valid_tiles]
            ranges.sort(key=lambda x: x[0])
            
            outliers_dir = args.output / "outliers"
            outliers_dir.mkdir(exist_ok=True)
            
            with open(outliers_dir / "flattest_tiles.txt", "w") as f:
                f.write("Tiles with smallest height range (potentially erased)\n")
                f.write("=" * 50 + "\n\n")
                for rng, tile in ranges[:50]:
                    f.write(f"{tile['tile_name']}: range={rng:.2f}, min={tile['height_min']:.1f}, max={tile['height_max']:.1f}\n")
            
            with open(outliers_dir / "steepest_tiles.txt", "w") as f:
                f.write("Tiles with largest height range (dramatic terrain)\n")
                f.write("=" * 50 + "\n\n")
                for rng, tile in reversed(ranges[-50:]):
                    f.write(f"{tile['tile_name']}: range={rng:.2f}, min={tile['height_min']:.1f}, max={tile['height_max']:.1f}\n")
            
            buried = sorted(valid_tiles, key=lambda t: t["height_min"])
            with open(outliers_dir / "deepest_tiles.txt", "w") as f:
                f.write("Tiles pushed deepest underground (hidden content?)\n")
                f.write("=" * 50 + "\n\n")
                for tile in buried[:100]:
                    f.write(f"{tile['tile_name']}: min={tile['height_min']:.1f}, max={tile['height_max']:.1f}\n")
            
            print(f"Outlier analysis saved to {outliers_dir}")
    
    # Generate histogram
    try:
        generate_height_histogram(tiles, args.output / "height_distribution.png")
        print(f"Height distribution histogram saved to {args.output / 'height_distribution.png'}")
    except ImportError:
        print("(Skipping histogram - matplotlib not available)")
    
    print("\nArchaeology complete!")


if __name__ == "__main__":
    main()

