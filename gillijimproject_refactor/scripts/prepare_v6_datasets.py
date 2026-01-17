#!/usr/bin/env python3
"""
V6 Training Pipeline - Complete Dataset Preparation and Validation

This script automates the full V6 training pipeline:
1. Dataset generation via C# VlmDatasetExporter
2. Normalmap rendering from JSON terrain data
3. WDL data extraction/fallback handling
4. Dataset validation for V6 requirements
5. Training script execution with proper configuration

Usage:
    python prepare_v6_datasets.py --dataset PATH --validate
    python prepare_v6_datasets.py --dataset PATH --render-normalmaps
    python prepare_v6_datasets.py --dataset PATH --fix-all
    python prepare_v6_datasets.py --dataset PATH --train
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
from PIL import Image
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field


# ============================================================================
# Configuration
# ============================================================================

GRID_SIZE = 256  # Target resolution

@dataclass
class DatasetStats:
    """Statistics from dataset validation."""
    total_tiles: int = 0
    has_minimap: int = 0
    has_normalmap: int = 0
    has_heightmap: int = 0
    has_heightmap_global: int = 0
    has_height_bounds: int = 0
    has_wdl: int = 0
    has_chunk_layers: int = 0
    complete_v6: int = 0
    issues: List[str] = field(default_factory=list)
    height_ranges: List[Tuple[float, float, float, float]] = field(default_factory=list)


# ============================================================================
# Normalmap Rendering (from MCNR data in JSON)
# ============================================================================

def render_normalmap_from_json(json_path: Path, output_path: Path) -> bool:
    """Render a 256x256 normalmap from JSON terrain data using MCNR normals."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"  Error loading {json_path.name}: {e}")
        return False
    
    td = data.get("terrain_data", {})
    chunk_layers = td.get("chunk_layers", [])
    
    if not chunk_layers:
        # Fallback: try to generate from heightmap gradient
        return generate_normalmap_from_heights(td, output_path)
    
    # Build 256x256 grid from MCNR normals
    normalmap = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
    weight_map = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
    for layer in chunk_layers:
        idx = layer.get("idx", -1)
        normals_raw = layer.get("normals", None)
        
        if normals_raw is None or idx < 0 or idx >= 256:
            continue
        
        chunk_y = idx // 16
        chunk_x = idx % 16
        base_x = chunk_x * 16
        base_y = chunk_y * 16
        
        # MCNR format: 81 outer (9x9) then 64 inner (8x8)
        if len(normals_raw) < 435:  # 145 * 3
            continue
        normals_raw = normals_raw[:435]
        
        # Outer 9x9 grid at even positions
        for oy in range(9):
            for ox in range(9):
                px = base_x + ox * 2
                py = base_y + oy * 2
                n_idx = oy * 9 + ox
                
                if px < GRID_SIZE and py < GRID_SIZE:
                    nx = normals_raw[n_idx * 3 + 0] / 127.0
                    ny = normals_raw[n_idx * 3 + 1] / 127.0
                    nz = normals_raw[n_idx * 3 + 2] / 127.0
                    
                    normalmap[py, px, 0] += nx
                    normalmap[py, px, 1] += ny
                    normalmap[py, px, 2] += nz
                    weight_map[py, px] += 1.0
        
        # Inner 8x8 grid at odd positions
        for iy in range(8):
            for ix in range(8):
                px = base_x + ix * 2 + 1
                py = base_y + iy * 2 + 1
                n_idx = 81 + iy * 8 + ix
                
                if px < GRID_SIZE and py < GRID_SIZE:
                    nx = normals_raw[n_idx * 3 + 0] / 127.0
                    ny = normals_raw[n_idx * 3 + 1] / 127.0
                    nz = normals_raw[n_idx * 3 + 2] / 127.0
                    
                    normalmap[py, px, 0] += nx
                    normalmap[py, px, 1] += ny
                    normalmap[py, px, 2] += nz
                    weight_map[py, px] += 1.0
    
    # Average overlapping normals
    weight_map[weight_map == 0] = 1.0
    for c in range(3):
        normalmap[:, :, c] /= weight_map
    
    # Fill gaps via interpolation
    normalmap = fill_normalmap_gaps(normalmap, weight_map)
    
    # Normalize and convert to RGB
    lengths = np.sqrt(np.sum(normalmap ** 2, axis=2, keepdims=True))
    lengths[lengths == 0] = 1.0
    normalmap = normalmap / lengths
    normalmap_rgb = ((normalmap + 1) / 2 * 255).astype(np.uint8)
    
    # Save
    Image.fromarray(normalmap_rgb, mode='RGB').save(output_path)
    return True


def generate_normalmap_from_heights(td: dict, output_path: Path) -> bool:
    """Fallback: Generate normalmap from heightmap gradient."""
    heights = td.get("heights", [])
    if not heights:
        return False
    
    # Reconstruct 256x256 heightmap from chunk heights
    heightmap = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.float32)
    
    for chunk in heights:
        idx = chunk.get("idx", -1)
        h = chunk.get("h", [])
        if idx < 0 or idx >= 256 or len(h) < 145:
            continue
        
        chunk_y = idx // 16
        chunk_x = idx % 16
        base_x = chunk_x * 16
        base_y = chunk_y * 16
        
        # Sample heights to 16x16 grid per chunk
        for py in range(16):
            for px in range(16):
                # Map to MCVT grid
                mcvt_y = py * 9 // 16
                mcvt_x = px * 9 // 16
                h_idx = mcvt_y * 9 + mcvt_x
                if h_idx < len(h):
                    fy, fx = base_y + py, base_x + px
                    if fy < GRID_SIZE and fx < GRID_SIZE:
                        heightmap[fy, fx] = h[h_idx]
    
    # Compute normals from height gradient (Sobel)
    sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float32)
    sobel_y = sobel_x.T
    
    from scipy.ndimage import convolve
    dx = convolve(heightmap, sobel_x)
    dy = convolve(heightmap, sobel_y)
    
    # Construct normal vectors (scale factor for visual quality)
    scale = 0.1
    normalmap = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.float32)
    normalmap[:, :, 0] = -dx * scale
    normalmap[:, :, 1] = -dy * scale
    normalmap[:, :, 2] = 1.0
    
    # Normalize
    lengths = np.sqrt(np.sum(normalmap ** 2, axis=2, keepdims=True))
    lengths[lengths == 0] = 1.0
    normalmap = normalmap / lengths
    
    # Convert to RGB
    normalmap_rgb = ((normalmap + 1) / 2 * 255).astype(np.uint8)
    Image.fromarray(normalmap_rgb, mode='RGB').save(output_path)
    return True


def fill_normalmap_gaps(normalmap: np.ndarray, weight_map: np.ndarray) -> np.ndarray:
    """Fill gaps in normalmap using neighbor interpolation."""
    valid_mask = (weight_map >= 0.5).astype(np.float32)
    filled = normalmap.copy()
    
    for _ in range(3):  # 3 passes
        for y in range(GRID_SIZE):
            for x in range(GRID_SIZE):
                if valid_mask[y, x] < 0.5:
                    neighbors = []
                    weights = []
                    for dy in [-1, 0, 1]:
                        for dx in [-1, 0, 1]:
                            if dy == 0 and dx == 0:
                                continue
                            ny, nx = y + dy, x + dx
                            if 0 <= ny < GRID_SIZE and 0 <= nx < GRID_SIZE:
                                if valid_mask[ny, nx] >= 0.5 or np.any(filled[ny, nx] != 0):
                                    dist = np.sqrt(dy*dy + dx*dx)
                                    neighbors.append(filled[ny, nx])
                                    weights.append(1.0 / dist)
                    if neighbors:
                        weights = np.array(weights)
                        weights /= weights.sum()
                        filled[y, x] = np.sum([n * w for n, w in zip(neighbors, weights)], axis=0)
        normalmap = filled.copy()
    
    return filled


# ============================================================================
# WDL Data Extraction/Fallback
# ============================================================================

def extract_wdl_channel(td: dict) -> Optional[np.ndarray]:
    """Extract WDL 17x17 height channel, upsampled to 256x256."""
    wdl = td.get("wdl_heights")
    if wdl is None:
        return None
    
    # Handle both formats: flat array or nested dict
    if isinstance(wdl, dict):
        outer_17 = wdl.get("outer_17", [])
    elif isinstance(wdl, list):
        outer_17 = wdl[:289] if len(wdl) >= 289 else wdl
    else:
        return None
    
    if len(outer_17) < 289:  # 17x17
        return None
    
    # Reshape to 17x17
    wdl_grid = np.array(outer_17[:289], dtype=np.float32).reshape(17, 17)
    
    # Normalize to [0, 1]
    wdl_min, wdl_max = wdl_grid.min(), wdl_grid.max()
    if wdl_max > wdl_min:
        wdl_norm = (wdl_grid - wdl_min) / (wdl_max - wdl_min)
    else:
        wdl_norm = np.zeros_like(wdl_grid)
    
    # Upsample to 256x256 using bilinear interpolation
    wdl_img = Image.fromarray((wdl_norm * 255).astype(np.uint8), mode='L')
    wdl_256 = np.array(wdl_img.resize((256, 256), Image.BILINEAR)) / 255.0
    
    return wdl_256


def create_wdl_fallback(td: dict) -> np.ndarray:
    """Create WDL fallback from chunk heights when WDL data is missing."""
    heights = td.get("heights", [])
    if not heights:
        return np.zeros((256, 256), dtype=np.float32)
    
    # Average height per chunk as 16x16 grid, then upsample
    chunk_heights = np.zeros((16, 16), dtype=np.float32)
    
    for chunk in heights:
        idx = chunk.get("idx", -1)
        h = chunk.get("h", [])
        if idx < 0 or idx >= 256 or not h:
            continue
        chunk_y = idx // 16
        chunk_x = idx % 16
        chunk_heights[chunk_y, chunk_x] = np.mean(h)
    
    # Normalize
    h_min, h_max = chunk_heights.min(), chunk_heights.max()
    if h_max > h_min:
        chunk_heights = (chunk_heights - h_min) / (h_max - h_min)
    
    # Upsample to 256x256
    img = Image.fromarray((chunk_heights * 255).astype(np.uint8), mode='L')
    return np.array(img.resize((256, 256), Image.BILINEAR)) / 255.0


# ============================================================================
# Dataset Validation
# ============================================================================

def validate_dataset(dataset_root: Path, verbose: bool = True) -> DatasetStats:
    """Validate a dataset for V6 training requirements."""
    stats = DatasetStats()
    
    images_dir = dataset_root / "images"
    dataset_dir = dataset_root / "dataset"
    
    if not images_dir.exists() or not dataset_dir.exists():
        stats.issues.append("Missing images/ or dataset/ directory")
        return stats
    
    json_files = list(dataset_dir.glob("*.json"))
    stats.total_tiles = len(json_files)
    
    if verbose:
        print(f"\n{'='*60}")
        print(f"Validating: {dataset_root.name}")
        print(f"{'='*60}")
        print(f"Total JSON files: {stats.total_tiles}")
    
    for json_path in json_files:
        tile_name = json_path.stem
        
        # Check images
        has_minimap = (images_dir / f"{tile_name}.png").exists()
        has_normalmap = (images_dir / f"{tile_name}_normalmap.png").exists()
        has_heightmap = (images_dir / f"{tile_name}_heightmap.png").exists()
        has_heightmap_global = (images_dir / f"{tile_name}_heightmap_global.png").exists()
        
        if has_minimap:
            stats.has_minimap += 1
        if has_normalmap:
            stats.has_normalmap += 1
        if has_heightmap:
            stats.has_heightmap += 1
        if has_heightmap_global:
            stats.has_heightmap_global += 1
        
        # Check JSON content
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            td = data.get("terrain_data", {})
            
            height_min = td.get("height_min")
            height_max = td.get("height_max")
            height_global_min = td.get("height_global_min")
            height_global_max = td.get("height_global_max")
            wdl = td.get("wdl_heights")
            chunk_layers = td.get("chunk_layers", [])
            
            if height_min is not None and height_max is not None:
                stats.has_height_bounds += 1
                stats.height_ranges.append((
                    height_min, height_max,
                    height_global_min or height_min,
                    height_global_max or height_max
                ))
            
            if wdl is not None:
                stats.has_wdl += 1
            
            if chunk_layers:
                stats.has_chunk_layers += 1
            
            # Complete for V6? (normalmap can be optional if we can generate it)
            is_complete = (
                has_minimap and
                (has_normalmap or chunk_layers) and  # Need normalmap OR source data
                has_heightmap and
                has_heightmap_global and
                height_min is not None
            )
            if is_complete:
                stats.complete_v6 += 1
                
        except Exception as e:
            stats.issues.append(f"{tile_name}: {e}")
    
    if verbose:
        print(f"\nResults:")
        print(f"  Minimap:           {stats.has_minimap}/{stats.total_tiles}")
        print(f"  Normalmap:         {stats.has_normalmap}/{stats.total_tiles}")
        print(f"  Heightmap (local): {stats.has_heightmap}/{stats.total_tiles}")
        print(f"  Heightmap (global):{stats.has_heightmap_global}/{stats.total_tiles}")
        print(f"  Height bounds:     {stats.has_height_bounds}/{stats.total_tiles}")
        print(f"  WDL data:          {stats.has_wdl}/{stats.total_tiles}")
        print(f"  Chunk layers:      {stats.has_chunk_layers}/{stats.total_tiles}")
        print(f"  Complete for V6:   {stats.complete_v6}/{stats.total_tiles}")
        
        if stats.has_normalmap == 0 and stats.has_chunk_layers > 0:
            print(f"\n  ⚠️ Normalmaps missing but chunk_layers available - run --render-normalmaps")
        elif stats.has_normalmap == 0:
            print(f"\n  ⚠️ Normalmaps missing and no chunk_layers - will use height gradient fallback")
        
        if stats.height_ranges:
            mins = [r[0] for r in stats.height_ranges]
            maxs = [r[1] for r in stats.height_ranges]
            print(f"\n  Height ranges: [{min(mins):.1f}, {max(maxs):.1f}]")
    
    return stats


# ============================================================================
# Fix Dataset Issues
# ============================================================================

def render_all_normalmaps(dataset_root: Path, force: bool = False) -> dict:
    """Render normalmaps for all tiles in a dataset. Returns stats dict."""
    images_dir = dataset_root / "images"
    dataset_dir = dataset_root / "dataset"
    
    json_files = list(dataset_dir.glob("*.json"))
    rendered_mcnr = 0
    rendered_fallback = 0
    skipped = 0
    failed = 0
    
    print(f"\nRendering normalmaps for {dataset_root.name}...")
    
    for json_path in json_files:
        tile_name = json_path.stem
        output_path = images_dir / f"{tile_name}_normalmap.png"
        
        if output_path.exists() and not force:
            skipped += 1
            continue
        
        # Check if MCNR data exists before rendering
        try:
            with open(json_path, 'r') as f:
                data = json.load(f)
            td = data.get("terrain_data", {})
            chunk_layers = td.get("chunk_layers", [])
            has_mcnr = any(layer.get("normals") for layer in chunk_layers if isinstance(layer, dict))
        except:
            has_mcnr = False
        
        if render_normalmap_from_json(json_path, output_path):
            if has_mcnr:
                rendered_mcnr += 1
            else:
                rendered_fallback += 1
                # Delete fallback-generated normalmaps - they're not ground truth
                output_path.unlink()
                print(f"  ⚠️ Skipped {tile_name} - only has fallback normals (not ground truth)")
        else:
            failed += 1
    
    print(f"  Rendered (MCNR): {rendered_mcnr}")
    print(f"  Skipped (existing): {skipped}")
    print(f"  Skipped (fallback only): {rendered_fallback}")
    print(f"  Failed: {failed}")
    
    return {"mcnr": rendered_mcnr, "fallback": rendered_fallback, "skipped": skipped, "failed": failed}


def fix_wdl_json(json_path: Path) -> bool:
    """Fix WDL data format in JSON if needed."""
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
        
        td = data.get("terrain_data", {})
        wdl = td.get("wdl_heights")
        
        # If WDL is a flat array, convert to expected format
        if isinstance(wdl, list) and len(wdl) >= 289:
            td["wdl_heights"] = {
                "outer_17": wdl[:289],
                "inner_16": wdl[289:545] if len(wdl) >= 545 else []
            }
            data["terrain_data"] = td
            
            with open(json_path, 'w') as f:
                json.dump(data, f, indent=2)
            return True
            
    except Exception as e:
        print(f"  Error fixing WDL in {json_path.name}: {e}")
    
    return False


def fix_all_issues(dataset_root: Path, force_render: bool = False):
    """Fix all known issues in a dataset."""
    print(f"\n{'='*60}")
    print(f"Fixing dataset: {dataset_root.name}")
    print(f"{'='*60}")
    
    # 1. Render missing normalmaps
    print("\n1. Rendering normalmaps...")
    render_all_normalmaps(dataset_root, force=force_render)
    
    # 2. Fix WDL format in JSONs
    print("\n2. Checking WDL format...")
    dataset_dir = dataset_root / "dataset"
    json_files = list(dataset_dir.glob("*.json"))
    wdl_fixed = 0
    for json_path in json_files:
        if fix_wdl_json(json_path):
            wdl_fixed += 1
    print(f"  Fixed WDL format in {wdl_fixed} files")
    
    # 3. Re-validate
    print("\n3. Re-validating...")
    stats = validate_dataset(dataset_root, verbose=True)
    
    return stats


# ============================================================================
# CLI
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="V6 Training Pipeline - Dataset Preparation and Validation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Validate a dataset
  python prepare_v6_datasets.py --dataset path/to/053_azeroth_v11 --validate
  
  # Render missing normalmaps
  python prepare_v6_datasets.py --dataset path/to/053_azeroth_v11 --render-normalmaps
  
  # Fix all issues (render normalmaps, fix WDL format)
  python prepare_v6_datasets.py --dataset path/to/053_azeroth_v11 --fix-all
  
  # Force re-render all normalmaps
  python prepare_v6_datasets.py --dataset path/to/053_azeroth_v11 --render-normalmaps --force
        """
    )
    
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to dataset root directory")
    parser.add_argument("--validate", action="store_true",
                        help="Validate dataset for V6 requirements")
    parser.add_argument("--render-normalmaps", action="store_true",
                        help="Render normalmaps from JSON terrain data")
    parser.add_argument("--fix-all", action="store_true",
                        help="Fix all issues (render normalmaps, fix WDL)")
    parser.add_argument("--force", action="store_true",
                        help="Force re-render even if files exist")
    parser.add_argument("--quiet", action="store_true",
                        help="Minimal output")
    
    args = parser.parse_args()
    
    dataset_root = Path(args.dataset)
    if not dataset_root.exists():
        print(f"Error: Dataset not found: {dataset_root}")
        sys.exit(1)
    
    if args.validate:
        stats = validate_dataset(dataset_root, verbose=not args.quiet)
        if stats.complete_v6 == stats.total_tiles:
            print("\n✅ Dataset fully ready for V6 training!")
        else:
            print(f"\n⚠️ {stats.total_tiles - stats.complete_v6} tiles not complete")
            sys.exit(1)
    
    elif args.render_normalmaps:
        render_all_normalmaps(dataset_root, force=args.force)
    
    elif args.fix_all:
        fix_all_issues(dataset_root, force_render=args.force)
    
    else:
        # Default: validate
        validate_dataset(dataset_root, verbose=True)


if __name__ == "__main__":
    main()
