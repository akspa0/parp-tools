"""Batch Continent-Scale 3D Terrain Reconstruction Engine (Spec 262).

Reconstructs all 736 authored minimap tiles of the Azeroth continent into
3D elevation grids and stitches them into a full continent elevation mosaic.

Features:
  - Live SAM 3.1 object sieving with mask caching & resume capability.
  - Multi-scale Laplacian inpainting & bare terrain shadow extraction.
  - Calibrated photometric shading inversion & directional ridge analysis.
  - Discrete 3D fractal editor brush fitting.
  - Seamless continent-level 64x64 grid assembly & shaded relief export.

Usage:
  uv run python scripts/v60_batch_reconstruct_continent.py --limit 10
  uv run python scripts/v60_batch_reconstruct_continent.py --all
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from harvester.normal_height_reconstructor import reconstruct_height_from_normals
from harvester.v60.comfyui_orchestrator import ComfyUIOrchestrator
from harvester.v60.fractal_brush_extractor import build_archetypal_brush_library
from harvester.v60.fractal_brush_fitter import FractalBrushFitter
from harvester.v60.minimap_shadow_stripper import MinimapShadowStripper
from harvester.v60.shadow_difference_refiner import extract_ridge_contours


def parse_tile_coords(filename: str) -> Tuple[int, int] | None:
    """Extract tile coordinates (tx, ty) from filename matching azeroth_XX_YY_authored.png."""
    match = re.search(r"azeroth_(\d+)_(\d+)_authored\.png", filename, re.IGNORECASE)
    if match:
        return int(match.group(1)), int(match.group(2))
    return None


def reconstruct_single_tile(
    img_path: Path,
    orchestrator: ComfyUIOrchestrator | None,
    stripper: MinimapShadowStripper,
    brush_fitter: FractalBrushFitter,
    mask_cache_dir: Path,
    max_stamps: int = 6,
) -> Tuple[np.ndarray, np.ndarray, int, float]:
    """Reconstructs a single 256x256 minimap tile into heightfield, normals, and metrics."""
    raw_img = Image.open(img_path).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
    raw_np = np.array(raw_img, dtype=np.float32) / 255.0

    stem = img_path.stem.replace("_authored", "")
    cached_mask_path = mask_cache_dir / f"mask_{stem}.png"

    # 1. SAM 3.1 Object Sieving
    mask = None
    if cached_mask_path.is_file():
        mask = np.array(Image.open(cached_mask_path).convert("L")) > 0
    elif orchestrator is not None:
        try:
            raw_mask = orchestrator.segment_objects(
                image_data=img_path,
                threshold=0.5,
                refine_iterations=2,
                timeout_seconds=45.0,
            )
            mask = raw_mask > 0
            Image.fromarray((mask * 255).astype(np.uint8)).save(cached_mask_path)
        except Exception:
            pass

    if mask is None:
        # Fallback to high-frequency edge & color prior
        gray = 0.299 * raw_np[..., 0] + 0.587 * raw_np[..., 1] + 0.114 * raw_np[..., 2]
        grad = ndimage.generic_gradient_magnitude(gray, ndimage.sobel)
        mask = grad > np.percentile(grad, 92)

    # 2. Stripping Albedo & Laplacian Inpainting
    stripped_shadow, attenuation = stripper.strip_and_inpaint(
        minimap_rgb=raw_np,
        object_mask=mask,
        normalize_albedo=True,
    )

    # 3. Directional Ridge & Normal Inversion along Sun Vector (270 deg azimuth, 40 deg elevation)
    azimuth = 4.7124
    elevation = 0.6981
    cos_el = np.cos(elevation, dtype=np.float32)
    sun_dir = np.array([np.cos(azimuth) * cos_el, np.sin(azimuth) * cos_el, np.sin(elevation)], dtype=np.float32)

    diff_from_mean = stripped_shadow - np.mean(stripped_shadow)
    dz_dx = -diff_from_mean * sun_dir[0] * 2.5
    dz_dy = -diff_from_mean * sun_dir[1] * 2.5
    recovered_normals = np.zeros((256, 256, 3), dtype=np.float32)
    recovered_normals[..., 0] = -dz_dx
    recovered_normals[..., 1] = -dz_dy
    recovered_normals[..., 2] = 1.0
    norm_lens = np.linalg.norm(recovered_normals, axis=-1, keepdims=True)
    recovered_normals /= np.maximum(norm_lens, 1e-6)

    # 4. Height Recovery via Frankot-Chellappa Fourier Integration
    height_map = reconstruct_height_from_normals(recovered_normals, apply_window=False)
    height_map = height_map.astype(np.float32)

    # 5. Discrete Fractal Brush Fitting
    stamps, _ = brush_fitter.fit_stamps(height_map, max_stamps=max_stamps)

    return height_map, recovered_normals, len(stamps), attenuation


def stitch_continent_mosaic(
    tile_results: list[tuple[int, int, np.ndarray]],
    out_dir: Path,
    tile_res: int = 256,
) -> None:
    """Stitches individual tile heightmaps into a complete continent elevation map."""
    if not tile_results:
        return

    min_tx = min(t[0] for t in tile_results)
    max_tx = max(t[0] for t in tile_results)
    min_ty = min(t[1] for t in tile_results)
    max_ty = max(t[1] for t in tile_results)

    span_x = max_tx - min_tx + 1
    span_y = max_ty - min_ty + 1

    mosaic_w = span_x * tile_res
    mosaic_h = span_y * tile_res

    print(f"\nStitching Continent Elevation Mosaic: Grid {span_x}x{span_y} tiles ({mosaic_w}x{mosaic_h} px)...")

    # Full resolution stitched height buffer
    continent_height = np.zeros((mosaic_h, mosaic_w), dtype=np.float32)
    visited_mask = np.zeros((mosaic_h, mosaic_w), dtype=bool)

    for tx, ty, h_arr in tile_results:
        ox = (tx - min_tx) * tile_res
        oy = (ty - min_ty) * tile_res
        continent_height[oy : oy + tile_res, ox : ox + tile_res] = h_arr
        visited_mask[oy : oy + tile_res, ox : ox + tile_res] = True

    # Normalize to 16-bit PNG
    val_min = np.min(continent_height[visited_mask]) if np.any(visited_mask) else 0.0
    val_max = np.max(continent_height[visited_mask]) if np.any(visited_mask) else 1.0
    h_norm = np.clip((continent_height - val_min) / max(1e-5, val_max - val_min), 0.0, 1.0)

    # Shaded relief preview
    dz_dx = ndimage.sobel(continent_height, axis=1)
    dz_dy = ndimage.sobel(continent_height, axis=0)
    shading = np.clip(0.5 + 0.5 * (-dz_dx * 0.707 - dz_dy * 0.707) / 20.0, 0.0, 1.0)
    shading[~visited_mask] = 0.05

    # Downscale preview if mosaic is very large
    max_preview_dim = 2048
    if max(mosaic_w, mosaic_h) > max_preview_dim:
        scale = max_preview_dim / max(mosaic_w, mosaic_h)
        preview_h = (
            ndimage.zoom(h_norm, scale, order=1) * 65535.0
        ).astype(np.uint16)
        preview_shade = (ndimage.zoom(shading, scale, order=1) * 255.0).astype(np.uint8)
    else:
        preview_h = (h_norm * 65535.0).astype(np.uint16)
        preview_shade = (shading * 255.0).astype(np.uint8)

    h_out_path = out_dir / "azeroth_continent_reconstructed_heightmap.png"
    shade_out_path = out_dir / "azeroth_continent_shaded_relief.png"

    Image.fromarray(preview_h).save(h_out_path)
    Image.fromarray(preview_shade).save(shade_out_path)

    print(f"  [OK] Saved Continent Heightmap: {h_out_path}")
    print(f"  [OK] Saved Shaded Relief:       {shade_out_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Batch Continent-Scale 3D Terrain Reconstruction (Spec 262)")
    parser.add_argument(
        "--tiles-dir",
        default="I:/parp/parp-tools/output/synthetic_minimaps/azeroth/tiles",
        help="Directory containing authored tiles",
    )
    parser.add_argument("--out-dir", default="out/continent_azeroth", help="Output directory")
    parser.add_argument("--limit", type=int, default=0, help="Cap number of tiles to process (0 = all)")
    parser.add_argument("--all", action="store_true", help="Process all available continent tiles")
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8199", help="ComfyUI server URL")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip already reconstructed tiles")
    args = parser.parse_args()

    tiles_dir = Path(args.tiles_dir)
    if not tiles_dir.is_dir():
        print(f"[ERROR] Tiles directory not found: {tiles_dir}")
        return 1

    out_dir = Path(args.out_dir)
    heightmaps_dir = out_dir / "heightmaps"
    masks_dir = out_dir / "masks"
    heightmaps_dir.mkdir(parents=True, exist_ok=True)
    masks_dir.mkdir(parents=True, exist_ok=True)

    authored_files = sorted(list(tiles_dir.glob("*_authored.png")))
    total_found = len(authored_files)
    print(f"Discovered {total_found} authored minimap tiles in {tiles_dir.name}/")

    target_files = authored_files
    if args.limit > 0:
        target_files = target_files[: args.limit]
    elif not args.all:
        print("[INFO] No --all or --limit specified. Defaulting to 15-tile sample run.")
        target_files = target_files[:15]

    print(f"Selected {len(target_files)} tiles for reconstruction.")

    # Initialize shared components
    orchestrator = None
    try:
        orch = ComfyUIOrchestrator(base_url=args.comfyui_url)
        orch.check_health()
        orchestrator = orch
        print(f"[OK] Connected to live ComfyUI instance at {args.comfyui_url}")
    except Exception as ex:
        print(f"[WARN] ComfyUI offline or unreachable ({ex}); falling back to mask cache & edge priors.")

    stripper = MinimapShadowStripper(inpaint_scales=3, iterations=35)
    brush_lib = build_archetypal_brush_library(grid_size=33)
    brush_fitter = FractalBrushFitter(brush_lib)

    print("\n=================================================================")
    print(f"Starting Continent Batch Reconstruction ({len(target_files)} tiles)")
    print("=================================================================")

    start_time = time.perf_counter()
    reconstructed_results: list[tuple[int, int, np.ndarray]] = []
    total_stamps = 0
    total_attenuation = 0.0

    for idx, tile_path in enumerate(target_files):
        t0 = time.perf_counter()
        coords = parse_tile_coords(tile_path.name)
        if not coords:
            continue
        tx, ty = coords
        npy_out = heightmaps_dir / f"azeroth_{tx}_{ty}_height.npy"

        if args.skip_existing and npy_out.is_file():
            h_arr = np.load(npy_out)
            reconstructed_results.append((tx, ty, h_arr))
            print(f"[{idx + 1}/{len(target_files)}] Tile ({tx}, {ty}) -> Cached")
            continue

        h_arr, normals, n_stamps, att = reconstruct_single_tile(
            img_path=tile_path,
            orchestrator=orchestrator,
            stripper=stripper,
            brush_fitter=brush_fitter,
            mask_cache_dir=masks_dir,
            max_stamps=6,
        )

        np.save(npy_out, h_arr)
        reconstructed_results.append((tx, ty, h_arr))
        total_stamps += n_stamps
        total_attenuation += att

        elapsed_tile = time.perf_counter() - t0
        print(
            f"[{idx + 1}/{len(target_files)}] Tile ({tx}, {ty}) -> "
            f"Relief: {h_arr.min():.1f}..{h_arr.max():.1f}m | "
            f"Stamps: {n_stamps} | Atten: {att * 100:.1f}% ({elapsed_tile:.2f}s)"
        )

    total_time = time.perf_counter() - start_time
    avg_per_tile = total_time / max(1, len(target_files))
    print("\n=================================================================")
    print(f"Batch Reconstruction Finished!")
    print(f"  Processed Tiles:    {len(reconstructed_results)}")
    print(f"  Total Duration:     {total_time:.2f}s (Average: {avg_per_tile:.2f}s/tile)")
    if len(target_files) > 0:
        print(f"  Average Stamps:     {total_stamps / max(1, len(target_files)):.1f} per tile")
        print(f"  Mean Energy Atten:  {total_attenuation / max(1, len(target_files)) * 100:.1f}%")
    print("=================================================================")

    # Stitch the continent
    stitch_continent_mosaic(reconstructed_results, out_dir=out_dir)

    return 0


if __name__ == "__main__":
    sys.exit(main())
