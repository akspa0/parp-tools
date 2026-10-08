"""One-shot 3D Terrain Reconstruction CLI from Minimap Images (Spec 262).

Pipeline:
  1. SAM 3.1 Object Sieving (excision of roofs, doodads, roads via ComfyUI or mask)
  2. Multi-Scale Laplacian Inpainting & Albedo Stripping (extract bare terrain shadow)
  3. Directional Ridge & Hessian Curvature Extraction
  4. Shading Inversion to Normals (via calibrated photometric solar law)
  5. 3D Fractal Editor Brush Fitting & Height Integration
  6. Materialize 3D Mesh (OBJ + MTL + Textures) and 4-up Diagnostic Quilt

Usage:
  uv run python scripts/v60_reconstruct_minimap.py --image v18_minimap_row04851.png
  uv run python scripts/v60_reconstruct_minimap.py --image path/to/minimap.png --export-obj
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from harvester.normal_height_reconstructor import reconstruct_height_from_normals
from harvester.v60.comfyui_orchestrator import ComfyUIConnectionError, ComfyUIOrchestrator
from harvester.v60.fractal_brush_extractor import build_archetypal_brush_library
from harvester.v60.fractal_brush_fitter import FractalBrushFitter, render_stamps_composition
from harvester.v60.minimap_lighting_solver import render_synthetic_shadow
from harvester.v60.minimap_shadow_stripper import MinimapShadowStripper
from harvester.v60.shadow_difference_refiner import (
    ShadowDifferenceRefiner,
    extract_ridge_contours,
)


from harvester.v60.mesh_exporter import export_glb_mesh, export_obj_mesh
from harvester.v60.sam_minimap_sieve import SamMinimapSieve


def build_diagnostic_quilt(
    minimap_img: Image.Image,
    mask_arr: np.ndarray,
    stripped_arr: np.ndarray,
    normals_arr: np.ndarray,
    ridge_arr: np.ndarray,
) -> Image.Image:
    """Compose a 4-panel diagnostic visual comparison sheet."""
    h, w = 256, 256
    quilt = Image.new("RGB", (w * 4, h))

    # 1. Raw Minimap
    quilt.paste(minimap_img.resize((w, h)), (0, 0))

    # 2. SAM Object Sieve Mask (cyan on raw)
    mask_vis = np.array(minimap_img.resize((w, h)).convert("RGB"))
    m_bool = mask_arr > 0
    mask_vis[m_bool] = (mask_vis[m_bool] * 0.3 + np.array([0, 220, 255]) * 0.7).astype(np.uint8)
    quilt.paste(Image.fromarray(mask_vis), (w, 0))

    # 3. Bare Residual Shadow
    strip_vis = (np.clip(stripped_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    quilt.paste(Image.fromarray(strip_vis).convert("RGB"), (w * 2, 0))

    # 4. Normal Map + Ridge Overlay
    norm_vis = ((normals_arr + 1.0) * 0.5 * 255.0).astype(np.uint8)
    if np.any(ridge_arr > 0):
        r_bool = ridge_arr > 0
        norm_vis[r_bool] = [255, 60, 60]  # Red ridges
    quilt.paste(Image.fromarray(norm_vis), (w * 3, 0))

    return quilt


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot 3D Terrain Reconstruction from Minimap Image (Spec 262)")
    parser.add_argument("--image", default="v18_minimap_row04851.png", help="Path to minimap crop (256x256)")
    parser.add_argument("--out-dir", default="out", help="Output directory")
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8199", help="ComfyUI server URL")
    parser.add_argument("--max-stamps", type=int, default=8, help="Maximum discrete 3D fractal editor brush stamps")
    parser.add_argument("--height-scale", type=float, default=45.0, help="Vertical relief scaling in yards")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        # Fallback to local file in parent
        cand = Path(__file__).resolve().parent.parent / args.image
        if cand.is_file():
            img_path = cand
        else:
            print(f"[ERROR] Image not found: {args.image}")
            return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(f"Spec 262: 3D Terrain Reconstruction Pipeline")
    print(f"Target Image: {img_path.name}")
    print("=================================================================")

    # 1. Load image
    raw_img = Image.open(img_path).convert("RGB")
    raw_img = raw_img.resize((256, 256), Image.Resampling.LANCZOS)
    raw_np = np.array(raw_img, dtype=np.float32) / 255.0

    # 2. SAM 3.1 Object Sieving
    mask_cached = out_dir / f"test_mask_{img_path.stem}.png"
    mask = None

    print("\n[Stage 1/5] SAM 3.1 Minimap Object Sieving...")
    try:
        orchestrator = ComfyUIOrchestrator(base_url=args.comfyui_url)
        print(f"  Querying live ComfyUI instance at {args.comfyui_url}...")
        mask = orchestrator.segment_objects(
            image_data=img_path,
            threshold=0.5,
            refine_iterations=2,
            timeout_seconds=45.0,
        )
        print(f"  Live SAM 3.1 Sieve Success: {int((mask > 0).sum())} non-terrain pixels identified.")
    except Exception as ex:
        print(f"  ComfyUI live query fallback ({ex}).")
        if mask_cached.is_file():
            print(f"  Loading cached SAM mask from {mask_cached}...")
            mask = np.array(Image.open(mask_cached).convert("L")) > 0
        else:
            print("  Using heuristic rooftop chroma anomaly fallback (roads strictly preserved)...")
            mask = SamMinimapSieve.heuristic_color_sieve(raw_np)

    # 3. Multi-Scale Laplacian Inpainting & Albedo Stripping
    print("\n[Stage 2/5] Stripping Albedo & Laplacian Inpainting...")
    stripper = MinimapShadowStripper(inpaint_scales=3, iterations=40)
    stripped_shadow, attenuation = stripper.strip_and_inpaint(
        minimap_rgb=raw_np,
        object_mask=mask,
        normalize_albedo=True,
    )
    print(f"  Bare shadow residual extracted: {attenuation * 100.0:.1f}% high-frequency object energy attenuated.")

    # 4. Directional Ridge & Hessian Curvature Extraction
    print("\n[Stage 3/5] Hessian Ridge Analysis & Normal Inversion...")
    azimuth = 4.7124  # 270 deg (standard Blizzard solar azimuth)
    elevation = 0.6981  # 40 deg
    cos_el = np.cos(elevation, dtype=np.float32)
    sun_dir = np.array([np.cos(azimuth) * cos_el, np.sin(azimuth) * cos_el, np.sin(elevation)], dtype=np.float32)

    # Convert stripped shadow variations into slope gradients along sun axis
    diff_from_mean = stripped_shadow - np.mean(stripped_shadow)
    dz_dx = -diff_from_mean * sun_dir[0] * 2.5
    dz_dy = -diff_from_mean * sun_dir[1] * 2.5
    recovered_normals = np.zeros((256, 256, 3), dtype=np.float32)
    recovered_normals[..., 0] = -dz_dx
    recovered_normals[..., 1] = -dz_dy
    recovered_normals[..., 2] = 1.0
    norm_lens = np.linalg.norm(recovered_normals, axis=-1, keepdims=True)
    recovered_normals /= np.maximum(norm_lens, 1e-6)

    ridge_mask, crest_pts = extract_ridge_contours(stripped_shadow, sigma=1.2, curvature_threshold=0.015)
    print(f"  Recovered surface normals: {recovered_normals.shape}")
    print(f"  Extracted {len(crest_pts)} directional ridge/crest contours.")

    # 5. Height Integration & 3D Fractal Editor Brush Fitting
    print("\n[Stage 4/5] 3D Fractal Editor Brush Fitting & Height Recovery...")
    integrated_height = reconstruct_height_from_normals(recovered_normals, apply_window=False)
    integrated_height = integrated_height.astype(np.float32)

    # Fit discrete 3D Blizzard editor brushes to residual deformations
    brush_lib = build_archetypal_brush_library(grid_size=33)
    fitter = FractalBrushFitter(brush_lib)
    stamps, residual_energy = fitter.fit_stamps(integrated_height, max_stamps=args.max_stamps)
    print(f"  Fitted {len(stamps)} discrete 3D fractal editor brush stamps.")
    for idx, st in enumerate(stamps):
        print(f"    Stamp #{idx + 1}: Brush={st.brush_id}, Pos={st.center_xy}, Amplitude={st.amplitude:.3f}")

    # Upsample to 257x257 (standard WoW ADT chunk/tile vertex resolution)
    h_257 = ndimage.zoom(integrated_height, (257.0 / 256.0, 257.0 / 256.0), order=1)

    # 6. Artifact Serialization
    print("\n[Stage 5/5] Exporting 3D Mesh & Visual Diagnostics...")
    stem = img_path.stem
    obj_out = out_dir / f"{stem}_reconstructed.obj"
    tex_out = out_dir / f"{stem}_texture.png"
    hmap_out = out_dir / f"{stem}_heightmap.png"
    quilt_out = out_dir / f"{stem}_diagnostic_quilt.png"

    # Save texture PNG
    raw_img.save(tex_out)

    # Save 16-bit Heightmap PNG
    h_norm_uint16 = (
        (h_257 - np.min(h_257)) / max(1e-5, (np.max(h_257) - np.min(h_257))) * 65535.0
    ).astype(np.uint16)
    Image.fromarray(h_norm_uint16).save(hmap_out)

    # Export 3D OBJ & GLB Meshes
    export_obj_mesh(h_257, tex_out, obj_out, height_scale=args.height_scale)
    glb_out = out_dir / f"{stem}_reconstructed.glb"
    export_glb_mesh(h_257, raw_img, glb_out, height_scale=args.height_scale)
    print(f"  [OK] Exported 3D Wavefront OBJ:  {obj_out}")
    print(f"  [OK] Exported 3D glTF GLB:        {glb_out}")
    print(f"  [OK] Exported 16-bit Heightmap:  {hmap_out}")

    # Build and save 4-panel diagnostic quilt
    quilt_img = build_diagnostic_quilt(
        raw_img,
        mask,
        stripped_shadow,
        recovered_normals,
        ridge_mask,
    )
    quilt_img.save(quilt_out)
    print(f"  [OK] Exported Diagnostic Quilt:  {quilt_out}")

    print("\n=================================================================")
    print("Reconstruction Complete!")
    print(f"View the 3D mesh: {obj_out}")
    print(f"View the visual quilt: {quilt_out}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
