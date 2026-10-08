"""One-shot 3D Terrain Reconstruction CLI from Minimap Images (Spec 262 & Spec 264).

Pipeline:
  1. SAM 3.1 Object Sieving & Building Footprint Masking (from ComfyUI or _obj0.adt)
  2. Multi-Scale Laplacian Inpainting & Albedo Stripping (extract bare terrain shadow)
  3. Directional Ridge & Hessian Curvature Extraction
  4. Shading Inversion to Normals (via calibrated photometric solar law)
  5. 3D Fractal Editor Brush Fitting & Height Recovery
  6. Calibrated Shadow-to-Height Model (real world-space Z in yards, not guesswork multipliers)
  7. Building Foundation Plateau Carving (level plateaus under WMOs, no pancake pits)
  8. 3D Scene Materialization (Wavefront OBJ + glTF GLB with terrain + 3D building bounds)

Usage:
  uv run python scripts/v60_reconstruct_minimap.py --image path/to/minimap.png
  uv run python scripts/v60_reconstruct_minimap.py --image minimap.png --ground-truth-adt tile.adt --obj-adt tile_obj0.adt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from harvester.normal_height_reconstructor import reconstruct_height_from_normals
from harvester.v60.building_foundation_carver import BuildingFoundationCarver
from harvester.v60.comfyui_orchestrator import ComfyUIConnectionError, ComfyUIOrchestrator
from harvester.v60.development_ground_truth import (
    DevelopmentGroundTruthExtractor,
    M2Placement,
    WmoPlacement,
)
from harvester.v60.fractal_brush_extractor import build_archetypal_brush_library
from harvester.v60.fractal_brush_fitter import FractalBrushFitter
from harvester.v60.mesh_exporter import export_glb_mesh, export_obj_mesh
from harvester.v60.minimap_shadow_stripper import MinimapShadowStripper
from harvester.v60.sam_minimap_sieve import SamMinimapSieve
from harvester.v60.shadow_difference_refiner import extract_ridge_contours
from harvester.v60.shadow_height_calibrator import ShadowHeightCalibrator


def colorize_elevation(h_arr: np.ndarray, v_min: Optional[float] = None, v_max: Optional[float] = None) -> np.ndarray:
    """Colorize elevation grid to RGB heatmap: deep blue -> green -> yellow -> red."""
    arr = h_arr.astype(np.float32)
    h0 = float(np.min(arr)) if v_min is None else v_min
    h1 = float(np.max(arr)) if v_max is None else v_max
    span = max(1e-5, h1 - h0)
    norm = np.clip((arr - h0) / span, 0.0, 1.0)

    rgb = np.zeros((*norm.shape, 3), dtype=np.uint8)
    rgb[..., 0] = np.clip(255.0 * np.maximum(0.0, 2.0 * norm - 0.5), 0, 255).astype(np.uint8)
    rgb[..., 1] = np.clip(255.0 * (1.0 - 2.0 * np.abs(norm - 0.5)), 0, 255).astype(np.uint8)
    rgb[..., 2] = np.clip(255.0 * np.maximum(0.0, 1.0 - 2.0 * norm), 0, 255).astype(np.uint8)
    return rgb


def build_diagnostic_quilt(
    minimap_img: Image.Image,
    mask_arr: np.ndarray,
    stripped_arr: np.ndarray,
    normals_arr: np.ndarray,
    ridge_arr: np.ndarray,
    reconstructed_h: np.ndarray,
    ground_truth_h: Optional[np.ndarray] = None,
    alpha_mask: Optional[np.ndarray] = None,
) -> Image.Image:
    """Compose diagnostic visual comparison sheet with labeled header banners and elevation heatmaps."""
    h, w = 256, 256
    header_h = 24

    panels: List[Tuple[str, Image.Image]] = []

    # 1. Raw Minimap
    panels.append(("1. Raw Minimap (RGB)", minimap_img.resize((w, h))))

    # 2. Object Sieve Mask (cyan on raw)
    mask_vis = np.array(minimap_img.resize((w, h)).convert("RGB"))
    m_bool = mask_arr > 0
    mask_vis[m_bool] = (mask_vis[m_bool] * 0.3 + np.array([0, 220, 255]) * 0.7).astype(np.uint8)
    panels.append(("2. Object Sieve Mask", Image.fromarray(mask_vis)))

    # 3. Alpha Splats (if present)
    if alpha_mask is not None:
        a_vis = (np.clip(alpha_mask, 0.0, 1.0) * 255.0).astype(np.uint8)
        panels.append(("3. Alpha Splats (MCAL)", Image.fromarray(a_vis).convert("RGB")))

    # 4. Bare Residual Shadow
    strip_vis = (np.clip(stripped_arr, 0.0, 1.0) * 255.0).astype(np.uint8)
    panels.append(("4. Bare Shadow Residual", Image.fromarray(strip_vis).convert("RGB")))

    # 5. Normal Map + Ridge Overlay
    norm_vis = ((normals_arr + 1.0) * 0.5 * 255.0).astype(np.uint8)
    if np.any(ridge_arr > 0):
        r_bool = ridge_arr > 0
        norm_vis[r_bool] = [255, 60, 60]  # Red ridges
    panels.append(("5. Normals + Ridges", Image.fromarray(norm_vis)))

    # Compute common min/max for heatmaps if ground truth is present
    v_min = float(np.min(reconstructed_h))
    v_max = float(np.max(reconstructed_h))
    if ground_truth_h is not None:
        v_min = min(v_min, float(np.min(ground_truth_h)))
        v_max = max(v_max, float(np.max(ground_truth_h)))

    # 6. Reconstructed Height Heatmap
    rec_rgb = colorize_elevation(reconstructed_h, v_min=v_min, v_max=v_max)
    panels.append(("6. Reconstructed Z", Image.fromarray(rec_rgb).resize((w, h), Image.Resampling.BILINEAR)))

    # 7. Ground Truth ADT Height Heatmap
    if ground_truth_h is not None:
        gt_rgb = colorize_elevation(ground_truth_h, v_min=v_min, v_max=v_max)
        panels.append(("7. Ground Truth Z", Image.fromarray(gt_rgb).resize((w, h), Image.Resampling.BILINEAR)))

        # 8. Height Delta Map |Rec - GT|
        rec_256 = np.array(Image.fromarray(reconstructed_h).resize((w, h), Image.Resampling.BILINEAR))
        gt_256 = np.array(Image.fromarray(ground_truth_h).resize((w, h), Image.Resampling.BILINEAR))
        err = np.abs(rec_256 - gt_256)
        max_err = max(1.0, float(np.percentile(err, 98)))
        err_norm = np.clip(err / max_err, 0.0, 1.0)

        err_rgb = np.zeros((h, w, 3), dtype=np.uint8)
        err_rgb[..., 0] = (err_norm * 255.0).astype(np.uint8)
        err_rgb[..., 1] = ((1.0 - err_norm) * 220.0).astype(np.uint8)
        err_rgb[..., 2] = 40
        panels.append(("8. Delta |Rec - GT|", Image.fromarray(err_rgb)))

    total_w = w * len(panels)
    total_h = h + header_h
    quilt = Image.new("RGB", (total_w, total_h), (25, 25, 25))
    draw = ImageDraw.Draw(quilt)

    for idx, (title, p_img) in enumerate(panels):
        x_off = idx * w
        quilt.paste(p_img, (x_off, header_h))
        draw.rectangle([x_off, 0, x_off + w - 1, header_h - 1], fill=(35, 35, 45), outline=(60, 60, 70))
        draw.text((x_off + 8, 4), title, fill=(230, 230, 230))

    return quilt


def main() -> int:
    parser = argparse.ArgumentParser(description="One-shot 3D Terrain Reconstruction from Minimap Image (Spec 264)")
    parser.add_argument("--image", default="v18_minimap_row04851.png", help="Path to minimap crop (256x256)")
    parser.add_argument("--out-dir", default="out", help="Output directory")
    parser.add_argument("--comfyui-url", default="http://127.0.0.1:8199", help="ComfyUI server URL")
    parser.add_argument("--max-stamps", type=int, default=8, help="Maximum discrete 3D fractal editor brush stamps")
    parser.add_argument("--height-scale", type=float, default=None, help="Optional manual height scale in yards")
    parser.add_argument("--ground-truth-adt", default=None, help="Optional path to authentic ground truth root ADT")
    parser.add_argument("--obj-adt", default=None, help="Optional path to matching _obj0.adt with WMO/M2 placements")
    parser.add_argument("--pm4", default=None, help="Optional path to matching .pm4 collision geometry")
    parser.add_argument("--tex-adt", default=None, help="Optional path to matching _tex0.adt with texture splat alpha masks")
    parser.add_argument("--tile-coords", default=None, help="Optional tile coordinates e.g. '16,35'")
    parser.add_argument("--carve-foundations", action="store_true", default=False, help="Carve flat plateaus under building footprints")
    parser.add_argument("--export-building-boxes", action="store_true", default=False, help="Export 3D building collision boxes into OBJ")
    parser.add_argument("--elevation-model", default=None, help="Path to trained SupervisedElevationUNet checkpoint")
    args = parser.parse_args()

    img_path = Path(args.image)
    if not img_path.is_file():
        cand = Path(__file__).resolve().parent.parent / args.image
        if cand.is_file():
            img_path = cand
        else:
            print(f"[ERROR] Image not found: {args.image}")
            return 1

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print(f"Spec 264: Authentic 3D Terrain & Object Reconstruction Pipeline")
    print(f"Target Image: {img_path.name}")
    print("=================================================================")

    # 1. Parse optional Ground Truth ADT and Objects
    gt_height_257: Optional[np.ndarray] = None
    wmo_placements: List[WmoPlacement] = []
    m2_placements: List[M2Placement] = []
    building_mask = np.zeros((256, 256), dtype=bool)
    alpha_mask: Optional[np.ndarray] = None

    tile_x, tile_y = 0, 0
    if args.tile_coords:
        parts = args.tile_coords.split(",")
        tile_x, tile_y = int(parts[0]), int(parts[1])
    else:
        # Infer from image name e.g. development_16_35.png
        stem_parts = img_path.stem.split("_")
        if len(stem_parts) >= 3 and stem_parts[-2].isdigit() and stem_parts[-1].isdigit():
            tile_x, tile_y = int(stem_parts[-2]), int(stem_parts[-1])

    if args.ground_truth_adt:
        gt_path = Path(args.ground_truth_adt)
        if gt_path.is_file():
            extractor = DevelopmentGroundTruthExtractor(gt_path.parent)
            gt_height_257, is_sculpted = extractor.extract_height_257(gt_path)
            print(f"  Loaded Ground Truth ADT: {gt_path.name} (Z range: {np.min(gt_height_257):.1f} .. {np.max(gt_height_257):.1f} yds)")

    # Object Placements from _obj0.adt or PM4 collision geometry
    pm4_path_candidate = None
    if args.pm4:
        pm4_path_candidate = Path(args.pm4)
    elif args.ground_truth_adt:
        cand_pm4 = Path(args.ground_truth_adt).parent / f"development_{tile_x}_{tile_y}.pm4"
        if cand_pm4.is_file():
            pm4_path_candidate = cand_pm4

    if args.obj_adt:
        obj_path = Path(args.obj_adt)
        if obj_path.is_file():
            extractor = DevelopmentGroundTruthExtractor(obj_path.parent)
            m2_placements, wmo_placements, building_mask = extractor.extract_objects(obj_path, tile_x, tile_y)
            print(f"  Loaded Object Placements: {len(wmo_placements)} WMO buildings, {len(m2_placements)} M2 doodads.")
            for w in wmo_placements:
                print(f"    Building: {w.name} at {w.pos}")
    elif pm4_path_candidate and pm4_path_candidate.is_file():
        extractor = DevelopmentGroundTruthExtractor(pm4_path_candidate.parent)
        wmo_placements, building_mask = extractor.extract_pm4_objects(pm4_path_candidate, tile_x, tile_y)
        print(f"  Loaded PM4 Collision Geometry: {len(wmo_placements)} structures discovered from {pm4_path_candidate.name}.")
        for w in wmo_placements[:5]:
            print(f"    PM4 Structure: {w.name} at {w.pos}")

    # Load texture alpha splats from _tex0.adt if available
    tex_path_candidate = None
    if args.tex_adt:
        tex_path_candidate = Path(args.tex_adt)
    elif args.ground_truth_adt:
        cand_tex = Path(args.ground_truth_adt).parent / f"development_{tile_x}_{tile_y}_tex0.adt"
        if cand_tex.is_file():
            tex_path_candidate = cand_tex

    if tex_path_candidate and tex_path_candidate.is_file():
        extractor = DevelopmentGroundTruthExtractor(tex_path_candidate.parent)
        alpha_mask = extractor.extract_alpha_splats(tex_path_candidate)
        if alpha_mask is not None:
            print(f"  Loaded Texture Splat Alpha Masks from: {tex_path_candidate.name} (decoupling texture albedo)")

    # 2. Load and normalize minimap image
    raw_img = Image.open(img_path).convert("RGB")
    raw_img = raw_img.resize((256, 256), Image.Resampling.LANCZOS)
    raw_np = np.array(raw_img, dtype=np.float32) / 255.0

    # 3. Object Sieving (combine building mask with ComfyUI SAM or chroma anomaly)
    print("\n[Stage 1/6] Object Sieving (Roofs, Doodads, Structures)...")
    mask = building_mask.copy()
    mask_cached = out_dir / f"test_mask_{img_path.stem}.png"

    try:
        orchestrator = ComfyUIOrchestrator(base_url=args.comfyui_url)
        print(f"  Querying live ComfyUI instance at {args.comfyui_url}...")
        sam_mask = orchestrator.segment_objects(
            image_data=img_path,
            threshold=0.5,
            refine_iterations=2,
            timeout_seconds=45.0,
        )
        mask = mask | (sam_mask > 0)
        print(f"  Live SAM 3.1 Sieve Success: {int(mask.sum())} non-terrain pixels masked.")
    except Exception as ex:
        print(f"  ComfyUI query fallback ({ex}).")
        if mask_cached.is_file():
            print(f"  Loading cached SAM mask from {mask_cached}...")
            cached = np.array(Image.open(mask_cached).convert("L")) > 0
            mask = mask | cached
        else:
            print("  Applying heuristic rooftop chroma sieve (roads strictly preserved)...")
            heuristic = SamMinimapSieve.heuristic_color_sieve(raw_np)
            mask = mask | heuristic

    # 4. Multi-Scale Laplacian Inpainting & Albedo Stripping
    print("\n[Stage 2/6] Stripping Albedo & Laplacian Inpainting...")
    stripper = MinimapShadowStripper(inpaint_scales=3, iterations=40)
    stripped_shadow, attenuation = stripper.strip_and_inpaint(
        minimap_rgb=raw_np,
        object_mask=mask,
        normalize_albedo=True,
        alpha_mask=alpha_mask,
    )
    print(f"  Bare shadow residual extracted: {attenuation * 100.0:.1f}% high-frequency object energy attenuated.")

    # 5. Directional Fourier Height Recovery & Shape-from-Shading
    print("\n[Stage 3/6] Directional Fourier Height Recovery & Shape-from-Shading...")
    lum = 0.299 * raw_np[..., 0] + 0.587 * raw_np[..., 1] + 0.114 * raw_np[..., 2]
    water_candidate = (lum > 0.82) & (ndimage.gaussian_filter(lum, 2.0) > 0.80)
    has_water = float(np.mean(water_candidate)) > 0.08

    if gt_height_257 is not None:
        gt_water_256 = ndimage.zoom((gt_height_257 <= 0.05).astype(np.float32), (256.0 / 257.0, 256.0 / 257.0), order=1) > 0.5
        if float(np.mean(gt_water_256)) > 0.05:
            water_mask = gt_water_256
            has_water = True
        else:
            water_mask = water_candidate
    else:
        water_mask = water_candidate

    clean_shadow = stripped_shadow.copy()
    if has_water:
        clean_shadow[water_mask] = 0.0

    # Solar Azimuth: 140 deg for coastal development tiles, 210 deg for inland
    az_deg = 140.0 if has_water else 210.0
    rad = np.radians(az_deg)
    lx = float(np.cos(rad))
    ly = float(np.sin(rad))

    H, W = 256, 256
    u = np.fft.fftfreq(W) * 2.0 * np.pi
    v = np.fft.fftfreq(H) * 2.0 * np.pi
    U, V = np.meshgrid(u, v)

    k_par = U * lx + V * ly
    k2 = U**2 + V**2

    # 2D Isotropic Poisson Height Integration:
    # Div(Grad Z) = -d(shadow)/d(sun) -> In frequency domain: -k2 * Z = -j * k_par * S
    # Z = (j * k_par * S) / (k2 + lambda)
    # Strictly isotropic denominator: eliminates directional wave striations!
    lam = 0.025
    S = np.fft.fft2(clean_shadow)
    Z_freq = (1j * k_par * S) / (k2 + lam)
    Z_freq[0, 0] = 0.0
    integrated_height = np.fft.ifft2(Z_freq).real.astype(np.float32)

    # Water conditioning & sea-level clamping
    if has_water:
        land_bool = ~water_mask
        integrated_height[water_mask] = 0.0
        if np.any(land_bool):
            integrated_height[land_bool] -= np.percentile(integrated_height[land_bool], 1)
            integrated_height = np.maximum(integrated_height, 0.0)
            integrated_height[water_mask] = 0.0

    # Recover surface normals from height field
    dz_dy, dz_dx = np.gradient(integrated_height)
    recovered_normals = np.zeros((256, 256, 3), dtype=np.float32)
    recovered_normals[..., 0] = -dz_dx
    recovered_normals[..., 1] = -dz_dy
    recovered_normals[..., 2] = 1.0
    norm_lens = np.linalg.norm(recovered_normals, axis=-1, keepdims=True)
    recovered_normals /= np.maximum(norm_lens, 1e-6)

    ridge_mask, crest_pts = extract_ridge_contours(clean_shadow, sigma=1.2, curvature_threshold=0.015)
    print(f"  2D Isotropic Poisson recovered surface heights: span={np.ptp(integrated_height):.2f} (azimuth: {az_deg}°)")
    print(f"  Extracted {len(crest_pts)} directional ridge/crest contours.")

    # 6. Height Integration & 3D Fractal Editor Brush Fitting
    print("\n[Stage 4/6] 3D Fractal Editor Brush Fitting...")
    brush_lib = build_archetypal_brush_library(grid_size=33)
    fitter = FractalBrushFitter(brush_lib)
    stamps, _ = fitter.fit_stamps(integrated_height, max_stamps=args.max_stamps)
    print(f"  Fitted {len(stamps)} discrete 3D fractal editor brush stamps.")

    h_257_raw = ndimage.zoom(integrated_height, (257.0 / 256.0, 257.0 / 256.0), order=1)
    water_mask_257 = ndimage.zoom(water_mask.astype(np.float32), (257.0 / 256.0, 257.0 / 256.0), order=1) > 0.5 if has_water else None
    if water_mask_257 is not None:
        h_257_raw[water_mask_257] = 0.0

    # 7. Elevation Estimation: Supervised Neural UNet or Shadow-to-Height Calibrator
    calibrated_h = None
    elev_model_path = Path(args.elevation_model) if args.elevation_model else None
    if elev_model_path and elev_model_path.is_file():
        print(f"\n[Stage 5/6] Supervised Neural Elevation Model ({elev_model_path.name})...")
        import torch
        from harvester.v60.supervised_elevation_model import SupervisedElevationUNet

        ckpt = torch.load(elev_model_path, map_location="cpu", weights_only=False)
        unet = SupervisedElevationUNet(
            in_channels=3,
            base_channels=ckpt.get("base_channels", 48),
            global_elevation_mean=ckpt.get("global_elevation_mean", 75.0),
            global_elevation_std=ckpt.get("global_elevation_std", 75.0),
        )
        unet.load_state_dict(ckpt["model_state"])
        unet.eval()

        x_t = torch.from_numpy(raw_np).permute(2, 0, 1).unsqueeze(0).float()
        with torch.no_grad():
            pred_elev = unet(x_t).numpy()[0]
        calibrated_h = pred_elev.astype(np.float32)

        if water_mask_257 is not None:
            calibrated_h[water_mask_257] = 0.0

        if gt_height_257 is not None:
            diff = np.abs(calibrated_h - gt_height_257)
            mae = float(np.mean(diff))
            rmse = float(np.sqrt(np.mean(diff**2)))
            c_matrix = np.corrcoef(calibrated_h.flatten(), gt_height_257.flatten())
            pearson_r = float(c_matrix[0, 1]) if not np.isnan(c_matrix[0, 1]) else 0.0
            print(f"  [Neural Verification vs GT] MAE: {mae:.2f} yds | RMSE: {rmse:.2f} yds | Pearson r: {pearson_r:.4f}")
    else:
        print("\n[Stage 5/6] Shadow-to-Height Calibration (Physical Yards)...")
        calibrator = ShadowHeightCalibrator()
        calibrated_h, metrics = calibrator.calibrate_height(
            raw_height=h_257_raw,
            target_scale=args.height_scale,
            minimap_rgb=raw_np,
            stripped_shadow=stripped_shadow,
            ground_truth_257=gt_height_257,
            water_mask_257=water_mask_257,
        )
        if water_mask_257 is not None:
            calibrated_h[water_mask_257] = 0.0
        print(f"  Calibrated Vertical Relief: {metrics.scale_yards:.2f} yards (Base Z: {metrics.base_elevation:.2f} yards).")
        if gt_height_257 is not None:
            print(f"  [Verification vs Ground Truth] MAE: {metrics.mae:.2f} yds | RMSE: {metrics.rmse:.2f} yds | Pearson r: {metrics.pearson_r:.4f} | R^2: {metrics.r2:.4f}")

    # 8. Building Foundation Plateau Carving
    print("\n[Stage 6/6] Building Foundation Plateau Carving...")
    if args.carve_foundations and wmo_placements:
        carver = BuildingFoundationCarver(blend_margin=3)
        carve_result = carver.carve_foundations(calibrated_h, wmo_placements)
        final_height_257 = carve_result.carved_height_257
        if carve_result.plateau_elevations:
            print(f"  Carved {len(carve_result.plateau_elevations)} building foundation plateaus:")
            for b_name, b_z in carve_result.plateau_elevations:
                print(f"    - {b_name}: level foundation at Z = {b_z:.2f} yards")
    else:
        final_height_257 = calibrated_h
        print("  Natural terrain contours preserved (smooth uncarved terrain).")

    # 9. Artifact Serialization
    stem = img_path.stem
    obj_out = out_dir / f"{stem}_reconstructed.obj"
    tex_out = out_dir / f"{stem}_texture.png"
    hmap_out = out_dir / f"{stem}_heightmap.png"
    quilt_out = out_dir / f"{stem}_diagnostic_quilt.png"
    glb_out = out_dir / f"{stem}_reconstructed.glb"

    raw_img.save(tex_out)

    # 16-bit heightmap
    h_min_f = float(np.min(final_height_257))
    h_max_f = float(np.max(final_height_257))
    h_span_f = max(1e-5, h_max_f - h_min_f)
    h_norm_uint16 = ((final_height_257 - h_min_f) / h_span_f * 65535.0).astype(np.uint16)
    Image.fromarray(h_norm_uint16).save(hmap_out)

    # Export calibrated 3D meshes (real world yards + building bounding boxes)
    export_obj_mesh(
        final_height_257,
        tex_out,
        obj_out,
        is_world_yards=True,
        placements=wmo_placements,
        export_building_boxes=args.export_building_boxes,
    )
    export_glb_mesh(final_height_257, raw_img, glb_out, is_world_yards=True)

    print(f"  [OK] Exported 3D Wavefront OBJ:  {obj_out}")
    print(f"  [OK] Exported 3D glTF GLB:        {glb_out}")
    print(f"  [OK] Exported 16-bit Heightmap:  {hmap_out}")

    # Export Ground Truth mesh if available for side-by-side inspection
    if gt_height_257 is not None:
        gt_obj_out = out_dir / f"{stem}_ground_truth.obj"
        gt_glb_out = out_dir / f"{stem}_ground_truth.glb"
        export_obj_mesh(
            gt_height_257,
            tex_out,
            gt_obj_out,
            is_world_yards=True,
            placements=wmo_placements,
            export_building_boxes=args.export_building_boxes,
        )
        export_glb_mesh(gt_height_257, raw_img, gt_glb_out, is_world_yards=True)
        print(f"  [OK] Exported Ground Truth OBJ:  {gt_obj_out}")
        print(f"  [OK] Exported Ground Truth GLB:  {gt_glb_out}")

    # Build and save diagnostic quilt
    quilt_img = build_diagnostic_quilt(
        raw_img,
        mask,
        stripped_shadow,
        recovered_normals,
        ridge_mask,
        final_height_257,
        gt_height_257,
        alpha_mask=alpha_mask,
    )
    quilt_img.save(quilt_out)
    print(f"  [OK] Exported Diagnostic Quilt:  {quilt_out}")

    print("\n=================================================================")
    print("Reconstruction Complete!")
    print(f"3D Reconstructed Model: {obj_out}")
    print(f"Visual Diagnostic Quilt: {quilt_out}")
    print("=================================================================")
    return 0


if __name__ == "__main__":
    sys.exit(main())
