"""End-to-End Multi-Tile Quilt Canvas Terrain Reconstruction CLI (Spec 268).

Pipeline:
  Stage 1: Quilt Canvas Stitcher & Global Coordinate Solver (AC-001)
  Stage 2: Minimap Albedo De-Mixing & Dynamic MCAL Layer Stacker (AC-002)
  Stage 3: Bare Terrain Shadow Sieve & WDL Macro-Trestle Lattice Quilt (AC-003, AC-004)
  Stage 4: 36x Inches-Scale Refiner & 3D Fractal Pastes/Scars (AC-005, AC-006)
  Stage 5: Monolithic ADT Materialization (100% chunks) & Multi-Tile OBJ/GLB Export (AC-007)

Usage:
  uv run python scripts/v60_reconstruct_quilt.py --tiles 16_32,16_33 --out-dir output/quilt_development
  uv run python scripts/v60_reconstruct_quilt.py --bbox 16,32,17,33 --out-dir output/quilt_2x2
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

# Ensure src/ is on sys.path
src_dir = Path(__file__).resolve().parent.parent / "src"
if str(src_dir) not in sys.path:
    sys.path.insert(0, str(src_dir))

from harvester.v60.mcal_layer_decipherer import McalLayerDecipherer
from harvester.v60.quilt_adt_materializer import QuiltAdtMaterializer
from harvester.v60.quilt_canvas_assembler import QuiltCanvasAssembler
from harvester.v60.quilt_fractal_refiner import QuiltFractalRefiner
from harvester.v60.wdl_quilt_synthesizer import WdlQuiltSynthesizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("reconstruct_quilt")


def parse_tile_list(tiles_str: str) -> List[Tuple[int, int]]:
    """Parse comma-separated 'tx_ty' strings into (tx, ty) tuples."""
    coords = []
    for part in tiles_str.split(","):
        part = part.strip()
        if not part:
            continue
        tokens = part.split("_")
        if len(tokens) == 2:
            coords.append((int(tokens[0]), int(tokens[1])))
    return coords


def parse_bbox(bbox_str: str) -> List[Tuple[int, int]]:
    """Parse 'min_tx,min_ty,max_tx,max_ty' into grid of (tx, ty) coordinates."""
    parts = [int(p.strip()) for p in bbox_str.split(",")]
    if len(parts) != 4:
        raise ValueError(f"Expected 4 values for bbox, got: {bbox_str}")
    min_tx, min_ty, max_tx, max_ty = parts
    coords = []
    for tx in range(min_tx, max_tx + 1):
        for ty in range(min_ty, max_ty + 1):
            coords.append((tx, ty))
    return coords


def find_minimap_image(tx: int, ty: int, search_dirs: List[Path]) -> Optional[Path]:
    """Search candidate directories for minimap image for tile (tx, ty)."""
    candidates = [
        f"map_{tx}_{ty}.png",
        f"development_{tx}_{ty}.png",
        f"minimap_{tx}_{ty}.png",
        f"tile_{tx}_{ty}.png",
        f"0_{tx}_{ty}.png",
    ]
    for d in search_dirs:
        for c in candidates:
            p = d / c
            if p.is_file():
                return p
    return None


def find_template_adt(tx: int, ty: int, search_dirs: List[Path]) -> Optional[Path]:
    """Search candidate directories for template ADT for tile (tx, ty)."""
    candidates = [
        f"development_{tx}_{ty}.adt",
        f"map_{tx}_{ty}.adt",
    ]
    for d in search_dirs:
        for c in candidates:
            p = d / c
            if p.is_file():
                return p
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description="Multi-Tile Quilt Canvas Terrain Reconstruction")
    parser.add_argument("--tiles", type=str, default=None, help="Comma-separated tile list (e.g. '16_32,16_33')")
    parser.add_argument("--bbox", type=str, default=None, help="Bounding box 'min_tx,min_ty,max_tx,max_ty'")
    parser.add_argument("--map-name", type=str, default="development", help="Continent/map name")
    parser.add_argument("--out-dir", type=Path, default=Path("output/quilt_reconstructed"), help="Output directory")
    parser.add_argument("--minimap-dir", type=Path, default=None, help="Directory containing minimap images")
    parser.add_argument("--template-dir", type=Path, default=None, help="Directory containing template ADT files")
    parser.add_argument("--export-mesh", action="store_true", default=True, help="Export continuous 3D OBJ and GLB meshes")
    parser.add_argument("--d1-checkpoint", type=Path, default=Path("checkpoints/d1_best.pt"), help="Model D1 checkpoint")
    parser.add_argument("--wdl-checkpoint", type=Path, default=Path("checkpoints/minimap_to_wdl_best.pt"), help="MinimapToWdlNet checkpoint")
    parser.add_argument("--trestle-checkpoint", type=Path, default=Path("../output/models/trestle_elevation_v1.pt"), help="TrestleElevationUNet checkpoint")
    parser.add_argument("--synth-wdl", type=Path, default=Path("../output/development_synthesized_wdl.npz"), help="Path to synthesized development WDL NPZ (Spec 266)")
    args = parser.parse_args()

    # Determine tile list
    tiles: List[Tuple[int, int]] = []
    if args.tiles:
        tiles.extend(parse_tile_list(args.tiles))
    elif args.bbox:
        tiles.extend(parse_bbox(args.bbox))
    else:
        # Default benchmark alpine massif
        tiles = [(16, 32), (16, 33)]

    if not tiles:
        logger.error("No valid tiles specified.")
        return 1

    logger.info("Starting Quilt Canvas Reconstruction for %d tiles: %s", len(tiles), tiles)
    start_time = time.time()

    # Search paths for assets
    minimap_search = [
        p for p in [
            args.minimap_dir,
            Path("../test_data/original_development/World/Textures/Minimap"),
            Path("test_data/original_development/World/Textures/Minimap"),
            Path("minimaps/development"),
            Path("../test_data/development_minimaps"),
            Path("test_data/development_minimaps"),
            Path("../output/development_minimaps"),
            Path("output/development_minimaps"),
        ] if p is not None
    ]

    template_search = [
        p for p in [
            args.template_dir,
            Path("../test_data/WoWMuseum/335-dev/World/Maps/development"),
            Path("../test_data/original_development/World/Maps/development"),
            Path("../test_data/original_development/WDT-ADT"),
        ] if p is not None
    ]

    # Initialize Stage components
    assembler = QuiltCanvasAssembler(tiles)
    mcal_decipherer = McalLayerDecipherer()
    wdl_synthesizer = WdlQuiltSynthesizer()
    fractal_refiner = QuiltFractalRefiner(subcell_factor=4)

    # Load minimaps and initialize Stage 1
    tile_minimaps: Dict[Tuple[int, int], np.ndarray] = {}
    for tx, ty in tiles:
        img_p = find_minimap_image(tx, ty, minimap_search)
        if img_p and img_p.is_file():
            pil_img = Image.open(img_p).convert("RGB")
            if pil_img.size != (256, 256):
                pil_img = pil_img.resize((256, 256), Image.Resampling.LANCZOS)
            arr = np.asarray(pil_img, dtype=np.uint8)
            logger.info("Loaded minimap for tile (%d, %d) from %s", tx, ty, img_p)
        else:
            logger.warning("No minimap found for (%d, %d), using synthetic fallback.", tx, ty)
            arr = np.full((256, 256, 3), 128, dtype=np.uint8)

        tile_minimaps[(tx, ty)] = arr
        assembler.add_tile(tx, ty, minimap_rgb=arr)

    stitched_minimap, bounds = assembler.stitch_minimap_quilt()
    logger.info("Stage 1: Stitched Quilt Minimap Canvas (%d x %d pixels)", bounds.pixel_width, bounds.pixel_height)

    # Stage 2: Minimap Albedo De-Mixing & MCAL Layer Deciphering
    logger.info("Stage 2: De-mixing minimap albedo and deciphering MCAL layers...")
    tile_albedos: Dict[Tuple[int, int], np.ndarray] = {}
    tile_chunk_layers: Dict[Tuple[int, int], List[DecipheredChunkLayers]] = {}

    for tx, ty in tiles:
        mm = tile_minimaps[(tx, ty)]
        if args.d1_checkpoint.is_file():
            weights, albedo, illum, meta = mcal_decipherer.predict_d1_neural_layers(
                mm, checkpoint_path=args.d1_checkpoint
            )
        else:
            weights, albedo, illum = mcal_decipherer.demix_pixel_albedo(mm)

        chunk_layers = mcal_decipherer.build_chunk_layers(weights)
        tile_albedos[(tx, ty)] = albedo
        tile_chunk_layers[(tx, ty)] = chunk_layers

    # Stage 3: Bare Shadow Sieve & WDL Macro Trestle
    logger.info("Stage 3: Extracting bare terrain shadows and synthesizing WDL lattices...")
    tile_wdl_17: Dict[Tuple[int, int], np.ndarray] = {}
    tile_shadows: Dict[Tuple[int, int], np.ndarray] = {}

    from scipy import ndimage
    from harvester.v60.wdl_elevation_calibrator import WdlElevationParser

    # Search for authentic WDL file for map
    wdl_search = [
        p for p in [
            Path("../test_data/original_development/World/Maps") / args.map_name / f"{args.map_name}.wdl",
            Path("test_data/original_development/World/Maps") / args.map_name / f"{args.map_name}.wdl",
            Path("../test_data/original_development/WDT-ADT") / f"{args.map_name}.wdl",
            Path("test_data/original_development/WDT-ADT") / f"{args.map_name}.wdl",
            Path("../test_data/WoWMuseum/335-dev/World/Maps") / args.map_name / f"{args.map_name}.wdl",
        ] if p.is_file()
    ]
    wdl_parser: Optional[WdlElevationParser] = None
    if wdl_search:
        try:
            wdl_parser = WdlElevationParser(wdl_search[0])
            logger.info("Found authentic WDL archive: %s", wdl_search[0])
        except Exception as e:
            logger.warning("Failed to open WDL archive %s: %s", wdl_search[0], e)

    # Load TrestleWdlSynthesizer database and device
    import torch
    from harvester.v60.trestle_wdl_synthesizer import TrestleWdlSynthesizer

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    trestle_synth: Optional[TrestleWdlSynthesizer] = None
    synth_candidates = [
        args.synth_wdl,
        Path("../output/development_synthesized_wdl.npz"),
        Path("output/development_synthesized_wdl.npz"),
    ]
    synth_p = next((p for p in synth_candidates if p and p.is_file()), None)
    if synth_p:
        try:
            trestle_synth = TrestleWdlSynthesizer.load_npz(synth_p)
            logger.info("Loaded TrestleWdlSynthesizer macro-lattice database from %s", synth_p)
        except Exception as e:
            logger.warning("Failed to load synthesized WDL database from %s: %s", synth_p, e)

    for tx, ty in tiles:
        mm = tile_minimaps[(tx, ty)]
        alb = tile_albedos[(tx, ty)]
        shadow_res = wdl_synthesizer.extract_bare_terrain_shadow(mm, alb)
        tile_shadows[(tx, ty)] = shadow_res.bare_shadow_256

        # 1. Attempt authentic WDL coarse elevation first
        h17 = None
        if wdl_parser and wdl_parser.has_tile_data(tx, ty):
            h17 = wdl_parser.extract_tile_17(tx, ty)
            if h17 is not None:
                logger.info("  Tile (%d, %d): loaded authentic WDL 17x17 lattice (elev range: %.1f to %.1f yds)",
                            tx, ty, float(h17.min()), float(h17.max()))

        # 2. Attempt transferred Northrend lineage macro-trestle from TrestleWdlSynthesizer
        if h17 is None and trestle_synth and trestle_synth.has_tile(tx, ty):
            cand_h17 = trestle_synth.get_trestle_17(tx, ty)
            # Ensure candidate has valid authored land data (not unauthored ocean void -652 yds)
            if cand_h17 is not None and float(cand_h17.min()) > -500.0 and float(cand_h17.max() - cand_h17.min()) > 1.0:
                h17 = cand_h17
                meta = trestle_synth.get_metadata(tx, ty) or {}
                logger.info("  Tile (%d, %d): loaded synthesized WDL macro-trestle (%s, sim=%.3f, span: %.1f yds, elev: %.1f..%.1f)",
                            tx, ty, meta.get("origin_tile", "Transferred"), meta.get("similarity", 1.0),
                            float(h17.max() - h17.min()), float(h17.min()), float(h17.max()))

        # 3. For unauthored minimap-only tiles: reconstruct coarse 17x17 lattice from 2D Poisson SfS and tie-edge anchor
        if h17 is None:
            rad = np.radians(220.0)
            lx = float(np.cos(rad))
            ly = float(np.sin(rad))
            H_s, W_s = shadow_res.bare_shadow_256.shape
            u = np.fft.fftfreq(W_s) * 2.0 * np.pi
            v = np.fft.fftfreq(H_s) * 2.0 * np.pi
            U, V = np.meshgrid(u, v)
            k_par = U * lx + V * ly
            k2 = U**2 + V**2
            S = np.fft.fft2(shadow_res.bare_shadow_256)
            Z_freq = (1j * k_par * S) / (k2 + 0.025)
            Z_freq[0, 0] = 0.0
            z_sfs = np.fft.ifft2(Z_freq).real.astype(np.float32)

            h17_rel = ndimage.zoom(z_sfs, (17.0 / 256.0, 17.0 / 256.0), order=1)[:17, :17]
            span_rel = max(1e-4, float(np.ptp(h17_rel)))
            h17_scaled = (h17_rel - float(h17_rel.min())) / span_rel * 200.0

            # Tie-edge boundary anchoring to already resolved adjacent quilt tile
            anchor_offset = 100.0
            if (tx, ty - 1) in tile_wdl_17:
                north_bottom = tile_wdl_17[(tx, ty - 1)][16, :]
                anchor_offset = float(np.mean(north_bottom)) - float(np.mean(h17_scaled[0, :]))
            elif (tx - 1, ty) in tile_wdl_17:
                west_right = tile_wdl_17[(tx - 1, ty)][:, 16]
                anchor_offset = float(np.mean(west_right)) - float(np.mean(h17_scaled[:, 0]))
            elif (tx, ty + 1) in tile_wdl_17:
                south_top = tile_wdl_17[(tx, ty + 1)][0, :]
                anchor_offset = float(np.mean(south_top)) - float(np.mean(h17_scaled[16, :]))
            elif (tx + 1, ty) in tile_wdl_17:
                east_left = tile_wdl_17[(tx + 1, ty)][:, 0]
                anchor_offset = float(np.mean(east_left)) - float(np.mean(h17_scaled[:, 16]))

            h17 = h17_scaled + anchor_offset
            logger.info("  Tile (%d, %d): inferred WDL lattice from Poisson SfS & quilt seam anchor (range: %.1f to %.1f yds, span: %.1f yds)",
                        tx, ty, float(h17.min()), float(h17.max()), float(np.ptp(h17)))

        tile_wdl_17[(tx, ty)] = h17

    # Assemble unified global WDL macro-trestle lattice across the quilt
    global_wdl, bounds = wdl_synthesizer.assemble_global_wdl_lattice(tile_wdl_17, bounds)
    logger.info("Assembled Global WDL Lattice: %d x %d vertices across %d x %d tiles",
                global_wdl.shape[1], global_wdl.shape[0], bounds.width_tiles, bounds.height_tiles)

    # Interpolate global WDL lattice to continuous elevation canvas with aligned corners
    # (guarantees shared 1px edges tie adjacent tiles together with exact C0/C1 continuity)
    global_elevation = wdl_synthesizer.interpolate_global_wdl_to_canvas(global_wdl, bounds)
    logger.info("Interpolated Continuous Global Elevation Canvas: %d x %d vertices (elev range: %.1f to %.1f yds)",
                global_elevation.shape[1], global_elevation.shape[0], float(global_elevation.min()), float(global_elevation.max()))

    # Stage 4: Inches-Scale Refinement & Fractal Pastes/Scars (with Neural Trestle Carving)
    logger.info("Stage 4: Executing neural carving, inches-scale refinement, and 3D fractal fitting...")
    tile_elevations_257: Dict[Tuple[int, int], np.ndarray] = {}

    from harvester.v60.trestle_elevation_model import TrestleElevationUNet
    from harvester.v60.trestle_dataset import compute_photometric_normals

    trestle_net: Optional[TrestleElevationUNet] = None
    trestle_candidates = [
        args.trestle_checkpoint,
        Path("../output/models/trestle_elevation_v1.pt"),
        Path("output/models/trestle_elevation_v1.pt"),
    ]
    trestle_ckpt = next((p for p in trestle_candidates if p and p.is_file()), None)
    if trestle_ckpt:
        try:
            t_data = torch.load(trestle_ckpt, map_location=device, weights_only=False)
            t_base_ch = t_data.get("base_channels", 32)
            trestle_net = TrestleElevationUNet(in_channels=6, base_channels=t_base_ch).to(device)
            trestle_net.load_state_dict(t_data["model_state_dict"])
            trestle_net.eval()
            logger.info("Loaded TrestleElevationUNet model from %s (val MAE: %.2f yds)",
                        trestle_ckpt, t_data.get("val_mae", 0.0))
        except Exception as e:
            logger.warning("Failed to load TrestleElevationUNet from %s: %s", trestle_ckpt, e)

    for tx, ty in tiles:
        # Slice macro elevation directly from continuous global canvas
        base_h = wdl_synthesizer.slice_tile_from_global_canvas(global_elevation, tx, ty, bounds)
        shadow = tile_shadows[(tx, ty)]

        # Neural residual carving via TrestleElevationUNet (Spec 266)
        if trestle_net is not None:
            rgb_f = tile_minimaps[(tx, ty)].astype(np.float32) / 255.0
            normals = compute_photometric_normals(rgb_f)
            cur_wdl_256 = ndimage.zoom(base_h, (256.0 / 257.0, 256.0 / 257.0), order=1)[:256, :256]
            norm_trestle = ((cur_wdl_256 - 150.0) / 350.0).astype(np.float32)
            feat_np = np.concatenate([rgb_f, norm_trestle[..., None], normals], axis=-1)
            feat_t = torch.from_numpy(feat_np).permute(2, 0, 1).unsqueeze(0).float().to(device)
            with torch.no_grad():
                delta_z_t, _ = trestle_net(feat_t)
                delta_z = delta_z_t.cpu().numpy()[0]
            # Taper outer border by 1px so exact quilt boundary tie-edges are preserved
            delta_z[0, :] *= 0.0
            delta_z[-1, :] *= 0.0
            delta_z[:, 0] *= 0.0
            delta_z[:, -1] *= 0.0
            base_h += delta_z
            logger.info("  Tile (%d, %d): carved neural residual delta_z (range: %.2f to %.2f yds)",
                        tx, ty, float(delta_z.min()), float(delta_z.max()))

        # Discover archeological 3D fractal editor brush stamps and fossil scars
        sculpted_inches, stamps, scars = fractal_refiner.sculpt_inches_canvas(
            base_elevation_257=base_h,
            residual_shadow_256=shadow,
            max_brush_stamps=8,
        )
        logger.info("  Tile (%d, %d): discovered %d authoring brush stamps, %d fossil scars",
                    tx, ty, len(stamps), len(scars))

        tile_elevations_257[(tx, ty)] = base_h

    # Stage 1 Seam Boundary Relaxation across quilt
    logger.info("Enforcing C0/C1 boundary seam continuity across quilt...")
    stitched_elevations = assembler.solve_seam_boundaries(tile_elevations_257, margin=8)
    metrics = assembler.verify_seam_continuity(stitched_elevations)
    logger.info("  Boundary Seam Metrics: max height step = %.6f yds, normal alignment = %.6f",
                metrics["max_height_step_yards"], metrics["mean_normal_cosine_similarity"])

    # Update global elevation canvas from seamless stitched elevations
    global_elevation, _ = assembler.assemble_global_elevation_canvas(stitched_elevations)

    # Stage 5: Monolithic ADT Materialization & 3D Mesh Export
    logger.info("Stage 5: Materializing monolithic ADT files and 3D meshes...")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for tx, ty in tiles:
        adt_out = args.out_dir / f"{args.map_name}_{tx}_{ty}.adt"
        template = find_template_adt(tx, ty, template_search)
        final_h = stitched_elevations[(tx, ty)]

        if template and template.is_file():
            logger.info("  Patching existing authentic ADT: %s -> %s", template.name, adt_out.name)
            QuiltAdtMaterializer.patch_monolithic_adt(
                template_path=template,
                out_path=adt_out,
                h257=final_h,
                preserve_authentic_mccv=True,
                chunk_layers=tile_chunk_layers.get((tx, ty)),
            )
        else:
            logger.info("  Constructing new monolithic LK ADT: %s", adt_out.name)
            mtex_list = mcal_decipherer.generate_mtex_manifest(tile_chunk_layers[(tx, ty)])
            QuiltAdtMaterializer.construct_monolithic_lk_adt(
                out_path=adt_out,
                h257=final_h,
                tile_x=tx,
                tile_y=ty,
                texture_names=mtex_list,
            )

    # Export continuous multi-tile 3D meshes textured with aerial minimap imagery
    if args.export_mesh:
        mesh_base = args.out_dir / f"{args.map_name}_quilt_{bounds.min_tx}_{bounds.min_ty}_to_{bounds.max_tx}_{bounds.max_ty}"
        logger.info("Exporting continuous quilt 3D models to %s (.obj, .glb)", mesh_base.name)
        QuiltAdtMaterializer.export_quilt_3d_mesh(
            mesh_base.with_suffix(".obj"),
            global_elevation,
            texture_path_or_img=stitched_minimap,
        )
        QuiltAdtMaterializer.export_quilt_3d_mesh(
            mesh_base.with_suffix(".glb"),
            global_elevation,
            texture_path_or_img=stitched_minimap,
        )

    elapsed = time.time() - start_time
    logger.info("Quilt Canvas Reconstruction completed in %.2fs. All assets written to %s", elapsed, args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
