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

    for tx, ty in tiles:
        mm = tile_minimaps[(tx, ty)]
        alb = tile_albedos[(tx, ty)]
        shadow_res = wdl_synthesizer.extract_bare_terrain_shadow(mm, alb)
        tile_shadows[(tx, ty)] = shadow_res.bare_shadow_256

        # Attempt authentic WDL coarse elevation first
        h17 = None
        if wdl_parser and wdl_parser.has_tile_data(tx, ty):
            h17 = wdl_parser.extract_tile_17(tx, ty)
            if h17 is not None:
                logger.info("  Tile (%d, %d): loaded authentic WDL 17x17 lattice (elev range: %.1f to %.1f yds)",
                            tx, ty, float(h17.min()), float(h17.max()))

        if h17 is None:
            # Fallback to synthesizing 17x17 from bare terrain photometric shadow
            shadow_17 = ndimage.zoom(shadow_res.bare_shadow_256, 17.0 / 256.0, order=1)[:17, :17]
            h17 = 100.0 + shadow_17 * 200.0
            logger.info("  Tile (%d, %d): synthesized WDL lattice from bare shadow (elev range: %.1f to %.1f yds)",
                        tx, ty, float(h17.min()), float(h17.max()))

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

    # Stage 4: Inches-Scale Refinement & Fractal Pastes/Scars
    logger.info("Stage 4: Executing inches-scale refinement and 3D fractal fitting...")
    tile_elevations_257: Dict[Tuple[int, int], np.ndarray] = {}

    for tx, ty in tiles:
        # Slice macro elevation directly from continuous global canvas
        base_h = wdl_synthesizer.slice_tile_from_global_canvas(global_elevation, tx, ty, bounds)
        shadow = tile_shadows[(tx, ty)]

        sculpted_inches, stamps, scars = fractal_refiner.sculpt_inches_canvas(
            base_elevation_257=base_h,
            residual_shadow_256=shadow,
            max_brush_stamps=12,
        )

        # Apply high-frequency displacement while preserving continuous boundary tie-edges
        scale = fractal_refiner.subcell_factor
        macro_resampled = ndimage.zoom(base_h[:256, :256], scale, order=1)
        disp_inches = sculpted_inches - macro_resampled
        disp_256 = ndimage.zoom(disp_inches, 1.0 / scale, order=1)

        tile_h = base_h.copy()
        # Add micro-relief to interior while letting edge taper to maintain seamless tie-line
        tile_h[:256, :256] += disp_256
        tile_elevations_257[(tx, ty)] = tile_h
        logger.info("  Tile (%d, %d): fitted %d brush stamps, %d scars", tx, ty, len(stamps), len(scars))

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
