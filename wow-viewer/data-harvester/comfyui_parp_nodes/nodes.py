"""parp-tools ComfyUI Custom Node Suite for WoW ADT & 3D Terrain Refinement (Spec 267).

Provides native ComfyUI nodes for:
  - WoW_AdtLoader: Loads minimaps, coarse WDL trestle lattices, and authentic object masks.
  - WoW_ObjectSieveConditioner: Sieves WMO/M2 footprints while preserving painted annotations and roads.
  - WoW_HardZCalibrator: Anchors unconstrained generative 3D foundation depth/heightfields to authentic WDL yards.
  - WoW_MeshExporter: Exports watertight North-up Cartesian OBJ/GLB meshes with authentic Blizzard world coordinates.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
import torch

from harvester.v60.development_ground_truth import DevelopmentGroundTruthExtractor
from harvester.v60.mesh_exporter import export_glb_mesh, export_obj_mesh
from harvester.v60.minimap_shadow_stripper import MinimapShadowStripper
from harvester.v60.terrain_feature_synthesizer import (
    compute_surface_normals,
    synthesize_multiband_terrain,
)
from harvester.v60.trestle_wdl_synthesizer import TrestleWdlSynthesizer
from harvester.v60.wdl_elevation_calibrator import WdlElevationParser


class WoW_AdtLoader:
    """Loads 256x256 minimap RGB, 257x257 WDL trestle elevation, and authentic _obj0 placement masks."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "minimap_path": ("STRING", {"default": "development_16_33.png", "multiline": False}),
            },
            "optional": {
                "adt_path": ("STRING", {"default": "", "multiline": False}),
                "wdl_path": ("STRING", {"default": "", "multiline": False}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "trestle_elevation", "object_mask", "tile_coords")
    FUNCTION = "load_adt_data"
    CATEGORY = "parp-tools/WoW"

    def load_adt_data(
        self,
        minimap_path: str,
        adt_path: str = "",
        wdl_path: str = "",
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str]:
        mmap_p = Path(minimap_path)
        if not mmap_p.is_file():
            raise FileNotFoundError(f"Minimap file not found: {minimap_path}")

        # Load 256x256 RGB minimap
        raw_img = Image.open(mmap_p).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
        img_np = np.array(raw_img, dtype=np.float32) / 255.0
        # ComfyUI IMAGE format is [B, H, W, C] in [0, 1]
        image_tensor = torch.from_numpy(img_np).unsqueeze(0).float()

        # Parse tile coordinates from stem (e.g. development_16_33 -> (16, 33))
        tile_x, tile_y = 0, 0
        parts = mmap_p.stem.split("_")
        if len(parts) >= 3 and parts[-2].isdigit() and parts[-1].isdigit():
            tile_x, tile_y = int(parts[-2]), int(parts[-1])
        coords_str = f"{tile_x},{tile_y}"

        # Load authentic objects from _obj0.adt if available
        obj_mask = np.zeros((256, 256), dtype=np.float32)
        adt_cand = Path(adt_path) if adt_path else None
        if adt_cand and not adt_cand.is_file():
            # Search candidate dirs
            adt_cand = None

        if not adt_cand:
            # Check adjacent directories
            for cand_dir in [mmap_p.parent.parent / "Maps" / "development", mmap_p.parent]:
                c = cand_dir / f"{mmap_p.stem}_obj0.adt"
                if c.is_file():
                    adt_cand = c
                    break

        if adt_cand and adt_cand.is_file():
            try:
                extractor = DevelopmentGroundTruthExtractor(adt_cand.parent)
                _, _, building_mask = extractor.extract_objects(
                    adt_cand, tile_x, tile_y, minimap_rgb=img_np * 255.0
                )
                obj_mask = building_mask.astype(np.float32)
            except Exception:
                pass

        mask_tensor = torch.from_numpy(obj_mask).unsqueeze(0).float()

        # Load WDL trestle elevation
        trestle_np = np.zeros((257, 257), dtype=np.float32)
        wdl_cand = Path(wdl_path) if wdl_path else None
        if not wdl_cand or not wdl_cand.is_file():
            # Check adjacent directory
            for c_wdl in [mmap_p.parent / "development.wdl", mmap_p.parent.parent / "Maps" / "development" / "development.wdl"]:
                if c_wdl.is_file():
                    wdl_cand = c_wdl
                    break

        if wdl_cand and wdl_cand.is_file():
            try:
                if wdl_cand.suffix == ".npz":
                    synth = TrestleWdlSynthesizer.load_npz(wdl_cand)
                    t257 = synth.get_trestle_257(tile_x, tile_y)
                    if t257 is not None:
                        trestle_np = t257
                else:
                    parser = WdlElevationParser(wdl_cand)
                    if parser.has_tile_data(tile_x, tile_y):
                        trestle_np = parser.extract_tile_257(tile_x, tile_y, order=1)
            except Exception:
                pass

        trestle_tensor = torch.from_numpy(trestle_np).unsqueeze(0).float()
        return (image_tensor, trestle_tensor, mask_tensor, coords_str)


class WoW_ObjectSieveConditioner:
    """Filters minimap imagery by authentic object placements while preserving painted roads and text."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "image": ("IMAGE", {}),
                "object_mask": ("MASK", {}),
                "inpaint_iterations": ("INT", {"default": 30, "min": 0, "max": 100, "step": 5}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("conditioned_image", "clean_mask")
    FUNCTION = "sieve_objects"
    CATEGORY = "parp-tools/WoW"

    def sieve_objects(
        self,
        image: torch.Tensor,
        object_mask: torch.Tensor,
        inpaint_iterations: int = 30,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # image is [B, H, W, C], object_mask is [B, H, W]
        img_np = image[0].cpu().numpy().copy()
        mask_np = (object_mask[0].cpu().numpy() > 0.5)

        if inpaint_iterations > 0 and np.any(mask_np):
            from harvester.v60.minimap_shadow_stripper import multiscale_laplacian_inpaint
            cleaned_rgb = multiscale_laplacian_inpaint(
                img_np, mask_np, num_scales=2, iterations_per_scale=inpaint_iterations
            )
        else:
            cleaned_rgb = img_np

        out_img = torch.from_numpy(cleaned_rgb).unsqueeze(0).float()
        out_mask = torch.from_numpy(mask_np.astype(np.float32)).unsqueeze(0).float()
        return (out_img, out_mask)


class WoW_HardZCalibrator:
    """Warp and anchor unit-space generative depth/heightfields to authentic WDL yards."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "generative_depth": ("IMAGE", {}),
                "trestle_elevation": ("MASK", {}),
                "target_relief_yards": ("FLOAT", {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.5}),
                "ridge_boost_yards": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 50.0, "step": 0.5}),
            },
            "optional": {
                "water_mask": ("MASK", {}),
                "object_mask": ("MASK", {}),
                "alpha_mask": ("MASK", {}),
            },
        }

    RETURN_TYPES = ("MASK", "FLOAT", "FLOAT", "FLOAT")
    RETURN_NAMES = ("calibrated_height", "z_min", "z_max", "relief_span")
    FUNCTION = "calibrate_elevation"
    CATEGORY = "parp-tools/WoW"

    def calibrate_elevation(
        self,
        generative_depth: torch.Tensor,
        trestle_elevation: torch.Tensor,
        target_relief_yards: float = 3.5,
        ridge_boost_yards: float = 1.5,
        water_mask: Optional[torch.Tensor] = None,
        object_mask: Optional[torch.Tensor] = None,
        alpha_mask: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, float, float, float]:
        from harvester.v60.building_foundation_carver import BuildingFoundationCarver

        # generative_depth: [B, H, W, C] or [B, H, W], trestle_elevation: [B, 257, 257]
        gen_np = generative_depth[0].squeeze().cpu().numpy()
        if gen_np.ndim == 3:
            gen_np = gen_np[..., 0]  # Take first channel
        trestle_np = trestle_elevation[0].cpu().numpy()

        w_mask_np = None
        if water_mask is not None:
            w_mask_np = water_mask[0].cpu().numpy() > 0.5
            if w_mask_np.shape != (257, 257):
                w_mask_np = ndimage.zoom(w_mask_np.astype(np.float32), (257.0 / w_mask_np.shape[0], 257.0 / w_mask_np.shape[1]), order=0) > 0.5

        obj_mask_np = None
        if object_mask is not None:
            obj_mask_np = object_mask[0].cpu().numpy() > 0.5
            if obj_mask_np.shape != (257, 257):
                obj_mask_np = ndimage.zoom(obj_mask_np.astype(np.float32), (257.0 / obj_mask_np.shape[0], 257.0 / obj_mask_np.shape[1]), order=0) > 0.5

        alpha_mask_np = None
        if alpha_mask is not None:
            alpha_mask_np = np.clip(alpha_mask[0].cpu().numpy().astype(np.float32), 0.0, 1.0)
            if alpha_mask_np.shape != (257, 257):
                alpha_mask_np = ndimage.zoom(alpha_mask_np, (257.0 / alpha_mask_np.shape[0], 257.0 / alpha_mask_np.shape[1]), order=1)[:257, :257]

        # Extract macro-meso relief from generative depth with sigma=12.0 cut
        gen_res = ndimage.zoom(gen_np, (257.0 / gen_np.shape[0], 257.0 / gen_np.shape[1]), order=1)[:257, :257]
        gen_clean = gen_res.copy()
        if obj_mask_np is not None and np.any(obj_mask_np):
            dil = ndimage.binary_dilation(obj_mask_np, iterations=2)
            perim = dil & (~obj_mask_np)
            fill_z = float(np.median(gen_clean[perim])) if np.any(perim) else float(np.median(gen_clean))
            gen_clean[obj_mask_np] = fill_z

        gen_lp = ndimage.gaussian_filter(gen_clean, sigma=12.0)
        gen_hp = gen_clean - gen_lp

        # Attenuate texture splat bleeding if alpha mask is provided
        if alpha_mask_np is not None:
            gen_hp *= (1.0 - 0.85 * alpha_mask_np)

        if obj_mask_np is not None:
            gen_hp[obj_mask_np] = 0.0

        land_bool = ~w_mask_np if w_mask_np is not None else np.ones((257, 257), dtype=bool)
        if obj_mask_np is not None:
            land_sample = land_bool & (~obj_mask_np)
        else:
            land_sample = land_bool

        p95 = float(np.percentile(np.abs(gen_hp[land_sample]), 95)) if np.any(land_sample) else float(np.percentile(np.abs(gen_hp), 95))
        scale = float(target_relief_yards / max(1e-5, p95))
        scaled_relief = gen_hp * scale

        calibrated_h = trestle_np.astype(np.float32) + scaled_relief

        # Level terrain under masked objects to the median average height of surrounding terrain
        if obj_mask_np is not None and np.any(obj_mask_np):
            carver = BuildingFoundationCarver(blend_margin=3)
            calibrated_h = carver.carve_masked_objects(calibrated_h, obj_mask_np)

        if w_mask_np is not None:
            calibrated_h[w_mask_np] = 0.0
            if np.any(land_bool):
                calibrated_h[land_bool] = np.maximum(calibrated_h[land_bool], 0.1)

        z_min = float(np.min(calibrated_h))
        z_max = float(np.max(calibrated_h))
        span = float(z_max - z_min)

        out_tensor = torch.from_numpy(calibrated_h).unsqueeze(0).float()
        return (out_tensor, z_min, z_max, span)


class WoW_MeshExporter:
    """Exports ComfyUI heightfields and meshes into North-up Cartesian OBJ/GLB models."""

    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "height_map": ("MASK", {}),
                "texture_image": ("IMAGE", {}),
                "output_dir": ("STRING", {"default": "output/comfyui_exports", "multiline": False}),
                "file_stem": ("STRING", {"default": "development_tile", "multiline": False}),
            },
            "optional": {
                "export_obj": ("BOOLEAN", {"default": True}),
                "export_glb": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("obj_path", "glb_path")
    FUNCTION = "export_meshes"
    CATEGORY = "parp-tools/WoW"

    def export_meshes(
        self,
        height_map: torch.Tensor,
        texture_image: torch.Tensor,
        output_dir: str,
        file_stem: str,
        export_obj: bool = True,
        export_glb: bool = True,
    ) -> Tuple[str, str]:
        out_d = Path(output_dir)
        out_d.mkdir(parents=True, exist_ok=True)

        h_np = height_map[0].cpu().numpy().astype(np.float32)
        if h_np.shape != (257, 257):
            h_np = ndimage.zoom(h_np, (257.0 / h_np.shape[0], 257.0 / h_np.shape[1]), order=1)[:257, :257]

        tex_np = (texture_image[0].cpu().numpy() * 255.0).clip(0, 255).astype(np.uint8)
        tex_img = Image.fromarray(tex_np)
        tex_path = out_d / f"{file_stem}_texture.png"
        tex_img.save(tex_path)

        obj_path = str(out_d / f"{file_stem}.obj")
        glb_path = str(out_d / f"{file_stem}.glb")

        if export_obj:
            export_obj_mesh(h_np, tex_path, Path(obj_path), is_world_yards=True)
        if export_glb:
            export_glb_mesh(h_np, tex_img, Path(glb_path), is_world_yards=True)

        return (obj_path, glb_path)
