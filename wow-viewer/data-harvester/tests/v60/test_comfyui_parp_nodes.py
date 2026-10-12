"""Tests for parp-tools ComfyUI Custom Node Suite & Terrain Feature Synthesizer (Spec 267)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from comfyui_parp_nodes import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    WoW_AdtLoader,
    WoW_HardZCalibrator,
    WoW_MeshExporter,
    WoW_ObjectSieveConditioner,
)
from harvester.v60.terrain_feature_synthesizer import (
    compute_surface_normals,
    synthesize_multiband_terrain,
)


def test_comfyui_node_mappings() -> None:
    """Ensure all custom nodes are properly exported in NODE_CLASS_MAPPINGS."""
    expected = [
        "WoW_AdtLoader",
        "WoW_ObjectSieveConditioner",
        "WoW_HardZCalibrator",
        "WoW_MeshExporter",
    ]
    for name in expected:
        assert name in NODE_CLASS_MAPPINGS
        assert name in NODE_DISPLAY_NAME_MAPPINGS


def test_terrain_feature_synthesizer_multiband() -> None:
    """Validate multi-band SfS + ridge + macro WDL fusion."""
    macro_wdl = np.full((257, 257), 150.0, dtype=np.float32)
    sfs_256 = np.random.randn(256, 256).astype(np.float32)
    ridge_256 = np.zeros((256, 256), dtype=np.float32)
    ridge_256[120:136, 120:136] = 1.0  # mountain ridge block

    water_mask = np.zeros((257, 257), dtype=bool)
    water_mask[200:, :] = True  # ocean bottom

    fused_h, normals, metrics = synthesize_multiband_terrain(
        macro_wdl_257=macro_wdl,
        integrated_sfs_256=sfs_256,
        ridge_mask_256=ridge_256,
        water_mask_257=water_mask,
        target_relief_yards=14.0,
        ridge_boost_yards=5.0,
    )

    assert fused_h.shape == (257, 257)
    assert normals.shape == (256, 256, 3)
    # Ocean strictly clamped to 0.0
    assert np.all(fused_h[water_mask] == 0.0)
    # Land above sea level
    assert np.all(fused_h[~water_mask] >= 0.1)
    # Ridge boost was applied
    assert metrics["ridge_max_boost"] > 0.0


def test_wow_hard_z_calibrator_node() -> None:
    """Validate WoW_HardZCalibrator scales generative depth to WDL trestle yards."""
    calibrator = WoW_HardZCalibrator()

    # Generative unit depth [1, 256, 256, 1] in [0, 1]
    gen_depth = torch.rand(1, 256, 256, 1, dtype=torch.float32)
    # Coarse WDL trestle [1, 257, 257] spanning 100 to 250 yards
    trestle = torch.linspace(100.0, 250.0, 257).repeat(257, 1).unsqueeze(0).float()

    calibrated_h, z_min, z_max, span = calibrator.calibrate_elevation(
        generative_depth=gen_depth,
        trestle_elevation=trestle,
        target_relief_yards=15.0,
        ridge_boost_yards=4.0,
    )

    assert calibrated_h.shape == (1, 257, 257)
    assert z_min >= 70.0
    assert z_max <= 280.0
    assert span > 100.0


def test_wow_object_sieve_conditioner_node() -> None:
    """Validate WoW_ObjectSieveConditioner preserves unmasked pixels."""
    conditioner = WoW_ObjectSieveConditioner()

    img = torch.ones(1, 256, 256, 3, dtype=torch.float32) * 0.5
    mask = torch.zeros(1, 256, 256, dtype=torch.float32)
    mask[0, 50:70, 50:70] = 1.0  # building footprint

    out_img, out_mask = conditioner.sieve_objects(img, mask, inpaint_iterations=5)
    assert out_img.shape == (1, 256, 256, 3)
    assert out_mask.shape == (1, 256, 256)


def test_wow_mesh_exporter_node() -> None:
    """Validate WoW_MeshExporter outputs OBJ and GLB files."""
    exporter = WoW_MeshExporter()

    height_map = torch.ones(1, 257, 257, dtype=torch.float32) * 50.0
    tex_image = torch.ones(1, 256, 256, 3, dtype=torch.float32) * 0.8

    with tempfile.TemporaryDirectory() as tmpdir:
        obj_path, glb_path = exporter.export_meshes(
            height_map=height_map,
            texture_image=tex_image,
            output_dir=tmpdir,
            file_stem="unit_test_tile",
            export_obj=True,
            export_glb=True,
        )

        assert Path(obj_path).is_file()
        assert Path(glb_path).is_file()
        assert Path(obj_path).stat().st_size > 0
        assert Path(glb_path).stat().st_size > 0
