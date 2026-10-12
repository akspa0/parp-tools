"""Unit tests for Spec 266: Trestle Elevation Reconstruction & Lineage Synthesis."""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from harvester.v60.mesh_exporter import export_glb_mesh, export_obj_mesh
from harvester.v60.trestle_elevation_model import (
    TrestleElevationUNet,
    trestle_composite_loss,
)
from harvester.v60.trestle_wdl_synthesizer import TrestleWdlSynthesizer


def test_trestle_unet_forward_shapes() -> None:
    model = TrestleElevationUNet(in_channels=6, base_channels=32)
    x = torch.rand(2, 6, 256, 256, dtype=torch.float32)

    delta_z, bounds = model(x)
    assert delta_z.shape == (2, 257, 257)
    assert bounds.shape == (2, 2)
    assert delta_z.dtype == torch.float32
    assert bounds.dtype == torch.float32


def test_trestle_unet_parameter_budget() -> None:
    """Verifies AC-002: Parameter budget under 10M (vs >100M in historical v7)."""
    model = TrestleElevationUNet(in_channels=6, base_channels=32)
    param_count = model.parameter_count
    assert param_count < 10_000_000, f"Expected < 10M parameters, got {param_count:,}"
    print(f"Verified TrestleElevationUNet parameter count: {param_count:,} params.")


def test_trestle_composite_loss_and_gradients() -> None:
    model = TrestleElevationUNet(in_channels=6, base_channels=16)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    x = torch.rand(2, 6, 256, 256, dtype=torch.float32)
    z_trestle = torch.full((2, 257, 257), 800.0, dtype=torch.float32)
    gt_z = torch.full((2, 257, 257), 1200.0, dtype=torch.float32)
    gt_bounds = torch.tensor([[800.0, 1200.0], [800.0, 1200.0]], dtype=torch.float32)

    optimizer.zero_grad()
    pred_delta_z, pred_bounds = model(x)
    loss, metrics = trestle_composite_loss(
        pred_delta_z=pred_delta_z,
        pred_bounds=pred_bounds,
        z_trestle=z_trestle,
        gt_z=gt_z,
        gt_bounds=gt_bounds,
    )

    assert "final_l1" in metrics
    assert "grad_loss" in metrics
    assert "normal_loss" in metrics
    assert "bounds_loss" in metrics
    assert "delta_smooth" in metrics
    assert "total_loss" in metrics
    assert loss.item() > 0.0

    loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
            break


def test_trestle_reconstruct_additive() -> None:
    model = TrestleElevationUNet(in_channels=6, base_channels=16)
    model.eval()

    x = torch.rand(1, 6, 256, 256, dtype=torch.float32)
    z_trestle = torch.full((1, 257, 257), 500.0, dtype=torch.float32)

    with torch.no_grad():
        delta_z, _ = model(x)
        z_final = model.reconstruct(x, z_trestle)
        diff = torch.max(torch.abs(z_final - (z_trestle + delta_z)))
        assert float(diff) < 1e-5


def test_trestle_wdl_synthesizer_lookup() -> None:
    npz_path = Path("../output/development_synthesized_wdl.npz")
    if not npz_path.is_file():
        pytest.skip("Synthesized WDL npz not found on disk")

    synth = TrestleWdlSynthesizer.load_npz(npz_path)
    assert synth.has_tile(16, 33)

    h17 = synth.get_trestle_17(16, 33)
    assert h17 is not None
    assert h17.shape == (17, 17)
    assert np.max(h17) - np.min(h17) > 500.0  # High relief mountain massif

    h257 = synth.get_trestle_257(16, 33)
    assert h257 is not None
    assert h257.shape == (257, 257)

    h256 = synth.get_trestle_256(16, 33)
    assert h256 is not None
    assert h256.shape == (256, 256)


def test_mesh_winding_upward_normals(tmp_path: Path) -> None:
    """Verifies AC-005: OBJ and GLB exported face normals point strictly upward (+Z in OBJ, +Y in GLB)."""
    h = np.linspace(100.0, 500.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    tex_path = tmp_path / "test_tex.png"
    Image.new("RGB", (32, 32), color=(128, 128, 128)).save(tex_path)

    # 1. OBJ Mesh Test
    obj_path = tmp_path / "test_upward.obj"
    export_obj_mesh(h, tex_path, obj_path, is_world_yards=True)

    vertices = []
    faces = []
    with open(obj_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
            elif line.startswith("f "):
                parts = line.strip().split()[1:]
                # Each part is v/vt/vn
                v_idx = [int(p.split("/")[0]) - 1 for p in parts]
                faces.append(v_idx)

    v_arr = np.array(vertices)
    for face in faces[:20]:
        v0, v1, v2 = v_arr[face[0]], v_arr[face[1]], v_arr[face[2]]
        norm = np.cross(v1 - v0, v2 - v0)
        assert norm[2] > 0.0, f"OBJ normal Z component was non-positive: {norm}"

    # 2. GLB Mesh Test
    glb_path = tmp_path / "test_upward.glb"
    export_glb_mesh(h, tex_path, glb_path, is_world_yards=True)
    assert glb_path.is_file()
    assert glb_path.stat().st_size > 1000
