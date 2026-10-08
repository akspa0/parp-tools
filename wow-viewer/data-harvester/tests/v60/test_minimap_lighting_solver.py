"""Tests for minimap lighting and specular calibration solver."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v60.minimap_lighting_solver import (
    compute_photometric_mae,
    render_synthetic_shadow,
    solve_minimap_lighting,
)


def test_render_synthetic_shadow_flat():
    # Surface pointing straight up along +Z (0, 0, 1)
    normals = np.zeros((16, 16, 3), dtype=np.float32)
    normals[..., 2] = 1.0

    # Solar overhead: azimuth 0, elevation pi/2
    rendered = render_synthetic_shadow(
        normals=normals,
        solar_azimuth=0.0,
        solar_elevation=float(np.pi / 2.0),
        ambient=0.2,
        diffuse=0.8,
        specular_intensity=0.0,
    )

    assert rendered.shape == (16, 16)
    # Cos theta = N . L = 1.0, so rendered should be ambient + diffuse = 1.0
    np.testing.assert_allclose(rendered, 1.0, atol=1e-5)


def test_render_specular_lobe():
    normals = np.zeros((16, 16, 3), dtype=np.float32)
    normals[..., 2] = 1.0

    # Half angle for overhead sun and overhead camera is (0, 0, 1)
    # N . H = 1.0, specular lobe is fully active
    with_spec = render_synthetic_shadow(
        normals=normals,
        solar_azimuth=0.0,
        solar_elevation=float(np.pi / 2.0),
        ambient=0.2,
        diffuse=0.5,
        specular_intensity=0.3,
        specular_power=16.0,
    )
    without_spec = render_synthetic_shadow(
        normals=normals,
        solar_azimuth=0.0,
        solar_elevation=float(np.pi / 2.0),
        ambient=0.2,
        diffuse=0.5,
        specular_intensity=0.0,
    )

    assert float(np.mean(with_spec)) > float(np.mean(without_spec))


def test_solve_minimap_lighting_synthetic_recovery():
    # Synthesize ground truth slope normals
    h, w = 32, 32
    normals = np.zeros((h, w, 3), dtype=np.float32)
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)
    normals[..., 0] = xx * 0.3
    normals[..., 1] = yy * 0.3
    normals[..., 2] = 1.0
    norm_len = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals = normals / norm_len

    gt_azimuth = float(np.pi / 4.0)  # 45 deg
    gt_elevation = float(np.pi / 3.0)  # 60 deg
    gt_ambient = 0.35
    gt_diffuse = 0.65

    target = render_synthetic_shadow(
        normals=normals,
        solar_azimuth=gt_azimuth,
        solar_elevation=gt_elevation,
        ambient=gt_ambient,
        diffuse=gt_diffuse,
    )

    profile = solve_minimap_lighting(
        normals=normals,
        observed_minimap=target,
        build="0.5.3.3368",
        map_name="Azeroth",
    )

    assert profile.converged is True
    assert profile.photometric_mae < 0.05
    assert abs(profile.ambient_intensity - gt_ambient) < 0.15
    assert abs(profile.diffuse_intensity - gt_diffuse) < 0.15
