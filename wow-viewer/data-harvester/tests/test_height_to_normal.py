"""Tests for spec 077 Phase 6 (US5) analytic normals from predicted height."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest
import torch
import zarr
import zarr.codecs
import zarr.storage

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _entry in (_REPO_ROOT, _SRC_DIR, _SCRIPTS_DIR):
    _entry_str = str(_entry)
    if _entry_str not in sys.path:
        sys.path.insert(0, _entry_str)

from harvester.height_to_normal import (  # noqa: E402
    analytic_normal_difference,
    analytic_normals_from_height,
)

CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=1, shuffle="bitshuffle")


# --- numpy -----------------------------------------------------------------

def test_analytic_normals_constant_height_all_point_up() -> None:
    h = np.full((64, 64), 10.0, dtype=np.float32)
    n = analytic_normals_from_height(h)
    assert n.shape == (64, 64, 3)
    # z component should be ~1.0 everywhere; x and y should be ~0.
    np.testing.assert_allclose(n[..., 0], 0.0, atol=1e-5)
    np.testing.assert_allclose(n[..., 1], 0.0, atol=1e-5)
    np.testing.assert_allclose(n[..., 2], 1.0, atol=1e-5)
    # Unit-length check.
    norms = np.linalg.norm(n, axis=-1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_analytic_normals_slope_along_x_points_negative_x() -> None:
    # Height increases along x: surface tilts in +x direction, so the
    # normal should tilt in -x.
    x = np.linspace(0, 1, 32, dtype=np.float32)
    h = np.broadcast_to(x[None, :], (32, 32)).copy()
    n = analytic_normals_from_height(h)
    # Pick a center sample away from the border.
    sample = n[16, 16]
    assert sample[0] < 0.0  # negative x
    assert abs(sample[1]) < 0.1
    assert sample[2] > 0.0


def test_analytic_normals_numpy_batch_is_channel_first_and_unit_length() -> None:
    x = np.linspace(0, 1, 16, dtype=np.float32)
    h = np.stack([
        np.broadcast_to(x[None, :], (16, 16)),
        np.zeros((16, 16), dtype=np.float32),
    ], axis=0)
    n = analytic_normals_from_height(h)
    assert n.shape == (2, 3, 16, 16)
    norms = np.linalg.norm(n, axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)
    assert n[0, 0, 8, 8] < 0.0
    np.testing.assert_allclose(n[1, :, 8, 8], [0.0, 0.0, 1.0], atol=1e-5)


def test_analytic_normals_too_small_height_returns_unit_z() -> None:
    h = np.zeros((1, 1), dtype=np.float32)
    n = analytic_normals_from_height(h)
    assert n.shape == (1, 1, 3)
    np.testing.assert_allclose(n[0, 0], [0.0, 0.0, 1.0], atol=1e-5)


# --- torch -----------------------------------------------------------------

def test_analytic_normals_torch_matches_numpy() -> None:
    rng = np.random.default_rng(7)
    h_np = rng.standard_normal((32, 32)).astype(np.float32)
    h_torch = torch.from_numpy(h_np)
    n_np = analytic_normals_from_height(h_np)
    n_torch = analytic_normals_from_height(h_torch).numpy()
    np.testing.assert_allclose(n_np, n_torch, atol=1e-5)


def test_analytic_normals_torch_4d_input() -> None:
    h = torch.zeros(2, 1, 16, 16)
    h[..., 5:10, 5:10] = 1.0  # raised plateau
    n = analytic_normals_from_height(h)
    assert n.shape == (2, 3, 16, 16)
    # Plateau interior: z component dominates.
    sample = n[0, :, 7, 7]
    assert sample[2] > 0.9


# --- sanity check on small deltas ------------------------------------------

def test_analytic_normal_difference_small_for_small_height_delta() -> None:
    rng = np.random.default_rng(11)
    h = rng.standard_normal((32, 32)).astype(np.float32) * 0.1
    h_perturbed = h + rng.standard_normal((32, 32)).astype(np.float32) * 0.001
    diff = analytic_normal_difference(h, h_perturbed)
    assert diff < 0.1  # tiny perturbation -> tiny angular error


def test_analytic_normal_difference_large_for_inverted_height() -> None:
    h = np.linspace(0, 16, 32, dtype=np.float32)
    h = np.broadcast_to(h[None, :], (32, 32)).copy()
    h_inv = h[:, ::-1].copy()  # mirror x
    diff = analytic_normal_difference(h, h_inv)
    # Mirror should produce a much larger angular error.
    assert diff > 0.5


def test_analytic_normal_difference_supports_torch_4d_channel_first() -> None:
    h = torch.zeros(2, 1, 16, 16)
    h[:, :, :, 4:] = 1.0
    diff = analytic_normal_difference(h, h.clone())
    assert diff == pytest.approx(0.0, abs=1e-5)
