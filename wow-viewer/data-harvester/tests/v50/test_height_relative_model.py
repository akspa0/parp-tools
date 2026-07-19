"""Spec 112 T018: the relative-height target must be provably invariant to per-tile altitude
offset (FR-007 — the rejected lane's structural flaw made impossible by construction), exactly
invertible including the flat-tile floor, and the lean model must produce the right shapes with
gradients flowing."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.v50.height_relative_model import (
    HEIGHT_GRID,
    RANGE_FLOOR,
    TARGET_CONTRACT_VERSION,
    HeightRelativeNet,
    decode_relative_height,
    encode_relative_height,
    height_loss,
)


def test_constant_offset_leaves_the_target_unchanged():
    # Quarter-unit values and integer offsets are exactly representable. The contract promises
    # byte-identical targets for an altitude shift, so exercise that promise directly rather than
    # hiding a regression behind a tolerance.
    rng = np.random.default_rng(3)
    height = rng.integers(-800, 801, size=(257, 257)).astype(np.float64) * 0.25

    base, _, _ = encode_relative_height(height)
    for offset in (100.0, -531.25, 4096.0):
        shifted, _, _ = encode_relative_height(height + offset)
        np.testing.assert_array_equal(shifted, base)


def test_encode_decode_round_trips_world_heights():
    rng = np.random.default_rng(7)
    height = (rng.random((257, 257)) * 250.0 - 120.0).astype(np.float32)

    normalized, tile_min, tile_max = encode_relative_height(height)
    decoded = decode_relative_height(normalized, tile_min, tile_max)

    np.testing.assert_allclose(decoded, height, atol=1e-3)


def test_flat_tile_floor_is_well_defined_and_round_trips():
    flat = np.full((257, 257), 381.5, dtype=np.float32)  # PVPZone02-style plateau, sub-floor range

    normalized, tile_min, tile_max = encode_relative_height(flat)
    assert float(normalized.min()) == float(normalized.max()) == 0.0
    assert tile_max - tile_min < RANGE_FLOOR

    decoded = decode_relative_height(normalized, tile_min, tile_max)
    np.testing.assert_allclose(decoded, flat, atol=1e-4)


def test_near_flat_tile_retains_relief_without_amplifying_it():
    near_flat = np.linspace(381.5, 382.0, 257 * 257, dtype=np.float32).reshape(257, 257)

    normalized, tile_min, tile_max = encode_relative_height(near_flat)

    assert tile_max - tile_min < RANGE_FLOOR
    assert float(normalized.min()) == 0.0
    assert float(normalized.max()) == pytest.approx(0.5, abs=1e-6)
    np.testing.assert_allclose(
        decode_relative_height(normalized, tile_min, tile_max), near_flat, atol=1e-4
    )


def test_model_forward_shape_and_gradients():
    model = HeightRelativeNet(base=8)  # tiny variant for a CPU fixture
    x = torch.rand(2, 3, 256, 256)
    target = torch.rand(2, HEIGHT_GRID, HEIGHT_GRID)

    out = model(x)
    assert out.shape == (2, HEIGHT_GRID, HEIGHT_GRID)
    detached = out.detach()
    assert float(detached.min()) >= 0.0 and float(detached.max()) <= 1.0

    loss = height_loss(out, target)
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_contract_version_is_pinned():
    assert TARGET_CONTRACT_VERSION == "v112.1"
