"""Unit tests for Spec 120 OBB Dataset Builder (T004)."""

from __future__ import annotations

import numpy as np

from harvester.spec120.obb_dataset import compute_spatial_split, convert_targets_to_array


def test_compute_spatial_split() -> None:
    """Verify 4x4 block spatial isolation in spatial split."""
    # Generate tiles across multiple 4x4 blocks
    sample_tiles = [(x, y) for x in range(30, 38) for y in range(30, 38)]
    train_tiles, val_tiles = compute_spatial_split(sample_tiles, val_ratio=0.25, seed=42)

    assert len(train_tiles) + len(val_tiles) == len(sample_tiles)
    assert len(val_tiles) > 0

    # Ensure no tile block overlap
    train_blocks = {(tx // 4, ty // 4) for tx, ty in train_tiles}
    val_blocks = {(tx // 4, ty // 4) for tx, ty in val_tiles}

    assert train_blocks.isdisjoint(val_blocks)


def test_convert_targets_to_array() -> None:
    """Verify target conversion to fixed-size array."""
    targets = [
        {"class_id": 0, "cx_norm": 0.5, "cy_norm": 0.5, "w_norm": 0.1, "h_norm": 0.1, "angle_deg": 45.0},
        {"class_id": 1, "cx_norm": 0.2, "cy_norm": 0.3, "w_norm": 0.05, "h_norm": 0.05, "angle_deg": 90.0},
    ]

    arr = convert_targets_to_array(targets, max_targets=4)
    assert arr.shape == (4, 6)

    # First row check
    np.testing.assert_allclose(arr[0], [0.0, 0.5, 0.5, 0.1, 0.1, 45.0])
    # Second row check
    np.testing.assert_allclose(arr[1], [1.0, 0.2, 0.3, 0.05, 0.05, 90.0])
    # Unused row check (padding value -1.0)
    assert np.all(arr[2:] == -1.0)
