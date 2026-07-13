from __future__ import annotations

import numpy as np
import torch

from harvester.spec102.m0 import (
    PRECISE_MASK_KEY,
    M0ObjectMask,
    clean_minimap_with_mask,
    precise_object_target_256,
    segmentation_loss,
)


def test_m0_is_single_output_and_small() -> None:
    model = M0ObjectMask()
    output = model(torch.zeros(2, 3, 256, 256))
    parameters = sum(parameter.numel() for parameter in model.parameters())
    assert output.shape == (2, 1, 256, 256)
    assert 3_000_000 <= parameters <= 12_000_000


def test_m0_loss_is_finite_and_backpropagates() -> None:
    model = M0ObjectMask(base_channels=8)
    logits = model(torch.rand(1, 3, 32, 32))
    target = torch.zeros_like(logits)
    target[:, :, 8:16, 8:16] = 1.0
    loss, parts = segmentation_loss(logits, target)
    loss.backward()
    assert torch.isfinite(loss)
    assert set(parts) == {"bce", "dice_loss"}
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_cleaner_is_identity_without_mask_and_deterministic_with_mask() -> None:
    rgb = np.arange(256 * 256 * 3, dtype=np.uint8).reshape(256, 256, 3)
    empty = np.zeros((256, 256), dtype=bool)
    assert np.array_equal(clean_minimap_with_mask(rgb, empty), rgb)
    mask = empty.copy()
    mask[100:110, 100:110] = True
    first = clean_minimap_with_mask(rgb, mask)
    second = clean_minimap_with_mask(rgb, mask)
    assert np.array_equal(first, second)
    assert not np.array_equal(first[mask], rgb[mask])


def test_precise_mask_projection_uses_all_four_257_grid_corners() -> None:
    precise = np.zeros((257, 257), dtype=np.float32)
    precise[1, 1] = 0.75
    projected = precise_object_target_256(precise)
    assert projected.shape == (256, 256)
    assert np.array_equal(projected[:2, :2], np.full((2, 2), 0.75, dtype=np.float32))
    assert projected[2:, 2:].max() == 0.0


def test_precise_mask_projection_rejects_reduced_mask() -> None:
    with np.testing.assert_raises_regex(ValueError, PRECISE_MASK_KEY):
        precise_object_target_256(np.zeros((256, 256), dtype=np.float32))
