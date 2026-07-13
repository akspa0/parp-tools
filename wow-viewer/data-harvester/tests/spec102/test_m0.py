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
from harvester.spec102.m0_validation import M0ValidationSample, render_m0_validation_panel


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


def test_validation_panel_embeds_legend_metadata_and_agreement_column() -> None:
    probability = np.zeros((256, 256), dtype=np.float32)
    target = np.zeros((256, 256), dtype=np.float32)
    probability[10:20, 10:20] = 0.9
    target[15:25, 15:25] = 1.0
    panel = render_m0_validation_panel([
        M0ValidationSample(
            row=42, build="3_3_5_12340", map_name="Northrend", tile_x=17, tile_y=24,
            source_rgb=np.zeros((256, 256, 3), dtype=np.uint8),
            probability=probability, target=target,
        )
    ], split="validation_map", epoch=12, threshold=0.5, checkpoint_label="best.pt")
    assert panel.size == (1024, 362)
    pixels = np.asarray(panel)
    agreement = pixels[76:332, 768:1024]
    assert np.any(np.all(agreement == (40, 220, 90), axis=-1))
    assert np.any(np.all(agreement == (240, 65, 65), axis=-1))
    assert np.any(np.all(agreement == (70, 135, 255), axis=-1))
