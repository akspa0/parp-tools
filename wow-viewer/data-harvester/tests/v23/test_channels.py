from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.v23.channels import CHANNEL_ORDER, InputMode, build_channel_tensor

pytestmark = pytest.mark.v23


def _synthetic_tile() -> dict[str, np.ndarray]:
    tile = {
        "minimap_rgb": np.full((256, 256, 3), 128, dtype=np.uint8),
        "alpha_256": np.zeros((256, 256, 4), dtype=np.float32),
        "mcly_tileset_ids": np.zeros((16, 16, 4), dtype=np.int32),
        "normal_xyz": np.zeros((257, 257, 3), dtype=np.float32),
        "mcnr_mask_257": np.ones((257, 257), dtype=bool),
        "liquid_mask": np.zeros((256, 256), dtype=np.float32),
        "object_mask": np.zeros((257, 257), dtype=bool),
    }
    tile["alpha_256"][..., 0] = 1.0
    tile["mcly_tileset_ids"][..., 0] = 7
    return tile


def test_channel_order_matches_documented_indices() -> None:
    assert CHANNEL_ORDER == (
        "minimap_r",
        "minimap_g",
        "minimap_b",
        "alpha_0",
        "alpha_1",
        "alpha_2",
        "alpha_3",
        "tileset_pruned_0",
        "tileset_pruned_1",
        "tileset_pruned_2",
        "tileset_pruned_3",
        "normal_x",
        "normal_y",
        "normal_z",
        "terrain_valid_mask",
    )


def test_build_channel_tensor_full_mode_shape() -> None:
    tensor = build_channel_tensor(_synthetic_tile(), tileset_prune_table={7: 0})
    assert isinstance(tensor, torch.Tensor)
    assert tuple(tensor.shape) == (15, 256, 256)


def test_build_channel_tensor_minimap_only_shape() -> None:
    tensor = build_channel_tensor(_synthetic_tile(), mode=InputMode.MINIMAP_ONLY)
    assert tuple(tensor.shape) == (3, 256, 256)


def test_build_channel_tensor_minimap_alpha_shape() -> None:
    tensor = build_channel_tensor(_synthetic_tile(), mode=InputMode.MINIMAP_ALPHA)
    assert tuple(tensor.shape) == (7, 256, 256)


def test_build_channel_tensor_returns_channel_valid_mask() -> None:
    tensor, channel_valid_mask = build_channel_tensor(
        _synthetic_tile(),
        tileset_prune_table={7: 0},
        return_channel_valid_mask=True,
    )
    assert tuple(tensor.shape) == (15, 256, 256)
    assert tuple(channel_valid_mask.shape) == (15,)
    assert bool(channel_valid_mask.all())
