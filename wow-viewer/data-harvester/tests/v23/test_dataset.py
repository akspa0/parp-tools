from __future__ import annotations

import numpy as np
import pytest
import torch

from harvester.v23.channels import InputMode
from harvester.v23.dataset import V23HeightDataset

pytestmark = pytest.mark.v23


class _StubV22Dataset:
    def __init__(self, tiles: list[dict]) -> None:
        self.tiles = tiles
        self.tile_index = [
            {
                "tile_id": int(tile.get("tile_id", index)),
                "build": str(tile.get("build", "")),
                "map": str(tile.get("map", "")),
                "tile_x": int(tile.get("tile_x", -1)),
                "tile_y": int(tile.get("tile_y", -1)),
            }
            for index, tile in enumerate(tiles)
        ]

    def __len__(self) -> int:
        return len(self.tiles)

    def __getitem__(self, idx: int) -> dict:
        return self.tiles[idx]


def _base_tile() -> dict:
    tile = {
        "tile_id": 0,
        "build": "3_3_5_12340",
        "map": "Azeroth",
        "tile_x": 30,
        "tile_y": 48,
        "minimap_rgb": np.full((256, 256, 3), 200, dtype=np.uint8),
        "alpha_256": np.zeros((256, 256, 4), dtype=np.float32),
        "mcly_tileset_ids": np.zeros((16, 16, 4), dtype=np.int32),
        "normal_xyz": np.zeros((257, 257, 3), dtype=np.float32),
        "mcnr_mask_257": np.ones((257, 257), dtype=bool),
        "liquid_mask": np.zeros((256, 256), dtype=np.float32),
        "liquid_height": np.zeros((256, 256), dtype=np.float32),
        "height_257": np.full((257, 257), 5.0, dtype=np.float32),
        "object_mask": np.zeros((257, 257), dtype=bool),
        "object_precise_mask": np.zeros((257, 257), dtype=np.float32),
    }
    tile["alpha_256"][..., 0] = 1.0
    tile["mcly_tileset_ids"][..., 0] = 7
    return tile


def test_v23_dataset_getitem_returns_documented_shapes() -> None:
    dataset = V23HeightDataset(_StubV22Dataset([_base_tile()]), build="3_3_5_12340", tileset_prune_table={7: 0})
    sample = dataset[0]
    assert tuple(sample["input"].shape) == (15, 256, 256)
    assert tuple(sample["target"].shape) == (1, 257, 257)
    assert tuple(sample["valid_mask"].shape) == (1, 257, 257)
    assert tuple(sample["channel_valid_mask"].shape) == (15,)


def test_v23_dataset_liquid_override_uses_liquid_height() -> None:
    tile = _base_tile()
    tile["liquid_mask"][:4, :4] = 1.0
    tile["liquid_height"][:4, :4] = 17.0
    dataset = V23HeightDataset(_StubV22Dataset([tile]), build="3_3_5_12340", tileset_prune_table={7: 0})
    sample = dataset[0]
    assert torch.isclose(sample["target"][0, 0, 0], torch.tensor(17.0))
    assert torch.isclose(sample["target"][0, 100, 100], torch.tensor(5.0))


def test_v23_dataset_missing_normals_zero_fills_and_marks_invalid() -> None:
    tile = _base_tile()
    tile.pop("normal_xyz")
    dataset = V23HeightDataset(_StubV22Dataset([tile]), build="3_3_5_12340", tileset_prune_table={7: 0})
    sample = dataset[0]
    assert tuple(sample["input"].shape) == (15, 256, 256)
    assert torch.equal(sample["input"][11:14], torch.zeros((3, 256, 256), dtype=torch.float32))
    assert sample["channel_valid_mask"][11:14].tolist() == [False, False, False]


def test_v23_dataset_minimap_alpha_mode_drops_normal_channels() -> None:
    tile = _base_tile()
    tile.pop("normal_xyz")
    dataset = V23HeightDataset(
        _StubV22Dataset([tile]),
        build="3_3_5_12340",
        input_mode=InputMode.MINIMAP_ALPHA,
        tileset_prune_table={7: 0},
    )
    sample = dataset[0]
    assert tuple(sample["input"].shape) == (7, 256, 256)
    assert tuple(sample["channel_valid_mask"].shape) == (7,)


def test_v23_dataset_docstring_lists_channel_contract() -> None:
    doc = V23HeightDataset.__doc__ or ""
    assert "0..2" in doc
    assert "3..6" in doc
    assert "7..10" in doc
    assert "11..13" in doc
    assert "14" in doc
