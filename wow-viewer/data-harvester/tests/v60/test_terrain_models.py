from __future__ import annotations

import pytest
import torch

from harvester.v60.terrain_models import (
    DPT_SMALL_ID,
    PYRAMID_CNN_ID,
    SEGFORMER_B0_ID,
    TERRAIN_ARCHITECTURES,
    UNET_LITE_ID,
    TerrainModelError,
    build_terrain_model,
)


@pytest.mark.parametrize("architecture", TERRAIN_ARCHITECTURES)
def test_tiny_terrain_architecture_contract(architecture: str) -> None:
    torch.manual_seed(6001)
    model, identity = build_terrain_model(architecture, profile="tiny")
    model.eval()
    with torch.no_grad():
        output = model(torch.zeros((1, 1, 256, 256)))

    assert tuple(output.shape) == (1, 257, 257)
    assert torch.isfinite(output).all()
    assert float(output.min()) >= 0.0
    assert float(output.max()) <= 1.0
    assert identity["id"] == architecture
    assert identity["parameter_count"] > 0
    assert identity["weights"] == "random_init"
    assert identity["pretrained"] is False


def test_unknown_terrain_architecture_fails_closed() -> None:
    with pytest.raises(TerrainModelError, match="architecture must be one of"):
        build_terrain_model("depth_anything")


def test_architecture_ids_are_explicit() -> None:
    assert {UNET_LITE_ID, PYRAMID_CNN_ID, DPT_SMALL_ID, SEGFORMER_B0_ID} == set(
        TERRAIN_ARCHITECTURES
    )
