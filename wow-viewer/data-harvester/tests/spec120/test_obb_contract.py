"""Unit tests for Spec 120 OBB Contract (T004)."""

from __future__ import annotations

import pytest

from harvester.spec120.obb_contract import (
    ObbContractError,
    derive_coarse_class,
    format_sidecar_item,
    is_pixel_on_tile,
    placement_to_obb_target,
    tile_pixels_to_world,
    validate_sidecar_schema,
    world_to_tile_pixels,
)


def test_world_to_tile_pixels_round_trip() -> None:
    """Verify world_to_tile_pixels and tile_pixels_to_world round-trip accurately."""
    tile_x, tile_y = 32, 32
    original_world_x, original_world_y = 120.5, -450.25

    px, py = world_to_tile_pixels(original_world_x, original_world_y, tile_x, tile_y)
    reconstructed_x, reconstructed_y = tile_pixels_to_world(px, py, tile_x, tile_y)

    assert pytest.approx(reconstructed_x, abs=1e-4) == original_world_x
    assert pytest.approx(reconstructed_y, abs=1e-4) == original_world_y


def test_is_pixel_on_tile() -> None:
    """Verify tile boundary checks."""
    assert is_pixel_on_tile(128.0, 128.0) is True
    assert is_pixel_on_tile(0.0, 256.0) is True
    assert is_pixel_on_tile(-5.0, 100.0) is False
    assert is_pixel_on_tile(-5.0, 100.0, margin_px=10.0) is True


def test_derive_coarse_class() -> None:
    """Verify coarse class classification rules for 0.5.3 (wmo vs mdx)."""
    assert derive_coarse_class("modf", "World/wmo/Azeroth/Castle.wmo") == "wmo"
    assert derive_coarse_class("mddf", "World/azeroth/elwynn/passivedoodads/trees/elwynntree.mdx") == "mdx"


def test_placement_to_obb_target() -> None:
    """Verify OBB target parameter calculation and normalization."""
    target = placement_to_obb_target(
        world_x=0.0,
        world_y=0.0,
        tile_x=32,
        tile_y=32,
        extent_x_yards=20.8333,
        extent_y_yards=41.6666,
        rotation_deg=45.0,
        coarse_class="wmo",
    )

    assert target["class_id"] == 0
    assert target["coarse_class"] == "wmo"
    assert pytest.approx(target["cx_norm"], abs=1e-3) == 0.0
    assert pytest.approx(target["cy_norm"], abs=1e-3) == 0.0
    assert pytest.approx(target["w_px"], abs=0.5) == 10.0
    assert pytest.approx(target["h_px"], abs=0.5) == 20.0
    assert target["angle_deg"] == 45.0


def test_format_sidecar_item_and_validate() -> None:
    """Verify sidecar item formatting and schema validation."""
    item = format_sidecar_item(
        instance_id=101,
        position_px=(124.5, 88.2),
        world_pos=(1845.2, -432.1, 65.4),
        scale_px=(32.4, 28.1),
        scale_factor=1.05,
        rotation_deg=45.0,
        coarse_class="wmo",
        retrieved_asset="World/wmo/Castle.wmo",
        confidence=0.9412,
    )

    assert item["instance_id"] == 101
    assert item["position_px"] == [124.5, 88.2]
    assert item["world_position"] == [1845.2, -432.1, 65.4]

    # Schema validation pass
    assert validate_sidecar_schema([item]) is True

    # Schema validation failure
    invalid_item = dict(item)
    del invalid_item["confidence"]
    with pytest.raises(ObbContractError):
        validate_sidecar_schema([invalid_item])
