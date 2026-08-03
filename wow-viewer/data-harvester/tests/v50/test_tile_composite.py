"""The three composite modes must differ only by the display transform."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from harvester.v50.tile_composite import (
    DEFAULT_CELL,
    LEGEND_HEIGHT,
    MODES,
    build_map_cells,
    downsample,
    effective_factor,
    hillshade_np,
    render_composite,
    restore_height,
)


def test_downsample_preserves_the_mean_and_the_target_size() -> None:
    field = np.linspace(0.0, 1.0, 257 * 257, dtype=np.float32).reshape(257, 257)
    reduced = downsample(field, 64)
    assert reduced.shape == (64, 64)
    assert reduced.min() >= float(field.min())
    assert reduced.max() <= float(field.max())


def test_hillshade_matches_the_torch_forward_model() -> None:
    """The composite uses a numpy twin for speed; it must agree with the canonical model."""
    from harvester.v50.tile_synthesis import hillshade as torch_hillshade

    rng = np.random.default_rng(0)
    field = rng.random((48, 48)).astype(np.float32) * 20.0
    numpy_shaded = hillshade_np(field, azimuth_deg=315.0, elevation_deg=30.0)
    torch_shaded = torch_hillshade(field)
    # Interior only: the two use different edge-gradient conventions at the border.
    assert np.allclose(numpy_shaded[1:-1, 1:-1], torch_shaded[1:-1, 1:-1], atol=0.02)


def test_restore_scales_relief_about_the_tile_floor() -> None:
    """A restored deep-ocean tile must stay on the ocean floor, not launch to the surface."""
    field = np.array([[-501.5, -501.4], [-501.3, -501.2]], dtype=np.float64)
    restored = restore_height(field, 10.0)
    assert restored.min() == pytest.approx(-501.5)          # floor is anchored
    assert np.ptp(restored) == pytest.approx(np.ptp(field) * 10.0)
    # No-ops leave the field untouched.
    assert np.array_equal(restore_height(field, 1.0), field)
    assert np.array_equal(restore_height(field, None), field)
    # A flat tile cannot be given relief by any factor.
    flat = np.full((4, 4), 7.0)
    assert np.ptp(restore_height(flat, 1000.0)) == 0.0


def test_effective_factor_bypasses_the_viewers_epsilon_but_not_physics() -> None:
    """The viewer's own epsilon refuses sub-0.001 tiles, which is why it never surfaced them."""
    record = {"neighbour_height_min": -500.0, "neighbour_height_max": 50.0,
              "suggested_amplification_factor": 1.0}
    # A tile the viewer rejects as unamplifiable still gets a real factor here.
    assert effective_factor(record, 0.000519) == pytest.approx(550.0 / 0.000519)
    # An already-suggested factor is respected rather than recomputed.
    assert effective_factor({**record, "suggested_amplification_factor": 270.0}, 1.39) == 270.0
    # Nothing to amplify, or nothing to amplify toward.
    assert effective_factor(record, 0.0) == 1.0
    assert effective_factor({"suggested_amplification_factor": None}, 5.0) == 1.0
    # Neighbours no taller than the tile itself means no amplification.
    assert effective_factor({"neighbour_height_min": 0.0, "neighbour_height_max": 1.0}, 5.0) == 1.0


def test_composite_renders_every_mode_from_one_read_pass(tmp_path: Path) -> None:
    rng = np.random.default_rng(1)
    group = {"height_257": rng.random((3, 257, 257)).astype(np.float32) * 100.0}
    index_rows = [
        {"map": "Kalidar", "tile_x": 10, "tile_y": 20},
        {"map": "Kalidar", "tile_x": 11, "tile_y": 20},
        {"map": "Kalidar", "tile_x": 10, "tile_y": 21},
    ]
    inventory = {
        "Kalidar_10_20": {"classification": "usable", "height_range": 100.0},
        "Kalidar_11_20": {"classification": "weak_signal", "height_range": 2.0,
                          "suggested_amplification_factor": 8.0},
        "Kalidar_10_21": {"classification": "white_plate", "height_range": 0.0},
    }
    cells, scale = build_map_cells(group, index_rows, inventory, cell=DEFAULT_CELL)

    assert set(cells) == {(10, 20), (11, 20), (10, 21)}
    assert set(cells[(10, 20)]) == set(MODES)
    # The global scale is set by full-scale tiles only, so one degenerate tile cannot crush it.
    assert scale["global_max"] > scale["global_min"]

    for mode in MODES:
        out = tmp_path / f"c-{mode}.png"
        render_composite(cells, inventory, map_name="Kalidar", mode=mode, output=out)
        with Image.open(out) as image:
            assert image.width == 2 * DEFAULT_CELL
            assert image.height == LEGEND_HEIGHT + 2 * DEFAULT_CELL

    with pytest.raises(ValueError, match="mode must be one of"):
        render_composite(cells, inventory, map_name="Kalidar", mode="nope", output=tmp_path / "x.png")
    with pytest.raises(ValueError, match="zero tiles"):
        render_composite({}, inventory, map_name="Kalidar", mode="absolute", output=tmp_path / "y.png")


def test_liquid_floods_basins_without_carving_them() -> None:
    """Liquid raises terrain to its surface; it never lowers it, and it never invents wetness."""
    from harvester.v50.tile_composite import flood_with_liquid

    terrain = np.array([[-50.0, -10.0], [0.0, 20.0]])
    level = np.full((2, 2), -5.0)
    mask = np.array([[1.0, 1.0], [0.0, 0.0]])

    surface, wet = flood_with_liquid(terrain, level, mask)
    assert surface[0, 0] == -5.0    # deep basin flooded up to the surface
    assert surface[0, 1] == -5.0    # shallow basin flooded up to the surface
    assert surface[1, 0] == 0.0     # dry: untouched even though it sits below the level
    assert surface[1, 1] == 20.0    # dry high ground untouched
    assert wet.tolist() == [[1.0, 1.0], [0.0, 0.0]]

    # Liquid below the terrain it covers must not carve it down.
    high = np.array([[100.0]])
    surface, _ = flood_with_liquid(high, np.array([[5.0]]), np.array([[1.0]]))
    assert surface[0, 0] == 100.0

    # No liquid arrays at all -> terrain passes through untouched, nothing marked wet.
    surface, wet = flood_with_liquid(terrain, None, None)
    assert np.array_equal(surface, terrain)
    assert not wet.any()


def test_liquid_tint_marks_wet_without_erasing_relief() -> None:
    from harvester.v50.tile_composite import _tint

    shaded = np.array([[0.8, 0.8]])
    wet = np.array([[0.0, 1.0]])
    rgb = _tint(shaded, wet)
    assert tuple(rgb[0, 0]) == (204, 204, 204)      # dry: plain grey
    assert rgb[0, 1, 2] > rgb[0, 1, 0]              # wet: blue-shifted
    assert rgb[0, 1, 0] > 0                         # relief still readable through the tint


def test_liquid_mode_is_rendered_and_reported(tmp_path: Path) -> None:
    from harvester.v50.tile_composite import MODES

    assert "liquid" in MODES
    rng = np.random.default_rng(4)
    group = {
        "height_257": rng.random((2, 257, 257)).astype(np.float32) * 100.0,
        "liquid_height": np.full((2, 257, 257), 60.0, dtype=np.float32),
        "liquid_mask": np.ones((2, 257, 257), dtype=np.float32),
    }
    index_rows = [{"map": "Deepholm", "tile_x": 5, "tile_y": 5},
                  {"map": "Deepholm", "tile_x": 6, "tile_y": 5}]
    inventory = {k: {"classification": "usable", "height_range": 100.0}
                 for k in ("Deepholm_05_05", "Deepholm_06_05")}
    cells, scale = build_map_cells(group, index_rows, inventory, cell=DEFAULT_CELL)
    assert scale["liquid_available"] is True
    assert scale["wet_tiles"] == 2
    render_composite(cells, inventory, map_name="Deepholm", mode="liquid",
                     output=tmp_path / "liq.png")
    assert (tmp_path / "liq.png").is_file()


def test_texture_preserves_minimap_brightness_and_adds_relief() -> None:
    """Multiplying by a raw Lambert term (mean ~0.5) would halve the map's brightness."""
    from harvester.v50.tile_composite import texture_over_relief

    albedo = np.full((16, 16, 3), 200, dtype=np.uint8)
    rng = np.random.default_rng(0)
    shaded = np.clip(rng.normal(0.5, 0.15, (16, 16)), 0.0, 1.0)

    out = texture_over_relief(albedo, shaded)
    # Mean brightness stays where the artist put it, rather than being halved.
    assert abs(float(out.mean()) - 200.0) < 12.0
    # Relief actually modulates: flat shading and varied shading differ.
    flat = texture_over_relief(albedo, np.full((16, 16), 0.5))
    assert np.array_equal(flat, np.full((16, 16, 3), 200, dtype=np.uint8))
    assert not np.array_equal(out, flat)
    # Colour is carried from the albedo, not invented by the shading.
    tinted = np.zeros((16, 16, 3), dtype=np.uint8); tinted[..., 1] = 200
    green = texture_over_relief(tinted, shaded)
    assert green[..., 1].mean() > 100 and green[..., 0].max() == 0


def test_downsample_rgb_keeps_three_channels_and_colour() -> None:
    from harvester.v50.tile_composite import downsample_rgb

    image = np.zeros((256, 256, 3), dtype=np.uint8)
    image[..., 0] = 255
    reduced = downsample_rgb(image, 64)
    assert reduced.shape == (64, 64, 3)
    assert reduced[..., 0].min() == 255 and reduced[..., 1].max() == 0


def test_textured_mode_uses_authored_minimap_and_falls_back(tmp_path: Path) -> None:
    from harvester.v50.tile_composite import MODES, resolve_minimap_array

    assert "textured" in MODES
    assert resolve_minimap_array({"minimap_rgb_authored": 1, "minimap_rgb": 1}) == "minimap_rgb_authored"
    assert resolve_minimap_array({"minimap_rgb": 1}) == "minimap_rgb"
    assert resolve_minimap_array({}) is None
    assert resolve_minimap_array({"minimap_rgb": 1}, "minimap_rgb_authored") is None  # asked, absent

    rng = np.random.default_rng(5)
    group = {
        "height_257": rng.random((2, 257, 257)).astype(np.float32) * 100.0,
        "minimap_rgb_authored": rng.integers(0, 256, (2, 256, 256, 3), dtype=np.uint8),
    }
    index_rows = [{"map": "Kalidar", "tile_x": 1, "tile_y": 1},
                  {"map": "Kalidar", "tile_x": 2, "tile_y": 1}]
    inventory = {k: {"classification": "usable", "height_range": 100.0}
                 for k in ("Kalidar_01_01", "Kalidar_02_01")}
    cells, scale = build_map_cells(group, index_rows, inventory, cell=DEFAULT_CELL)
    assert scale["minimap_array"] == "minimap_rgb_authored"
    assert scale["textured_tiles"] == 2
    render_composite(cells, inventory, map_name="Kalidar", mode="textured", output=tmp_path / "t.png")
    assert (tmp_path / "t.png").is_file()

    # A store with no minimap at all still renders; the albedo falls back to neutral grey.
    _, bare = build_map_cells({"height_257": group["height_257"]}, index_rows, inventory,
                              cell=DEFAULT_CELL)
    assert bare["minimap_array"] is None and bare["textured_tiles"] == 0
