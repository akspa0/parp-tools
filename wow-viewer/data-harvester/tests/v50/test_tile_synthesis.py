"""The synthesis must amplify a weak signal without inventing one."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from harvester.v50.tile_synthesis import (
    MOSAIC_TILE,
    PANEL_TITLES,
    autostretch,
    hillshade,
    normals_rgb,
    render_mosaic,
    render_tile_sheet,
)


def test_autostretch_gives_a_millimetre_of_relief_full_dynamic_range() -> None:
    """The whole point: 0.4m of relief must read like 400m would, or we cannot see what is there."""
    weak = np.linspace(10.0, 10.4, 64 * 64, dtype=np.float32).reshape(64, 64)
    image, lo, hi, true_zero = autostretch(weak)

    assert true_zero is False
    assert lo == 10.0
    assert hi > 10.39
    assert float(image.min()) == 0.0
    assert float(image.max()) == 1.0


def test_autostretch_separates_bit_exact_flat_from_merely_tiny() -> None:
    """'Exactly zero' and '8e-06' are different findings; float64 keeps them apart."""
    flat = np.full((16, 16), 42.0, dtype=np.float32)
    image, lo, hi, true_zero = autostretch(flat)
    assert true_zero is True
    assert lo == hi == 42.0
    assert np.allclose(image, 0.5)  # a flat plate, not amplified noise

    tiny = np.full((16, 16), 42.0, dtype=np.float32)
    tiny[0, 0] = np.float32(42.0 + 1e-5)
    image, lo, hi, true_zero = autostretch(tiny)
    assert true_zero is False
    assert 0.0 < hi - lo < 1e-4
    assert float(image.max()) == 1.0


def test_hillshade_responds_to_slope_and_not_to_offset() -> None:
    ramp = np.tile(np.linspace(0.0, 1.0, 32, dtype=np.float32), (32, 1))
    shaded = hillshade(ramp)
    assert shaded.shape == ramp.shape
    assert np.isfinite(shaded).all()
    assert 0.0 <= float(shaded.min()) and float(shaded.max()) <= 1.0
    # A constant offset is not relief; the shading must not move.
    assert np.allclose(shaded, hillshade(ramp + 500.0), atol=1e-6)
    # A flat field carries no gradient, so it cannot shade like a slope.
    assert not np.allclose(shaded, hillshade(np.zeros_like(ramp)))


def test_normals_rgb_blacks_out_vertices_with_no_mcnr() -> None:
    normals = np.zeros((8, 8, 3), dtype=np.float32)
    normals[..., 2] = 1.0
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4, :] = True

    image = normals_rgb(normals, mask, amplify=False)
    assert image.shape == (8, 8, 3)
    assert (image[4:, :] == 0).all()          # no MCNR vertex -> black, never a fake up-normal
    assert (image[:4, :, 2] == 255).all()     # z=+1 -> full blue in tangent-space encoding


def test_normals_amplification_reveals_the_bulk_not_just_the_outlier() -> None:
    """Long-tailed tilt: scaling to the max hides the typical vertex, which is the whole failure."""
    rng = np.random.default_rng(0)
    mask = np.ones((32, 32), dtype=bool)
    weak = np.zeros((32, 32, 3), dtype=np.float32)
    weak[..., 2] = 1.0
    # Typical tilt ~0.008 with one 0.3 outlier — the real distribution measured on Kalimdor_33_12.
    weak[..., 0] = rng.normal(0.0, 0.008, size=(32, 32)).astype(np.float32)
    weak[0, 0, 0] = 0.3

    raw = normals_rgb(weak, mask, amplify=False)
    amplified = normals_rgb(weak, mask, amplify=True)
    typical = np.abs(amplified[..., 0].astype(int) - 128)
    assert np.abs(raw[..., 0].astype(int) - 128).mean() < 2.0   # raw: invisible
    assert typical.mean() > 20.0                                 # amplified: clearly visible
    assert amplified[0, 0, 0] == 255                             # outlier still saturates

    # A perfectly flat tile has no deviation to stretch, so amplification invents nothing.
    flat = np.zeros((8, 8, 3), dtype=np.float32)
    flat[..., 2] = 1.0
    flat_mask = np.ones((8, 8), dtype=bool)
    assert np.array_equal(
        normals_rgb(flat, flat_mask, amplify=True), normals_rgb(flat, flat_mask, amplify=False)
    )


def test_tile_sheet_and_mosaic_are_written(tmp_path: Path) -> None:
    panels = {title: np.full((257, 257, 3), 60, dtype=np.uint8) for title in PANEL_TITLES}
    render_tile_sheet(panels, ["Azeroth_42_39", "range 0", "weak 0/256"], tmp_path / "t.png")
    with Image.open(tmp_path / "t.png") as image:
        assert image.width == 4 * 257

    records = [
        {"tile_x": 10, "tile_y": 20, "classification": "weak_signal",
         "_mosaic": np.full((MOSAIC_TILE, MOSAIC_TILE, 3), 90, dtype=np.uint8)},
        {"tile_x": 12, "tile_y": 22, "classification": "white_plate",
         "_mosaic": np.full((MOSAIC_TILE, MOSAIC_TILE, 3), 30, dtype=np.uint8)},
    ]
    render_mosaic(records, "Azeroth", tmp_path / "m.png")
    with Image.open(tmp_path / "m.png") as image:
        # Tiles are placed at TRUE grid coordinates, so the gap between 10 and 12 is preserved.
        assert image.width == 3 * MOSAIC_TILE + 64
        assert image.height == 3 * MOSAIC_TILE + 64
