from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from harvester.v50.universal_relief_contract import (
    build_terrain_mesh,
    prepare_raster,
    raster_to_rgb,
    stitch_relief,
    write_obj,
)


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_prepare_raster_accepts_common_modes_and_preserves_coverage(mode: str) -> None:
    if mode == "L":
        pixels = np.arange(37 * 91, dtype=np.uint8).reshape(37, 91)
    else:
        channels = 4 if mode == "RGBA" else 3
        pixels = np.zeros((37, 91, channels), dtype=np.uint8)
        pixels[..., 0] = 123
        if channels == 4:
            pixels[..., 3] = 127
    image = Image.fromarray(pixels, mode=mode)

    prepared = prepare_raster(image, tile_size=32, overlap=8)

    assert prepared.source_rgb_hwc.shape == (37, 91, 3)
    assert prepared.transform.original_width == 91
    assert prepared.transform.original_height == 37
    assert len(prepared.tiles) > 1
    assert all(tile.rgb_chw.shape == (3, 32, 32) for tile in prepared.tiles)
    assert all(np.isfinite(tile.rgb_chw).all() for tile in prepared.tiles)


def test_small_rgba_raster_is_edge_padded_and_alpha_composited() -> None:
    pixels = np.zeros((5, 9, 4), dtype=np.uint8)
    pixels[..., :3] = (250, 10, 20)
    pixels[..., 3] = 0
    image = Image.fromarray(pixels, mode="RGBA")

    prepared = prepare_raster(
        image,
        tile_size=16,
        overlap=4,
        alpha_background=(7, 8, 9),
    )

    assert prepared.source_rgb_hwc.shape == (5, 9, 3)
    assert np.all(prepared.source_rgb_hwc == np.array([7, 8, 9], dtype=np.uint8))
    assert prepared.transform.padded_width == 16
    assert prepared.transform.padded_height == 16
    assert len(prepared.tiles) == 1


def test_float_grayscale_is_normalized_without_nonfinite_values() -> None:
    pixels = np.array([[0.0, 1.0], [np.nan, np.inf]], dtype=np.float32)
    rgb = raster_to_rgb(Image.fromarray(pixels, mode="F"))

    assert rgb.shape == (2, 2, 3)
    assert rgb.dtype == np.uint8
    assert np.isfinite(rgb).all()


def test_overlapping_predictions_stitch_to_exact_original_aspect() -> None:
    image = Image.fromarray(np.zeros((45, 83, 3), dtype=np.uint8), mode="RGB")
    prepared = prepare_raster(image, tile_size=32, overlap=8)
    predictions = []
    for tile in prepared.tiles:
        x_values = np.arange(tile.x, tile.x + 32, dtype=np.float32)
        predictions.append(np.broadcast_to(x_values[None, :], (32, 32)))

    relief = stitch_relief(predictions, prepared.transform, normalize=False)

    expected = np.broadcast_to(
        np.arange(83, dtype=np.float32)[None, :] + prepared.transform.pad_left,
        (45, 83),
    )
    np.testing.assert_allclose(relief, expected, atol=1e-4)


def test_blank_relief_builds_finite_continuous_mesh_with_complete_uvs() -> None:
    mesh = build_terrain_mesh(np.zeros((7, 11), dtype=np.float32), extent_x=10.0)

    assert mesh.vertices.shape == (77, 3)
    assert mesh.normals.shape == (77, 3)
    assert mesh.uvs.shape == (77, 2)
    assert mesh.faces.shape == (120, 3)
    assert np.isfinite(mesh.vertices).all()
    assert np.isfinite(mesh.normals).all()
    assert np.allclose(mesh.normals, np.array([0.0, 1.0, 0.0], dtype=np.float32))
    assert tuple(mesh.uvs.min(axis=0)) == (0.0, 0.0)
    assert tuple(mesh.uvs.max(axis=0)) == (1.0, 1.0)
    assert mesh.extent_z == pytest.approx(6.0)


def test_mesh_faces_are_valid_and_obj_references_uvs_and_normals(tmp_path) -> None:
    relief = np.arange(20, dtype=np.float32).reshape(4, 5)
    mesh = build_terrain_mesh(relief, extent_x=4.0, vertical_scale=2.0)

    assert mesh.faces.min() == 0
    assert mesh.faces.max() == 19
    assert np.linalg.norm(mesh.normals, axis=1) == pytest.approx(1.0)

    output = write_obj(mesh, tmp_path / "terrain.obj", texture_filename="source.png")
    obj_text = output.read_text(encoding="utf-8")
    mtl_text = output.with_suffix(".mtl").read_text(encoding="utf-8")
    assert "mtllib terrain.mtl" in obj_text
    assert "f 1/1/1" in obj_text
    assert "map_Kd source.png" in mtl_text


def test_invalid_tile_and_mesh_contracts_fail_loudly() -> None:
    image = Image.fromarray(np.zeros((20, 20, 3), dtype=np.uint8), mode="RGB")
    with pytest.raises(ValueError, match="overlap"):
        prepare_raster(image, tile_size=16, overlap=16)
    with pytest.raises(ValueError, match="at least 2x2"):
        build_terrain_mesh(np.zeros((1, 3), dtype=np.float32))
