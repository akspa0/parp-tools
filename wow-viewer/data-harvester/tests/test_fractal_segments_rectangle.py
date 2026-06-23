from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

_SRC_DIR = Path(__file__).resolve().parent.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_segments import (  # noqa: E402
    detect_rectangle_pages,
)


def _canvas(tmp_path: Path, shape: tuple[int, int, int]) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "canvas.zarr"), mode="w")
    root.create_array("alpha_256", shape=shape, dtype=np.float32, fill_value=0.0)
    root.create_array("tile_id_256", shape=(shape[0], shape[1]), dtype=np.int32, fill_value=0)
    root.create_array("alpha_layer_indices", data=np.arange(shape[2], dtype=np.int32))
    root.attrs["layout"] = {"build": "0_5_3_3368", "map_name": "Azeroth"}
    return root


def test_detect_rectangle_pages_finds_solid_rectangle(tmp_path: Path) -> None:
    canvas = _canvas(tmp_path, (64, 64, 1))
    canvas["alpha_256"][10:30, 5:25, 0] = 1.0
    regions = detect_rectangle_pages(canvas, threshold=0.5, min_area=100, min_extent=0.85)
    assert len(regions) == 1
    region = regions[0]
    assert region.curation_label == "rectangle_page"
    assert region.bbox_xywh == (5, 10, 20, 20)
    assert region.area == 400


def test_detect_rectangle_pages_ignores_low_extent_shape(tmp_path: Path) -> None:
    canvas = _canvas(tmp_path, (64, 64, 1))
    # A diagonal line: bbox area much larger than filled area.
    for i in range(20):
        canvas["alpha_256"][i, i, 0] = 1.0
    regions = detect_rectangle_pages(canvas, threshold=0.5, min_area=10, min_extent=0.85)
    assert len(regions) == 0


def test_detect_rectangle_pages_respects_aspect_ratio(tmp_path: Path) -> None:
    canvas = _canvas(tmp_path, (64, 256, 1))
    canvas["alpha_256"][0:8, 0:128, 0] = 1.0
    regions = detect_rectangle_pages(canvas, threshold=0.5, min_area=32, min_extent=0.85, max_aspect_ratio=20.0)
    assert len(regions) == 1
    regions = detect_rectangle_pages(canvas, threshold=0.5, min_area=32, min_extent=0.85, max_aspect_ratio=8.0)
    assert len(regions) == 0


def test_detect_rectangle_pages_limits_output(tmp_path: Path) -> None:
    canvas = _canvas(tmp_path, (128, 128, 1))
    # Two rectangles, only keep top one because of max_regions_per_layer=1.
    canvas["alpha_256"][0:20, 0:20, 0] = 1.0
    canvas["alpha_256"][40:50, 40:50, 0] = 1.0
    regions = detect_rectangle_pages(canvas, threshold=0.5, min_area=50, min_extent=0.85, max_regions_per_layer=1)
    assert len(regions) == 1
