from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_canvas import (  # noqa: E402
    CanvasTileRecord,
    alpha_origin,
    alpha_pixel_to_canvas,
    build_canvas_layout,
    compact_tile_limit,
    height_origin,
    height_vertex_to_canvas,
    mcly_cell_to_canvas,
    mcly_origin,
)


def _record(tile_id: int, tile_x: int, tile_y: int) -> CanvasTileRecord:
    return CanvasTileRecord(
        build="0_5_3_3368",
        map_name="Azeroth",
        tile_id=tile_id,
        tile_x=tile_x,
        tile_y=tile_y,
        has_alpha_256=True,
        has_height_257=True,
        has_normal_xyz=True,
        has_mcly_texture_ids=True,
        has_mcly_layer_mask=True,
    )


def test_canvas_layout_uses_tile_extents() -> None:
    layout = build_canvas_layout([_record(1, 30, 40), _record(2, 31, 41)])

    assert layout.min_tile_x == 30
    assert layout.min_tile_y == 40
    assert layout.tile_count_x == 2
    assert layout.tile_count_y == 2
    assert layout.alpha_shape == (512, 512)
    assert layout.height_shape == (513, 513)
    assert layout.mcly_shape == (32, 32)


def test_coordinate_transforms_align_tile_seams() -> None:
    left = _record(1, 30, 40)
    right = _record(2, 31, 40)
    layout = build_canvas_layout([left, right])

    assert alpha_origin(left, layout) == (0, 0)
    assert alpha_origin(right, layout) == (256, 0)
    assert alpha_pixel_to_canvas(right, layout, 0, 12) == (256, 12)

    assert height_origin(left, layout) == (0, 0)
    assert height_origin(right, layout) == (256, 0)
    assert height_vertex_to_canvas(left, layout, 256, 7) == height_vertex_to_canvas(right, layout, 0, 7)

    assert mcly_origin(left, layout) == (0, 0)
    assert mcly_origin(right, layout) == (16, 0)
    assert mcly_cell_to_canvas(right, layout, 0, 3) == (16, 3)


def test_canvas_shapes_are_array_compatible() -> None:
    layout = build_canvas_layout([_record(1, 30, 40), _record(2, 31, 41)])

    alpha = np.zeros((*layout.alpha_shape, 4), dtype=np.float32)
    height = np.zeros(layout.height_shape, dtype=np.float32)
    mcly = np.zeros((*layout.mcly_shape, 4), dtype=np.int32)

    assert alpha.shape == (512, 512, 4)
    assert height.shape == (513, 513)
    assert mcly.shape == (32, 32, 4)


def test_compact_tile_limit_prefers_spatially_close_row_window() -> None:
    records = [
        _record(1, 10, 1),
        _record(2, 50, 1),
        _record(3, 51, 1),
        _record(4, 52, 1),
        _record(5, 53, 1),
        _record(6, 0, 2),
        _record(7, 100, 2),
        _record(8, 200, 2),
        _record(9, 300, 2),
    ]

    selected = compact_tile_limit(records, 4)

    assert [(record.tile_x, record.tile_y) for record in selected] == [
        (50, 1),
        (51, 1),
        (52, 1),
        (53, 1),
    ]
