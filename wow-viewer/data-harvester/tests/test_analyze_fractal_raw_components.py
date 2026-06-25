from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import zarr

_SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from analyze_fractal_raw_components import (  # noqa: E402
    process_map_in_strips,
    resolve_target_maps,
)

from harvester.fractal_canvas import CanvasTileRecord  # noqa: E402


def _record(tile_id: int, tile_x: int, tile_y: int) -> CanvasTileRecord:
    return CanvasTileRecord(
        build="0_5_3_3368",
        map_name="Azeroth",
        tile_id=tile_id,
        tile_x=tile_x,
        tile_y=tile_y,
        has_alpha_256=True,
        has_height_257=False,
        has_normal_xyz=False,
        has_mcly_texture_ids=False,
        has_mcly_layer_mask=False,
    )


def _make_source_zarr(tmp_path: Path, records: list[CanvasTileRecord]) -> zarr.Group:
    root = zarr.open_group(str(tmp_path / "source.zarr"), mode="w")
    root.create_array("alpha_256", shape=(len(records), 256, 256, 4), dtype=np.float32, fill_value=0.0)
    return root


def test_resolve_target_maps_expands_all() -> None:
    assert resolve_target_maps(["all"], ["Azeroth", "Kalimdor"]) == ["Azeroth", "Kalimdor"]


def test_resolve_target_maps_keeps_explicit_names() -> None:
    assert resolve_target_maps(["Azeroth"], ["Azeroth", "Kalimdor"]) == ["Azeroth"]


def test_process_map_in_strips_matches_dense_path(tmp_path: Path) -> None:
    records = [_record(0, 0, 0), _record(1, 1, 0), _record(2, 2, 0)]
    source = _make_source_zarr(tmp_path, records)
    # Component spanning tile 0-1 on layer 0.
    source["alpha_256"][0, 50:100, 200:256, 0] = 1.0
    source["alpha_256"][1, 50:100, 0:56, 0] = 1.0
    # Component spanning tile 1-2 on layer 0.
    source["alpha_256"][1, 150:200, 200:256, 0] = 1.0
    source["alpha_256"][2, 150:200, 0:56, 0] = 1.0

    canvas_dir = tmp_path / "canvas"
    segments_dir = tmp_path / "segments"
    regions = process_map_in_strips(
        source,
        records,
        (0, 1, 2, 3),
        canvas_dir,
        segments_dir,
        threshold=0.5,
        min_area=64,
        min_footprint_px=8,
        max_regions_per_layer=1000,
        strip_tiles=2,
        overlap_alpha_tiles=1,
        no_overlay=True,
    )

    assert len(regions) == 2
    assert all(region.curation_label == "raw_component" for region in regions)
    bboxes = sorted(region.bbox_xywh for region in regions)
    # First component at y=50, second at y=150; widths/heights should match.
    assert bboxes[0][1] == 50
    assert bboxes[1][1] == 150


def test_process_map_in_strips_dedupes_across_overlap(tmp_path: Path) -> None:
    records = [_record(0, 0, 0), _record(1, 1, 0), _record(2, 2, 0)]
    source = _make_source_zarr(tmp_path, records)
    # One component that crosses the strip boundary between tile 1 and 2.
    source["alpha_256"][1, 100:156, 200:256, 0] = 1.0
    source["alpha_256"][2, 100:156, 0:56, 0] = 1.0

    canvas_dir = tmp_path / "canvas"
    segments_dir = tmp_path / "segments"
    regions = process_map_in_strips(
        source,
        records,
        (0, 1, 2, 3),
        canvas_dir,
        segments_dir,
        threshold=0.5,
        min_area=64,
        min_footprint_px=8,
        max_regions_per_layer=1000,
        strip_tiles=2,
        overlap_alpha_tiles=1,
        no_overlay=True,
    )

    assert len(regions) == 1
    assert regions[0].bbox_xywh[0] == 256 + 200
    assert regions[0].bbox_xywh[1] == 100


def test_process_map_in_strips_can_skip_raw_segmentation_for_macro_mode(tmp_path: Path) -> None:
    records = [_record(0, 0, 0), _record(1, 1, 0)]
    source = _make_source_zarr(tmp_path, records)
    source["alpha_256"][0, 50:100, 200:256, 0] = 1.0
    source["alpha_256"][1, 50:100, 0:56, 0] = 1.0

    canvas_dir = tmp_path / "canvas"
    segments_dir = tmp_path / "segments"
    regions = process_map_in_strips(
        source,
        records,
        (0, 1, 2, 3),
        canvas_dir,
        segments_dir,
        threshold=0.5,
        min_area=64,
        min_footprint_px=8,
        max_regions_per_layer=1000,
        strip_tiles=2,
        overlap_alpha_tiles=1,
        no_overlay=True,
        skip_raw_segments=True,
    )

    assert regions == []
    assert (canvas_dir / "canvas.zarr").exists()
    assert (segments_dir / "fractal_regions.jsonl").exists()
