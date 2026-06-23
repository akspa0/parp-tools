from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.fractal_raw_analysis import (  # noqa: E402
    build_pattern_catalog,
    raw_component_pattern_id,
)


def test_raw_component_pattern_id_dedupes_identical_binary_crops() -> None:
    a = np.zeros((8, 8), dtype=np.float32)
    b = np.zeros((8, 8), dtype=np.float32)
    a[2:6, 3:5] = 0.8
    b[2:6, 3:5] = 1.0

    assert raw_component_pattern_id(a) == raw_component_pattern_id(b)


def test_pattern_catalog_groups_cross_build_members() -> None:
    rows = [
        _region("a", "pat_same", "0_5_3_3368"),
        _region("b", "pat_same", "3_3_5_12340"),
        _region("c", "pat_other", "3_3_5_12340"),
    ]

    catalog = build_pattern_catalog(rows)  # type: ignore[arg-type]

    assert catalog[0]["pattern_id"] == "pat_same"
    assert catalog[0]["member_count"] == 2
    assert catalog[0]["build_count"] == 2


def _region(region_id: str, pattern_id: str, build: str) -> object:
    from harvester.fractal_raw_analysis import RawComponentFingerprint

    return RawComponentFingerprint(
        region_id=region_id,
        pattern_id=pattern_id,
        build=build,
        map_name="Azeroth",
        layer_idx=1,
        layer_slot=1,
        bbox_xywh=(0, 0, 8, 8),
        area=16,
        crop_w=8,
        crop_h=8,
        alpha_mean=0.5,
        alpha_max=1.0,
        tile_coverage_count=1,
        tile_coverage=[{"tile_id": 1, "pixel_count": 16}],
        mcly_texture_ids=[7],
        mcly_active_layers=[1],
    )
