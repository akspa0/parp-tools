"""Parity with WeakSignalDetector.cs, and the partition guarantee (nothing is dropped)."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from harvester.v50.tile_inventory import (
    CLASSIC_COMPRESSION_FACTOR,
    analyze_chunks,
    classify_information,
    classify_severity,
    classify_tile,
    estimate_factor_from_ranges,
    is_weak_signal,
    mcnr_tilted_fraction,
    summarize,
    surviving_height_levels,
    write_inventory,
)


def test_level_count_separates_squeezed_terrain_from_squeezed_nothing() -> None:
    """Amplitude alone cannot tell them apart; two real tiles both measure ~0.5 units of range but
    hold 2 and 27,132 distinct heights respectively. Only the second is recoverable terrain."""
    plateau = np.zeros((64, 64), dtype=np.float32)
    plateau[32:] = np.float32(0.512079)
    detailed = np.linspace(0.0, 0.512079, 64 * 64, dtype=np.float32).reshape(64, 64)

    assert float(np.ptp(plateau)) == float(np.ptp(detailed))  # identical amplitude
    assert surviving_height_levels(plateau) == 2
    assert surviving_height_levels(detailed) > 4000
    assert classify_information(surviving_height_levels(plateau)) == "trace"
    assert classify_information(surviving_height_levels(detailed)) == "rich_terrain"


def test_information_class_bands() -> None:
    assert classify_information(1) == "bit_exact_flat"
    assert classify_information(0) == "bit_exact_flat"
    assert classify_information(8) == "trace"
    assert classify_information(9) == "coarse_terrain"
    assert classify_information(64) == "coarse_terrain"
    assert classify_information(65) == "rich_terrain"
    assert surviving_height_levels(np.full((8, 8), 3.0, dtype=np.float32)) == 1


def test_weak_signal_thresholds_match_the_viewer() -> None:
    # WeakSignalDetector.Analyze: hasTerrainData && isCompressed && nearZeroBand
    assert is_weak_signal(-2.0, 3.0) is True          # compressed, near zero
    assert is_weak_signal(0.0, 0.0) is False          # flat: no terrain data
    assert is_weak_signal(0.0, 400.0) is False        # full-scale relief, not compressed
    assert is_weak_signal(-1200.0, -1198.0) is False  # compressed but far off the zero band


def test_severity_bands_match_the_viewer() -> None:
    assert classify_severity(0.2, True) == "high"
    assert classify_severity(1.0, True) == "medium"
    assert classify_severity(9.0, True) == "low"
    assert classify_severity(0.2, False) == "none"


def test_factor_estimate_scales_toward_the_reference_range() -> None:
    # A 3m-relief tile beside 300m-relief neighbours wants ~100x, not the blanket era constant.
    assert estimate_factor_from_ranges(0.0, 3.0, 0.0, 300.0) == 100.0
    # Reference no larger than observed => no amplification.
    assert estimate_factor_from_ranges(0.0, 300.0, 0.0, 300.0) == 1.0
    # Degenerate inputs must not divide by zero.
    assert estimate_factor_from_ranges(0.0, 0.0, 0.0, 300.0) == 1.0
    assert estimate_factor_from_ranges(0.0, 3.0, 5.0, 5.0) == 1.0
    # An observed range at or below epsilon is unamplifiable, not infinitely amplifiable.
    assert estimate_factor_from_ranges(0.0, 0.001, -8000.0, 500.0) == 1.0
    # Clamped to MaxFactor once the ratio runs away.
    assert estimate_factor_from_ranges(0.0, 0.01, -8000.0, 500.0) == 512.0
    assert CLASSIC_COMPRESSION_FACTOR == 33.334


def test_chunk_analysis_separates_weak_from_blank() -> None:
    height = np.zeros((257, 257), dtype=np.float32)
    # Chunks stride by 16 but span 17, so adjacent chunks SHARE their edge vertices (same as the
    # viewer). Keeping the relief inside 0..15 leaves the shared row/column at 16 flat, so exactly
    # one chunk sees it.
    height[0:16, 0:16] = np.linspace(1.0, 4.0, 16 * 16, dtype=np.float32).reshape(16, 16)
    result = analyze_chunks(height)
    assert result["weak_chunk_count"] == 1
    assert result["blank_chunk_count"] == 255
    assert result["chunk_range_max"] > 3.9

    # Relief written across a shared edge is seen by every chunk touching that edge.
    spill = np.zeros((257, 257), dtype=np.float32)
    spill[0:17, 0:17] = np.linspace(1.0, 4.0, 17 * 17, dtype=np.float32).reshape(17, 17)
    assert analyze_chunks(spill)["weak_chunk_count"] == 4


def test_tilted_fraction_ignores_unwritten_vertices() -> None:
    """Unwritten MCNR vertices are (0,0,0); an unmasked test reads those as 'tilted' and reports
    ~100% on a flat tile. Masking is what makes the measurement mean anything."""
    normals = np.zeros((8, 8, 3), dtype=np.float32)
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4, :] = True
    normals[:4, :, 2] = 1.0  # every REAL vertex is straight up
    assert mcnr_tilted_fraction(normals, mask) == 0.0

    normals[0, 0] = (0.6, 0.0, 0.8)  # one genuinely tilted real vertex
    assert mcnr_tilted_fraction(normals, mask) == 1.0 / 32.0
    assert mcnr_tilted_fraction(normals, np.zeros((8, 8), dtype=bool)) == 0.0


def test_classification_partitions_every_tile() -> None:
    assert classify_tile(has_height=True, has_minimap=True, height_range=300.0, weak=False) == "usable"
    assert classify_tile(has_height=True, has_minimap=False, height_range=300.0, weak=False) == "terrain_no_minimap"
    assert classify_tile(has_height=True, has_minimap=True, height_range=3.0, weak=True) == "weak_signal_with_minimap"
    assert classify_tile(has_height=False, has_minimap=True, height_range=0.0, weak=False) == "white_plate_with_minimap"
    assert classify_tile(has_height=False, has_minimap=False, height_range=0.0, weak=False) == "white_plate"


def test_inventory_writes_csv_and_json_without_dropping_rows(tmp_path: Path) -> None:
    rows = [
        {"tile_key": "Azeroth_01_01", "map": "Azeroth", "tile_x": 1, "tile_y": 1, "row_id": 0,
         "classification": "usable", "has_height_257": True, "has_minimap_rgb": True,
         "height_min": 0.0, "height_max": 300.0, "height_range": 300.0, "is_weak_signal": False,
         "weak_severity": "none", "strong_neighbour_count": 0, "neighbours": {"W": None}},
        {"tile_key": "Azeroth_02_01", "map": "Azeroth", "tile_x": 2, "tile_y": 1, "row_id": 1,
         "classification": "white_plate", "has_height_257": False, "has_minimap_rgb": False,
         "height_min": 0.0, "height_max": 0.0, "height_range": 0.0, "is_weak_signal": False,
         "weak_severity": "none", "strong_neighbour_count": 1, "neighbours": {"W": "Azeroth_01_01"}},
    ]
    summaries = summarize(rows)
    write_inventory(rows, summaries, tmp_path)

    with (tmp_path / "tiles.csv").open(encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    assert [r["tile_key"] for r in csv_rows] == ["Azeroth_01_01", "Azeroth_02_01"]

    # The blank tile is present and labelled, never filtered out.
    assert summaries["white_plates"] == ["Azeroth_02_01"]
    assert summaries["no_minimap"] == ["Azeroth_02_01"]
    assert summaries["by_classification"] == {"usable": 1, "white_plate": 1}
    assert json.loads((tmp_path / "tiles.json").read_text(encoding="utf-8"))["tiles"][1]["neighbours"]["W"] == "Azeroth_01_01"


def test_era_band_can_suppress_every_detection_so_both_tests_are_recorded() -> None:
    """4.0.0 Kalimdor sits at |Z| ~440-520; the alpha-calibrated |Z|<50 band rejected all 71 of its
    compressed tiles, reporting zero weak-signal tiles on a map that plainly has them."""
    from harvester.v50.tile_inventory import is_compressed_range

    # A compressed tile at Cataclysm altitude.
    lo, hi = -515.0, -505.0
    assert is_compressed_range(lo, hi) is True
    assert is_weak_signal(lo, hi) is False                       # alpha default: suppressed
    assert is_weak_signal(lo, hi, near_zero_band=600.0) is True  # era-appropriate band: found
    assert is_weak_signal(lo, hi, near_zero_band=float("inf")) is True

    # The band still does its job at alpha altitudes: full-scale relief is never "weak".
    assert is_compressed_range(0.0, 400.0) is False
    assert is_weak_signal(0.0, 400.0, near_zero_band=float("inf")) is False
