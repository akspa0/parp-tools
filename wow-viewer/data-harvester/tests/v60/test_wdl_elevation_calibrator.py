"""Unit tests for WDL elevation parser & scale calibrator (Spec 265)."""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

from harvester.v60.wdl_elevation_calibrator import (
    WdlElevationCalibrator,
    WdlElevationParser,
)


def create_synthetic_wdl(tmp_path: Path, tile_x: int = 16, tile_y: int = 33) -> Path:
    """Create a minimal authentic-format WDL file with one MARE chunk."""
    wdl_path = tmp_path / "test.wdl"

    # Header: MVER chunk
    mver = b"REVM" + struct.pack("<II", 4, 0x12)

    # MAOF chunk: 4096 offsets
    maof_header = b"FOAM" + struct.pack("<I", 4096 * 4)
    offsets = [0] * 4096
    tile_offset = len(mver) + len(maof_header) + 4096 * 4
    offsets[tile_x * 64 + tile_y] = tile_offset
    maof_data = struct.pack("<4096I", *offsets)

    # MARE chunk: 289 int16 outer + 256 int16 inner = 545 int16 = 1090 bytes
    mare_header = b"ERAM" + struct.pack("<I", 1090)
    outer_heights = [int(i * 1.5) for i in range(289)]
    inner_heights = [0] * 256
    mare_data = struct.pack("<289h", *outer_heights) + struct.pack("<256h", *inner_heights)

    full_bytes = mver + maof_header + maof_data + mare_header + mare_data
    wdl_path.write_bytes(full_bytes)
    return wdl_path


def test_wdl_parser_synthetic(tmp_path: Path) -> None:
    wdl_file = create_synthetic_wdl(tmp_path, tile_x=10, tile_y=20)
    parser = WdlElevationParser(wdl_file)

    assert parser.has_tile_data(10, 20) is True
    assert parser.has_tile_data(0, 0) is False

    h17 = parser.extract_tile_17(10, 20)
    assert h17 is not None
    assert h17.shape == (17, 17)
    assert h17[0, 0] == 0.0

    h257 = parser.extract_tile_257(10, 20)
    assert h257 is not None
    assert h257.shape == (257, 257)


def test_wdl_parser_authentic_development() -> None:
    cand = Path("test_data/original_development/World/Maps/development/development.wdl")
    if not cand.is_file():
        cand = Path("../test_data/original_development/World/Maps/development/development.wdl")
    if not cand.is_file():
        pytest.skip("Authentic development.wdl not found in test environment.")

    parser = WdlElevationParser(cand)
    assert parser.has_tile_data(16, 33) is True

    h17 = parser.extract_tile_17(16, 33)
    assert h17 is not None
    assert h17.shape == (17, 17)
    assert float(np.min(h17)) == 0.0
    assert float(np.max(h17)) == 427.0

    h257 = parser.extract_tile_257(16, 33)
    assert h257 is not None
    assert h257.shape == (257, 257)


def test_wdl_calibrator_linear_regression() -> None:
    calibrator = WdlElevationCalibrator()

    # Create synthetic WDL heights
    y, x = np.mgrid[0:257, 0:257]
    wdl_h = (x * 1.5 + y * 0.8).astype(np.float32)

    # Neural prediction has same shape but squashed amplitude (scale 0.1, offset 50)
    pred_h = (wdl_h * 0.1 + 50.0).astype(np.float32)

    calibrated, metrics = calibrator.calibrate(pred_h, wdl_h)
    assert metrics["method"] == "linear_regression"
    assert pytest.approx(metrics["slope"], rel=1e-2) == 10.0
    assert pytest.approx(metrics["offset"], abs=2.0) == -500.0
    assert pytest.approx(metrics["calibrated_span_yards"], rel=1e-2) == float(np.ptp(wdl_h))


def test_wdl_calibrator_std_ratio_fallback() -> None:
    calibrator = WdlElevationCalibrator()

    wdl_h = np.random.normal(loc=100.0, scale=30.0, size=(257, 257)).astype(np.float32)
    # Completely uncorrelated prediction with squashed variance
    pred_h = np.random.normal(loc=10.0, scale=3.0, size=(257, 257)).astype(np.float32)

    calibrated, metrics = calibrator.calibrate(pred_h, wdl_h, min_correlation_threshold=0.5)
    assert metrics["method"] == "std_ratio"
    assert pytest.approx(metrics["slope"], rel=1e-1) == 10.0
    assert pytest.approx(np.std(calibrated), rel=1e-1) == np.std(wdl_h)


def test_wdl_calibrator_water_pinning() -> None:
    calibrator = WdlElevationCalibrator()

    wdl_h = np.full((257, 257), 100.0, dtype=np.float32)
    pred_h = np.full((257, 257), 20.0, dtype=np.float32)

    # Bottom half is water
    water = np.zeros((257, 257), dtype=bool)
    water[128:, :] = True

    calibrated, metrics = calibrator.calibrate(pred_h, wdl_h, water_mask=water)
    assert np.all(calibrated[water] == 0.0)
    assert np.all(calibrated[~water] >= 0.0)
