"""WDL (World Detail Level) Macro-Elevation Parser & Scale Calibrator (Spec 265).

Extracts authentic coarse 17x17 elevation lattices from Blizzard WDL files
and calibrates high-frequency neural elevation predictions to real world yards,
eliminating regression-to-the-mean amplitude compression.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from scipy import ndimage


class WdlElevationParser:
    """Parser for Blizzard WDL (World Detail Level) coarse terrain lattices."""

    FOURCC_MVER = 0x4D564552  # 'MVER'
    FOURCC_MAOF = 0x4D414F46  # 'MAOF'
    FOURCC_MARE = 0x4D415245  # 'MARE'

    def __init__(self, wdl_path: Path | str):
        self.wdl_path = Path(wdl_path)
        self._data: Optional[bytes] = None
        self._offsets: Optional[np.ndarray] = None
        self._version: int = 0
        self._load()

    def _load(self) -> None:
        if not self.wdl_path.is_file():
            raise FileNotFoundError(f"WDL file not found: {self.wdl_path}")

        data = self.wdl_path.read_bytes()
        self._data = data

        if len(data) < 20:
            raise ValueError(f"WDL file too small ({len(data)} bytes): {self.wdl_path}")

        # Locate MAOF chunk (4096 uint32 offsets for 64x64 grid)
        # Scan for 'FOAM' (little-endian 'MAOF')
        maof_pos = data.find(b"FOAM")
        if maof_pos == -1:
            raise ValueError(f"MAOF chunk not found in WDL: {self.wdl_path}")

        maof_size = struct.unpack_from("<I", data, maof_pos + 4)[0]
        if maof_size < 4096 * 4 or maof_pos + 8 + 4096 * 4 > len(data):
            raise ValueError(f"Invalid MAOF size {maof_size} in WDL: {self.wdl_path}")

        offsets = struct.unpack_from("<4096I", data, maof_pos + 8)
        self._offsets = np.array(offsets, dtype=np.uint32)

    def has_tile_data(self, tile_x: int, tile_y: int) -> bool:
        """Check whether WDL contains coarse elevation for tile (x, y)."""
        if self._offsets is None or not (0 <= tile_x < 64 and 0 <= tile_y < 64):
            return False
        # WDL MAOF is row-major: row * 64 + col = tile_y * 64 + tile_x
        idx = tile_y * 64 + tile_x
        return bool(self._offsets[idx] > 0)

    def extract_tile_17(self, tile_x: int, tile_y: int) -> Optional[np.ndarray]:
        """Extract coarse 17x17 outer elevation lattice in yards for tile (x, y)."""
        if not self.has_tile_data(tile_x, tile_y) or self._data is None or self._offsets is None:
            return None

        # WDL MAOF is row-major: row * 64 + col = tile_y * 64 + tile_x
        idx = tile_y * 64 + tile_x
        offset = int(self._offsets[idx])
        if offset + 8 + 289 * 2 > len(self._data):
            return None

        magic = self._data[offset : offset + 4]
        if magic != b"ERAM":  # Little-endian 'MARE'
            return None

        # Outer lattice: 289 int16 values = 17x17 grid
        raw_heights = struct.unpack_from("<289h", self._data, offset + 8)
        h17 = np.array(raw_heights, dtype=np.float32).reshape(17, 17)
        return h17

    def extract_tile_257(self, tile_x: int, tile_y: int, order: int = 1) -> Optional[np.ndarray]:
        """Extract and upsample coarse WDL lattice to (257, 257) matching ADT vertex dimensions."""
        h17 = self.extract_tile_17(tile_x, tile_y)
        if h17 is None:
            return None

        # Resample 17x17 to 257x257 (16 chunks x 16 units = 256 cells + 1 border vertex)
        h257 = ndimage.zoom(h17, (257.0 / 17.0, 257.0 / 17.0), order=order)
        return h257.astype(np.float32)


class WdlElevationCalibrator:
    """Calibrates fine neural elevation predictions against coarse WDL terrain envelopes."""

    @staticmethod
    def find_wdl_for_tile(
        map_dir: Path | str,
        map_name: Optional[str] = None,
    ) -> Optional[Path]:
        """Find matching .wdl file in map directory or parent Maps directory."""
        dir_path = Path(map_dir)
        if map_name:
            cand = dir_path / f"{map_name}.wdl"
            if cand.is_file():
                return cand
            cand_parent = dir_path.parent / map_name / f"{map_name}.wdl"
            if cand_parent.is_file():
                return cand_parent

        # Search for any .wdl in directory
        for p in dir_path.glob("*.wdl"):
            if p.is_file():
                return p
        if dir_path.parent.is_dir():
            for p in dir_path.parent.glob("**/*.wdl"):
                if p.is_file():
                    return p
        return None

    def calibrate(
        self,
        pred_h: np.ndarray,
        wdl_h: np.ndarray,
        water_mask: Optional[np.ndarray] = None,
        min_correlation_threshold: float = 0.35,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Calibrate neural elevation using coarse WDL ground-truth envelope.

        Parameters
        ----------
        pred_h : np.ndarray
            Predicted elevation lattice, shape (257, 257).
        wdl_h : np.ndarray
            Upsampled coarse WDL elevation lattice, shape (257, 257).
        water_mask : Optional[np.ndarray]
            Boolean mask of water pixels to exclude from regression and clamp.
        min_correlation_threshold : float
            Threshold below which variance scaling is preferred over least squares.

        Returns
        -------
        Tuple[np.ndarray, Dict[str, Any]]
            Calibrated elevation in real yards and calibration metrics dictionary.
        """
        assert pred_h.shape == (257, 257), f"Expected (257, 257), got {pred_h.shape}"
        assert wdl_h.shape == (257, 257), f"Expected (257, 257), got {wdl_h.shape}"

        valid_mask = np.ones((257, 257), dtype=bool)
        if water_mask is not None:
            valid_mask = valid_mask & (~water_mask)

        p_flat = pred_h[valid_mask].astype(np.float64)
        w_flat = wdl_h[valid_mask].astype(np.float64)

        if len(p_flat) < 100:
            p_flat = pred_h.flatten().astype(np.float64)
            w_flat = wdl_h.flatten().astype(np.float64)

        # Compute Pearson correlation
        p_std = float(np.std(p_flat))
        w_std = float(np.std(w_flat))

        if p_std > 1e-4 and w_std > 1e-4:
            r = float(np.corrcoef(p_flat, w_flat)[0, 1])
            if np.isnan(r):
                r = 0.0
        else:
            r = 0.0

        raw_span = float(np.ptp(pred_h))
        wdl_span = float(np.ptp(wdl_h))

        # Determine scale factor (slope a) and offset b
        if r >= min_correlation_threshold and p_std > 1e-4:
            # Linear least squares regression
            p = np.polyfit(p_flat, w_flat, 1)
            slope_a = float(p[0])
            offset_b = float(p[1])
            method = "linear_regression"
        else:
            # Standard deviation / total variation ratio matching
            slope_a = w_std / max(1e-4, p_std)
            offset_b = float(np.mean(w_flat) - slope_a * np.mean(p_flat))
            method = "std_ratio"

        # Apply calibration
        calibrated = (slope_a * pred_h + offset_b).astype(np.float32)

        # Sea-level / water pinning if water is present
        if water_mask is not None and np.any(water_mask):
            # Pin lowest land point above sea level (0.0 yds)
            land_mask = ~water_mask
            if np.any(land_mask):
                p1 = float(np.percentile(calibrated[land_mask], 1.0))
                if p1 < 0.0:
                    calibrated[land_mask] -= p1
                calibrated[land_mask] = np.maximum(calibrated[land_mask], 0.0)
            calibrated[water_mask] = 0.0

        calibrated_span = float(np.ptp(calibrated))

        metrics = {
            "method": method,
            "slope": slope_a,
            "offset": offset_b,
            "pearson_r_wdl": r,
            "raw_span_yards": raw_span,
            "calibrated_span_yards": calibrated_span,
            "wdl_span_yards": wdl_span,
        }

        return calibrated, metrics
