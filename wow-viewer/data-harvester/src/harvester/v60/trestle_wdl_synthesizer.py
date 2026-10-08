"""WDL Trestle Macro-Elevation Synthesizer (Spec 266).

Synthesizes complete 17x17 WDL macro-elevation lattices across the development map
by transferring authentic Wrath 3.0.1.8303 Northrend WDL lattices based on visual
prototype match confidence (similarity >= 0.85).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import ndimage


class TrestleWdlSynthesizer:
    """Synthesizes and queries coarse WDL elevation lattices for development tiles."""

    def __init__(
        self,
        matches_path: Path | str,
        northrend_wdl_path: Path | str,
        min_similarity: float = 0.85,
    ):
        self.matches_path = Path(matches_path)
        self.northrend_wdl_path = Path(northrend_wdl_path)
        self.min_similarity = float(min_similarity)

        # Storage for 64x64 grid
        # grid_17[x, y] -> shape (17, 17) float32 array or None
        self._grid_17: Dict[Tuple[int, int], np.ndarray] = {}
        self._metadata: Dict[Tuple[int, int], Dict[str, any]] = {}

        self._build_synthesis()

    def _build_synthesis(self) -> None:
        if not self.matches_path.is_file():
            raise FileNotFoundError(f"Matches ledger not found: {self.matches_path}")
        if not self.northrend_wdl_path.is_file():
            raise FileNotFoundError(f"Northrend WDL npz not found: {self.northrend_wdl_path}")

        # Load Northrend WDL data
        nr_npz = np.load(self.northrend_wdl_path)
        nr_coords = nr_npz["tile_xy"]  # (736, 2)
        nr_outer = nr_npz["outer"]      # (736, 17, 17)

        nr_lut: Dict[Tuple[int, int], np.ndarray] = {}
        for idx in range(len(nr_coords)):
            x, y = int(nr_coords[idx, 0]), int(nr_coords[idx, 1])
            nr_lut[(x, y)] = nr_outer[idx]

        # Load match ledger
        with open(self.matches_path, "r", encoding="utf-8") as f:
            matches_data = json.load(f)

        transferred_count = 0
        for entry in matches_data:
            dev_x = int(entry["dev_x"])
            dev_y = int(entry["dev_y"])
            sim = float(entry["similarity"])
            nr_x = int(entry["nr_x"])
            nr_y = int(entry["nr_y"])

            if sim >= self.min_similarity and (nr_x, nr_y) in nr_lut:
                lattice_17 = nr_lut[(nr_x, nr_y)].copy()
                self._grid_17[(dev_x, dev_y)] = lattice_17
                self._metadata[(dev_x, dev_y)] = {
                    "source": "northrend_lineage",
                    "similarity": sim,
                    "origin_tile": entry.get("northrend_tile", f"Northrend_{nr_x}_{nr_y}"),
                    "origin_xy": (nr_x, nr_y),
                    "vertical_span": float(np.max(lattice_17) - np.min(lattice_17)),
                }
                transferred_count += 1

        print(
            f"[TrestleWdlSynthesizer] Transferred {transferred_count} WDL lattices "
            f"(sim >= {self.min_similarity:.2f}) from Northrend 3.0.1 to development map."
        )

    def has_tile(self, tile_x: int, tile_y: int) -> bool:
        """Return True if synthesized WDL lattice exists for (tile_x, tile_y)."""
        return (tile_x, tile_y) in self._grid_17

    def get_trestle_17(self, tile_x: int, tile_y: int) -> Optional[np.ndarray]:
        """Get coarse (17, 17) elevation lattice in yards."""
        lattice = self._grid_17.get((tile_x, tile_y))
        if lattice is not None:
            return lattice.copy()
        return None

    def get_trestle_257(
        self,
        tile_x: int,
        tile_y: int,
        order: int = 3,
    ) -> Optional[np.ndarray]:
        """Get (257, 257) bicubic upsampled elevation trestle in yards."""
        h17 = self.get_trestle_17(tile_x, tile_y)
        if h17 is None:
            return None
        # Upsample 17x17 to 257x257
        # Zoom factor = (257 - 1) / (17 - 1) = 256 / 16 = 16.0
        # Zooming 17 to 257 with grid endpoints matching:
        h257 = ndimage.zoom(h17, (257.0 / 17.0, 257.0 / 17.0), order=order)
        # Ensure exact shape
        if h257.shape != (257, 257):
            h257 = h257[:257, :257]
        return h257.astype(np.float32)

    def get_trestle_256(
        self,
        tile_x: int,
        tile_y: int,
        order: int = 3,
    ) -> Optional[np.ndarray]:
        """Get (256, 256) bicubic upsampled elevation trestle in yards."""
        h17 = self.get_trestle_17(tile_x, tile_y)
        if h17 is None:
            return None
        h256 = ndimage.zoom(h17, (256.0 / 17.0, 256.0 / 17.0), order=order)
        if h256.shape != (256, 256):
            h256 = h256[:256, :256]
        return h256.astype(np.float32)

    def get_metadata(self, tile_x: int, tile_y: int) -> Optional[Dict[str, any]]:
        return self._metadata.get((tile_x, tile_y))

    def save_npz(self, output_path: Path | str) -> Path:
        """Export all synthesized lattices to a consolidated NPZ archive."""
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)

        keys = sorted(self._grid_17.keys())
        tile_xy = np.array(keys, dtype=np.int32)
        outer = np.stack([self._grid_17[k] for k in keys], axis=0).astype(np.float32)

        similarities = np.array([self._metadata[k]["similarity"] for k in keys], dtype=np.float32)
        origin_tiles = np.array([self._metadata[k]["origin_tile"] for k in keys], dtype=str)
        sources = np.array([self._metadata[k]["source"] for k in keys], dtype=str)

        np.savez_compressed(
            out,
            tile_xy=tile_xy,
            outer=outer,
            similarity=similarities,
            origin_tile=origin_tiles,
            source=sources,
            version=np.array([1], dtype=np.int32),
        )
        print(f"[TrestleWdlSynthesizer] Saved {len(keys)} tiles to {out}")
        return out

    @classmethod
    def load_npz(cls, npz_path: Path | str) -> TrestleWdlSynthesizer:
        """Load pre-synthesized NPZ archive directly."""
        p = Path(npz_path)
        if not p.is_file():
            raise FileNotFoundError(f"Synthesized NPZ not found: {p}")

        instance = cls.__new__(cls)
        instance.matches_path = p
        instance.northrend_wdl_path = p
        instance.min_similarity = 0.0
        instance._grid_17 = {}
        instance._metadata = {}

        data = np.load(p)
        tile_xy = data["tile_xy"]
        outer = data["outer"]
        sims = data.get("similarity", np.ones(len(tile_xy), dtype=np.float32))
        origins = data.get("origin_tile", np.array([""] * len(tile_xy)))
        sources = data.get("source", np.array(["northrend_lineage"] * len(tile_xy)))

        for i in range(len(tile_xy)):
            tx, ty = int(tile_xy[i, 0]), int(tile_xy[i, 1])
            instance._grid_17[(tx, ty)] = outer[i]
            instance._metadata[(tx, ty)] = {
                "source": str(sources[i]),
                "similarity": float(sims[i]),
                "origin_tile": str(origins[i]),
                "vertical_span": float(np.max(outer[i]) - np.min(outer[i])),
            }

        return instance
