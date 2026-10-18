"""PyTorch Dataset & DataLoader for Clean Minimap / Elevation Pairs (Spec 274).

Loads paired tuples:
- input: real_minimap_rgb (3, 256, 256) float32 in [0, 1]
- target 1: synthetic_hillshade_dxt1 (3, 256, 256) float32 in [0, 1]
- target 2: wdl_lattice (1, 33, 33) float32 in yards
- target 3: height_257 (1, 257, 257) float32 in yards
with synchronous geometric and photometric augmentations.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
import zarr

logger = logging.getLogger(__name__)


class CleanPairDataset(Dataset):
    """PyTorch Dataset reading synchronized real minimap and synthetic terrain pairs."""

    def __init__(
        self,
        store_path: Path | str,
        split: Optional[str] = "train",
        augment: bool = False,
    ):
        self.store_path = Path(store_path)
        if not self.store_path.exists():
            raise FileNotFoundError(f"Clean pair Zarr store not found: {self.store_path}")

        index_file = self.store_path / "index.parquet"
        if not index_file.exists():
            raise FileNotFoundError(f"index.parquet missing from store: {self.store_path}")

        tbl = pq.read_table(index_file)
        rows = tbl.to_pylist()

        if split is not None:
            self.rows = [r for r in rows if str(r.get("split", "")).lower() == split.lower()]
        else:
            self.rows = rows

        self.augment = augment
        self._group: Optional[zarr.Group] = None

    def _get_group(self) -> zarr.Group:
        if self._group is None:
            self._group = zarr.open_group(str(self.store_path), mode="r")
        return self._group

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.rows[idx]
        orig_idx = int(row["index"])
        g = self._get_group()

        # Load raw arrays
        real_rgb = np.asarray(g["real_minimap_rgb"][orig_idx], dtype=np.float32) / 255.0  # (256, 256, 3)
        synth_dxt1 = np.asarray(g["synthetic_hillshade_dxt1"][orig_idx], dtype=np.float32) / 255.0  # (256, 256, 3)
        wdl = np.asarray(g["wdl_lattice_33"][orig_idx], dtype=np.float32)  # (33, 33)
        h257 = np.asarray(g["height_257"][orig_idx], dtype=np.float32)  # (257, 257)

        # Synchronous data augmentation
        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                real_rgb = np.fliplr(real_rgb).copy()
                synth_dxt1 = np.fliplr(synth_dxt1).copy()
                wdl = np.fliplr(wdl).copy()
                h257 = np.fliplr(h257).copy()

            # Random vertical flip
            if np.random.rand() > 0.5:
                real_rgb = np.flipud(real_rgb).copy()
                synth_dxt1 = np.flipud(synth_dxt1).copy()
                wdl = np.flipud(wdl).copy()
                h257 = np.flipud(h257).copy()

            # Input-only subtle photometric jitter (learning exposure invariance)
            if np.random.rand() > 0.5:
                gamma = float(np.random.uniform(0.90, 1.10))
                brightness = float(np.random.uniform(-0.05, 0.05))
                real_rgb = np.clip((real_rgb ** gamma) + brightness, 0.0, 1.0)

        # Transpose images to channels-first (C, H, W)
        real_tensor = torch.from_numpy(real_rgb).permute(2, 0, 1).float()  # (3, 256, 256)
        synth_tensor = torch.from_numpy(synth_dxt1).permute(2, 0, 1).float()  # (3, 256, 256)
        wdl_tensor = torch.from_numpy(wdl).unsqueeze(0).float()  # (1, 33, 33)
        h257_tensor = torch.from_numpy(h257).unsqueeze(0).float()  # (1, 257, 257)

        return {
            "real_minimap": real_tensor,
            "synthetic_hillshade": synth_tensor,
            "wdl_lattice": wdl_tensor,
            "height_257": h257_tensor,
            "map": str(row.get("map", "")),
            "tile_x": int(row.get("tile_x", -1)),
            "tile_y": int(row.get("tile_y", -1)),
            "relief_delta_z": float(row.get("relief_delta_z", 0.0)),
            "split": str(row.get("split", "")),
        }
