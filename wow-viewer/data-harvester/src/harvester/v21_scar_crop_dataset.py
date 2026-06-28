"""Filtered full-tile scar dataset using miner index for candidate tile selection.

Unlike the original V21ScarMaskDataset (which includes all tiles, scar or not),
this dataset filters to only tiles that the miner found scar candidates in.
Training is on full 256×256 tiles — no cropping, no reshaping.

Ground truth is the per-pixel binary scar mask (alpha layer 1-3 threshold)
on the full tile. The miner's role is just tile-level presence filtering.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import zarr
from torch.utils.data import Dataset


class V21ScarFilteredDataset(Dataset):
    """Full-tile scar dataset filtered by miner candidate index."""

    def __init__(
        self,
        dataset_dir: str | Path,
        scar_dir: str | Path,
        builds: list[str] | None = None,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        augment: bool = False,
        alpha_threshold: float = 0.05,
    ) -> None:
        self._dataset_dir_str = str(Path(dataset_dir).resolve())
        scar_dir = Path(scar_dir)
        self.augment = augment and split == "train"
        self.alpha_threshold = alpha_threshold
        self._rng = np.random.RandomState(seed)
        self._stores: dict[str, zarr.Group] = {}

        # Read miner index to find which (build, tile_id) pairs have scars
        scar_zarr = scar_dir / "tile_to_scars.zarr"
        if not scar_zarr.exists():
            raise FileNotFoundError(f"Scar index: {scar_zarr}")
        scar_root = zarr.open_group(str(scar_zarr), mode="r")
        tile_offset = scar_root["tile_offset"][:]
        # tile_offset[tid] != tile_offset[tid+1] => tile has >=1 candidate
        total_tiles = int(scar_root.attrs.get("total_tiles", 0))
        scar_tile_ids = set(
            tid for tid in range(min(total_tiles + 1, len(tile_offset) - 1))
            if int(tile_offset[tid]) != int(tile_offset[tid + 1])
        )

        # Load JSONL for build+metadata per tile
        jsonl_path = scar_dir / "candidates.jsonl"
        build_tile_seen: set[tuple[str, int]] = set()
        if jsonl_path.exists():
            with jsonl_path.open("r", encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    b = str(row.get("build", ""))
                    tid = int(row.get("tile_id", -1))
                    if tid >= 0 and (builds is None or b in builds):
                        build_tile_seen.add((b, tid))

        # Also add tile_ids from zarr index where we don't have JSONL
        if not build_tile_seen:
            for tid in scar_tile_ids:
                if builds:
                    for b in builds:
                        build_tile_seen.add((b, tid))
                else:
                    build_tile_seen.add(("", tid))

        dataset_dir = Path(self._dataset_dir_str)
        build_names = builds or sorted(
            d.stem.replace(".zarr", "") for d in dataset_dir.glob("*.zarr")
        )
        build_stems = set(build_names)

        # Verify tiles exist in V18 stores
        self._entries: list[dict] = []
        for bt, tid in sorted(build_tile_seen):
            if bt not in build_stems:
                continue
            zp = dataset_dir / f"{bt}.zarr"
            if not zp.exists():
                continue
            store = zarr.storage.LocalStore(str(zp), read_only=True)
            root = zarr.open_group(store=store, mode="r")
            n_tiles = root["minimap_rgb"].shape[0] if "minimap_rgb" in root else 0
            if tid < n_tiles:
                self._entries.append({"build": bt, "tile_id": tid})

        if not self._entries:
            raise ValueError("No scar-filtered tiles found")

        n = len(self._entries)
        indices = np.arange(n)
        self._rng.shuffle(indices)
        split_idx = max(1, int(n * (1 - val_fraction)))
        if split == "train":
            self._indices = indices[:split_idx]
        elif split == "val":
            self._indices = indices[split_idx:]
        else:
            self._indices = indices

    def _get_store(self, build: str) -> zarr.Group | None:
        if build in self._stores:
            return self._stores[build]
        zp = Path(self._dataset_dir_str) / f"{build}.zarr"
        if not zp.exists():
            return None
        store = zarr.storage.LocalStore(str(zp), read_only=True)
        root = zarr.open_group(store=store, mode="r")
        self._stores[build] = root
        return root

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        entry = self._entries[int(self._indices[idx])]
        build = entry["build"]
        tile_id = entry["tile_id"]
        root = self._get_store(build)
        if root is None:
            return self._fallback_zeros()

        minimap = torch.from_numpy(root["minimap_rgb"][tile_id].astype(np.float32) / 255.0).permute(2, 0, 1)
        alpha = torch.from_numpy(root["alpha_256"][tile_id].astype(np.float32))
        scar = (alpha[:, :, 1:4].max(dim=2).values >= self.alpha_threshold).float().unsqueeze(0)

        if self.augment:
            flip_h = self._rng.rand() > 0.5
            flip_v = self._rng.rand() > 0.5
            rot = self._rng.randint(0, 4)
            if flip_h:
                minimap = minimap.flip(-1)
                scar = scar.flip(-1)
            if flip_v:
                minimap = minimap.flip(-2)
                scar = scar.flip(-2)
            if rot:
                minimap = torch.rot90(minimap, rot, dims=(1, 2))
                scar = torch.rot90(scar, rot, dims=(1, 2))
            noise = torch.randn_like(minimap) * 0.02
            minimap = (minimap + noise).clamp(0.0, 1.0)
            brightness = 0.85 + self._rng.rand() * 0.30
            minimap = (minimap * brightness).clamp(0.0, 1.0)

        return {
            "input": minimap.float(),
            "minimap": minimap.float(),
            "scar_mask": scar.float(),
            "build": build,
            "tile_id": tile_id,
        }

    def _fallback_zeros(self) -> dict:
        return {
            "input": torch.zeros(3, 256, 256, dtype=torch.float32),
            "minimap": torch.zeros(3, 256, 256, dtype=torch.float32),
            "scar_mask": torch.zeros(1, 256, 256, dtype=torch.float32),
            "build": "",
            "tile_id": -1,
        }
