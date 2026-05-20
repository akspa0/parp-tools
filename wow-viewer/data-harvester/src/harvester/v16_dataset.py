"""V16Dataset — PyTorch Dataset reading from consolidated Zarr stores.

Reads from V16 Zarr stores produced by build_v16_dataset.py.
Each build is a single .zarr directory with flat arrays indexed by tile_id.
The Parquet index provides map/tile coordinates and has_* flags.

Supports geometric augmentation (hflip/vflip/rot90) with correct normal transforms.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
import zarr.storage
from torch.utils.data import Dataset


class V16Dataset(Dataset):
    def __init__(
        self,
        dataset_dir: str | Path,
        builds: list[str] | None = None,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        augment: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.augment = augment and split == "train"
        self._rng = np.random.RandomState(seed)

        self._stores: dict[str, zarr.Group] = {}
        self._index_entries: list[dict] = []

        build_dirs = builds or [d.stem.replace(".zarr", "") for d in sorted(self.dataset_dir.glob("*.zarr"))]

        for build in build_dirs:
            zarr_path = self.dataset_dir / f"{build}.zarr"
            if not zarr_path.exists():
                continue
            store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
            root = zarr.open_group(store=store, mode="r")
            self._stores[build] = root

            index_path = zarr_path / "index.parquet"
            if not index_path.exists():
                continue
            table = pq.read_table(str(index_path))
            for i in range(table.num_rows):
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                row["_build"] = build
                self._index_entries.append(row)

        if not self._index_entries:
            raise ValueError(f"No index entries found in {self.dataset_dir}")

        n_val = int(len(self._index_entries) * val_fraction)
        rng = np.random.RandomState(seed)
        indices = rng.permutation(len(self._index_entries))
        if split == "val":
            self._indices = sorted(indices[:n_val])
        else:
            self._indices = sorted(indices[n_val:])

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        entry = self._index_entries[self._indices[idx]]
        build = entry["_build"]
        source_build = str(entry.get("build") or build)
        tile_id = entry["tile_id"]
        root = self._stores[build]

        minimap = root["minimap_rgb"][tile_id].astype(np.float32) / 255.0
        height_raw = root["height_257"][tile_id].astype(np.float32)
        h_mean = float(entry["height_mean"])
        h_std = float(entry["height_std"])
        height = (height_raw - h_mean) / (h_std + 1e-8)

        has_normals = bool(entry.get("has_normal_xyz", False))
        has_alpha = bool(entry.get("has_alpha_256", False))
        has_holes = bool(entry.get("has_holes_16", False))
        has_liquid = bool(entry.get("has_liquid_mask", False))
        has_instance = bool(entry.get("has_object_instance_mask", False))
        has_mcly = bool(entry.get("has_mcly_texture_ids", False))

        if has_normals:
            normals = root["normal_xyz"][tile_id].astype(np.float32)
            normal_mask = root["normal_mask"][tile_id].astype(np.float32)
        else:
            normals = np.zeros((257, 257, 3), dtype=np.float32)
            normal_mask = np.zeros((257, 257), dtype=np.float32)

        if has_alpha:
            alpha = root["alpha_256"][tile_id].astype(np.float32)
            alpha = np.clip(alpha, 0.0, 1.0)
        else:
            alpha = np.zeros((256, 256, 4), dtype=np.float32)

        if has_holes:
            holes = root["holes_16"][tile_id].astype(np.float32)
        else:
            holes = np.zeros((16, 16), dtype=np.float32)

        if has_liquid:
            liquid_mask = root["liquid_mask"][tile_id].astype(np.float32)
            liquid_mask = np.clip(liquid_mask, 0.0, 1.0)
            liquid_height = root["liquid_height"][tile_id].astype(np.float32)
        else:
            liquid_mask = np.zeros((256, 256), dtype=np.float32)
            liquid_height = np.zeros((256, 256), dtype=np.float32)

        obj_mask = root["object_mask"][tile_id].astype(np.float32)
        # Use filtered mask for terrain loss weighting (excludes trees, includes WMOs)
        if "object_filtered_mask" in root:
            weight = 1.0 - root["object_filtered_mask"][tile_id].astype(np.float32)
        else:
            weight = 1.0 - obj_mask

        if has_instance and "object_instance_mask" in root:
            instance_mask = root["object_instance_mask"][tile_id].astype(np.int64)
        else:
            instance_mask = np.zeros((257, 257), dtype=np.int64)

        if has_mcly and "mcly_texture_ids" in root and "mcly_layer_mask" in root:
            mcly_ids = root["mcly_texture_ids"][tile_id].astype(np.int64)
            mcly_ids = np.clip(mcly_ids, 0, 15)
            mcly_mask = root["mcly_layer_mask"][tile_id].astype(np.float32)
        else:
            mcly_ids = np.zeros((16, 16, 4), dtype=np.int64)
            mcly_mask = np.zeros((16, 16, 4), dtype=np.float32)

        if self.augment:
            xform = self._rng.randint(0, 8)
            if xform & 1:
                minimap = minimap[:, ::-1]
                height = height[:, ::-1]
                normals = normals[:, ::-1]
                normals[..., 0] = -normals[..., 0]
                normal_mask = normal_mask[:, ::-1]
                alpha = alpha[:, ::-1]
                holes = holes[:, ::-1]
                liquid_mask = liquid_mask[:, ::-1]
                liquid_height = liquid_height[:, ::-1]
                weight = weight[:, ::-1]
                instance_mask = instance_mask[:, ::-1]
                mcly_ids = mcly_ids[:, ::-1]
                mcly_mask = mcly_mask[:, ::-1]
            if xform & 2:
                minimap = minimap[::-1]
                height = height[::-1]
                normals = normals[::-1]
                normals[..., 1] = -normals[..., 1]
                normal_mask = normal_mask[::-1]
                alpha = alpha[::-1]
                holes = holes[::-1]
                liquid_mask = liquid_mask[::-1]
                liquid_height = liquid_height[::-1]
                weight = weight[::-1]
                instance_mask = instance_mask[::-1]
                mcly_ids = mcly_ids[::-1]
                mcly_mask = mcly_mask[::-1]
            if xform & 4:
                minimap = np.rot90(minimap, k=1)
                height = np.rot90(height, k=1)
                normals = np.rot90(normals, k=1)
                old_nx = normals[..., 0].copy()
                normals[..., 0] = normals[..., 1]
                normals[..., 1] = -old_nx
                normal_mask = np.rot90(normal_mask, k=1)
                alpha = np.rot90(alpha, k=1)
                holes = np.rot90(holes, k=1)
                liquid_mask = np.rot90(liquid_mask, k=1)
                liquid_height = np.rot90(liquid_height, k=1)
                weight = np.rot90(weight, k=1)
                instance_mask = np.rot90(instance_mask, k=1)
                mcly_ids = np.rot90(mcly_ids, k=1)
                mcly_mask = np.rot90(mcly_mask, k=1)

        return {
            "input": torch.from_numpy(minimap.copy()).permute(2, 0, 1),
            "height": torch.from_numpy(height.copy()).unsqueeze(0),
            "normals": torch.from_numpy(normals.copy()).permute(2, 0, 1),
            "normal_mask": torch.from_numpy(normal_mask.copy()).unsqueeze(0),
            "alpha": torch.from_numpy(alpha.copy()).permute(2, 0, 1),
            "holes": torch.from_numpy(holes.copy()).unsqueeze(0),
            "liquid": torch.from_numpy(liquid_mask.copy()).unsqueeze(0),
            "liquid_height": torch.from_numpy(liquid_height.copy()).unsqueeze(0),
            "weight": torch.from_numpy(weight.copy()).unsqueeze(0),
            "instance_mask": torch.from_numpy(instance_mask.copy()).unsqueeze(0).long(),
            "mcly_ids": torch.from_numpy(mcly_ids.copy()).long(),
            "mcly_mask": torch.from_numpy(mcly_mask.copy()),
            "has_normals": has_normals,
            "has_alpha": has_alpha,
            "has_holes": has_holes,
            "has_liquid": has_liquid,
            "has_instance": has_instance,
            "has_mcly": has_mcly,
            "meta_build": source_build,
            "meta_store": str(build),
            "meta_map": str(entry.get("map", "")),
            "meta_tile_id": int(tile_id),
            "meta_tile_x": int(entry.get("tile_x") if entry.get("tile_x") is not None else -1),
            "meta_tile_y": int(entry.get("tile_y") if entry.get("tile_y") is not None else -1),
        }
