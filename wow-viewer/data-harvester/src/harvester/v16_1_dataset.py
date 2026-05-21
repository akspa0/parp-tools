"""V16.1 dataset helpers built on the V16 Zarr corpus contract.

This keeps the V16 storage format as the truth surface while exposing the extra
signals needed by the split-and-link V16.1 model family.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
import zarr.storage
from torch.utils.data import Dataset


def _build_split_indices(n_items: int, split: str, val_fraction: float, seed: int) -> list[int]:
    n_val = int(n_items * val_fraction)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_items)
    if split == "val":
        return sorted(indices[:n_val].tolist())
    return sorted(indices[n_val:].tolist())


def _flags_to_liquid_type(flags_16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert MCNK flags into a coarse liquid-type grid.

    Classes:
      0 none
      1 water
      2 ocean
      3 magma
      4 slime
    """

    flags = flags_16.astype(np.int32, copy=False)
    out = np.zeros(flags.shape, dtype=np.int64)
    valid = ((flags & 0x3C) != 0).astype(np.float32)
    out[(flags & 0x04) != 0] = 1
    out[(flags & 0x08) != 0] = 2
    out[(flags & 0x10) != 0] = 3
    out[(flags & 0x20) != 0] = 4
    return out, valid


def _crop_257_to_256(x: np.ndarray) -> np.ndarray:
    return x[:256, :256]


def _downsample_256_to_16(x: np.ndarray) -> np.ndarray:
    arr = x[:256, :256]
    reshaped = arr.reshape(16, 16, 16, 16)
    return reshaped.mean(axis=(1, 3)).astype(np.float32, copy=False)


class V161Dataset(Dataset):
    """Read V16 Zarr stores and expose richer signals for V16.1 trainers."""

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

        self._indices = _build_split_indices(len(self._index_entries), split=split, val_fraction=val_fraction, seed=seed)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | bool | int | str]:
        entry = self._index_entries[self._indices[idx]]
        build = entry["_build"]
        root = self._stores[build]
        tile_id = int(entry["tile_id"])

        minimap = root["minimap_rgb"][tile_id].astype(np.float32) / 255.0
        height_raw = root["height_257"][tile_id].astype(np.float32)
        h_mean = float(entry["height_mean"])
        h_std = float(entry["height_std"]) + 1e-8
        height_norm = (height_raw - h_mean) / h_std

        normals = root["normal_xyz"][tile_id].astype(np.float32) if bool(entry.get("has_normal_xyz", False)) else np.zeros((257, 257, 3), dtype=np.float32)
        normal_mask = root["normal_mask"][tile_id].astype(np.float32) if bool(entry.get("has_normal_xyz", False)) and "normal_mask" in root else np.zeros((257, 257), dtype=np.float32)
        alpha = root["alpha_256"][tile_id].astype(np.float32) if bool(entry.get("has_alpha_256", False)) else np.zeros((256, 256, 4), dtype=np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)
        holes = root["holes_16"][tile_id].astype(np.float32) if bool(entry.get("has_holes_16", False)) else np.zeros((16, 16), dtype=np.float32)
        liquid_mask = root["liquid_mask"][tile_id].astype(np.float32) if bool(entry.get("has_liquid_mask", False)) else np.zeros((256, 256), dtype=np.float32)
        liquid_mask = np.clip(liquid_mask, 0.0, 1.0)
        liquid_height = root["liquid_height"][tile_id].astype(np.float32) if bool(entry.get("has_liquid_height", False)) and "liquid_height" in root else np.zeros((256, 256), dtype=np.float32)
        mcly_ids = root["mcly_texture_ids"][tile_id].astype(np.int64) if bool(entry.get("has_mcly_texture_ids", False)) and "mcly_texture_ids" in root else np.zeros((16, 16, 4), dtype=np.int64)
        mcly_ids = np.clip(mcly_ids, 0, 15)
        mcly_mask = root["mcly_layer_mask"][tile_id].astype(np.float32) if bool(entry.get("has_mcly_layer_mask", False)) and "mcly_layer_mask" in root else np.zeros((16, 16, 4), dtype=np.float32)
        mcnk_flags_16 = root["mcnk_flags_16"][tile_id].astype(np.int32) if "mcnk_flags_16" in root else np.zeros((16, 16), dtype=np.int32)
        liquid_type_16, liquid_type_valid_16 = _flags_to_liquid_type(mcnk_flags_16)

        object_filtered = root["object_filtered_mask"][tile_id].astype(np.float32) if "object_filtered_mask" in root else root["object_mask"][tile_id].astype(np.float32)
        weight_257 = 1.0 - np.clip(object_filtered, 0.0, 1.0)
        weight_256 = _crop_257_to_256(weight_257)
        weight_16 = _downsample_256_to_16(weight_256)
        mddf_mask = root["mddf_mask"][tile_id].astype(np.float32) if "mddf_mask" in root else np.zeros((257, 257), dtype=np.float32)
        modf_mask = root["modf_mask"][tile_id].astype(np.float32) if "modf_mask" in root else np.zeros((257, 257), dtype=np.float32)

        if self.augment:
            xform = int(self._rng.randint(0, 8))
            if xform & 1:
                minimap = minimap[:, ::-1]
                height_raw = height_raw[:, ::-1]
                height_norm = height_norm[:, ::-1]
                normals = normals[:, ::-1]
                normals[..., 0] = -normals[..., 0]
                normal_mask = normal_mask[:, ::-1]
                alpha = alpha[:, ::-1]
                holes = holes[:, ::-1]
                liquid_mask = liquid_mask[:, ::-1]
                liquid_height = liquid_height[:, ::-1]
                mcly_ids = mcly_ids[:, ::-1]
                mcly_mask = mcly_mask[:, ::-1]
                mcnk_flags_16 = mcnk_flags_16[:, ::-1]
                liquid_type_16 = liquid_type_16[:, ::-1]
                liquid_type_valid_16 = liquid_type_valid_16[:, ::-1]
                weight_257 = weight_257[:, ::-1]
                weight_256 = weight_256[:, ::-1]
                weight_16 = weight_16[:, ::-1]
                mddf_mask = mddf_mask[:, ::-1]
                modf_mask = modf_mask[:, ::-1]
            if xform & 2:
                minimap = minimap[::-1]
                height_raw = height_raw[::-1]
                height_norm = height_norm[::-1]
                normals = normals[::-1]
                normals[..., 1] = -normals[..., 1]
                normal_mask = normal_mask[::-1]
                alpha = alpha[::-1]
                holes = holes[::-1]
                liquid_mask = liquid_mask[::-1]
                liquid_height = liquid_height[::-1]
                mcly_ids = mcly_ids[::-1]
                mcly_mask = mcly_mask[::-1]
                mcnk_flags_16 = mcnk_flags_16[::-1]
                liquid_type_16 = liquid_type_16[::-1]
                liquid_type_valid_16 = liquid_type_valid_16[::-1]
                weight_257 = weight_257[::-1]
                weight_256 = weight_256[::-1]
                weight_16 = weight_16[::-1]
                mddf_mask = mddf_mask[::-1]
                modf_mask = modf_mask[::-1]
            if xform & 4:
                minimap = np.rot90(minimap, k=1)
                height_raw = np.rot90(height_raw, k=1)
                height_norm = np.rot90(height_norm, k=1)
                normals = np.rot90(normals, k=1)
                old_nx = normals[..., 0].copy()
                normals[..., 0] = normals[..., 1]
                normals[..., 1] = -old_nx
                normal_mask = np.rot90(normal_mask, k=1)
                alpha = np.rot90(alpha, k=1)
                holes = np.rot90(holes, k=1)
                liquid_mask = np.rot90(liquid_mask, k=1)
                liquid_height = np.rot90(liquid_height, k=1)
                mcly_ids = np.rot90(mcly_ids, k=1)
                mcly_mask = np.rot90(mcly_mask, k=1)
                mcnk_flags_16 = np.rot90(mcnk_flags_16, k=1)
                liquid_type_16 = np.rot90(liquid_type_16, k=1)
                liquid_type_valid_16 = np.rot90(liquid_type_valid_16, k=1)
                weight_257 = np.rot90(weight_257, k=1)
                weight_256 = np.rot90(weight_256, k=1)
                weight_16 = np.rot90(weight_16, k=1)
                mddf_mask = np.rot90(mddf_mask, k=1)
                modf_mask = np.rot90(modf_mask, k=1)

        return {
            "input": torch.from_numpy(minimap.copy()).permute(2, 0, 1),
            "height_raw": torch.from_numpy(height_raw.copy()).unsqueeze(0),
            "height_norm": torch.from_numpy(height_norm.copy()).unsqueeze(0),
            "height_mean": torch.tensor(h_mean, dtype=torch.float32),
            "height_std": torch.tensor(h_std, dtype=torch.float32),
            "normals": torch.from_numpy(normals.copy()).permute(2, 0, 1),
            "normal_mask": torch.from_numpy(normal_mask.copy()).unsqueeze(0),
            "alpha": torch.from_numpy(alpha.copy()).permute(2, 0, 1),
            "holes": torch.from_numpy(holes.copy()).unsqueeze(0),
            "liquid_mask": torch.from_numpy(liquid_mask.copy()).unsqueeze(0),
            "liquid_height": torch.from_numpy(liquid_height.copy()).unsqueeze(0),
            "liquid_type_16": torch.from_numpy(liquid_type_16.copy()).long(),
            "liquid_type_valid_16": torch.from_numpy(liquid_type_valid_16.copy()).unsqueeze(0),
            "mcly_ids": torch.from_numpy(mcly_ids.copy()).long(),
            "mcly_mask": torch.from_numpy(mcly_mask.copy()),
            "mcnk_flags_16": torch.from_numpy(mcnk_flags_16.copy()).long(),
            "weight_257": torch.from_numpy(weight_257.copy()).unsqueeze(0),
            "weight_256": torch.from_numpy(weight_256.copy()).unsqueeze(0),
            "weight_16": torch.from_numpy(weight_16.copy()).unsqueeze(0),
            "mddf_mask": torch.from_numpy(mddf_mask.copy()).unsqueeze(0),
            "modf_mask": torch.from_numpy(modf_mask.copy()).unsqueeze(0),
            "has_normals": bool(entry.get("has_normal_xyz", False)),
            "has_alpha": bool(entry.get("has_alpha_256", False)),
            "has_holes": bool(entry.get("has_holes_16", False)),
            "has_liquid": bool(entry.get("has_liquid_mask", False)),
            "has_mcly": bool(entry.get("has_mcly_texture_ids", False)),
            "meta_build": str(entry.get("build") or build),
            "meta_store": str(build),
            "meta_map": str(entry.get("map", "")),
            "meta_tile_id": tile_id,
            "meta_tile_x": int(entry.get("tile_x") if entry.get("tile_x") is not None else -1),
            "meta_tile_y": int(entry.get("tile_y") if entry.get("tile_y") is not None else -1),
        }
