"""V21 scar-mask dataset built from patched V18 Zarr stores."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from torch.utils.data import Dataset


def make_scar_mask(alpha: np.ndarray, layers: tuple[int, ...] = (1, 2, 3), threshold: float = 0.05) -> np.ndarray:
    """Return a single 256x256 binary mask for authored alpha scars."""
    arr = np.asarray(alpha, dtype=np.float32)
    if arr.ndim != 3:
        raise ValueError(f"alpha must be HxWxL, got {arr.shape}")
    valid_layers = [layer for layer in layers if 0 <= int(layer) < arr.shape[2]]
    if not valid_layers:
        return np.zeros(arr.shape[:2], dtype=np.float32)
    return (arr[:, :, valid_layers].max(axis=2) > float(threshold)).astype(np.float32, copy=False)


def _split_indices(n_items: int, split: str, val_fraction: float, seed: int) -> list[int]:
    rng = np.random.RandomState(int(seed))
    indices = rng.permutation(n_items)
    n_val = max(1, int(round(n_items * float(val_fraction)))) if n_items > 1 else 0
    if split == "val":
        return sorted(indices[:n_val].tolist())
    if split == "train":
        return sorted(indices[n_val:].tolist()) if n_val else sorted(indices.tolist())
    raise ValueError(f"split must be train or val, got {split}")


class V21ScarMaskDataset(Dataset):
    """Read minimaps and derived scar masks from patched V18 Zarr stores."""

    def __init__(
        self,
        dataset_dir: str | Path,
        builds: list[str] | None = None,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 74,
        threshold: float = 0.05,
        layers: tuple[int, ...] = (1, 2, 3),
        max_tiles: int | None = None,
        augment: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.threshold = float(threshold)
        self.layers = tuple(int(layer) for layer in layers)
        self.augment = bool(augment and split == "train")
        self._rng = np.random.RandomState(int(seed))
        self._stores: dict[str, zarr.Group] = {}
        entries: list[dict] = []

        build_names = builds or [path.name.removesuffix(".zarr") for path in sorted(self.dataset_dir.glob("*.zarr")) if path.is_dir()]
        for build in build_names:
            zarr_path = self.dataset_dir / f"{build}.zarr"
            if not zarr_path.exists():
                continue
            store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
            root = zarr.open_group(store=store, mode="r")
            if "minimap_rgb" not in root or "alpha_256" not in root:
                continue
            self._stores[build] = root
            index_path = zarr_path / "index.parquet"
            if index_path.exists():
                table = pq.read_table(str(index_path))
                for row_idx in range(table.num_rows):
                    if "has_alpha_256" in table.column_names and not bool(table.column("has_alpha_256")[row_idx].as_py()):
                        continue
                    entries.append(
                        {
                            "build": build,
                            "tile_id": int(table.column("tile_id")[row_idx].as_py()) if "tile_id" in table.column_names else row_idx,
                            "map": str(table.column("map")[row_idx].as_py()) if "map" in table.column_names else "unknown",
                            "tile_x": int(table.column("tile_x")[row_idx].as_py() or -1) if "tile_x" in table.column_names else -1,
                            "tile_y": int(table.column("tile_y")[row_idx].as_py() or -1) if "tile_y" in table.column_names else -1,
                        }
                    )
            else:
                for tile_id in range(int(root["alpha_256"].shape[0])):
                    entries.append({"build": build, "tile_id": tile_id, "map": "unknown", "tile_x": -1, "tile_y": -1})

        if max_tiles is not None:
            entries = entries[: int(max_tiles)]
        if not entries:
            raise ValueError(f"No V21 scar-mask entries found under {self.dataset_dir}")
        self._entries = entries
        self._indices = _split_indices(len(entries), split, val_fraction, seed)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | int | str]:
        entry = self._entries[self._indices[idx]]
        root = self._stores[str(entry["build"])]
        tile_id = int(entry["tile_id"])
        minimap = root["minimap_rgb"][tile_id].astype(np.float32) / 255.0
        alpha = np.clip(root["alpha_256"][tile_id].astype(np.float32), 0.0, 1.0)
        scar = make_scar_mask(alpha, self.layers, self.threshold)

        if self.augment:
            xform = int(self._rng.randint(0, 8))
            if xform & 1:
                minimap = minimap[:, ::-1]
                scar = scar[:, ::-1]
            if xform & 2:
                minimap = minimap[::-1]
                scar = scar[::-1]
            if xform & 4:
                minimap = np.rot90(minimap, k=1)
                scar = np.rot90(scar, k=1)

        return {
            "input": torch.from_numpy(minimap.copy()).permute(2, 0, 1),
            "scar_mask": torch.from_numpy(scar.copy()).unsqueeze(0),
            "meta_build": str(entry["build"]),
            "meta_map": str(entry["map"]),
            "meta_tile_id": int(entry["tile_id"]),
            "meta_tile_x": int(entry["tile_x"]),
            "meta_tile_y": int(entry["tile_y"]),
        }
