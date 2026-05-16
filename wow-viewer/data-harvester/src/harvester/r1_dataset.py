"""R1Dataset — PyTorch Dataset for V14 Model R1: Terrain Reconstruction.

Returns (residual, height_257, hole_mask_16, liquid_mask_256).
Residual is computed from ground-truth MCAL alphas via the compositor,
matching what the D2 subtraction step would produce if D1 were perfect.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .compositor import composite_synthetic_minimap, compute_residual


def _downsample_alpha(alpha: np.ndarray, target_size: int = 256) -> np.ndarray:
    h, w = alpha.shape[:2]
    if h == target_size:
        return alpha
    factor = h // target_size
    new_shape = (target_size, factor, target_size, factor) + alpha.shape[2:]
    return alpha.reshape(new_shape).mean(axis=1).mean(axis=2)


def _build_shard_index(
    shard_root: Path,
    validation_selection_path: Path,
) -> tuple[list[Path], list[Path]]:
    with open(validation_selection_path, encoding="utf-8") as f:
        selection = json.load(f)

    val_set: set[str] = set()
    for entry in selection.get("selections", []):
        original_path = entry.get("path", "")
        if original_path:
            p = Path(original_path)
            if len(p.parts) >= 3:
                key = str(Path(p.parts[-3]) / p.parts[-2] / p.parts[-1])
                val_set.add(key)

    train_paths: list[Path] = []
    val_paths: list[Path] = []

    for npz_path in sorted(shard_root.glob("*/*/*.npz")):
        key = str(Path(npz_path.parts[-3]) / npz_path.parts[-2] / npz_path.parts[-1])
        if key in val_set:
            val_paths.append(npz_path)
        else:
            train_paths.append(npz_path)

    return train_paths, val_paths


R1_REQUIRED = frozenset({"minimap_rgb_256", "height_257", "hole_mask_16"})


class R1Dataset(Dataset):
    """PyTorch Dataset for V14 Model R1.

    Args:
        shard_root: shards/<build>/<map>/*.npz
        validation_selection_path: holdout JSON
        split: 'train' or 'val'
        max_samples: randomly subsample to this many eligible shards
        seed: random seed for subsampling
    """

    def __init__(
        self,
        shard_root: str | Path,
        validation_selection_path: str | Path,
        split: str = "train",
        max_samples: int | None = None,
        seed: int = 42,
    ) -> None:
        shard_root = Path(shard_root)
        validation_selection_path = Path(validation_selection_path)

        train_paths, val_paths = _build_shard_index(shard_root, validation_selection_path)
        self._paths = train_paths if split == "train" else val_paths
        self._eligible: list[int] = []
        self._max_samples = max_samples
        self._rng = random.Random(seed)

    def _ensure_index(self) -> None:
        if self._eligible:
            return
        for i, p in enumerate(self._paths):
            try:
                with np.load(p) as data:
                    keys = set(data.files)
                    has_mm = "minimap_rgb_256" in keys
                    has_h = "height_257" in keys
                    has_holes = "hole_mask_16" in keys
                    has_alpha = "mcal_alpha_pack_256" in keys or "mcal_alpha_pack" in keys
                    if has_mm and has_h and has_holes and has_alpha:
                        self._eligible.append(i)
            except OSError:
                continue
        if self._max_samples is not None and len(self._eligible) > self._max_samples:
            self._eligible = self._rng.sample(self._eligible, self._max_samples)
            self._eligible.sort()

    def __len__(self) -> int:
        self._ensure_index()
        return len(self._eligible)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_index()
        path = self._paths[self._eligible[idx]]

        with np.load(path) as data:
            minimap = data["minimap_rgb_256"].astype(np.float32) / 255.0
            height_257 = data["height_257"].astype(np.float32)
            holes_16 = data["hole_mask_16"].astype(np.float32)

            # Liquid: try unified_liquid_mask, then mh2o/mclq fallback
            liquid = data.get("unified_liquid_mask", None)
            if liquid is None:
                liquid = data.get("mh2o_type_mask", None)
                if liquid is not None:
                    liquid = (liquid > 0).astype(np.float32)
            if liquid is None:
                liquid = data.get("mclq_type_mask", None)
                if liquid is not None:
                    liquid = (liquid > 0).astype(np.float32)
            if liquid is None or liquid.ndim != 2:
                liquid = np.zeros((257, 257), dtype=np.float32)

            # Downsample liquid 257→256
            if liquid.shape[0] == 257:
                liquid = liquid[:256, :256]

            alpha_key = "mcal_alpha_pack_256" if "mcal_alpha_pack_256" in data else "mcal_alpha_pack"
            alpha_pack = data[alpha_key].astype(np.float32)
            if alpha_pack.shape[0] != 256:
                alpha_pack = _downsample_alpha(alpha_pack, 256)
            if alpha_pack.max() > 1.5:
                alpha_pack /= 255.0
            alpha_pack = alpha_pack.clip(0, 1)

        # Compute residual
        synthetic = composite_synthetic_minimap(alpha_pack)
        residual = compute_residual(minimap, synthetic)

        # Normalise height to zero-mean unit-variance per tile
        h_mean = height_257.mean()
        h_std = height_257.std() + 1e-8
        height_norm = (height_257 - h_mean) / h_std

        # To tensors
        inp = torch.from_numpy(residual.copy()).permute(2, 0, 1)  # (3, 256, 256)
        hgt = torch.from_numpy(height_norm.copy()).unsqueeze(0)   # (1, 257, 257)
        hol = torch.from_numpy(holes_16.copy()).unsqueeze(0)       # (1, 16, 16)
        liq = torch.from_numpy(liquid.copy()).unsqueeze(0)         # (1, 256, 256)

        return inp, hgt, hol, liq
