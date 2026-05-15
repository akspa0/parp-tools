"""D1Dataset — PyTorch Dataset for V14 Model D1: Tileset Decomposition.

Loads NPZ shards from the staged native shard directory, splits into training
and validation using the existing validation_selection.json holdout, and provides
(d1_input, d1_targets) tensors for training.

D1 Input:  minimap_rgb_256  (3, 256, 256) float32 in [0, 1]
D1 Target: tileset_layer_1  (3, 256, 256) float32
           tileset_layer_2  (3, 256, 256) float32
           alpha_mask_1     (1, 256, 256) float32
           alpha_mask_2     (1, 256, 256) float32
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from .compositor import compute_d1_targets


def _downsample_alpha(alpha: np.ndarray, target_size: int = 256) -> np.ndarray:
    """Bilinearly downsample MCAL alpha from source resolution to target_size x target_size.

    Handles the stale-shard issue where 0.7.0 and 3.3.5 shards store 1024x1024
    alpha data under the mcal_alpha_pack_256 key.
    """
    h, w = alpha.shape[:2]
    if h == target_size and w == target_size:
        return alpha
    # Use simple area-based downsampling equivalent to bilinear interpolation
    # Factor is integer (1024/256 = 4)
    factor = h // target_size
    if factor * target_size != h:
        # Non-integer factor — fall back to corner sampling
        indices_y = np.linspace(0, h - 1, target_size, dtype=np.int32)
        indices_x = np.linspace(0, w - 1, target_size, dtype=np.int32)
        return alpha[np.ix_(indices_y, indices_x)]
    # 4D area reduction: reshape and mean
    new_shape = (target_size, factor, target_size, factor) + alpha.shape[2:]
    return alpha.reshape(new_shape).mean(axis=1).mean(axis=2)


def _build_shard_index(
    shard_root: Path,
    validation_selection_path: Path,
) -> tuple[list[Path], list[Path]]:
    """Scan shard_root for D1-ready shards and split into train/val using the holdout JSON.

    Returns (train_paths, val_paths) sorted for reproducibility.
    Validation shards are identified by the 'path' field in validation_selection.json.
    """
    with open(validation_selection_path, encoding="utf-8") as f:
        selection = json.load(f)

    # Collect the set of original shard paths used as validation holdout
    val_set: set[str] = set()
    for entry in selection.get("selections", []):
        original_path = entry.get("path", "")
        if original_path:
            val_set.add(str(Path(original_path).resolve()))

    train_paths: list[Path] = []
    val_paths: list[Path] = []

    for npz_path in sorted(shard_root.glob("*/*/*.npz")):
        abs_path = str(npz_path.resolve())
        if abs_path in val_set:
            target_list = val_paths
        else:
            target_list = train_paths

        target_list.append(npz_path)

    return train_paths, val_paths


class D1Dataset(Dataset):
    """PyTorch Dataset for V14 Model D1: Tileset Decomposition.

    Args:
        shard_root: Root directory containing shards/<build>/<map>/*.npz.
        validation_selection_path: Path to validation_selection.json for train/val split.
        split: 'train' or 'val'.
    """

    def __init__(
        self,
        shard_root: str | Path,
        validation_selection_path: str | Path,
        split: str = "train",
    ) -> None:
        shard_root = Path(shard_root)
        validation_selection_path = Path(validation_selection_path)

        train_paths, val_paths = _build_shard_index(shard_root, validation_selection_path)

        if split == "train":
            self._paths = train_paths
        elif split == "val":
            self._paths = val_paths
        else:
            raise ValueError(f"Unknown split '{split}'. Use 'train' or 'val'.")

        self._eligible: list[int] = []
        self._dropout_indices: list[int] = []
        self._strict: bool = True

    @property
    def all_paths(self) -> list[Path]:
        return list(self._paths)

    @property
    def eligible_paths(self) -> list[Path]:
        self._ensure_index()
        return [self._paths[i] for i in self._eligible]

    @property
    def dropout_paths(self) -> list[Path]:
        self._ensure_index()
        return [self._paths[i] for i in self._dropout_indices]

    @property
    def strict(self) -> bool:
        return self._strict

    @strict.setter
    def strict(self, value: bool) -> None:
        self._strict = value

    def _ensure_index(self) -> None:
        """Lazily build the eligibility index on first access."""
        if self._eligible or self._dropout_indices:
            return
        for i, p in enumerate(self._paths):
            try:
                with np.load(p) as data:
                    has_mm = "minimap_rgb_256" in data
                    has_alpha = "mcal_alpha_pack_256" in data or "mcal_alpha_pack" in data
                    has_mcly = "mcly_texture_ids" in data
                    if has_mm and has_alpha and has_mcly:
                        self._eligible.append(i)
                    else:
                        self._dropout_indices.append(i)
            except OSError:
                self._dropout_indices.append(i)

    def __len__(self) -> int:
        self._ensure_index()
        return len(self._eligible)

    def _load_sample(
        self, npz_path: Path
    ) -> tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ]:
        """Load a single shard and return D1 input/target arrays.

        Returns:
            (minimap, tileset_1, tileset_2, alpha_1, alpha_2) — all float32 [0,1].
            Returns None elements if the shard is missing required keys.
        """
        with np.load(npz_path) as data:
            minimap = data["minimap_rgb_256"].astype(np.float32) / 255.0

            alpha_key = (
                "mcal_alpha_pack_256" if "mcal_alpha_pack_256" in data else "mcal_alpha_pack"
            )
            alpha_pack = data[alpha_key].astype(np.float32)

            # Handle stale 1024x1024 alpha storage
            if alpha_pack.shape[0] != 256:
                alpha_pack = _downsample_alpha(alpha_pack, 256)

            # Alpha values are stored as uint8 [0,255]; normalise to [0,1]
            if alpha_pack.max() > 1.5:
                alpha_pack = alpha_pack / 255.0

            alpha_pack = alpha_pack.clip(0.0, 1.0)

        t1, t2, a1, a2 = compute_d1_targets(alpha_pack, minimap)
        return minimap, t1, t2, a1, a2

    def __getitem__(
        self, idx: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self._ensure_index()
        if self._strict and idx >= len(self._eligible):
            raise IndexError(
                f"Attempted to access dropout shard at index {idx} "
                f"(only {len(self._eligible)} eligible). "
                "Run audit first or set dataset.strict = False."
            )
        real_idx = self._eligible[idx]
        path = self._paths[real_idx]
        minimap, t1, t2, a1, a2 = self._load_sample(path)

        # Convert to CHW for PyTorch
        inputs = torch.from_numpy(minimap.copy()).permute(2, 0, 1)  # (3, 256, 256)
        target_t1 = torch.from_numpy(t1.copy()).permute(2, 0, 1)  # (3, 256, 256)
        target_t2 = torch.from_numpy(t2.copy()).permute(2, 0, 1)  # (3, 256, 256)
        target_a1 = torch.from_numpy(a1.copy()).unsqueeze(0)  # (1, 256, 256)
        target_a2 = torch.from_numpy(a2.copy()).unsqueeze(0)  # (1, 256, 256)

        return inputs, target_t1, target_t2, target_a1, target_a2


def create_d1_dataloaders(
    shard_root: str | Path,
    validation_selection_path: str | Path,
    batch_size: int = 16,
    num_workers: int = 0,
    train_val_split: float = 0.9,
) -> tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create training and validation DataLoaders for D1.

    Uses the validation_selection.json holdout for splitting:
    - Training: all D1-ready shards NOT in the holdout
    - Validation: the pre-selected holdout tiles

    Falls back to a random 90/10 split if fewer than 10 validation shards are eligible.
    """
    shard_root = Path(shard_root)
    validation_selection_path = Path(validation_selection_path)

    train_paths, val_paths = _build_shard_index(shard_root, validation_selection_path)

    # Check eligibility of validation shards
    eligible_val: list[Path] = []
    for p in val_paths:
        try:
            with np.load(p) as data:
                has_mm = "minimap_rgb_256" in data
                has_alpha = "mcal_alpha_pack_256" in data or "mcal_alpha_pack" in data
                has_mcly = "mcly_texture_ids" in data
                if has_mm and has_alpha and has_mcly:
                    eligible_val.append(p)
        except OSError:
            continue

    train_ds = D1Dataset(shard_root, validation_selection_path, split="train")
    val_ds = D1Dataset(shard_root, validation_selection_path, split="val")

    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=True,
    )

    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
        pin_memory=True,
    )

    return train_loader, val_loader
