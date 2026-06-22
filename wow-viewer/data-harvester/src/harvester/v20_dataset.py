"""V20 Multi-Modal Chained dataset.

Extends V16.1 dataset by loading and exposing:
- liquid_type_256: pixel-aligned liquid classes [0-4]
- ground_intent_height: normalized continuous ground height under structures
- object_precise_mask_256/257: structural footprint masks
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from harvester.v16_1_dataset import V161Dataset


def _flip_lr(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 2:
        w_dim = -2 if tensor.shape[-1] == 4 else -1
        return torch.flip(tensor, dims=[w_dim])
    return tensor


def _flip_ud(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 2:
        h_dim = -3 if tensor.shape[-1] == 4 else -2
        return torch.flip(tensor, dims=[h_dim])
    return tensor


def _rot90(tensor: torch.Tensor) -> torch.Tensor:
    if tensor.ndim >= 2:
        h_dim, w_dim = (-3, -2) if tensor.shape[-1] == 4 else (-2, -1)
        return torch.rot90(tensor, k=1, dims=[h_dim, w_dim])
    return tensor


def augment_sample_tensors(sample: Dict[str, Any], xform: int) -> Dict[str, Any]:
    spatial_keys = [
        "input", "height_raw", "height_norm", "normals", "normal_mask",
        "alpha", "liquid_mask", "liquid_height", "weight_257", "weight_256",
        "object_presence_257", "object_roof_mask_256", "object_roof_weight_256",
        "object_roof_weight_257", "alpha_painted_256", "terrain_valid_mask_257",
        "liquid_type_256", "ground_intent_height", "object_precise_mask_256",
        "object_precise_mask_257"
    ]
    coarse_keys = [
        "holes", "liquid_type_16", "liquid_type_valid_16", "mcly_ids", "mcly_mask",
        "mcnk_flags_16", "weight_16", "mcly_any_16"
    ]

    if xform & 1:  # Flip Left-Right
        for k in spatial_keys:
            if k in sample and isinstance(sample[k], torch.Tensor):
                sample[k] = _flip_lr(sample[k])
        for k in coarse_keys:
            if k in sample and isinstance(sample[k], torch.Tensor):
                sample[k] = _flip_lr(sample[k])
        if "normals" in sample and isinstance(sample["normals"], torch.Tensor):
            sample["normals"][0] = -sample["normals"][0]

    if xform & 2:  # Flip Up-Down
        for k in spatial_keys:
            if k in sample and isinstance(sample[k], torch.Tensor):
                sample[k] = _flip_ud(sample[k])
        for k in coarse_keys:
            if k in sample and isinstance(sample[k], torch.Tensor):
                sample[k] = _flip_ud(sample[k])
        if "normals" in sample and isinstance(sample["normals"], torch.Tensor):
            sample["normals"][1] = -sample["normals"][1]

    if xform & 4:  # Rotate 90
        for k in spatial_keys:
            if k in sample and isinstance(sample[k], torch.Tensor):
                sample[k] = _rot90(sample[k])
        for k in coarse_keys:
            if k in sample and isinstance(sample[k], torch.Tensor):
                sample[k] = _rot90(sample[k])
        if "normals" in sample and isinstance(sample["normals"], torch.Tensor):
            old_nx = sample["normals"][0].clone()
            sample["normals"][0] = sample["normals"][1]
            sample["normals"][1] = -old_nx

    return sample


class V20Dataset(V161Dataset):
    """V20 dataset with multimodal outputs for chained reconstruction.

    Exposes:
    - input: [3, 256, 256] minimap RGB (or [6, 256, 256] with normals)
    - liquid_type_256: [1, 256, 256] target classes [0-4]
    - ground_intent_height: [1, 257, 257] normalized clean heighttarget
    - object_precise_mask_256: [1, 256, 256] precise structural footprints
    - object_precise_mask_257: [1, 257, 257] precise structural footprints
    """

    def __init__(
        self,
        dataset_root: Path,
        builds: Optional[List[str]] = None,
        include_maps: Optional[List[str]] = None,
        exclude_maps: Optional[List[str]] = None,
        input_channels: int = 3,
        augment: bool = True,
        limit: Optional[int] = None,
        curation_manifest: Optional[str | Path] = None,
        curation_min_terrain_validity: float = 0.0,
        curation_min_minimap_usefulness: float = 0.0,
        curation_reject_what_plate: bool = True,
        val_fraction: float = 0.1,
        split: str = "train",
        seed: int = 42,
        **kwargs,
    ):
        if input_channels not in (3, 6):
            raise ValueError(f"V20Dataset: input_channels must be 3 or 6, got {input_channels}")

        self.input_channels = input_channels
        self.augment_v20 = augment and split == "train"

        # Initialize base dataset with augment=False to handle flips/rotations uniformly
        super().__init__(
            dataset_dir=dataset_root,
            builds=builds,
            split=split,
            val_fraction=val_fraction,
            seed=seed,
            augment=False,
            curation_manifest=curation_manifest,
            height_channel=False,
            object_roof_channel=False,
            curation_min_terrain_validity=curation_min_terrain_validity,
            curation_min_minimap_usefulness=curation_min_minimap_usefulness,
            curation_reject_what_plate=curation_reject_what_plate,
            **kwargs,
        )

        if limit is not None and limit > 0:
            self._indices = self._indices[:limit]

        print(f"V20Dataset: {len(self)} samples, input_channels={input_channels}, augment={self.augment_v20}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        # Call base class to load raw data without augmentations
        sample = super().__getitem__(index)

        # Retrieve index metadata
        entry = self._index_entries[self._indices[index]]
        build = entry["_build"]
        root = self._stores[build]
        tile_id = int(entry["tile_id"])

        h_mean = float(entry["height_mean"])
        h_std = float(entry["height_std"]) + 1e-8

        # 1. Fetch liquid_type_256
        if "liquid_type_256" in root:
            liquid_type = root["liquid_type_256"][tile_id].astype(np.int64)
        else:
            # Fallback to zero
            liquid_type = np.zeros((256, 256), dtype=np.int64)

        # 2. Fetch ground_intent_height_257 and normalize
        if "ground_intent_height_257" in root:
            ground_h_raw = root["ground_intent_height_257"][tile_id].astype(np.float32)
        else:
            # Fallback to raw height
            ground_h_raw = root["height_257"][tile_id].astype(np.float32)

        ground_h_norm = (ground_h_raw - h_mean) / h_std

        # Fetch actual precision object masks directly from Zarr
        if "object_precise_mask" in root:
            precise_mask_257 = root["object_precise_mask"][tile_id].astype(np.float32)
            precise_mask_257 = np.clip(precise_mask_257, 0.0, 1.0)
        elif "object_roof_mask" in root:
            roof_mask_256 = root["object_roof_mask"][tile_id].astype(np.float32)
            roof_mask_256 = np.clip(roof_mask_256, 0.0, 1.0)
            precise_mask_257 = np.pad(roof_mask_256, ((0, 1), (0, 1)), mode="edge")
        else:
            precise_mask_257 = np.zeros((257, 257), dtype=np.float32)

        precise_mask_256 = precise_mask_257[:256, :256]

        object_mask_256 = torch.from_numpy(precise_mask_256).unsqueeze(0).float()
        object_mask_257 = torch.from_numpy(precise_mask_257).unsqueeze(0).float()

        # Append V20 specific signals
        sample["liquid_type_256"] = torch.from_numpy(liquid_type).unsqueeze(0).long()
        sample["ground_intent_height"] = torch.from_numpy(ground_h_norm).unsqueeze(0).float()
        sample["object_precise_mask_256"] = object_mask_256
        sample["object_precise_mask_257"] = object_mask_257

        # Adjust input channel count
        minimap_rgb = sample["input"][:3]
        if self.input_channels == 6:
            normals = sample["normals"]
            normals_256 = normals[:, :256, :256]
            sample["input"] = torch.cat([minimap_rgb, normals_256], dim=0)
        else:
            sample["input"] = minimap_rgb

        # 3. Apply uniform augmentations if enabled
        if self.augment_v20:
            xform = int(self._rng.randint(0, 8))
            sample = augment_sample_tensors(sample, xform)

        return sample


def create_v20_dataset(
    dataset_root: Path,
    builds: Optional[List[str]] = None,
    include_maps: Optional[List[str]] = None,
    exclude_maps: Optional[List[str]] = None,
    input_channels: int = 3,
    augment: bool = True,
    limit: Optional[int] = None,
    curation_manifest: Optional[str | Path] = None,
    split: str = "train",
    val_fraction: float = 0.1,
    seed: int = 42,
) -> V20Dataset:
    return V20Dataset(
        dataset_root=dataset_root,
        builds=builds,
        include_maps=include_maps,
        exclude_maps=exclude_maps,
        input_channels=input_channels,
        augment=augment,
        limit=limit,
        curation_manifest=curation_manifest,
        split=split,
        val_fraction=val_fraction,
        seed=seed,
    )
