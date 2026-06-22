"""V19 dataset – minimal signal terrain reconstruction.

V19 builds on V16.1's Zarr stores and provides the minimal input set:
- minimap RGB (3ch, always present)
- optional normal map RGB (3ch, ~70% coverage)

Target: height_257 with liquid_height override where liquid_mask > 0.

No WDL prior. No object masks. No alpha layers. No brush masks.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import Dataset

from harvester.v16_1_dataset import V161Dataset, _interpolate_checkerboard_normals


class V19Dataset(V161Dataset):
    """V19 dataset inheriting from V161Dataset with minimal input channels.

    Provides:
    - input: [C, 256, 256] where C=3 (minimap only) or C=6 (minimap + normals)
    - target: [1, 257, 257] absolute height with liquid override
    - terrain_valid_mask: [1, 257, 257] loss weighting mask
    - height_mean, height_std: normalization params for denormalization
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
            raise ValueError(f"V19Dataset: input_channels must be 3 or 6, got {input_channels}")

        self.input_channels = input_channels
        super().__init__(
            dataset_dir=dataset_root,
            builds=builds,
            split=split,
            val_fraction=val_fraction,
            seed=seed,
            augment=augment,
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

        print(f"V19Dataset: {len(self)} samples, input_channels={input_channels}")

    def __getitem__(self, index: int) -> Dict[str, Any]:
        sample = super().__getitem__(index)

        minimap_rgb = sample["input"][:3]  # [3, 256, 256]

        if self.input_channels == 6:
            normals = sample["normals"]  # [3, 257, 257]
            normals_256 = normals[:, :256, :256]
            input_tensor = torch.cat([minimap_rgb, normals_256], dim=0)
        else:
            input_tensor = minimap_rgb

        height_norm = sample["height_norm"]  # [1, 257, 257] normalized to ~[0,1]
        liquid_mask = sample["liquid_mask"]  # [1, 256, 256]
        liquid_height = sample["liquid_height"]  # [1, 256, 256]
        height_mean = sample["height_mean"]
        height_std = sample["height_std"]

        target = height_norm.clone()
        liquid_mask_257 = torch.nn.functional.pad(
            liquid_mask, (0, 1, 0, 1), mode="replicate"
        )
        liquid_height_257 = torch.nn.functional.pad(
            liquid_height, (0, 1, 0, 1), mode="replicate"
        )
        water_pixels = liquid_mask_257.squeeze(0) > 0.5
        if water_pixels.any():
            water_height_raw = liquid_height_257.squeeze(0)[water_pixels]
            water_height_norm = (water_height_raw - height_mean) / (height_std + 1e-8)
            target[0, water_pixels] = water_height_norm

        terrain_valid = sample["terrain_valid_mask_257"]

        sample["input"] = input_tensor
        sample["target"] = target
        sample["terrain_valid_mask"] = terrain_valid
        return sample


def create_v19_dataset(
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
) -> V19Dataset:
    return V19Dataset(
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
