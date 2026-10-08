"""Residual Elevation Dataset Builder & Loader for Cascaded Refinement (Spec 265).

Extracts initial predictions Z_initial from Stage 1 SupervisedElevationUNet and computes
the residual target field ΔZ = Z_gt - Z_initial in physical yards across all 270 authentic tiles.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from harvester.v60.supervised_elevation_model import (
    SupervisedElevationUNet,
    compute_elevation_normals,
)


class ResidualElevationDataset(Dataset):
    """PyTorch Dataset for Stage 2 Residual Refinement."""

    def __init__(
        self,
        samples: List[Dict[str, np.ndarray | str]],
        augment: bool = False,
    ):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        item = self.samples[idx]
        rgb = item["rgb"]          # (256, 256, 3) in [0, 1]
        z_init = item["z_init"]    # (257, 257) in yards
        delta_z = item["delta_z"]  # (257, 257) in yards (target)
        stem = str(item["stem"])

        if self.augment:
            if np.random.rand() > 0.5:
                rgb = np.fliplr(rgb).copy()
                z_init = np.fliplr(z_init).copy()
                delta_z = np.fliplr(delta_z).copy()

        # Compute unit surface normals from z_initial for input conditioning
        z_init_t = torch.from_numpy(z_init).unsqueeze(0).float()  # (1, 257, 257)
        normals = compute_elevation_normals(z_init_t)[0]          # (3, 255, 255)

        # Pad normals to 256x256 to match RGB grid
        pad_normals = torch.nn.functional.pad(normals[:2], (0, 1, 0, 1), mode="replicate")  # (2, 256, 256)

        # Interpolate z_init to 256x256 and scale
        z_init_norm = (torch.nn.functional.interpolate(
            z_init_t.unsqueeze(0), size=(256, 256), mode="bilinear", align_corners=True
        )[0] / 100.0).float()  # (1, 256, 256)

        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float()  # (3, 256, 256)

        # Assemble 6-channel input feature tensor: RGB (3) + z_init_norm (1) + normals (2)
        features = torch.cat([rgb_t, z_init_norm, pad_normals], dim=0)  # (6, 256, 256)

        return {
            "features": features,
            "z_init": z_init_t.squeeze(0),
            "delta_z": torch.from_numpy(delta_z).float(),
            "z_gt": torch.from_numpy(item["z_gt"]).float(),
            "stem": stem,
        }


def build_or_load_residual_corpus(
    residual_cache_path: Path,
    base_corpus_path: Path,
    stage1_checkpoint_path: Path,
    device: str = "cpu",
    force_rebuild: bool = False,
) -> Tuple[List[Dict[str, np.ndarray | str]], List[Dict[str, np.ndarray | str]]]:
    """Generates residual dataset by running Stage 1 inference over base corpus."""
    residual_cache_path = Path(residual_cache_path)
    if residual_cache_path.is_file() and not force_rebuild:
        print(f"Loading cached residual corpus from {residual_cache_path}...")
        data = np.load(residual_cache_path, allow_pickle=True)
        return list(data["train"]), list(data["val"])

    print(f"Loading base elevation corpus from {base_corpus_path}...")
    base_data = np.load(base_corpus_path, allow_pickle=True)
    raw_train = list(base_data["train"])
    raw_val = list(base_data["val"])

    print(f"Loading Stage 1 model from {stage1_checkpoint_path}...")
    ckpt = torch.load(stage1_checkpoint_path, map_location=device, weights_only=False)
    stage1 = SupervisedElevationUNet(
        in_channels=3,
        base_channels=ckpt.get("base_channels", 48),
        global_elevation_mean=ckpt.get("global_elevation_mean", 75.0),
        global_elevation_std=ckpt.get("global_elevation_std", 75.0),
    ).to(device)
    stage1.load_state_dict(ckpt["model_state"])
    stage1.eval()

    def _process_split(split_data: List[Dict[str, np.ndarray | str]]) -> List[Dict[str, np.ndarray | str]]:
        processed = []
        with torch.no_grad():
            for item in split_data:
                rgb = item["rgb"]
                gt_z = item["z"]
                x = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device).float()
                pred_z = stage1(x).cpu().numpy()[0]  # (257, 257)

                delta_z = (gt_z - pred_z).astype(np.float32)
                processed.append({
                    "stem": item["stem"],
                    "rgb": rgb.astype(np.float32),
                    "z_init": pred_z.astype(np.float32),
                    "z_gt": gt_z.astype(np.float32),
                    "delta_z": delta_z,
                })
        return processed

    print(f"Inferring Stage 1 predictions and computing Delta Z across {len(raw_train)} train tiles...")
    res_train = _process_split(raw_train)
    print(f"Inferring Stage 1 predictions and computing Delta Z across {len(raw_val)} val tiles...")
    res_val = _process_split(raw_val)

    residual_cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        residual_cache_path,
        train=res_train,
        val=res_val,
    )
    print(f"Cached residual corpus to {residual_cache_path} ({len(res_train)} train, {len(res_val)} val).")
    return res_train, res_val
