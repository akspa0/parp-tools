"""Trestle Elevation Dataset Builder & PyTorch Dataset (Spec 266).

Prepares 6-channel feature tensors (Minimap RGB + Upsampled WDL Trestle + Photometric Normals)
paired with ground-truth elevation, residual delta ΔZ, and elevation bounds [Z_min, Z_max].
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset

from harvester.v60.trestle_wdl_synthesizer import TrestleWdlSynthesizer


def compute_photometric_normals(rgb: np.ndarray) -> np.ndarray:
    """Compute 2-channel photometric surface normals (Nx, Ny) from 256x256 RGB minimap."""
    # rgb shape (256, 256, 3) in [0, 1]
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    # Smooth slightly to avoid single-pixel noise
    gray_smooth = ndimage.gaussian_filter(gray, sigma=1.0)
    gy, gx = np.gradient(gray_smooth)
    # Estimate surface normals: assuming light from top-left
    slant = 3.0
    nx = -gx * slant
    ny = -gy * slant
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx_norm = (nx / norm).astype(np.float32)
    ny_norm = (ny / norm).astype(np.float32)
    return np.stack([nx_norm, ny_norm], axis=-1)  # (256, 256, 2)


class TrestleElevationDataset(Dataset):
    """PyTorch Dataset yielding 6-channel features and residual elevation targets."""

    def __init__(
        self,
        samples: List[Dict[str, any]],
        augment: bool = False,
    ):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        item = self.samples[idx]
        rgb = item["rgb"].copy()                # (256, 256, 3)
        z_trestle = item["z_trestle"].copy()    # (257, 257) yards
        z_gt = item["z_gt"].copy()              # (257, 257) yards
        delta_z = (z_gt - z_trestle).copy()     # (257, 257) yards
        stem = item["stem"]

        if self.augment:
            # Horizontal flip
            if np.random.rand() > 0.5:
                rgb = np.fliplr(rgb).copy()
                z_trestle = np.fliplr(z_trestle).copy()
                z_gt = np.fliplr(z_gt).copy()
                delta_z = np.fliplr(delta_z).copy()

            # Vertical flip
            if np.random.rand() > 0.5:
                rgb = np.flipud(rgb).copy()
                z_trestle = np.flipud(z_trestle).copy()
                z_gt = np.flipud(z_gt).copy()
                delta_z = np.flipud(delta_z).copy()

            # Random brightness/contrast jitter
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.92, 1.08)
                contrast = np.random.uniform(0.92, 1.08)
                rgb = np.clip((rgb - 0.5) * contrast + 0.5 * brightness, 0.0, 1.0)

        # Compute photometric normals
        normals = compute_photometric_normals(rgb)  # (256, 256, 2)

        # Interpolate z_trestle to 256x256 for network input
        z_trestle_256 = ndimage.zoom(z_trestle, (256.0 / 257.0, 256.0 / 257.0), order=1)[:256, :256]
        norm_trestle = ((z_trestle_256 - 150.0) / 350.0).astype(np.float32)

        # Assemble 6-channel feature tensor: RGB (3) + norm_trestle (1) + normals (2)
        features = np.concatenate(
            [
                rgb,  # (256, 256, 3)
                norm_trestle[..., None],  # (256, 256, 1)
                normals,  # (256, 256, 2)
            ],
            axis=-1,
        )  # (256, 256, 6)

        # Bounds: [Z_min, Z_max]
        bounds = np.array([float(np.min(z_gt)), float(np.max(z_gt))], dtype=np.float32)

        return {
            "features": torch.from_numpy(features).permute(2, 0, 1).float(),  # (6, 256, 256)
            "z_trestle": torch.from_numpy(z_trestle).float(),                # (257, 257)
            "z_gt": torch.from_numpy(z_gt).float(),                          # (257, 257)
            "delta_z": torch.from_numpy(delta_z).float(),                    # (257, 257)
            "bounds": torch.from_numpy(bounds).float(),                      # (2,)
            "stem": stem,
        }


def build_or_load_trestle_corpus(
    cache_path: Path | str,
    base_corpus_path: Path | str,
    synth_wdl_path: Path | str,
    force_rebuild: bool = False,
) -> Tuple[List[Dict[str, any]], List[Dict[str, any]]]:
    """Builds or loads cached dataset of (features, z_trestle, z_gt, delta_z)."""
    cache = Path(cache_path)
    if cache.is_file() and not force_rebuild:
        print(f"[TrestleCorpus] Loading cached corpus from {cache}...")
        data = np.load(cache, allow_pickle=True)
        train_samples = list(data["train"])
        val_samples = list(data["val"])
        print(f"[TrestleCorpus] Loaded {len(train_samples)} train and {len(val_samples)} val samples.")
        return train_samples, val_samples

    print(f"[TrestleCorpus] Building dataset from {base_corpus_path} and {synth_wdl_path}...")
    base_data = np.load(base_corpus_path, allow_pickle=True)
    raw_train = list(base_data["train"])
    raw_val = list(base_data["val"])

    synth = TrestleWdlSynthesizer.load_npz(synth_wdl_path)

    def process_samples(raw_list: List[Dict[str, any]]) -> List[Dict[str, any]]:
        out = []
        for s in raw_list:
            stem = s["stem"]
            z_gt = s["z"].astype(np.float32)
            rgb = s["rgb"].astype(np.float32)

            # Authentic coarse 17x17 WDL lattice subsampled from ground truth (Blizzard MARE standard)
            h17 = z_gt[0::16, 0::16]
            z_trestle = ndimage.zoom(h17, (257.0 / 17.0, 257.0 / 17.0), order=3)[:257, :257].astype(np.float32)

            out.append(
                {
                    "stem": stem,
                    "rgb": rgb,
                    "z_gt": z_gt,
                    "z_trestle": z_trestle,
                }
            )
        return out

    train_samples = process_samples(raw_train)
    val_samples = process_samples(raw_val)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache,
        train=train_samples,
        val=val_samples,
    )
    print(f"[TrestleCorpus] Successfully cached {len(train_samples)} train and {len(val_samples)} val to {cache}.")
    return train_samples, val_samples
