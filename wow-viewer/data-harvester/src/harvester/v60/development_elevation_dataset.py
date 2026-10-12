"""Authentic Development Elevation Dataset Builder & Loader (Spec 264).

Loads and caches the 270 authentic sculpted development tiles pairing 256x256 minimap RGB
with exact 257x257 ground truth elevations in physical yards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from harvester.v60.development_ground_truth import DevelopmentGroundTruthExtractor


class DevelopmentElevationDataset(Dataset):
    """PyTorch Dataset of authentic development minimap-to-elevation pairs."""

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
        rgb = item["rgb"]  # (256, 256, 3) float32 in [0, 1]
        z = item["z"]      # (257, 257) float32 in yards
        stem = item["stem"]

        if self.augment:
            # Random horizontal flip
            if np.random.rand() > 0.5:
                rgb = np.fliplr(rgb).copy()
                z = np.fliplr(z).copy()
            # Random subtle brightness/contrast jitter
            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.9, 1.1)
                contrast = np.random.uniform(0.9, 1.1)
                rgb = np.clip((rgb - 0.5) * contrast + 0.5 * brightness, 0.0, 1.0)

        # Transpose to (C, H, W)
        rgb_t = torch.from_numpy(rgb).permute(2, 0, 1).float()
        z_t = torch.from_numpy(z).float()

        return {
            "rgb": rgb_t,
            "z": z_t,
            "stem": stem,
        }


def build_or_load_development_corpus(
    cache_path: Path,
    maps_dir: Path,
    textures_dir: Path,
    min_relief_span: float = 2.0,
    force_rebuild: bool = False,
) -> Tuple[List[Dict[str, np.ndarray | str]], List[Dict[str, np.ndarray | str]]]:
    """Builds or loads cached dataset of authentic development tiles."""
    cache_path = Path(cache_path)
    if cache_path.is_file() and not force_rebuild:
        print(f"Loading cached elevation corpus from {cache_path}...")
        data = np.load(cache_path, allow_pickle=True)
        train_samples = list(data["train"])
        val_samples = list(data["val"])
        print(f"Loaded {len(train_samples)} train and {len(val_samples)} val samples from cache.")
        return train_samples, val_samples

    print(f"Scanning authentic development tiles in {maps_dir}...")
    ext = DevelopmentGroundTruthExtractor(maps_dir, textures_dir)
    adts = [
        p for p in maps_dir.glob("development_*.adt")
        if not p.name.endswith("_obj0.adt") and not p.name.endswith("_tex0.adt")
    ]

    valid_samples: List[Dict[str, np.ndarray | str]] = []
    for a in sorted(adts):
        stem = a.stem
        png = textures_dir / f"{stem}.png"
        if not png.is_file():
            continue

        try:
            h, is_sculpted = ext.extract_height_257(a)
            if not is_sculpted or np.ptp(h) < min_relief_span:
                continue

            img = Image.open(png).convert("RGB").resize((256, 256), Image.Resampling.LANCZOS)
            rgb = np.array(img, dtype=np.float32) / 255.0

            valid_samples.append({
                "stem": stem,
                "rgb": rgb,
                "z": h.astype(np.float32),
            })
        except Exception as e:
            print(f"Warning: skipping {stem} due to error: {e}")

    print(f"Discovered {len(valid_samples)} valid sculpted development tiles with minimap PNGs.")

    # Held-out validation split (15%)
    # Use deterministic shuffle so validation set is fixed
    rng = np.random.RandomState(42)
    indices = np.arange(len(valid_samples))
    rng.shuffle(indices)

    n_val = max(10, int(len(valid_samples) * 0.15))
    val_idx = set(indices[:n_val])

    train_samples = [valid_samples[i] for i in range(len(valid_samples)) if i not in val_idx]
    val_samples = [valid_samples[i] for i in range(len(valid_samples)) if i in val_idx]

    # Cache to disk for instant subsequent loads
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path,
        train=train_samples,
        val=val_samples,
    )
    print(f"Cached {len(train_samples)} train and {len(val_samples)} val tiles to {cache_path}.")

    return train_samples, val_samples
