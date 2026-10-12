"""Zarr-Backed Clean Elevation Dataset & Felwood Indentation Supervision (Spec 269).

Extracts 6-channel feature tensors and residual targets directly from the unified v60
Zarr datastore or per-build Zarr stores, providing supervision on authentic Kalimdor
and Azeroth terrain with explicit recognition of Felwood canyons and classic indentations
under the strict V7 architecture.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
from scipy import ndimage
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)


def is_felwood_tile(map_name: str, tile_x: int, tile_y: int) -> bool:
    """True if coordinates correspond to the authentic Felwood canyon corridor in Kalimdor."""
    return map_name.strip().lower() == "kalimdor" and (25 <= tile_x <= 29) and (19 <= tile_y <= 26)


def is_classic_indentation_tile(map_name: str, tile_x: int, tile_y: int) -> bool:
    """True if coordinates fall within known classic high-relief indentation corridors.

    Includes Felwood chasms, Ashenvale ravines, Thousand Needles canyons, and Searing Gorge.
    """
    m = map_name.strip().lower()
    if m == "kalimdor":
        # Felwood corridor
        if 25 <= tile_x <= 29 and 19 <= tile_y <= 26:
            return True
        # Ashenvale gorges
        if 25 <= tile_x <= 32 and 27 <= tile_y <= 34:
            return True
        # Thousand Needles canyon chasms
        if 28 <= tile_x <= 35 and 45 <= tile_y <= 52:
            return True
    elif m == "azeroth":
        # Searing Gorge / Badlands rift
        if 33 <= tile_x <= 40 and 36 <= tile_y <= 44:
            return True
    return False


def compute_photometric_normals_from_rgb(rgb: np.ndarray) -> np.ndarray:
    """Compute 2-channel photometric surface normals (Nx, Ny) from 256x256 RGB minimap."""
    if rgb.max() > 1.0:
        rgb = rgb.astype(np.float32) / 255.0
    gray = 0.299 * rgb[..., 0] + 0.587 * rgb[..., 1] + 0.114 * rgb[..., 2]
    gray_smooth = ndimage.gaussian_filter(gray, sigma=1.0)
    gy, gx = np.gradient(gray_smooth)
    slant = 3.0
    nx = -gx * slant
    ny = -gy * slant
    nz = np.ones_like(nx)
    norm = np.sqrt(nx * nx + ny * ny + nz * nz)
    nx_norm = (nx / norm).astype(np.float32)
    ny_norm = (ny / norm).astype(np.float32)
    return np.stack([nx_norm, ny_norm], axis=-1)  # (256, 256, 2)


class ZarrElevationDataset(Dataset):
    """PyTorch Dataset yielding 6-channel V7 features and residual elevation targets from Zarr."""

    def __init__(
        self,
        samples: List[Dict[str, Any]],
        augment: bool = False,
    ):
        self.samples = samples
        self.augment = augment

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str | bool]:
        item = self.samples[idx]
        rgb = item["rgb"].copy().astype(np.float32)
        if rgb.max() > 1.0:
            rgb /= 255.0

        z_gt = item["z_gt"].copy().astype(np.float32)          # (257, 257) yards
        z_trestle = item["z_trestle"].copy().astype(np.float32)  # (257, 257) yards
        delta_z = (z_gt - z_trestle).copy()                     # (257, 257) yards
        stem = item["stem"]
        is_felwood = item.get("is_felwood", False)

        if self.augment:
            if np.random.rand() > 0.5:
                rgb = np.fliplr(rgb).copy()
                z_trestle = np.fliplr(z_trestle).copy()
                z_gt = np.fliplr(z_gt).copy()
                delta_z = np.fliplr(delta_z).copy()

            if np.random.rand() > 0.5:
                rgb = np.flipud(rgb).copy()
                z_trestle = np.flipud(z_trestle).copy()
                z_gt = np.flipud(z_gt).copy()
                delta_z = np.flipud(delta_z).copy()

            if np.random.rand() > 0.5:
                brightness = np.random.uniform(0.92, 1.08)
                contrast = np.random.uniform(0.92, 1.08)
                rgb = np.clip((rgb - 0.5) * contrast + 0.5 * brightness, 0.0, 1.0)

        # Photometric normals from minimap RGB
        normals = compute_photometric_normals_from_rgb(rgb)  # (256, 256, 2)

        # Interpolate z_trestle to 256x256 for network input channel
        z_trestle_256 = ndimage.zoom(z_trestle, (256.0 / 257.0, 256.0 / 257.0), order=1)[:256, :256]
        norm_trestle = ((z_trestle_256 - 150.0) / 350.0).astype(np.float32)

        # 6-channel input tensor: RGB (3) + norm_trestle (1) + normals (2)
        features = np.concatenate(
            [
                rgb,
                norm_trestle[..., None],
                normals,
            ],
            axis=-1,
        )  # (256, 256, 6)

        bounds = np.array([float(np.min(z_gt)), float(np.max(z_gt))], dtype=np.float32)

        return {
            "features": torch.from_numpy(features).permute(2, 0, 1).float(),  # (6, 256, 256)
            "z_trestle": torch.from_numpy(z_trestle).float(),                # (257, 257)
            "z_gt": torch.from_numpy(z_gt).float(),                          # (257, 257)
            "delta_z": torch.from_numpy(delta_z).float(),                    # (257, 257)
            "bounds": torch.from_numpy(bounds).float(),                      # (2,)
            "stem": stem,
            "is_felwood": is_felwood,
        }


def extract_zarr_samples(
    store_path: Path | str,
    target_maps: Optional[List[str]] = None,
    min_relief: float = 5.0,
    focus_indentations: bool = False,
    indentation_weight: int = 3,
) -> List[Dict[str, Any]]:
    """Extract clean elevation samples from a unified or per-build v60 Zarr store."""
    store = Path(store_path)
    if not store.exists():
        raise FileNotFoundError(f"Zarr store not found: {store}")

    index_file = store / "index.parquet"
    if not index_file.exists():
        sub_stores = sorted(store.glob("*.zarr"))
        if sub_stores:
            all_samples: List[Dict[str, Any]] = []
            for s in sub_stores:
                all_samples.extend(extract_zarr_samples(
                    store_path=s,
                    target_maps=target_maps,
                    focus_indentations=focus_indentations,
                    min_relief=min_relief,
                    indentation_weight=indentation_weight,
                ))
            return all_samples
        raise FileNotFoundError(f"index.parquet not found in {store}")

    tbl = pq.read_table(index_file)
    rows = tbl.to_pylist()

    g = zarr.open_group(str(store), mode="r")

    # Discover available height array
    height_key = None
    for candidate in ("ground_intent_height_257", "height_257"):
        if candidate in g:
            height_key = candidate
            break

    if height_key is None:
        raise ValueError(f"Neither 'ground_intent_height_257' nor 'height_257' found in {store}")

    rgb_key = "minimap_rgb" if "minimap_rgb" in g else "minimap_rgb_256"
    if rgb_key not in g:
        raise ValueError(f"minimap_rgb not found in {store}")

    is_dedup = "canonical" in getattr(g[height_key], "keys", lambda: [])()

    height_canonical = g[height_key]["canonical"] if is_dedup else g[height_key]
    height_row_idx = g[height_key]["row_index"] if is_dedup else None

    rgb_is_dedup = "canonical" in getattr(g[rgb_key], "keys", lambda: [])()
    rgb_canonical = g[rgb_key]["canonical"] if rgb_is_dedup else g[rgb_key]
    rgb_row_idx = g[rgb_key]["row_index"] if rgb_is_dedup else None

    target_maps_set = {m.lower() for m in target_maps} if target_maps else None

    samples: List[Dict[str, Any]] = []

    for row_id, r in enumerate(rows):
        map_name = str(r.get("map", ""))
        if target_maps_set and map_name.lower() not in target_maps_set:
            continue

        tx = int(r.get("tile_x", -1))
        ty = int(r.get("tile_y", -1))
        bid = str(r.get("build_id", "3_3_5_12340"))

        # Fetch height
        if is_dedup:
            ptr = int(height_row_idx[row_id])
            if ptr < 0 or ptr >= height_canonical.shape[0]:
                continue
            h_arr = np.asarray(height_canonical[ptr], dtype=np.float32)
        else:
            if row_id >= height_canonical.shape[0]:
                continue
            h_arr = np.asarray(height_canonical[row_id], dtype=np.float32)

        # Check vertical relief (skip flat/unauthored ocean tiles)
        relief = float(np.max(h_arr) - np.min(h_arr))
        if relief < min_relief:
            continue

        # Fetch minimap RGB
        if rgb_is_dedup:
            r_ptr = int(rgb_row_idx[row_id])
            if r_ptr < 0 or r_ptr >= rgb_canonical.shape[0]:
                continue
            rgb_arr = np.asarray(rgb_canonical[r_ptr], dtype=np.float32)
        else:
            if row_id >= rgb_canonical.shape[0]:
                continue
            rgb_arr = np.asarray(rgb_canonical[row_id], dtype=np.float32)

        # Ensure correct shape
        if h_arr.shape != (257, 257) or rgb_arr.shape[:2] != (256, 256):
            continue

        # Compute coarse 17x17 WDL lattice and upsample (Blizzard MARE WDL standard)
        h17 = h_arr[0::16, 0::16]
        z_trestle = ndimage.zoom(h17, (257.0 / 17.0, 257.0 / 17.0), order=3)[:257, :257].astype(np.float32)

        felwood = is_felwood_tile(map_name, tx, ty)
        classic_indent = is_classic_indentation_tile(map_name, tx, ty)

        sample = {
            "stem": f"{bid}_{map_name}_{tx}_{ty}",
            "rgb": rgb_arr,
            "z_gt": h_arr,
            "z_trestle": z_trestle,
            "tile_x": tx,
            "tile_y": ty,
            "map": map_name,
            "build_id": bid,
            "is_felwood": felwood,
            "is_indentation": classic_indent,
        }

        # Weight indentation and Felwood tiles higher if requested
        repeat_count = indentation_weight if (focus_indentations and classic_indent) else 1
        for _ in range(repeat_count):
            samples.append(sample)

    logger.info(
        "Extracted %d samples from %s (target_maps=%s, Felwood=%d)",
        len(samples),
        store_path,
        target_maps,
        sum(1 for s in samples if s.get("is_felwood")),
    )
    return samples


def build_or_load_zarr_elevation_corpus(
    zarr_store_path: Path | str,
    cache_path: Optional[Path | str] = None,
    target_maps: Optional[List[str]] = None,
    val_split: float = 0.1,
    focus_indentations: bool = True,
    force_rebuild: bool = False,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Builds or loads train/validation splits from a v60 Zarr datastore."""
    if cache_path is not None:
        c_path = Path(cache_path)
        if c_path.is_file() and not force_rebuild:
            logger.info("Loading cached Zarr elevation corpus from %s...", c_path)
            data = np.load(c_path, allow_pickle=True)
            return list(data["train"]), list(data["val"])

    samples = extract_zarr_samples(
        store_path=zarr_store_path,
        target_maps=target_maps,
        focus_indentations=focus_indentations,
    )

    if not samples:
        raise ValueError(f"No usable tiles found in {zarr_store_path}")

    # Deterministic train/val split based on tile coordinate hash
    train_samples: List[Dict[str, Any]] = []
    val_samples: List[Dict[str, Any]] = []

    for s in samples:
        h = hash(f"{s['map']}_{s['tile_x']}_{s['tile_y']}") % 1000
        if h < int(val_split * 1000):
            val_samples.append(s)
        else:
            train_samples.append(s)

    if cache_path is not None:
        c_path = Path(cache_path)
        c_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            c_path,
            train=train_samples,
            val=val_samples,
        )
        logger.info("Cached %d train and %d val samples to %s", len(train_samples), len(val_samples), c_path)

    return train_samples, val_samples
