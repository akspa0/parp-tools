"""Paste-aware dataset: loads tile arrays, crops to candidate bbox, resizes to standard dims."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import zarr
from torch.utils.data import Dataset

from harvester.v16_1_dataset import V161Dataset

_TARGET_256 = (256, 256)
_TARGET_257 = (257, 257)
_TARGET_16 = (16, 16)

# Keys grouped by target spatial resolution after crop+resize
_KEYS_257 = {
    "height_raw", "height_norm", "normals", "normal_mask",
    "weight_257", "mddf_mask", "modf_mask", "object_presence_257",
    "terrain_valid_mask_257",
}
_KEYS_16 = {
    "holes", "mcnk_flags_16", "weight_16", "mcly_any_16",
    "liquid_type_16", "liquid_type_valid_16",
}
_KEYS_256 = {
    "input", "alpha", "liquid_mask", "liquid_height",
    "weight_256", "alpha_painted_256",
}
_SCALAR_OR_PASSTHRU = {
    "height_mean", "height_std", "what_plate_flag",
    "alpha_painted_cov", "mcly_cov",
    "curation_quality_score", "curation_usefulness_score",
    "curation_difficulty_score", "curation_difficulty_rank",
    "curation_difficulty_bucket",
    "has_normals", "has_alpha", "has_holes", "has_liquid", "has_mcly",
    "meta_build", "meta_store", "meta_map",
    "meta_tile_id", "meta_tile_x", "meta_tile_y",
}

# HWC auxiliary arrays (channels last) — pass through without spatial crop
_HWC_PASSTHRU = {"mcly_ids", "mcly_mask"}


def _crop_and_resize(tensor: torch.Tensor, bbox: tuple[int, int, int, int],
                     target_size: tuple[int, int]) -> torch.Tensor:
    x0, y0, x1, y1 = bbox
    h, w = tensor.shape[-2:]
    scale_y = h / 256.0
    scale_x = w / 256.0
    cy0 = max(0, int(round(y0 * scale_y)))
    cy1 = min(h, max(cy0 + 1, int(round(y1 * scale_y))))
    cx0 = max(0, int(round(x0 * scale_x)))
    cx1 = min(w, max(cx0 + 1, int(round(x1 * scale_x))))
    cropped = tensor[..., cy0:cy1, cx0:cx1]
    if cropped.shape[-2:] == target_size:
        return cropped.contiguous()
    needs_unsqueeze = cropped.dim() == 2
    if needs_unsqueeze:
        cropped = cropped.unsqueeze(0).unsqueeze(0)
    elif cropped.dim() == 3:
        cropped = cropped.unsqueeze(0)
    resized = F.interpolate(cropped.float(), size=target_size, mode="bilinear" if cropped.shape[1] <= 32 else "nearest", align_corners=False)
    if needs_unsqueeze:
        resized = resized.squeeze(0).squeeze(0)
    elif cropped.dim() == 4:
        resized = resized.squeeze(0)
    return resized.to(dtype=tensor.dtype)


class PasteAwareDataset(Dataset):
    """Samples paste candidates instead of random tile positions.

    Wraps a V161Dataset (which loads full tiles from Zarr) and adds
    per-candidate cropping + resize to standard model input sizes.
    """

    def __init__(
        self,
        base_dataset: V161Dataset,
        paste_dir: str | Path,
    ) -> None:
        self.base = base_dataset
        paste_dir = Path(paste_dir)

        store = zarr.storage.LocalStore(str(paste_dir / "tile_to_pastes.zarr"), read_only=True)
        paste_root = zarr.open_group(store=store, mode="r")
        self.paste_tile_offset: np.ndarray = paste_root["tile_offset"][:]
        self.paste_bboxes: np.ndarray = paste_root["tile_local_bbox"][:]
        self.paste_candidate_idx: np.ndarray = paste_root["candidate_idx"][:]

        self.candidate_meta: dict[int, dict] = {}
        cand_path = paste_dir / "candidates.jsonl"
        if cand_path.exists():
            with open(cand_path) as f:
                for line in f:
                    c = json.loads(line)
                    self.candidate_meta[int(c["candidate_id"])] = c

        # Build flat list of (base_idx, slot) pairs and matching synthetic
        # index entries for the common trainer's sampler/metadata hooks.
        self.entries: list[tuple[int, int]] = []
        self._index_entries: list[dict] = []
        for i in range(len(base_dataset._index_entries)):
            entry = base_dataset._index_entries[i]
            tile_id = int(entry["tile_id"])
            if tile_id >= len(self.paste_tile_offset) - 1:
                continue
            start = int(self.paste_tile_offset[tile_id])
            end = int(self.paste_tile_offset[tile_id + 1])
            for slot in range(start, end):
                self.entries.append((i, slot))
                self._index_entries.append(dict(entry))

        if not self.entries:
            raise ValueError(
                f"No paste entries — did the miner produce tile_to_pastes.zarr "
                f"with matching tile_ids? (paste_dir={paste_dir})"
            )
        self._indices = list(range(len(self.entries)))

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict:
        base_idx, slot = self.entries[idx]
        sample = self.base[base_idx]
        x0, y0, x1, y1 = [int(v) for v in self.paste_bboxes[slot].tolist()]
        bbox = (x0, y0, x1, y1)

        result: dict = {}
        for k, v in sample.items():
            if k in _HWC_PASSTHRU:
                result[k] = v
            elif isinstance(v, torch.Tensor) and v.dim() >= 2:
                if k in _KEYS_257:
                    result[k] = _crop_and_resize(v, bbox, _TARGET_257)
                elif k in _KEYS_16:
                    result[k] = _crop_and_resize(v, bbox, _TARGET_16)
                elif k in _KEYS_256:
                    result[k] = _crop_and_resize(v, bbox, _TARGET_256)
                else:
                    result[k] = v
            else:
                result[k] = v

        # Attach candidate metadata
        cid = int(self.paste_candidate_idx[slot])
        meta = self.candidate_meta.get(cid, {})
        result["paste_score"] = torch.tensor(float(meta.get("score_mean", 0.0)), dtype=torch.float32)
        result["paste_area"] = torch.tensor(int(meta.get("component_area", 0)), dtype=torch.int32)
        result["paste_candidate_id"] = cid

        return result
