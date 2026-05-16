"""V15Dataset — PyTorch Dataset for V15 terrain model.

Returns (minimap, height, normals, alpha, holes, liquid, object_weight).
Missing signals are returned as zero tensors and masked from loss.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _parse_key(data, key: str, default: np.ndarray | None = None) -> np.ndarray | None:
    if key in data:
        return data[key]
    return default


def _build_shard_index(
    shard_root: Path,
    validation_selection_path: Path,
) -> tuple[list[Path], list[Path]]:
    with open(validation_selection_path, encoding="utf-8") as f:
        selection = json.load(f)

    val_set: set[str] = set()
    for entry in selection.get("selections", []):
        p = Path(entry.get("path", ""))
        if len(p.parts) >= 3:
            val_set.add(str(Path(p.parts[-3]) / p.parts[-2] / p.parts[-1]))

    train_paths: list[Path] = []
    val_paths: list[Path] = []
    for npz_path in sorted(shard_root.glob("*/*/*.npz")):
        key = str(Path(npz_path.parts[-3]) / npz_path.parts[-2] / npz_path.parts[-1])
        (val_paths if key in val_set else train_paths).append(npz_path)
    return train_paths, val_paths


V15_REQUIRED = frozenset({"minimap_rgb_256", "height_257"})


class V15Dataset(Dataset):
    def __init__(
        self,
        shard_root: str | Path,
        validation_selection_path: str | Path,
        split: str = "train",
        max_samples: int | None = None,
        seed: int = 42,
        augment: bool = False,
    ) -> None:
        shard_root = Path(shard_root)
        validation_selection_path = Path(validation_selection_path)
        train, val = _build_shard_index(shard_root, validation_selection_path)
        self._paths = train if split == "train" else val
        self._eligible: list[int] = []
        self._max_samples = max_samples
        self._rng = random.Random(seed)
        self._augment = augment and split == "train"

    def _ensure_index(self) -> None:
        if self._eligible:
            return
        for i, p in enumerate(self._paths):
            try:
                with np.load(p) as data:
                    if "minimap_rgb_256" in data and "height_257" in data:
                        self._eligible.append(i)
            except OSError:
                continue
        if self._max_samples and len(self._eligible) > self._max_samples:
            self._eligible = sorted(self._rng.sample(self._eligible, self._max_samples))

    def __len__(self) -> int:
        self._ensure_index()
        return len(self._eligible)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        self._ensure_index()
        path = self._paths[self._eligible[idx]]
        with np.load(path, allow_pickle=False) as data:
            minimap = data["minimap_rgb_256"].astype(np.float32) / 255.0

            h = data["height_257"].astype(np.float32)
            h_mean = h.mean()
            h_std = h.std() + 1e-8
            height = (h - h_mean) / h_std

            nrm = _parse_key(data, "mcnr_normal_xyz")
            if nrm is not None:
                nrm = nrm.astype(np.float32)
                normal_mask = (np.abs(nrm).sum(axis=-1) > 1e-6).astype(np.float32)
                zero_mask = normal_mask < 0.5
                nrm[zero_mask] = [0.0, 0.0, 1.0]
                norms = np.linalg.norm(nrm, axis=-1, keepdims=True)
                norms = np.where(norms < 1e-6, 1.0, norms)
                nrm = nrm / norms
            else:
                normal_mask = np.zeros((257, 257), dtype=np.float32)
            has_normals = nrm is not None

            alp = _parse_key(data, "mcal_alpha_pack_256")
            if alp is None:
                alp = _parse_key(data, "mcal_alpha_pack")
            if alp is not None:
                alp = alp.astype(np.float32)
                if alp.shape[0] != 256:
                    alp = _downsample(alp, 256)
                if alp.max() > 1.5:
                    alp /= 255.0
                alp = np.clip(alp, 0, 1)
            has_alpha = alp is not None

            hol = _parse_key(data, "hole_mask_16")
            if hol is not None:
                hol = hol.astype(np.float32)
            has_holes = hol is not None

            liq = _parse_key(data, "unified_liquid_mask")
            if liq is not None:
                liq = liq.astype(np.float32)
                if liq.ndim == 3 and liq.shape[-1] == 1:
                    liq = liq.squeeze(-1)
                if liq.max() > 1.5:
                    liq /= 255.0
                liq = np.clip(liq, 0, 1)
            has_liquid = liq is not None

            mcly_ids = _parse_key(data, "mcly_texture_ids")
            mcly_mask = _parse_key(data, "mcly_layer_mask")
            has_mcly = mcly_ids is not None and mcly_mask is not None
            if has_mcly:
                mcly_ids = mcly_ids.astype(np.int64)
                mcly_ids = np.clip(mcly_ids, 0, 15)
                mcly_mask = mcly_mask.astype(np.float32)
            else:
                mcly_ids = np.zeros((16, 16, 4), dtype=np.int64)
                mcly_mask = np.zeros((16, 16, 4), dtype=np.float32)

            obj = _parse_key(data, "object_mask_257")
            if obj is not None:
                obj = obj.astype(np.float32)
            else:
                obj = np.zeros((257, 257), dtype=np.float32)

        if self._augment:
            xform = self._rng.randint(0, 7)
            if xform & 1:
                minimap = minimap[:, ::-1]
                height = height[:, ::-1]
                if nrm is not None:
                    nrm = nrm[:, ::-1]
                    nrm[..., 0] = -nrm[..., 0]
                    normal_mask = normal_mask[:, ::-1]
                else:
                    normal_mask = normal_mask[:, ::-1]
                if alp is not None:
                    alp = alp[:, ::-1]
                if hol is not None:
                    hol = hol[:, ::-1]
                liq = liq[:, ::-1] if liq is not None else None
                mcly_ids = mcly_ids[:, ::-1]
                mcly_mask = mcly_mask[:, ::-1]
                obj = obj[:, ::-1]
            if xform & 2:
                minimap = minimap[::-1]
                height = height[::-1]
                if nrm is not None:
                    nrm = nrm[::-1]
                    nrm[..., 1] = -nrm[..., 1]
                    normal_mask = normal_mask[::-1]
                else:
                    normal_mask = normal_mask[::-1]
                if alp is not None:
                    alp = alp[::-1]
                if hol is not None:
                    hol = hol[::-1]
                liq = liq[::-1] if liq is not None else None
                mcly_ids = mcly_ids[::-1]
                mcly_mask = mcly_mask[::-1]
                obj = obj[::-1]
            if xform & 4:
                minimap = np.rot90(minimap, k=1)
                height = np.rot90(height, k=1)
                if nrm is not None:
                    nrm = np.rot90(nrm, k=1)
                    old_x = nrm[..., 0].copy()
                    nrm[..., 0] = nrm[..., 1]
                    nrm[..., 1] = -old_x
                    normal_mask = np.rot90(normal_mask, k=1)
                else:
                    normal_mask = np.rot90(normal_mask, k=1)
                if alp is not None:
                    alp = np.rot90(alp, k=1)
                if hol is not None:
                    hol = np.rot90(hol, k=1)
                liq = np.rot90(liq, k=1) if liq is not None else None
                mcly_ids = np.rot90(mcly_ids, k=1)
                mcly_mask = np.rot90(mcly_mask, k=1)
                obj = np.rot90(obj, k=1)

        inp = torch.from_numpy(minimap.copy()).permute(2, 0, 1)
        hgt = torch.from_numpy(height.copy()).unsqueeze(0)
        nrm_t = torch.from_numpy(nrm.copy()).permute(2, 0, 1) if has_normals else torch.zeros(3, 257, 257)
        nm_mask = torch.from_numpy(normal_mask.copy()).unsqueeze(0)
        alp_t = torch.from_numpy(alp.copy()).permute(2, 0, 1) if has_alpha else torch.zeros(4, 256, 256)
        hol_t = torch.from_numpy(hol.copy()).unsqueeze(0) if has_holes else torch.zeros(1, 16, 16)
        liq_t = torch.from_numpy(liq.copy()).unsqueeze(0) if has_liquid else torch.zeros(1, 256, 256)
        mcly_t = torch.from_numpy(mcly_ids.copy())
        mcly_m = torch.from_numpy(mcly_mask.copy())
        wgt = torch.from_numpy(1.0 - obj).unsqueeze(0)

        return {
            "input": inp,
            "height": hgt,
            "normals": nrm_t,
            "normal_mask": nm_mask,
            "alpha": alp_t,
            "holes": hol_t,
            "liquid": liq_t,
            "mcly_ids": mcly_t,
            "mcly_mask": mcly_m,
            "weight": wgt,
            "has_normals": has_normals,
            "has_alpha": has_alpha,
            "has_holes": has_holes,
            "has_liquid": has_liquid,
            "has_mcly": has_mcly,
        }


def _downsample(arr: np.ndarray, size: int) -> np.ndarray:
    h = arr.shape[0]
    if h == size:
        return arr
    factor = h // size
    new_shape = (size, factor, size, factor) + arr.shape[2:]
    return arr.reshape(new_shape).mean(axis=1).mean(axis=2)