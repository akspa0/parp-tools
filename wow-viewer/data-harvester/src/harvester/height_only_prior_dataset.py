"""Spec 077 Phase 4 (US3) height-only terrain dataset.

Reads the spec 077 teacher-prior Zarr store as the model input and the
source V18 Zarr store for the authoritative ``height_257`` target plus the
``object_filtered_mask`` that gates the loss.

The dataset returns the exact ``HeightOnlyTrainingSample`` contract from
spec 077 data-model.md §3.1:

  * ``input_prior`` ``(C, 256, 256)`` float32 — suppressed RGB + mask + confidence
  * ``raw_minimap_rgb`` ``(3, 256, 256)`` float32 — unsuppressed minimap RGB for review
  * ``teacher_object_mask`` / ``teacher_object_confidence`` ``(1, 256, 256)`` float32 — deconstruction review bands
  * ``height_257`` ``(1, 257, 257)``  float32 — per-vertex terrain height
  * ``normal_xyz`` ``(3, 257, 257)`` float32 — optional V18 normal guidance target
  * ``normal_mask`` ``(1, 257, 257)`` float32 — optional normal-guidance validity mask
  * ``weight_257`` ``(1, 257, 257)``  float32 — terrain-valid weight (1.0 on
    terrain, 0.0 on object/filtered pixels)
  * ``meta_build`` / ``meta_map`` / ``meta_tile_id`` — provenance

Non-goals (spec 077 FR-013, FR-014, FR-023):
  * No normal, liquid, object, or other heads.
  * No shared-weight multitask loss.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import Dataset
import zarr
import zarr.storage

from harvester.terrain_augment import augment_sample, sample_transform


@dataclass(frozen=True)
class HeightOnlyPriorSample:
    input_prior: np.ndarray  # (C, 256, 256) float32 in [0, 1]
    height_257: np.ndarray   # (1, 257, 257) float32 in world units
    weight_257: np.ndarray   # (1, 257, 257) float32 in [0, 1]
    meta_build: str
    meta_map: str
    meta_tile_id: int


def _load_zarr_array(store_path: Path, key: str) -> np.ndarray | None:
    if not (store_path / key).exists():
        return None
    store = zarr.storage.LocalStore(str(store_path), read_only=True)
    root = zarr.open_group(store, mode="r")
    if key not in root:
        return None
    return np.asarray(root[key][:])


def _load_tiles_parquet(path: Path) -> list[dict]:
    if not path.exists():
        return []
    table = pq.read_table(str(path))
    return [
        {col: table.column(col)[idx].as_py() for col in table.column_names}
        for idx in range(table.num_rows)
    ]


def _nearest_resize(arr: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    h, w = arr.shape[-2], arr.shape[-1]
    if h == target_h and w == target_w:
        return arr
    ys = np.linspace(0, h - 1, target_h).astype(np.int64)
    xs = np.linspace(0, w - 1, target_w).astype(np.int64)
    if arr.ndim == 2:
        return arr[np.ix_(ys, xs)]
    return arr[..., np.ix_(ys, xs)[0], np.ix_(ys, xs)[1]]


class HeightOnlyPriorDataset(Dataset):
    """Height-only training/inference dataset over a teacher-prior Zarr store.

    Parameters
    ----------
    prior_path:
        Path to the teacher-prior ``<build>.zarr`` store written by
        ``build_teacher_prior_dataset.py``. Must contain
        ``processed_minimap_prior_256`` and ideally
        ``teacher_object_mask_256``.
    v18_path:
        Path to the source ``<build>.zarr`` V18 store. Used to read the
        authoritative ``height_257`` target and the
        ``object_filtered_mask`` weight. May be ``None`` for inference-only
        mode that only emits the prior inputs (target arrays are zeros).
    tile_filter:
        Optional iterable of ``tile_id`` values to restrict to. When
        ``None`` all available tiles are used.
    include_weight:
        If ``True`` (default), compute ``weight_257 = 1 - object_filtered_mask``.
        If the filtered mask is missing the weight is a constant 1.0.
    height_norm:
        If ``True`` (default), normalize ``height_257`` by its per-tile
        mean/std. The de-normalization is the caller's responsibility.
    augment:
        If ``True``, apply a random D4 (flip/rotate) transform to every
        spatial array in the returned sample. This is geometrically exact
        for terrain height (a scalar field) and tracks the
        ``[-dh/dx, -dh/dy, +1]`` normal convention so the normal-guidance
        target stays consistent. Intended for the train split only; leave
        ``False`` for validation so val loss is deterministic.
    augment_seed:
        Seed for the per-sample augmentation RNG. Ignored when
        ``augment`` is ``False``.
    """

    def __init__(
        self,
        prior_path: str | Path,
        v18_path: str | Path | None = None,
        tile_filter: list[int] | None = None,
        include_weight: bool = True,
        height_norm: bool = True,
        augment: bool = False,
        augment_seed: int = 0,
    ) -> None:
        self.prior_path = Path(prior_path)
        self.v18_path = Path(v18_path) if v18_path is not None else None
        self.include_weight = include_weight
        self.height_norm = height_norm
        self.augment = bool(augment)
        self._augment_rng = np.random.default_rng(int(augment_seed))

        prior_tensor = _load_zarr_array(self.prior_path, "processed_minimap_prior_256")
        if prior_tensor is None or prior_tensor.size == 0:
            raise ValueError(
                f"processed_minimap_prior_256 missing or empty under {self.prior_path}"
            )
        self.prior_tensor = prior_tensor  # (N, 256, 256, C)
        if self.prior_tensor.shape[1] != 256 or self.prior_tensor.shape[2] != 256:
            raise ValueError(
                f"Expected prior tensor shape (N, 256, 256, C); got {self.prior_tensor.shape}"
            )

        teacher_mask = _load_zarr_array(self.prior_path, "teacher_object_mask_256")
        self.teacher_mask = teacher_mask  # (N, 256, 256) uint8, may be None
        teacher_confidence = _load_zarr_array(self.prior_path, "teacher_object_confidence_256")
        self.teacher_confidence = teacher_confidence  # (N, 256, 256) uint8, may be None
        raw_minimap = _load_zarr_array(self.prior_path, "raw_minimap_rgb_256")
        self.raw_minimap = raw_minimap  # (N, 256, 256, 3) uint8, may be None

        self.height_257 = None
        self.normal_xyz = None
        self.normal_mask = None
        self.weight_257 = None
        if self.v18_path is not None and self.v18_path.exists():
            self.height_257 = _load_zarr_array(self.v18_path, "height_257")
            self.normal_xyz = _load_zarr_array(self.v18_path, "normal_xyz")
            self.normal_mask = _load_zarr_array(self.v18_path, "normal_mask")
            if include_weight:
                filtered = _load_zarr_array(self.v18_path, "object_filtered_mask")
                if filtered is not None and filtered.size:
                    self.weight_257 = (1.0 - np.clip(filtered, 0.0, 1.0)).astype(np.float32)
                else:
                    self.weight_257 = np.ones_like(
                        self.height_257, dtype=np.float32
                    ) if self.height_257 is not None else None

        # Build a tile_id -> prior-row-index lookup. The teacher-prior Zarr
        # was written in the same order as the V18 index, so the row index
        # equals the source tile_id unless a start-tile-id offset was used.
        all_tile_meta = _load_tiles_parquet(self.prior_path / "tiles.parquet")
        self._tile_id_to_index: dict[int, int] = {}
        for i, row in enumerate(all_tile_meta):
            self._tile_id_to_index[int(row.get("tile_id", i))] = i
        self.tile_meta = all_tile_meta
        if tile_filter is not None:
            keep = set(int(t) for t in tile_filter)
            self.tile_meta = [t for t in self.tile_meta if int(t.get("tile_id", -1)) in keep]

    def __len__(self) -> int:
        return len(self.tile_meta) if self.tile_meta else self.prior_tensor.shape[0]

    def _resolve_index(self, idx: int) -> tuple[int, dict]:
        if self.tile_meta:
            row = self.tile_meta[idx]
            tile_id = int(row.get("tile_id", idx))
            prior_index = self._tile_id_to_index.get(tile_id, idx)
            return prior_index, row
        return idx, {"build": self.prior_path.stem, "map": "", "tile_id": idx}

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        prior_index, row = self._resolve_index(idx)
        source_tile_id = int(row.get("tile_id", prior_index))
        prior = self.prior_tensor[prior_index]  # (256, 256, 5) uint8
        prior_chw = np.transpose(prior.astype(np.float32) / 255.0, (2, 0, 1))  # (5, 256, 256)
        if self.raw_minimap is not None and prior_index < self.raw_minimap.shape[0]:
            raw = self.raw_minimap[prior_index].astype(np.float32) / 255.0
            raw_chw = np.transpose(raw, (2, 0, 1))
        else:
            raw_chw = prior_chw[:3].copy()

        if self.teacher_mask is not None and prior_index < self.teacher_mask.shape[0]:
            teacher_mask = self.teacher_mask[prior_index].astype(np.float32)[None, :, :]
            if float(teacher_mask.max(initial=0.0)) > 1.0:
                teacher_mask = teacher_mask / 255.0
        else:
            teacher_mask = prior_chw[3:4].copy()

        if self.teacher_confidence is not None and prior_index < self.teacher_confidence.shape[0]:
            teacher_confidence = self.teacher_confidence[prior_index].astype(np.float32)[None, :, :]
            if float(teacher_confidence.max(initial=0.0)) > 1.0:
                teacher_confidence = teacher_confidence / 255.0
        else:
            teacher_confidence = prior_chw[4:5].copy()

        if self.height_257 is not None and source_tile_id < self.height_257.shape[0]:
            h = self.height_257[source_tile_id].astype(np.float32)  # (257, 257)
            if self.height_norm:
                h_mean = float(h.mean())
                h_std = float(h.std()) + 1e-6
                h = (h - h_mean) / h_std
            height = h[None, :, :]
        else:
            height = np.zeros((1, 257, 257), dtype=np.float32)

        if self.normal_xyz is not None and source_tile_id < self.normal_xyz.shape[0]:
            n = self.normal_xyz[source_tile_id].astype(np.float32)  # (257, 257, 3)
            if n.ndim == 3 and n.shape[-1] == 3:
                norm = np.linalg.norm(n, axis=2, keepdims=True)
                normal = np.transpose(n / np.clip(norm, 1e-8, None), (2, 0, 1))
            else:
                normal = np.zeros((3, 257, 257), dtype=np.float32)
        else:
            normal = np.zeros((3, 257, 257), dtype=np.float32)

        if self.normal_mask is not None and source_tile_id < self.normal_mask.shape[0]:
            nm = np.clip(self.normal_mask[source_tile_id].astype(np.float32), 0.0, 1.0)
            normal_mask = nm[None, :, :]
        elif self.normal_xyz is not None and source_tile_id < self.normal_xyz.shape[0]:
            normal_mask = np.ones((1, 257, 257), dtype=np.float32)
        else:
            normal_mask = np.zeros((1, 257, 257), dtype=np.float32)

        if self.weight_257 is not None and source_tile_id < self.weight_257.shape[0]:
            w = self.weight_257[source_tile_id]  # (257, 257)
            weight = w[None, :, :]
        elif self.height_257 is None:
            # In inference-only mode there is no authoritative target and no
            # valid loss surface, so emit a zero weight map.
            weight = np.zeros((1, 257, 257), dtype=np.float32)
        else:
            weight = np.ones((1, 257, 257), dtype=np.float32)

        sample = {
            "input_prior": prior_chw,
            "raw_minimap_rgb": raw_chw,
            "teacher_object_mask": teacher_mask,
            "teacher_object_confidence": teacher_confidence,
            "height_257": height,
            "normal_xyz": normal,
            "normal_mask": normal_mask,
            "weight_257": weight,
            "meta_build": str(row.get("build", self.prior_path.stem)),
            "meta_map": str(row.get("map_name", row.get("map", ""))),
            "meta_tile_id": source_tile_id,
            "meta_prior_row": prior_index,
            "meta_v18_row": source_tile_id,
        }
        if self.augment:
            transform = sample_transform(self._augment_rng)
            sample = augment_sample(sample, transform)
        return {
            "input_prior": torch.from_numpy(np.ascontiguousarray(sample["input_prior"])),
            "raw_minimap_rgb": torch.from_numpy(np.ascontiguousarray(sample["raw_minimap_rgb"])),
            "teacher_object_mask": torch.from_numpy(np.ascontiguousarray(sample["teacher_object_mask"])),
            "teacher_object_confidence": torch.from_numpy(np.ascontiguousarray(sample["teacher_object_confidence"])),
            "height_257": torch.from_numpy(np.ascontiguousarray(sample["height_257"])),
            "normal_xyz": torch.from_numpy(np.ascontiguousarray(sample["normal_xyz"])),
            "normal_mask": torch.from_numpy(np.ascontiguousarray(sample["normal_mask"])),
            "weight_257": torch.from_numpy(np.ascontiguousarray(sample["weight_257"])),
            "meta_build": sample["meta_build"],
            "meta_map": sample["meta_map"],
            "meta_tile_id": sample["meta_tile_id"],
            "meta_prior_row": sample["meta_prior_row"],
            "meta_v18_row": sample["meta_v18_row"],
        }


def dataset_summary(prior_path: str | Path) -> dict:
    """Return a small JSON-friendly summary of a teacher-prior store."""
    prior_path = Path(prior_path)
    store = zarr.storage.LocalStore(str(prior_path), read_only=True)
    root = zarr.open_group(store, mode="r")
    summary = {
        "path": str(prior_path),
        "schema": dict(root.attrs).get("schema", ""),
        "build": dict(root.attrs).get("build", prior_path.stem),
        "arrays": sorted(root.array_keys()),
    }
    return summary
