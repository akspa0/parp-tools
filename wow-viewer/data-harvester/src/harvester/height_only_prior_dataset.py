"""Spec 077 Phase 4 (US3) height-only terrain dataset.

Reads the spec 077 teacher-prior Zarr store as the model input and the
source V18 Zarr store for the authoritative ``height_257`` target plus the
best available object mask that gates the loss.

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
  * ``albedo_rgb`` ``(3, 256, 256)`` float32 — optional texture-identity
    guidance channel derived from MCAL alpha via the compositor. Present
    only when ``include_albedo=True``. Orthogonal to the suppressed-minimap
    RGB: it encodes which terrain layer each pixel belongs to, not the
    appearance. A plain image under the selected augmentation policy.
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

from harvester.compositor import composite_texture_identity_albedo
from harvester.terrain_augment import (
    SHADOW_SAFE_TRANSFORMS,
    TransformId,
    augment_sample,
    sample_transform,
)


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
    if arr.ndim == 3 and arr.shape[-1] <= 16:
        h, w = arr.shape[0], arr.shape[1]
        if h == target_h and w == target_w:
            return arr
        ys = np.linspace(0, h - 1, target_h).astype(np.int64)
        xs = np.linspace(0, w - 1, target_w).astype(np.int64)
        return arr[ys[:, None], xs[None, :], :]
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
        ``object_precise_mask`` / fallback object weight. May be ``None`` for inference-only
        mode that only emits the prior inputs (target arrays are zeros).
    albedo_path:
        Optional sidecar ``<build>.zarr`` store containing precomputed
        ``albedo_rgb_256``. When supplied with ``include_albedo=True``, this
        is preferred over lazy compositing from V18 ``alpha_256``.
    tile_filter:
        Optional iterable of ``tile_id`` values to restrict to. When
        ``None`` all available tiles are used.
    include_weight:
        If ``True`` (default), compute ``weight_257`` from the best available
        object gate: `object_precise_mask`, then `object_filtered_mask`, then
        `object_mask`. If every mask is missing the weight is a constant 1.0.
    height_norm:
        If ``True`` (default), normalize ``height_257`` by its per-tile
        mean/std. The de-normalization is the caller's responsibility.
    augment:
        If ``True``, sample from ``augment_transforms`` and apply that
        transform to every spatial array in the returned sample. The default
        transform set is shadow-safe identity only. Explicit D4 transforms
        are geometrically exact for terrain height (a scalar field) and track
        the ``[-dh/dx, -dh/dy, +1]`` normal convention, but are not canonical
        for baked minimap RGB. Intended for the train split only; leave
        ``False`` for validation so val loss is deterministic.
    augment_seed:
        Seed for the per-sample augmentation RNG. Ignored when
        ``augment`` is ``False``.
    augment_transforms:
        Allowed geometric transforms when ``augment`` is ``True``. Defaults
        to ``SHADOW_SAFE_TRANSFORMS`` (identity only) because baked minimap
        RGB has fixed-direction terrain lighting/shadows; D4 augmentation is
        available only when explicitly supplied by experiments that do not
        rely on orientation-sensitive appearance.
    include_albedo:
        If ``True``, include a per-tile ``albedo_rgb`` ``(3, 256, 256)``
        guidance channel. Precomputed ``albedo_rgb_256`` from
        ``albedo_path`` is preferred; otherwise the dataset falls back to
        deriving albedo from V18 ``alpha_256`` plus MCLY texture IDs via
        ``compositor.composite_texture_identity_albedo``. The albedo encodes
        terrain-layer identity and is orthogonal to the suppressed-minimap
        RGB. Tiles without precomputed albedo or alpha data emit zeros. The
        channel follows the selected augmentation transform like any other
        image.
    """

    def __init__(
        self,
        prior_path: str | Path,
        v18_path: str | Path | None = None,
        albedo_path: str | Path | None = None,
        tile_filter: list[int] | None = None,
        include_weight: bool = True,
        height_norm: bool = True,
        augment: bool = False,
        augment_seed: int = 0,
        augment_transforms: tuple[TransformId, ...] | None = None,
        include_albedo: bool = False,
    ) -> None:
        self.prior_path = Path(prior_path)
        self.v18_path = Path(v18_path) if v18_path is not None else None
        self.albedo_path = Path(albedo_path) if albedo_path is not None else None
        self.include_weight = include_weight
        self.height_norm = height_norm
        self.augment = bool(augment)
        self._augment_rng = np.random.default_rng(int(augment_seed))
        self.augment_transforms = (
            tuple(augment_transforms)
            if augment_transforms is not None
            else SHADOW_SAFE_TRANSFORMS
        )
        if not self.augment_transforms:
            raise ValueError("augment_transforms must contain at least one transform")
        self.include_albedo = bool(include_albedo)

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
        self.weight_mask_source = "none"
        self.alpha_256 = None
        self.mcly_texture_ids = None
        self.mcly_layer_mask = None
        self.albedo_rgb_256 = None
        if self.include_albedo and self.albedo_path is not None and self.albedo_path.exists():
            self.albedo_rgb_256 = _load_zarr_array(self.albedo_path, "albedo_rgb_256")
            if self.albedo_rgb_256 is None:
                print(
                    f"include_albedo=True but no albedo_rgb_256 array under {self.albedo_path}; "
                    "falling back to V18 alpha_256 if available.",
                    flush=True,
                )
        if self.v18_path is not None and self.v18_path.exists():
            self.height_257 = _load_zarr_array(self.v18_path, "height_257")
            self.normal_xyz = _load_zarr_array(self.v18_path, "normal_xyz")
            self.normal_mask = _load_zarr_array(self.v18_path, "normal_mask")
            if include_weight:
                object_gate = None
                for key in ("object_precise_mask", "object_filtered_mask", "object_mask"):
                    candidate = _load_zarr_array(self.v18_path, key)
                    if candidate is not None and candidate.size:
                        object_gate = candidate
                        self.weight_mask_source = key
                        break
                if object_gate is not None:
                    self.weight_257 = (1.0 - np.clip(object_gate, 0.0, 1.0)).astype(np.float32)
                else:
                    self.weight_257 = np.ones_like(
                        self.height_257, dtype=np.float32
                    ) if self.height_257 is not None else None
            if self.include_albedo and self.albedo_rgb_256 is None:
                precomputed_in_v18 = _load_zarr_array(self.v18_path, "albedo_rgb_256")
                if precomputed_in_v18 is not None:
                    self.albedo_rgb_256 = precomputed_in_v18
            if self.include_albedo and self.albedo_rgb_256 is None:
                self.alpha_256 = _load_zarr_array(self.v18_path, "alpha_256")
                self.mcly_texture_ids = _load_zarr_array(self.v18_path, "mcly_texture_ids")
                self.mcly_layer_mask = _load_zarr_array(self.v18_path, "mcly_layer_mask")
                if self.alpha_256 is not None:
                    # Normalize to [0, 1] float32 if stored as uint8 or >1.
                    if self.alpha_256.dtype != np.float32:
                        self.alpha_256 = self.alpha_256.astype(np.float32)
                    if float(self.alpha_256.max(initial=0.0)) > 1.5:
                        self.alpha_256 = self.alpha_256 / 255.0
                    self.alpha_256 = np.clip(self.alpha_256, 0.0, 1.0)
                else:
                    print(
                        f"include_albedo=True but no alpha_256 array under {self.v18_path}; "
                        "albedo_rgb will be zeros.",
                        flush=True,
                    )

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

        albedo_chw = None
        if self.include_albedo:
            if self.albedo_rgb_256 is not None and source_tile_id < self.albedo_rgb_256.shape[0]:
                albedo_tile = self.albedo_rgb_256[source_tile_id].astype(np.float32)
                if albedo_tile.shape[0] != 256 or albedo_tile.shape[1] != 256:
                    albedo_tile = _nearest_resize(albedo_tile, 256, 256)
                if float(albedo_tile.max(initial=0.0)) > 1.5:
                    albedo_tile = albedo_tile / 255.0
                albedo_tile = np.clip(albedo_tile, 0.0, 1.0)
                albedo_chw = np.transpose(albedo_tile, (2, 0, 1)).astype(np.float32)
            elif self.alpha_256 is not None and source_tile_id < self.alpha_256.shape[0]:
                alpha_tile = self.alpha_256[source_tile_id].astype(np.float32)
                # alpha_256 is stored at (256, 256, 4) or (128, 128, 4).
                # Resize to 256x256 if needed, then composite.
                if alpha_tile.shape[0] != 256 or alpha_tile.shape[1] != 256:
                    alpha_tile = _nearest_resize(alpha_tile, 256, 256)
                tex_tile = None
                layer_mask_tile = None
                if self.mcly_texture_ids is not None and source_tile_id < self.mcly_texture_ids.shape[0]:
                    tex_tile = self.mcly_texture_ids[source_tile_id].astype(np.int32)
                if self.mcly_layer_mask is not None and source_tile_id < self.mcly_layer_mask.shape[0]:
                    layer_mask_tile = self.mcly_layer_mask[source_tile_id].astype(np.float32)
                albedo_hwc = composite_texture_identity_albedo(alpha_tile, tex_tile, layer_mask_tile)  # (256, 256, 3)
                albedo_chw = np.transpose(albedo_hwc, (2, 0, 1)).astype(np.float32)
            else:
                albedo_chw = np.zeros((3, 256, 256), dtype=np.float32)

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
            "meta_weight_mask_source": self.weight_mask_source,
        }
        if albedo_chw is not None:
            sample["albedo_rgb"] = albedo_chw
        if self.augment:
            transform = sample_transform(self._augment_rng, self.augment_transforms)
            sample = augment_sample(sample, transform)
        out = {
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
            "meta_weight_mask_source": sample["meta_weight_mask_source"],
        }
        if "albedo_rgb" in sample:
            out["albedo_rgb"] = torch.from_numpy(np.ascontiguousarray(sample["albedo_rgb"]))
        return out


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
