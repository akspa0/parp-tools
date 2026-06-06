"""V16.1 dataset helpers built on the V16 Zarr corpus contract.

This keeps the V16 storage format as the truth surface while exposing the extra
signals needed by the split-and-link V16.1 model family.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
import zarr
import zarr.storage
from torch.utils.data import Dataset

from harvester.v16_curation import alpha_painted, is_blank_what_plate, load_curation_index, mcly_painted_coverage


def _build_split_indices(n_items: int, split: str, val_fraction: float, seed: int) -> list[int]:
    n_val = int(n_items * val_fraction)
    rng = np.random.RandomState(seed)
    indices = rng.permutation(n_items)
    if split == "val":
        return sorted(indices[:n_val].tolist())
    return sorted(indices[n_val:].tolist())


def _flags_to_liquid_type(flags_16: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Convert MCNK flags into a coarse liquid-type grid.

    Classes:
      0 none
      1 water
      2 ocean
      3 magma
      4 slime
    """

    flags = flags_16.astype(np.int32, copy=False)
    out = np.zeros(flags.shape, dtype=np.int64)
    valid = ((flags & 0x3C) != 0).astype(np.float32)
    out[(flags & 0x04) != 0] = 1
    out[(flags & 0x08) != 0] = 2
    out[(flags & 0x10) != 0] = 3
    out[(flags & 0x20) != 0] = 4
    return out, valid


def _crop_257_to_256(x: np.ndarray) -> np.ndarray:
    return x[:256, :256]


def _downsample_256_to_16(x: np.ndarray) -> np.ndarray:
    arr = x[:256, :256]
    reshaped = arr.reshape(16, 16, 16, 16)
    return reshaped.mean(axis=(1, 3)).astype(np.float32, copy=False)


def compose_terrain_valid_mask_257(
    *,
    normal_mask_257: np.ndarray,
    object_presence_257: np.ndarray,
    liquid_mask_256: np.ndarray,
    object_roof_weight_257: np.ndarray | None = None,
    what_plate: bool = False,
) -> np.ndarray:
    terrain_valid_257 = normal_mask_257.astype(np.float32, copy=True)
    terrain_valid_257 *= 1.0 - np.clip(object_presence_257.astype(np.float32, copy=False), 0.0, 1.0)
    if object_roof_weight_257 is not None:
        terrain_valid_257 *= np.clip(object_roof_weight_257.astype(np.float32, copy=False), 0.0, 1.0)
    liquid_mask_257 = np.pad(liquid_mask_256.astype(np.float32, copy=False), ((0, 1), (0, 1)), mode="edge")
    terrain_valid_257 *= 1.0 - (0.85 * np.clip(liquid_mask_257, 0.0, 1.0))
    if what_plate:
        terrain_valid_257[...] = 0.0
    return terrain_valid_257


def compose_object_loss_weights_257(
    *,
    object_presence_257: np.ndarray,
    object_roof_weight_257: np.ndarray | None = None,
) -> np.ndarray:
    weight_257 = 1.0 - np.clip(object_presence_257.astype(np.float32, copy=False), 0.0, 1.0)
    if object_roof_weight_257 is not None:
        weight_257 *= np.clip(object_roof_weight_257.astype(np.float32, copy=False), 0.0, 1.0)
    return weight_257.astype(np.float32, copy=False)


def _interpolate_checkerboard_normals(normals: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Fill MCNR checkerboard gaps by averaging valid cardinal neighbors.

    MCNR stores per-vertex normals on a checkerboard grid: positions where
    x%2 == y%2 are valid, positions where x%2 != y%2 are gaps (zero).
    Cardinal neighbors of a gap are always valid, so we average them and
    renormalize to unit length.
    """
    result = normals.copy()
    new_mask = mask.copy()
    h, w = normals.shape[:2]

    # Positions that need interpolation (currently zero / masked out)
    gaps = ~mask.astype(bool)

    # Accumulate valid cardinal neighbors
    neighbor_sum = np.zeros_like(normals)
    neighbor_count = np.zeros((h, w), dtype=np.int32)

    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = np.roll(normals, (dy, dx), axis=(0, 1))
        shifted_mask = np.roll(mask, (dy, dx), axis=(0, 1))
        # Zero out wrapped edges so we don't pull data from the opposite side
        if dy == -1:
            shifted_mask[0, :] = False
        elif dy == 1:
            shifted_mask[-1, :] = False
        if dx == -1:
            shifted_mask[:, 0] = False
        elif dx == 1:
            shifted_mask[:, -1] = False

        valid = shifted_mask.astype(bool)
        neighbor_sum[valid] += shifted[valid]
        neighbor_count[valid] += 1

    # Interpolate where we have at least one valid neighbor
    interp = gaps & (neighbor_count > 0)
    interp_vecs = neighbor_sum[interp]
    interp_mags = np.linalg.norm(interp_vecs, axis=-1, keepdims=True)
    interp_vecs = interp_vecs / np.maximum(interp_mags, 1e-8)

    result[interp] = interp_vecs
    new_mask[interp] = True
    return result, new_mask


class V161Dataset(Dataset):
    """Read V16 Zarr stores and expose richer signals for V16.1 trainers."""

    def __init__(
        self,
        dataset_dir: str | Path,
        builds: list[str] | None = None,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        augment: bool = False,
        curation_manifest: str | Path | None = None,
        height_channel: bool = False,
        object_roof_channel: bool = False,
        lightweight_object_gating: bool = False,
        curation_min_terrain_validity: float = 0.0,
        curation_min_minimap_usefulness: float = 0.0,
        curation_reject_what_plate: bool = False,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.augment = augment and split == "train"
        self.height_channel = bool(height_channel)
        self.object_roof_channel = bool(object_roof_channel)
        self.lightweight_object_gating = bool(lightweight_object_gating)
        self._rng = np.random.RandomState(seed)
        self._stores: dict[str, zarr.Group] = {}
        self._index_entries: list[dict] = []
        self._curation_manifest = Path(curation_manifest) if curation_manifest is not None else None
        self._curation_min_terrain_validity = float(curation_min_terrain_validity)
        self._curation_min_minimap_usefulness = float(curation_min_minimap_usefulness)
        self._curation_reject_what_plate = bool(curation_reject_what_plate)
        curation_index = load_curation_index(self._curation_manifest) if self._curation_manifest is not None else None

        build_dirs = builds or [d.stem.replace(".zarr", "") for d in sorted(self.dataset_dir.glob("*.zarr"))]
        for build in build_dirs:
            zarr_path = self.dataset_dir / f"{build}.zarr"
            if not zarr_path.exists():
                continue
            store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
            root = zarr.open_group(store=store, mode="r")
            self._stores[build] = root

            index_path = zarr_path / "index.parquet"
            if not index_path.exists():
                continue
            table = pq.read_table(str(index_path))
            for i in range(table.num_rows):
                row = {col: table.column(col)[i].as_py() for col in table.column_names}
                row["_build"] = build
                if curation_index is not None:
                    tile_id = int(row.get("tile_id", -1))
                    curation_row = curation_index.get((build, tile_id))
                    if curation_row is None or not bool(curation_row.get("keep", True)):
                        continue
                    row["_curation_profile"] = str(curation_row.get("profile", ""))
                    row["_curation_quality_score"] = float(curation_row.get("quality_score", 0.0) or 0.0)
                    row["_curation_usefulness_score"] = float(curation_row.get("usefulness_score", row["_curation_quality_score"]) or 0.0)
                    row["_curation_difficulty_score"] = float(curation_row.get("difficulty_score", row["_curation_quality_score"]) or 0.0)
                    row["_curation_difficulty_bucket"] = str(curation_row.get("difficulty_bucket", ""))
                    row["_curation_difficulty_rank"] = int(
                        curation_row["difficulty_rank"] if "difficulty_rank" in curation_row and curation_row["difficulty_rank"] is not None else -1
                    )
                    row["_curation_score_deformation_richness"] = float(curation_row.get("score_deformation_richness", 0.0) or 0.0)
                    row["_curation_score_normal_coverage"] = float(curation_row.get("score_normal_coverage", 0.0) or 0.0)
                    row["_curation_score_terrain_validity"] = float(curation_row.get("score_terrain_validity", 0.0) or 0.0)
                    row["_curation_score_painted_signal"] = float(curation_row.get("score_painted_signal", 0.0) or 0.0)
                    row["_curation_score_minimap_target_usefulness"] = float(curation_row.get("score_minimap_target_usefulness", 0.0) or 0.0)
                    row["_curation_normal_edge_f1"] = float(curation_row.get("normal_edge_f1", 0.0) or 0.0)
                    row["_curation_terrain_valid_cov"] = float(curation_row.get("terrain_valid_cov", 0.0) or 0.0)
                    row["_curation_minimap_gray_std"] = float(curation_row.get("minimap_gray_std", 0.0) or 0.0)
                    row["_curation_what_plate"] = bool(curation_row.get("what_plate", False))

                    if row["_curation_score_terrain_validity"] < self._curation_min_terrain_validity:
                        continue
                    if row["_curation_score_minimap_target_usefulness"] < self._curation_min_minimap_usefulness:
                        continue
                    if self._curation_reject_what_plate and row["_curation_what_plate"]:
                        continue
                self._index_entries.append(row)

        if not self._index_entries:
            raise ValueError(f"No index entries found in {self.dataset_dir}")

        self._indices = _build_split_indices(len(self._index_entries), split=split, val_fraction=val_fraction, seed=seed)

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor | bool | int | str]:
        entry = self._index_entries[self._indices[idx]]
        build = entry["_build"]
        root = self._stores[build]
        tile_id = int(entry["tile_id"])

        minimap = root["minimap_rgb"][tile_id].astype(np.float32) / 255.0
        height_raw = root["height_257"][tile_id].astype(np.float32)
        h_mean = float(entry["height_mean"])
        h_std = float(entry["height_std"]) + 1e-8
        height_norm = (height_raw - h_mean) / h_std

        normals = root["normal_xyz"][tile_id].astype(np.float32) if bool(entry.get("has_normal_xyz", False)) else np.zeros((257, 257, 3), dtype=np.float32)
        normal_mask = root["normal_mask"][tile_id].astype(np.float32) if bool(entry.get("has_normal_xyz", False)) and "normal_mask" in root else np.zeros((257, 257), dtype=np.float32)
        if bool(entry.get("has_normal_xyz", False)):
            normals, normal_mask = _interpolate_checkerboard_normals(normals, normal_mask)
        alpha = root["alpha_256"][tile_id].astype(np.float32) if bool(entry.get("has_alpha_256", False)) else np.zeros((256, 256, 4), dtype=np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)
        holes = root["holes_16"][tile_id].astype(np.float32) if bool(entry.get("has_holes_16", False)) else np.zeros((16, 16), dtype=np.float32)
        liquid_mask = root["liquid_mask"][tile_id].astype(np.float32) if bool(entry.get("has_liquid_mask", False)) else np.zeros((256, 256), dtype=np.float32)
        liquid_mask = np.clip(liquid_mask, 0.0, 1.0)
        liquid_height = root["liquid_height"][tile_id].astype(np.float32) if bool(entry.get("has_liquid_height", False)) and "liquid_height" in root else np.zeros((256, 256), dtype=np.float32)
        mcly_ids = root["mcly_texture_ids"][tile_id].astype(np.int64) if bool(entry.get("has_mcly_texture_ids", False)) and "mcly_texture_ids" in root else np.zeros((16, 16, 4), dtype=np.int64)
        mcly_ids = np.clip(mcly_ids, 0, 15)
        mcly_mask = root["mcly_layer_mask"][tile_id].astype(np.float32) if bool(entry.get("has_mcly_layer_mask", False)) and "mcly_layer_mask" in root else np.zeros((16, 16, 4), dtype=np.float32)
        mcnk_flags_16 = root["mcnk_flags_16"][tile_id].astype(np.int32) if "mcnk_flags_16" in root else np.zeros((16, 16), dtype=np.int32)
        liquid_type_16, liquid_type_valid_16 = _flags_to_liquid_type(mcnk_flags_16)

        mddf_mask = root["mddf_mask"][tile_id].astype(np.float32) if "mddf_mask" in root else np.zeros((257, 257), dtype=np.float32)
        modf_mask = root["modf_mask"][tile_id].astype(np.float32) if "modf_mask" in root else np.zeros((257, 257), dtype=np.float32)
        object_presence_257 = np.maximum(mddf_mask, modf_mask).astype(np.float32, copy=False)
        has_object_roof_mask = bool(entry.get("has_object_roof_mask", False)) and ("object_roof_mask" in root)
        object_roof_mask_256 = root["object_roof_mask"][tile_id].astype(np.float32) if has_object_roof_mask else np.zeros((256, 256), dtype=np.float32)
        object_roof_mask_256 = np.clip(object_roof_mask_256, 0.0, 1.0)
        object_roof_weight_256 = 1.0 - object_roof_mask_256
        object_roof_weight_257 = np.pad(object_roof_weight_256, ((0, 1), (0, 1)), mode="edge")
        object_roof_source = str(entry.get("object_roof_mask_source", "none"))
        # Focused height/normal runs do not need the heavy precise object mask.
        if self.lightweight_object_gating:
            object_filtered = 1.0 - compose_object_loss_weights_257(
                object_presence_257=object_presence_257,
                object_roof_weight_257=object_roof_weight_257,
            )
        elif "object_precise_mask" in root:
            object_filtered = root["object_precise_mask"][tile_id].astype(np.float32)
        elif "object_filtered_mask" in root:
            object_filtered = root["object_filtered_mask"][tile_id].astype(np.float32)
        else:
            object_filtered = root["object_mask"][tile_id].astype(np.float32)
        weight_257 = 1.0 - np.clip(object_filtered, 0.0, 1.0)
        weight_256 = _crop_257_to_256(weight_257)
        weight_16 = _downsample_256_to_16(weight_256)
        alpha_painted_256 = alpha_painted(alpha).astype(np.float32, copy=False)
        alpha_painted_cov = float((alpha_painted_256 >= 0.05).mean())
        mcly_cov = mcly_painted_coverage(mcly_mask)
        liquid_cov = float(liquid_mask.mean())
        object_cov = float(object_presence_257.mean())
        what_plate_flag = float(
            is_blank_what_plate(
                height_257=height_raw,
                alpha_cov=alpha_painted_cov,
                mcly_cov=mcly_cov,
                liquid_cov=liquid_cov,
                object_cov=object_cov,
            )
        )
        terrain_valid_mask_257 = compose_terrain_valid_mask_257(
            normal_mask_257=normal_mask,
            object_presence_257=object_presence_257,
            liquid_mask_256=liquid_mask,
            object_roof_weight_257=object_roof_weight_257,
            what_plate=what_plate_flag > 0.5,
        )
        mcly_any_16 = (mcly_mask.max(axis=2) > 0.05).astype(np.float32, copy=False)

        if self.augment:
            xform = int(self._rng.randint(0, 8))
            if xform & 1:
                minimap = minimap[:, ::-1]
                height_raw = height_raw[:, ::-1]
                height_norm = height_norm[:, ::-1]
                normals = normals[:, ::-1]
                normals[..., 0] = -normals[..., 0]
                normal_mask = normal_mask[:, ::-1]
                alpha = alpha[:, ::-1]
                holes = holes[:, ::-1]
                liquid_mask = liquid_mask[:, ::-1]
                liquid_height = liquid_height[:, ::-1]
                mcly_ids = mcly_ids[:, ::-1]
                mcly_mask = mcly_mask[:, ::-1]
                mcnk_flags_16 = mcnk_flags_16[:, ::-1]
                liquid_type_16 = liquid_type_16[:, ::-1]
                liquid_type_valid_16 = liquid_type_valid_16[:, ::-1]
                weight_257 = weight_257[:, ::-1]
                weight_256 = weight_256[:, ::-1]
                weight_16 = weight_16[:, ::-1]
                mddf_mask = mddf_mask[:, ::-1]
                modf_mask = modf_mask[:, ::-1]
                object_presence_257 = object_presence_257[:, ::-1]
                alpha_painted_256 = alpha_painted_256[:, ::-1]
                terrain_valid_mask_257 = terrain_valid_mask_257[:, ::-1]
                mcly_any_16 = mcly_any_16[:, ::-1]
                object_roof_mask_256 = object_roof_mask_256[:, ::-1]
                object_roof_weight_256 = object_roof_weight_256[:, ::-1]
                object_roof_weight_257 = object_roof_weight_257[:, ::-1]
            if xform & 2:
                minimap = minimap[::-1]
                height_raw = height_raw[::-1]
                height_norm = height_norm[::-1]
                normals = normals[::-1]
                normals[..., 1] = -normals[..., 1]
                normal_mask = normal_mask[::-1]
                alpha = alpha[::-1]
                holes = holes[::-1]
                liquid_mask = liquid_mask[::-1]
                liquid_height = liquid_height[::-1]
                mcly_ids = mcly_ids[::-1]
                mcly_mask = mcly_mask[::-1]
                mcnk_flags_16 = mcnk_flags_16[::-1]
                liquid_type_16 = liquid_type_16[::-1]
                liquid_type_valid_16 = liquid_type_valid_16[::-1]
                weight_257 = weight_257[::-1]
                weight_256 = weight_256[::-1]
                weight_16 = weight_16[::-1]
                mddf_mask = mddf_mask[::-1]
                modf_mask = modf_mask[::-1]
                object_presence_257 = object_presence_257[::-1]
                alpha_painted_256 = alpha_painted_256[::-1]
                terrain_valid_mask_257 = terrain_valid_mask_257[::-1]
                mcly_any_16 = mcly_any_16[::-1]
                object_roof_mask_256 = object_roof_mask_256[::-1]
                object_roof_weight_256 = object_roof_weight_256[::-1]
                object_roof_weight_257 = object_roof_weight_257[::-1]
            if xform & 4:
                minimap = np.rot90(minimap, k=1)
                height_raw = np.rot90(height_raw, k=1)
                height_norm = np.rot90(height_norm, k=1)
                normals = np.rot90(normals, k=1)
                old_nx = normals[..., 0].copy()
                normals[..., 0] = normals[..., 1]
                normals[..., 1] = -old_nx
                normal_mask = np.rot90(normal_mask, k=1)
                alpha = np.rot90(alpha, k=1)
                holes = np.rot90(holes, k=1)
                liquid_mask = np.rot90(liquid_mask, k=1)
                liquid_height = np.rot90(liquid_height, k=1)
                mcly_ids = np.rot90(mcly_ids, k=1)
                mcly_mask = np.rot90(mcly_mask, k=1)
                mcnk_flags_16 = np.rot90(mcnk_flags_16, k=1)
                liquid_type_16 = np.rot90(liquid_type_16, k=1)
                liquid_type_valid_16 = np.rot90(liquid_type_valid_16, k=1)
                weight_257 = np.rot90(weight_257, k=1)
                weight_256 = np.rot90(weight_256, k=1)
                weight_16 = np.rot90(weight_16, k=1)
                mddf_mask = np.rot90(mddf_mask, k=1)
                modf_mask = np.rot90(modf_mask, k=1)
                object_presence_257 = np.rot90(object_presence_257, k=1)
                alpha_painted_256 = np.rot90(alpha_painted_256, k=1)
                terrain_valid_mask_257 = np.rot90(terrain_valid_mask_257, k=1)
                mcly_any_16 = np.rot90(mcly_any_16, k=1)
                object_roof_mask_256 = np.rot90(object_roof_mask_256, k=1)
                object_roof_weight_256 = np.rot90(object_roof_weight_256, k=1)
                object_roof_weight_257 = np.rot90(object_roof_weight_257, k=1)

        minimap_t = torch.from_numpy(minimap.copy()).permute(2, 0, 1)
        if self.height_channel:
            height_norm_t = torch.from_numpy(height_norm[:256, :256].copy()).unsqueeze(0)
            input_tensor = torch.cat([minimap_t, height_norm_t], dim=0)
        elif self.object_roof_channel:
            object_roof_t = torch.from_numpy(object_roof_mask_256.copy()).unsqueeze(0)
            input_tensor = torch.cat([minimap_t, object_roof_t], dim=0)
        else:
            input_tensor = minimap_t
        return {
            "input": input_tensor,
            "height_raw": torch.from_numpy(height_raw.copy()).unsqueeze(0),
            "height_norm": torch.from_numpy(height_norm.copy()).unsqueeze(0),
            "height_mean": torch.tensor(h_mean, dtype=torch.float32),
            "height_std": torch.tensor(h_std, dtype=torch.float32),
            "normals": torch.from_numpy(normals.copy()).permute(2, 0, 1),
            "normal_mask": torch.from_numpy(normal_mask.copy()).unsqueeze(0),
            "alpha": torch.from_numpy(alpha.copy()).permute(2, 0, 1),
            "holes": torch.from_numpy(holes.copy()).unsqueeze(0),
            "liquid_mask": torch.from_numpy(liquid_mask.copy()).unsqueeze(0),
            "liquid_height": torch.from_numpy(liquid_height.copy()).unsqueeze(0),
            "liquid_type_16": torch.from_numpy(liquid_type_16.copy()).long(),
            "liquid_type_valid_16": torch.from_numpy(liquid_type_valid_16.copy()).unsqueeze(0),
            "mcly_ids": torch.from_numpy(mcly_ids.copy()).long(),
            "mcly_mask": torch.from_numpy(mcly_mask.copy()),
            "mcnk_flags_16": torch.from_numpy(mcnk_flags_16.copy()).long(),
            "weight_257": torch.from_numpy(weight_257.copy()).unsqueeze(0),
            "weight_256": torch.from_numpy(weight_256.copy()).unsqueeze(0),
            "weight_16": torch.from_numpy(weight_16.copy()).unsqueeze(0),
            "mddf_mask": torch.from_numpy(mddf_mask.copy()).unsqueeze(0),
            "modf_mask": torch.from_numpy(modf_mask.copy()).unsqueeze(0),
            "object_presence_257": torch.from_numpy(object_presence_257.copy()).unsqueeze(0),
            "object_roof_mask_256": torch.from_numpy(object_roof_mask_256.copy()).unsqueeze(0),
            "object_roof_weight_256": torch.from_numpy(object_roof_weight_256.copy()).unsqueeze(0),
            "object_roof_weight_257": torch.from_numpy(object_roof_weight_257.copy()).unsqueeze(0),
            "alpha_painted_256": torch.from_numpy(alpha_painted_256.copy()).unsqueeze(0),
            "terrain_valid_mask_257": torch.from_numpy(terrain_valid_mask_257.copy()).unsqueeze(0),
            "mcly_any_16": torch.from_numpy(mcly_any_16.copy()).unsqueeze(0),
            "what_plate_flag": torch.tensor(what_plate_flag, dtype=torch.float32),
            "alpha_painted_cov": torch.tensor(alpha_painted_cov, dtype=torch.float32),
            "mcly_cov": torch.tensor(mcly_cov, dtype=torch.float32),
            "curation_quality_score": torch.tensor(float(entry.get("_curation_quality_score", 0.0) or 0.0), dtype=torch.float32),
            "curation_usefulness_score": torch.tensor(float(entry.get("_curation_usefulness_score", 0.0) or 0.0), dtype=torch.float32),
            "curation_difficulty_score": torch.tensor(float(entry.get("_curation_difficulty_score", 0.0) or 0.0), dtype=torch.float32),
            "curation_difficulty_rank": torch.tensor(int(entry.get("_curation_difficulty_rank", -1)), dtype=torch.int64),
            "curation_difficulty_bucket": str(entry.get("_curation_difficulty_bucket", "")),
            "has_normals": bool(entry.get("has_normal_xyz", False)),
            "has_alpha": bool(entry.get("has_alpha_256", False)),
            "has_holes": bool(entry.get("has_holes_16", False)),
            "has_liquid": bool(entry.get("has_liquid_mask", False)),
            "has_mcly": bool(entry.get("has_mcly_texture_ids", False)),
            "has_object_roof_mask": has_object_roof_mask,
            "meta_build": str(entry.get("build") or build),
            "meta_store": str(build),
            "meta_map": str(entry.get("map", "")),
            "meta_tile_id": tile_id,
            "meta_tile_x": int(entry.get("tile_x") if entry.get("tile_x") is not None else -1),
            "meta_tile_y": int(entry.get("tile_y") if entry.get("tile_y") is not None else -1),
            "meta_object_roof_source": object_roof_source,
        }
