"""V16.2 dataset — extends V16.1.1 with richer object-guidance signals.

Reads from the same V16 Zarr stores but exposes additional channels that
trainers can use for terrain-aware loss gating and guidance:

  - object_filtered_mask (already in stores from patch-objects)
  - terrain_valid_mask_257 (computed: normal_mask * (1 - object_presence) * (1 - liquid))
  - object_visibility_mask (renderer-truth, optional, patched by V16.2 workflow)
  - no_object_minimap (renderer-truth, optional, patched by V16.2 workflow)

The V16.2 input tensor becomes 7 channels:
  channels 0-2: minimap RGB (existing)
  channel 3:    object_filtered_mask (terrain loss gate)
  channel 4:    terrain_valid_mask_257 (composite terrain validity)
  channel 5:    alpha_painted_256 (upsampled painted alpha)
  channel 6:    mcly_any_16 (upsampled MCLY presence)

When renderer-truth arrays are present, they are also returned as
supervision targets for the normal lane.
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


def _upsample_16_to_256(x: np.ndarray) -> np.ndarray:
    """Upsample 16x16 to 256x256 via repeat."""
    return np.repeat(np.repeat(x, 16, axis=0), 16, axis=1).astype(np.float32, copy=False)


class V162Dataset(Dataset):
    """Read V16 Zarr stores and expose V16.2 object-guidance signals.

    The input tensor is 7 channels:
      0-2: minimap RGB
      3:   object_filtered_mask (terrain loss gate)
      4:   terrain_valid_mask_257 (composite terrain validity)
      5:   alpha_painted_256 (upsampled painted alpha)
      6:   mcly_any_16 (upsampled MCLY presence)

    Renderer-truth signals (object_visibility_mask, no_object_minimap) are
    returned as separate supervision targets when present in the store.
    """

    # Number of extra guidance channels beyond the 3-channel minimap
    GUIDANCE_CHANNELS = 4

    def __init__(
        self,
        dataset_dir: str | Path,
        builds: list[str] | None = None,
        split: str = "train",
        val_fraction: float = 0.1,
        seed: int = 42,
        augment: bool = False,
        curation_manifest: str | Path | None = None,
    ) -> None:
        self.dataset_dir = Path(dataset_dir)
        self.augment = augment and split == "train"
        self._rng = np.random.RandomState(seed)
        self._stores: dict[str, zarr.Group] = {}
        self._index_entries: list[dict] = []
        self._curation_manifest = Path(curation_manifest) if curation_manifest is not None else None
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

        # --- Core signals (same as V16.1.1) ---
        minimap = root["minimap_rgb"][tile_id].astype(np.float32) / 255.0
        height_raw = root["height_257"][tile_id].astype(np.float32)
        h_mean = float(entry["height_mean"])
        h_std = float(entry["height_std"]) + 1e-8
        height_norm = (height_raw - h_mean) / h_std

        has_normals = bool(entry.get("has_normal_xyz", False))
        has_alpha = bool(entry.get("has_alpha_256", False))
        has_holes = bool(entry.get("has_holes_16", False))
        has_liquid = bool(entry.get("has_liquid_mask", False))
        has_instance = bool(entry.get("has_object_instance_mask", False))
        has_mcly = bool(entry.get("has_mcly_texture_ids", False))

        normals = root["normal_xyz"][tile_id].astype(np.float32) if has_normals and "normal_xyz" in root else np.zeros((257, 257, 3), dtype=np.float32)
        normal_mask = root["normal_mask"][tile_id].astype(np.float32) if has_normals and "normal_mask" in root else np.zeros((257, 257), dtype=np.float32)
        alpha = root["alpha_256"][tile_id].astype(np.float32) if has_alpha and "alpha_256" in root else np.zeros((256, 256, 4), dtype=np.float32)
        alpha = np.clip(alpha, 0.0, 1.0)
        holes = root["holes_16"][tile_id].astype(np.float32) if has_holes and "holes_16" in root else np.zeros((16, 16), dtype=np.float32)
        liquid_mask = root["liquid_mask"][tile_id].astype(np.float32) if has_liquid and "liquid_mask" in root else np.zeros((256, 256), dtype=np.float32)
        liquid_mask = np.clip(liquid_mask, 0.0, 1.0)
        liquid_height = root["liquid_height"][tile_id].astype(np.float32) if has_liquid and "liquid_height" in root else np.zeros((256, 256), dtype=np.float32)
        mcly_ids = root["mcly_texture_ids"][tile_id].astype(np.int64) if has_mcly and "mcly_texture_ids" in root else np.zeros((16, 16, 4), dtype=np.int64)
        mcly_ids = np.clip(mcly_ids, 0, 15)
        mcly_mask = root["mcly_layer_mask"][tile_id].astype(np.float32) if has_mcly and "mcly_layer_mask" in root else np.zeros((16, 16, 4), dtype=np.float32)
        mcnk_flags_16 = root["mcnk_flags_16"][tile_id].astype(np.int32) if "mcnk_flags_16" in root else np.zeros((16, 16), dtype=np.int32)
        liquid_type_16, liquid_type_valid_16 = _flags_to_liquid_type(mcnk_flags_16)

        # --- Object masks (existing in stores) ---
        # Prefer precise mask (rasterized WMO mesh) over filtered (coarse AABB)
        if "object_precise_mask" in root:
            object_filtered = root["object_precise_mask"][tile_id].astype(np.float32)
        elif "object_filtered_mask" in root:
            object_filtered = root["object_filtered_mask"][tile_id].astype(np.float32)
        else:
            object_filtered = root["object_mask"][tile_id].astype(np.float32)
        mddf_mask = root["mddf_mask"][tile_id].astype(np.float32) if "mddf_mask" in root else np.zeros((257, 257), dtype=np.float32)
        modf_mask = root["modf_mask"][tile_id].astype(np.float32) if "modf_mask" in root else np.zeros((257, 257), dtype=np.float32)
        object_presence_257 = np.maximum(mddf_mask, modf_mask).astype(np.float32, copy=False)

        # --- Derived guidance channels (computed, not stored) ---
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
        terrain_valid_mask_257 = normal_mask * (1.0 - np.clip(object_presence_257, 0.0, 1.0))
        terrain_valid_mask_257 *= (1.0 - np.clip(np.pad(liquid_mask, ((0, 1), (0, 1)), mode="edge"), 0.0, 1.0) * 0.85)
        if what_plate_flag > 0.5:
            terrain_valid_mask_257[...] = 0.0
        mcly_any_16 = (mcly_mask.max(axis=2) > 0.05).astype(np.float32, copy=False)

        # --- V16.2: Build 7-channel input ---
        # Channel 0-2: minimap RGB
        # Channel 3:   object_filtered_mask (257->256 crop)
        # Channel 4:   terrain_valid_mask_257 (257->256 crop)
        # Channel 5:   alpha_painted_256 (already 256)
        # Channel 6:   mcly_any_16 (upsampled to 256)
        guidance_ch3 = _crop_257_to_256(object_filtered).astype(np.float32)
        guidance_ch4 = _crop_257_to_256(terrain_valid_mask_257).astype(np.float32)
        guidance_ch5 = alpha_painted_256.astype(np.float32)
        guidance_ch6 = _upsample_16_to_256(mcly_any_16).astype(np.float32)

        input_7ch = np.stack([
            minimap[:, :, 0],
            minimap[:, :, 1],
            minimap[:, :, 2],
            guidance_ch3,
            guidance_ch4,
            guidance_ch5,
            guidance_ch6,
        ], axis=-1)  # (256, 256, 7)

        # --- Renderer-truth signals (optional, from V16.2 patch) ---
        has_object_visibility = "object_visibility_mask" in root
        has_no_object_minimap = "no_object_minimap" in root

        object_visibility = root["object_visibility_mask"][tile_id].astype(np.float32) if has_object_visibility else np.zeros((256, 256), dtype=np.float32)
        no_object_minimap = root["no_object_minimap"][tile_id].astype(np.float32) / 255.0 if has_no_object_minimap else np.zeros((256, 256, 3), dtype=np.float32)

        # --- Weight tensors (same as V16.1.1) ---
        weight_257 = 1.0 - np.clip(object_filtered, 0.0, 1.0)
        weight_256 = _crop_257_to_256(weight_257)
        weight_16 = _downsample_256_to_16(weight_256)

        # --- Augmentation ---
        if self.augment:
            xform = int(self._rng.randint(0, 8))
            if xform & 1:
                input_7ch = input_7ch[:, ::-1]
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
                object_visibility = object_visibility[:, ::-1]
                no_object_minimap = no_object_minimap[:, ::-1]
            if xform & 2:
                input_7ch = input_7ch[::-1]
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
                object_visibility = object_visibility[::-1]
                no_object_minimap = no_object_minimap[::-1]
            if xform & 4:
                input_7ch = np.rot90(input_7ch, k=1)
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
                object_visibility = np.rot90(object_visibility, k=1)
                no_object_minimap = np.rot90(no_object_minimap, k=1)

        # --- Build output dict ---
        result: dict[str, torch.Tensor | bool | int | str] = {
            # 7-channel input
            "input": torch.from_numpy(input_7ch.copy()).permute(2, 0, 1),  # (7, 256, 256)
            # Core terrain targets
            "height": torch.from_numpy(height_norm.copy()).unsqueeze(0),
            "height_raw": torch.from_numpy(height_raw.copy()).unsqueeze(0),
            "normals": torch.from_numpy(normals.copy()).permute(2, 0, 1),
            "normal_mask": torch.from_numpy(normal_mask.copy()).unsqueeze(0),
            "alpha": torch.from_numpy(alpha.copy()).permute(2, 0, 1),
            "holes": torch.from_numpy(holes.copy()).unsqueeze(0),
            "liquid": torch.from_numpy(liquid_mask.copy()).unsqueeze(0),
            "liquid_height": torch.from_numpy(liquid_height.copy()).unsqueeze(0),
            "liquid_type": torch.from_numpy(liquid_type_16.copy()).unsqueeze(0).long(),
            "liquid_type_valid": torch.from_numpy(liquid_type_valid_16.copy()).unsqueeze(0),
            # Loss weighting
            "weight_257": torch.from_numpy(weight_257.copy()).unsqueeze(0),
            "weight_256": torch.from_numpy(weight_256.copy()).unsqueeze(0),
            "weight_16": torch.from_numpy(weight_16.copy()).unsqueeze(0),
            # Object signals
            "object_filtered_mask": torch.from_numpy(object_filtered.copy()).unsqueeze(0),
            "mddf_mask": torch.from_numpy(mddf_mask.copy()).unsqueeze(0),
            "modf_mask": torch.from_numpy(modf_mask.copy()).unsqueeze(0),
            "object_presence_257": torch.from_numpy(object_presence_257.copy()).unsqueeze(0),
            # V16.2 guidance channels (also returned as separate tensors for inspection)
            "guidance_ch3_object_filtered": torch.from_numpy(guidance_ch3.copy()).unsqueeze(0),
            "guidance_ch4_terrain_valid": torch.from_numpy(guidance_ch4.copy()).unsqueeze(0),
            "guidance_ch5_alpha_painted": torch.from_numpy(guidance_ch5.copy()).unsqueeze(0),
            "guidance_ch6_mcly_any": torch.from_numpy(guidance_ch6.copy()).unsqueeze(0),
            # Texture decomposition
            "mcly_ids": torch.from_numpy(mcly_ids.copy()).long(),
            "mcly_mask": torch.from_numpy(mcly_mask.copy()),
            "mcnk_flags_16": torch.from_numpy(mcnk_flags_16.copy()),
            # V16.2 renderer-truth (optional supervision targets)
            "object_visibility_mask": torch.from_numpy(object_visibility.copy()).unsqueeze(0),
            "no_object_minimap": torch.from_numpy(no_object_minimap.copy()).permute(2, 0, 1),
            # Collation-compatible derived tensors (needed by loss functions)
            "terrain_valid_mask_257": torch.from_numpy(terrain_valid_mask_257.copy()).unsqueeze(0),
            "alpha_painted_256": torch.from_numpy(alpha_painted_256.copy()).unsqueeze(0),
            "mcly_any_16": torch.from_numpy(mcly_any_16.copy()).unsqueeze(0).float(),
            "alpha_painted_cov": torch.tensor([alpha_painted_cov], dtype=torch.float32),
            "mcly_cov": torch.tensor([mcly_cov], dtype=torch.float32),
            "what_plate_flag": torch.tensor([what_plate_flag], dtype=torch.float32),
            "liquid_mask": torch.from_numpy(liquid_mask.copy()).unsqueeze(0),
            # Presence flags
            "has_normals": has_normals,
            "has_alpha": has_alpha,
            "has_holes": has_holes,
            "has_liquid": has_liquid,
            "has_instance": has_instance,
            "has_mcly": has_mcly,
            "has_object_visibility": has_object_visibility,
            "has_no_object_minimap": has_no_object_minimap,
            # Metadata
            "meta_build": str(entry.get("build", build)),
            "meta_store": str(build),
            "meta_map": str(entry.get("map", "")),
            "meta_tile_id": int(tile_id),
            "meta_tile_x": int(entry.get("tile_x") if entry.get("tile_x") is not None else -1),
            "meta_tile_y": int(entry.get("tile_y") if entry.get("tile_y") is not None else -1),
            "meta_height_mean": h_mean,
            "meta_height_std": float(entry["height_std"]),
            "meta_what_plate_flag": what_plate_flag,
        }

        # Curation metadata if available
        if "_curation_difficulty_bucket" in entry:
            result["_curation_difficulty_bucket"] = str(entry["_curation_difficulty_bucket"])
            result["_curation_usefulness_score"] = float(entry.get("_curation_usefulness_score", 0.0))
            result["_curation_difficulty_score"] = float(entry.get("_curation_difficulty_score", 0.0))
            result["_curation_difficulty_rank"] = int(entry.get("_curation_difficulty_rank", -1))

        return result
