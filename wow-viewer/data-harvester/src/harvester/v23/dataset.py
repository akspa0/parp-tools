"""V22-to-V23 dataset adapter for Spec 089 Phase 1.

Channel contract:

- 0..2: minimap_rgb (ImageNet-normalized RGB)
- 3..6: alpha_256 layer weights
- 7..10: fixed tileset identity planes for prune-table indices 0..3
- 11..13: normal_xyz (257->256 crop)
- 14: terrain_valid_mask
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from harvester.v16_curation import load_curation_index
from harvester.v22_zarr_io import V22Dataset
from harvester.v23.channels import (
    InputMode,
    MissingMinimapError,
    build_channel_tensor,
    derive_terrain_valid_mask_257,
    load_tileset_prune_table,
)


def _resample_256_to_257(array_256: np.ndarray) -> np.ndarray:
    tensor = torch.from_numpy(np.ascontiguousarray(np.asarray(array_256, dtype=np.float32))).view(1, 1, 256, 256)
    resized = F.interpolate(tensor, size=(257, 257), mode="bicubic", align_corners=False)
    return resized.squeeze(0).squeeze(0).numpy().astype(np.float32, copy=False)


def _curation_float(row: Mapping[str, Any], key: str, default: float = 0.0) -> float:
    value = row.get(key, default)
    if value is None:
        return float(default)
    return float(value)


def _curation_bool(row: Mapping[str, Any], key: str, default: bool = False) -> bool:
    value = row.get(key, default)
    if value is None:
        return bool(default)
    return bool(value)


def _curation_mismatch_score(row: Mapping[str, Any]) -> float:
    """Rank validation candidates by mismatch-rich, non-blank curation signals."""
    difficulty = _curation_float(row, "difficulty_score")
    usefulness = _curation_float(row, "score_minimap_target_usefulness", _curation_float(row, "usefulness_score"))
    edge_iou_gap = 1.0 - _curation_float(row, "normal_edge_iou", 1.0)
    edge_f1_gap = 1.0 - _curation_float(row, "normal_edge_f1", 1.0)
    painted = _curation_float(row, "score_painted_signal", 0.0)
    return max(0.0, difficulty) + max(0.0, usefulness) + max(0.0, edge_iou_gap) + max(0.0, edge_f1_gap) + (0.25 * max(0.0, painted))


class V23HeightDataset(Dataset):
    """Read a V22 Zarr store and expose V23 training samples.

    Channel contract for ``input`` in ``full`` mode:

    - ``0..2``: ``minimap_rgb`` -> ``float32`` RGB, scaled by ``/255.0`` and
      normalized with ImageNet mean/std; missing minimap is fatal.
    - ``3..6``: ``alpha_256`` -> ``float32`` layer weights in ``[0,1]``;
      missing alpha zero-fills all four channels and marks them invalid in
      ``channel_valid_mask``.
    - ``7..10``: dominant-layer ``mcly_tileset_ids`` resolved through the
      prune table into four fixed identity planes for prune indices ``0..3``;
      missing tileset ids zero-fill the block and mark it invalid.
    - ``11..13``: ``normal_xyz`` -> ``float32`` XYZ cropped ``257->256``;
      missing normals zero-fill the block and mark it invalid.
    - ``14``: ``terrain_valid_mask`` derived from ``mcnr_mask_257``,
      ``liquid_mask``, and object-presence masks; always emitted as
      ``float32``.

    Each sample also contains:

    - ``input``: ``float32[C, 256, 256]`` V23 channel tensor
    - ``target`` / ``target_height``: ``float32[1, 257, 257]`` metric target
    - ``valid_mask``: ``float32[1, 257, 257]`` terrain-valid supervision mask
    - ``channel_valid_mask``: ``bool[C]`` source-availability mask aligned to
      the returned ``input`` tensor
    """

    def __init__(
        self,
        source: str | Path | Sequence[Mapping[str, Any]],
        *,
        build: str | None = None,
        maps: Sequence[str] | None = None,
        input_mode: str | InputMode = InputMode.FULL,
        tileset_prune_table: str | Path | Mapping[str, Any] | Mapping[int, int] | None = None,
        curation_manifest: str | Path | None = None,
        curation_min_terrain_validity: float = 0.20,
        curation_min_minimap_usefulness: float = 0.10,
        curation_max_liquid_coverage: float = 0.05,
        curation_reject_what_plate: bool = True,
    ) -> None:
        self.input_mode = InputMode.coerce(input_mode)
        self.tileset_prune_table = load_tileset_prune_table(tileset_prune_table)
        self._dataset = source if hasattr(source, "__getitem__") and hasattr(source, "__len__") and not isinstance(source, (str, Path)) else V22Dataset(source)  # type: ignore[arg-type]
        self._indices: list[int] = []
        self._curation_by_source_index: dict[int, dict[str, Any]] = {}
        curation_index = load_curation_index(curation_manifest) if curation_manifest is not None else None
        map_filter = {str(value) for value in maps or []}

        for idx in range(len(self._dataset)):  # type: ignore[arg-type]
            row = self._tile_row(idx)
            if build is None:
                build_matches = True
            else:
                build_matches = str(row.get("build", "")) == build
            map_matches = not map_filter or str(row.get("map", "")) in map_filter
            curation_row = None
            if curation_index is not None:
                tile_id = int(row.get("tile_id", idx))
                curation_row = curation_index.get((str(row.get("build", "")), tile_id))
                if curation_row is None or not bool(curation_row.get("keep", True)):
                    continue
                if _curation_float(curation_row, "score_terrain_validity") < float(curation_min_terrain_validity):
                    continue
                if _curation_float(curation_row, "score_minimap_target_usefulness") < float(curation_min_minimap_usefulness):
                    continue
                if _curation_float(curation_row, "liquid_cov") > float(curation_max_liquid_coverage):
                    continue
                if bool(curation_reject_what_plate) and _curation_bool(curation_row, "what_plate"):
                    continue
            if build_matches and map_matches:
                self._indices.append(idx)
                if curation_row is not None:
                    self._curation_by_source_index[idx] = dict(curation_row)

        if not self._indices:
            raise ValueError(f"No V23HeightDataset tiles matched build={build!r}, maps={sorted(map_filter)!r}")

    def __len__(self) -> int:
        return len(self._indices)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        source_idx = self._indices[idx]
        tile = self._dataset[source_idx]  # type: ignore[index]

        input_tensor, channel_valid_mask = build_channel_tensor(
            tile,
            self.input_mode,
            tileset_prune_table=self.tileset_prune_table,
            return_channel_valid_mask=True,
        )
        if input_tensor.shape[1:] != (256, 256):
            raise ValueError(f"V23 input tensor has unexpected shape {tuple(input_tensor.shape)}")

        if "height_257" not in tile:
            raise ValueError("V22 tile is missing height_257")
        target_height = np.asarray(tile["height_257"], dtype=np.float32)
        if target_height.shape != (257, 257):
            raise ValueError(f"height_257 must have shape (257, 257), got {target_height.shape}")

        liquid_mask_256 = np.asarray(tile.get("liquid_mask", np.zeros((256, 256), dtype=np.float32)), dtype=np.float32)
        if liquid_mask_256.shape != (256, 256):
            liquid_mask_256 = np.zeros((256, 256), dtype=np.float32)
        if "liquid_height" in tile and np.asarray(tile["liquid_height"]).shape == (256, 256):
            liquid_height_257 = _resample_256_to_257(np.asarray(tile["liquid_height"], dtype=np.float32))
            liquid_mask_257 = np.pad(np.clip(liquid_mask_256, 0.0, 1.0), ((0, 1), (0, 1)), mode="edge")
            target_height = np.where(liquid_mask_257 > 0.0, liquid_height_257, target_height).astype(np.float32, copy=False)

        valid_mask = derive_terrain_valid_mask_257(tile)
        sample = {
            "input": input_tensor,
            "target": torch.from_numpy(np.ascontiguousarray(target_height)).unsqueeze(0),
            "target_height": torch.from_numpy(np.ascontiguousarray(target_height)).unsqueeze(0),
            "valid_mask": torch.from_numpy(np.ascontiguousarray(valid_mask)).unsqueeze(0),
            "channel_valid_mask": channel_valid_mask,
            "tile_id": int(tile.get("tile_id", source_idx)),
            "build": str(tile.get("build", "")),
            "map": str(tile.get("map", "")),
            "tile_x": int(tile.get("tile_x", -1)) if "tile_x" in tile else -1,
            "tile_y": int(tile.get("tile_y", -1)) if "tile_y" in tile else -1,
        }
        curation = self._curation_by_source_index.get(source_idx)
        if curation is not None:
            sample.update(
                {
                    "curation_quality_score": torch.tensor(_curation_float(curation, "quality_score"), dtype=torch.float32),
                    "curation_usefulness_score": torch.tensor(
                        _curation_float(curation, "usefulness_score", _curation_float(curation, "quality_score")),
                        dtype=torch.float32,
                    ),
                    "curation_difficulty_score": torch.tensor(_curation_float(curation, "difficulty_score"), dtype=torch.float32),
                    "curation_difficulty_rank": torch.tensor(int(curation.get("difficulty_rank", -1) or -1), dtype=torch.int64),
                    "curation_difficulty_bucket": str(curation.get("difficulty_bucket", "")),
                    "curation_normal_edge_iou": torch.tensor(_curation_float(curation, "normal_edge_iou", 1.0), dtype=torch.float32),
                    "curation_normal_edge_f1": torch.tensor(_curation_float(curation, "normal_edge_f1", 1.0), dtype=torch.float32),
                    "curation_mismatch_score": torch.tensor(_curation_mismatch_score(curation), dtype=torch.float32),
                }
            )
        return sample

    def _tile_row(self, idx: int) -> Mapping[str, Any]:
        tile_index = getattr(self._dataset, "tile_index", None)
        if tile_index is not None:
            return tile_index[idx]
        return self._dataset[idx]  # type: ignore[index]

    def curation_metadata(self, local_idx: int) -> dict[str, Any] | None:
        """Return curation metadata for a dataset-local index when available."""
        source_idx = self._indices[int(local_idx)]
        row = self._curation_by_source_index.get(source_idx)
        return dict(row) if row is not None else None

    def curation_mismatch_score(self, local_idx: int) -> float:
        row = self.curation_metadata(local_idx)
        if row is None:
            return 0.0
        return _curation_mismatch_score(row)


__all__ = [
    "MissingMinimapError",
    "V23HeightDataset",
]
