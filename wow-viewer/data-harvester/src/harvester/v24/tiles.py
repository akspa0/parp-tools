"""Shared tile assembly for the V24 models.

Loads a V24 store row together with its V18 substrate arrays (joined via
``v18_row`` in the V24 index) and produces the per-tile signal dict both
stage adapters consume. All arrays are float32 NumPy; nothing touches torch
here so the loaders stay testable without a GPU.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import zarr

from harvester.v24 import lattice, store
from harvester.v24.clean_minimap import clean_minimap

HEIGHT_SCALE = 100.0  # world units -> model space for absolute heights
RESIDUAL_SCALE = 25.0  # world units -> model space for Stage B residuals


@dataclass
class TileRecord:
    """One tile's signals, ready for model input assembly."""

    row: int
    v18_row: int
    map_name: str
    tile_x: int
    tile_y: int
    audit_empty: bool
    real_available: bool
    cleaned_minimap: np.ndarray  # (256, 256, 3) float32 [0, 1]
    alpha: np.ndarray  # (256, 256, 4) float32
    normal: np.ndarray  # (257, 257, 3) float32
    mcnr_mask: np.ndarray  # (257, 257) float32
    object_mask: np.ndarray  # (257, 257) float32
    liquid_mask: np.ndarray  # (256, 256) float32
    holes: np.ndarray  # (16, 16) bool
    height: np.ndarray  # (257, 257) float32
    prior_outer: np.ndarray  # (17, 17) float32
    prior_inner: np.ndarray  # (16, 16) float32
    source_outer: np.ndarray  # (17, 17) uint8
    source_inner: np.ndarray  # (16, 16) uint8
    confidence_outer: np.ndarray  # (17, 17) float32
    confidence_inner: np.ndarray  # (16, 16) float32
    synth_outer: np.ndarray  # (17, 17) float32 (lattice sample of height)
    synth_inner: np.ndarray  # (16, 16) float32


class TileSource:
    """Reads V24 + V18 stores and yields TileRecords."""

    def __init__(self, v24_path: Path | str, v18_path: Path | str | None = None):
        self.v24 = store.open_v24_store(v24_path)
        self.index = store.read_index(v24_path)
        if v18_path is None:
            v18_path = self.v24.attrs.get("v18_store_path")
            if not v18_path:
                raise ValueError("V24 store has no v18_store_path attr; pass v18_path")
        self.v18 = zarr.open_group(str(v18_path), mode="r")
        self.has_no_object_minimap = "no_object_minimap" in self.v18

    def __len__(self) -> int:
        return len(self.index["tile_id"])

    def usable_rows(self) -> list[int]:
        """Rows that are trainable: not audit-empty and minimap present."""
        audit_empty = np.asarray(self.v24["wdl_prior_audit_empty"][:])
        return [r for r in range(len(self)) if not audit_empty[r]]

    def load(self, row: int) -> TileRecord:
        v18_row = int(self.index["v18_row"][row])
        minimap = np.asarray(self.v18["minimap_rgb"][v18_row])
        object_mask = np.asarray(self.v18["object_precise_mask"][v18_row], dtype=np.float32)
        rendered = (
            np.asarray(self.v18["no_object_minimap"][v18_row])
            if self.has_no_object_minimap
            else None
        )
        cleaned, _ = clean_minimap(minimap, object_mask, rendered)

        height = np.asarray(self.v18["height_257"][v18_row], dtype=np.float32)
        synth_outer, synth_inner = lattice.sample_lattice_from_height(height)

        return TileRecord(
            row=row,
            v18_row=v18_row,
            map_name=self.index["map"][row],
            tile_x=int(self.index["tile_x"][row]),
            tile_y=int(self.index["tile_y"][row]),
            audit_empty=bool(self.v24["wdl_prior_audit_empty"][row]),
            real_available=bool(self.v24["wdl_prior_real_available"][row]),
            cleaned_minimap=cleaned,
            alpha=np.asarray(self.v18["alpha_256"][v18_row], dtype=np.float32),
            normal=np.asarray(self.v18["normal_xyz"][v18_row], dtype=np.float32),
            mcnr_mask=np.asarray(self.v18["mcnr_mask_257"][v18_row], dtype=np.float32),
            object_mask=object_mask,
            liquid_mask=(
                np.asarray(self.v18["liquid_mask"][v18_row], dtype=np.float32)
                if "liquid_mask" in self.v18
                else np.zeros((256, 256), dtype=np.float32)
            ),
            holes=_normalize_holes(
                np.asarray(self.v18["holes_16"][v18_row]).astype(bool)
                if "holes_16" in self.v18
                else np.zeros((16, 16), dtype=bool)
            ),
            height=height,
            prior_outer=np.asarray(self.v24["wdl_prior_outer"][row], dtype=np.float32),
            prior_inner=np.asarray(self.v24["wdl_prior_inner"][row], dtype=np.float32),
            source_outer=np.asarray(self.v24["wdl_prior_source_outer"][row]),
            source_inner=np.asarray(self.v24["wdl_prior_source_inner"][row]),
            confidence_outer=np.asarray(
                self.v24["wdl_prior_confidence_outer"][row], dtype=np.float32
            ),
            confidence_inner=np.asarray(
                self.v24["wdl_prior_confidence_inner"][row], dtype=np.float32
            ),
            synth_outer=synth_outer,
            synth_inner=synth_inner,
        )


def _normalize_holes(holes_16: np.ndarray) -> np.ndarray:
    """Normalize the stored hole-mask polarity to True = hole.

    The V18 store's ``holes_16`` is all-True on ordinary terrain, while the C#
    harvester's ``hole_mask_16`` marks holes (a small minority of chunks) —
    the stored polarity is inverted. Real holes are always a minority, so a
    majority-True mask is flipped. Flagged as a dataset defect in the Spec 094
    V22/V18 audit; this keeps Stage B's loss gate usable either way.
    """
    if holes_16.mean() > 0.5:
        return ~holes_16
    return holes_16


def downsample_mean(image: np.ndarray, factor: int) -> np.ndarray:
    """Mean-pool a (H, W[, C]) array by an integer factor."""
    h, w = image.shape[:2]
    if h % factor or w % factor:
        raise ValueError(f"dims {image.shape[:2]} not divisible by {factor}")
    if image.ndim == 2:
        return image.reshape(h // factor, factor, w // factor, factor).mean(axis=(1, 3))
    return image.reshape(h // factor, factor, w // factor, factor, -1).mean(axis=(1, 3))


def pad_256_to_257(image: np.ndarray) -> np.ndarray:
    """Edge-replicate a 256-grid array to the 257 corner grid."""
    pad_width = ((0, 1), (0, 1)) + ((0, 0),) * (image.ndim - 2)
    return np.pad(image, pad_width, mode="edge")


def holes_to_257(holes_16: np.ndarray) -> np.ndarray:
    """Expand the 16x16 MCNK hole grid to the 257 corner grid (True = hole)."""
    expanded = np.repeat(np.repeat(holes_16, 16, axis=0), 16, axis=1)  # (256, 256)
    return pad_256_to_257(expanded.astype(np.float32)) > 0.5
