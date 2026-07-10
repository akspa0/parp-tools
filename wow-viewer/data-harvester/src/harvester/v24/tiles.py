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
    """Reads V24 + V18 stores and yields TileRecords.

    Supports two load paths:

    * **Random-access** (default): ``load(row)`` reads V18 data one array at a
      time via ``zarr[row]`` indexing.  Fast when the OS page cache is warm,
      but the FIRST epoch is slow if V18 chunks span many tiles (read
      amplification: every index fetch pulls the whole chunk from disk).
    * **Preloaded** (recommended for training): call ``preload(rows)`` *before*
      iterating.  This reads all needed V18 rows in a single contiguous Zarr
      slice (sequential I/O, no read amplification) then caches the block in
      memory.  Subsequent ``load(row)`` calls are dict lookups — sub-ms.

    Minimap loading order (first match wins):
    1. ``cleaned_minimap_256`` in the V24 store (pre-computed, preferred)
    2. Raw ``minimap_rgb`` from V18, normalized to float32 [0,1] (fallback)
    """

    _V18_ARRAYS = frozenset({
        "minimap_rgb", "object_precise_mask", "alpha_256", "normal_xyz",
        "mcnr_mask_257", "height_257", "liquid_mask", "holes_16",
    })

    def __init__(self, v24_path: Path | str, v18_path: Path | str | None = None):
        self.v24 = store.open_v24_store(v24_path)
        self.index = store.read_index(v24_path)
        if v18_path is None:
            v18_path = self.v24.attrs.get("v18_store_path")
            if not v18_path:
                raise ValueError("V24 store has no v18_store_path attr; pass v18_path")
        self.v18 = zarr.open_group(str(v18_path), mode="r")
        self.has_no_object_minimap = "no_object_minimap" in self.v18
        # Check for pre-computed cleaned minimaps in V24 store.
        self._has_cleaned = "cleaned_minimap_256" in self.v24
        # Preload cache — populated by preload(), consumed by load().
        self._v18_cache: dict[str, np.ndarray] = {}
        self._v18_offset: dict[int, int] = {}   # v18_row -> index into cache arrays
        self._v18_start_row: int = -1

    def __len__(self) -> int:
        return len(self.index["tile_id"])

    def usable_rows(self) -> list[int]:
        """Rows that are trainable: not audit-empty and minimap present."""
        audit_empty = np.asarray(self.v24["wdl_prior_audit_empty"][:])
        return [r for r in range(len(self)) if not audit_empty[r]]

    # ------------------------------------------------------------------
    # Preload API — call once before training to avoid per-load Zarr seeks
    # ------------------------------------------------------------------

    def preload(self, rows: list[int]) -> None:
        """Batch-read V18 data for *rows* in a single contiguous Zarr pass.

        All unique ``v18_row`` values are collected, sorted, and read via
        ``arr[lo:hi]`` — sequential chunk access that avoids the read
        amplification of per-row random indexing.  After this call,
        ``load(row)`` reads from the in-memory cache.
        """
        unique_v18 = sorted({int(self.index["v18_row"][r]) for r in rows})
        lo, hi = unique_v18[0], unique_v18[-1] + 1

        for name in sorted(self._V18_ARRAYS):
            if name not in self.v18:
                continue
            block = np.asarray(self.v18[name][lo:hi])
            self._v18_cache[name] = block

        self._v18_start_row = lo
        self._v18_offset = {v: i for i, v in enumerate(unique_v18)}

    def _v18_read(self, name: str, v18_row: int, dtype=None):
        """Read a single tile from V18, using the preload cache if available."""
        if self._v18_cache:
            # Fast path — contiguous preload block
            arr = self._v18_cache[name]
            idx = v18_row - self._v18_start_row
            raw = arr[idx]
        else:
            raw = np.asarray(self.v18[name][v18_row])
        if dtype is not None:
            return np.asarray(raw, dtype=dtype)
        return raw

    def _v18_has(self, name: str) -> bool:
        return name in self._v18_cache or name in self.v18

    # ------------------------------------------------------------------
    # Tile loading
    # ------------------------------------------------------------------

    def load(self, row: int) -> TileRecord:
        v18_row = int(self.index["v18_row"][row])

        # Minimap: prefer pre-computed cleaned from V24 store (one-time build).
        if self._has_cleaned:
            cleaned = np.asarray(self.v24["cleaned_minimap_256"][row], dtype=np.float32)
        else:
            minimap_raw = self._v18_read("minimap_rgb", v18_row)
            rgb = minimap_raw.astype(np.float32)
            if rgb.max() > 1.5:
                rgb = rgb / 255.0
            cleaned = rgb

        object_mask = self._v18_read("object_precise_mask", v18_row, dtype=np.float32)
        height = self._v18_read("height_257", v18_row, dtype=np.float32)
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
            alpha=self._v18_read("alpha_256", v18_row, dtype=np.float32),
            normal=self._v18_read("normal_xyz", v18_row, dtype=np.float32),
            mcnr_mask=self._v18_read("mcnr_mask_257", v18_row, dtype=np.float32),
            object_mask=object_mask,
            liquid_mask=(
                self._v18_read("liquid_mask", v18_row, dtype=np.float32)
                if self._v18_has("liquid_mask")
                else np.zeros((256, 256), dtype=np.float32)
            ),
            holes=_normalize_holes(
                self._v18_read("holes_16", v18_row).astype(bool)
                if self._v18_has("holes_16")
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


class MultiTileSource:
    """Concatenates multiple (V24, V18) store pairs into one flat, indexable corpus.

    Lets a training run span builds that only exist as separate V24 stores
    (e.g. an 0.5.3 alpha corpus and a 3.3.5 curated corpus) without merging
    the underlying Zarr data. Row indices are global offsets into the
    concatenation, so callers that only ever see plain ints (train/val
    splitting, shuffling) need no changes.
    """

    def __init__(self, pairs: list[tuple[Path | str, Path | str | None]]):
        if not pairs:
            raise ValueError("MultiTileSource requires at least one (v24, v18) pair")
        self.sources = [TileSource(v24, v18) for v24, v18 in pairs]
        self._offsets: list[int] = []
        offset = 0
        for source in self.sources:
            self._offsets.append(offset)
            offset += len(source)
        self._total = offset

    def __len__(self) -> int:
        return self._total

    def _locate(self, global_row: int) -> tuple[TileSource, int]:
        for i in range(len(self.sources) - 1, -1, -1):
            if global_row >= self._offsets[i]:
                return self.sources[i], global_row - self._offsets[i]
        raise IndexError(global_row)

    def usable_rows(self) -> list[int]:
        rows: list[int] = []
        for offset, source in zip(self._offsets, self.sources):
            rows.extend(offset + r for r in source.usable_rows())
        return rows

    def load(self, global_row: int) -> TileRecord:
        source, local_row = self._locate(global_row)
        record = source.load(local_row)
        record.row = global_row
        return record

    def preload(self, rows: list[int]) -> None:
        """Preload V18 data for *rows* across all sub-sources."""
        for offset, source in zip(self._offsets, self.sources):
            local_rows = [r - offset for r in rows if offset <= r < offset + len(source)]
            if local_rows:
                source.preload(local_rows)


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
