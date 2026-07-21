"""Spec 116 US4: relief-stratified evaluation + trivial baseline + reused-piece overlap.

Evaluation is reported **stratified by how much relief a region actually contains**, so model
quality is not hidden behind terrain a constant predictor already solves (FR-011 / SC-006). The
trivial (tile-mean) baseline is reported alongside each stratum, and promotion is refused for any
model that does not beat it on relief-bearing regions (FR-012 / SC-007).

This module also measures reused-piece overlap between train and held-out sets (FR-013 / SC-008):
for each held-out minimap block, count a match if any train block correlates at/above a threshold
under the 8 dihedral transforms -- the same measurement the spec's Motivation used to establish the
corpus is a discrete alphabet of reused pieces.
"""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

CHUNKS_PER_AXIS = 16
PIXELS_PER_CHUNK = 16
RELIEF_STD_THRESHOLD = 0.02  # per-chunk height std at/above which a chunk is "relief-bearing"
STRATUM_FLAT = 0
STRATUM_RELIEF = 1
DEFAULT_BLOCK = 32
DEFAULT_NCC_THRESHOLD = 0.99

# 8 dihedral transforms as (k_rot90, flip_ud) -- covers all square symmetries.
_DIHEDRAL = [(k, f) for k in range(4) for f in (False, True)]


class ReliefStratificationError(ValueError):
    """Raised when relief stratification cannot be produced honestly."""


def _apply_dihedral(block: np.ndarray, k: int, flip: bool) -> np.ndarray:
    b = np.rot90(block, k=k)
    return np.flipud(b) if flip else b


def chunk_strata(height_257: np.ndarray, threshold: float = RELIEF_STD_THRESHOLD) -> np.ndarray:
    """Per-chunk (16, 16) uint8 stratum labels: 0=flat, 1=relief-bearing.

    A chunk is relief-bearing when its 16x16 height block has std above ``threshold``.
    """
    h = np.asarray(height_257, dtype=np.float32)
    if h.shape != (257, 257):
        raise ReliefStratificationError(f"height_257 must be (257, 257), got {h.shape}")
    block = h[: CHUNKS_PER_AXIS * PIXELS_PER_CHUNK, : CHUNKS_PER_AXIS * PIXELS_PER_CHUNK]
    # std over each 16x16 block -> (16, 16)
    std = block.reshape(CHUNKS_PER_AXIS, PIXELS_PER_CHUNK, CHUNKS_PER_AXIS, PIXELS_PER_CHUNK).std(axis=(1, 3))
    return (std >= threshold).astype(np.uint8)


def _upsample_strata_to_pixels(strata: np.ndarray) -> np.ndarray:
    """(16, 16) strata -> (256, 256) by nearest repeat, to align with pixel-level targets."""
    return np.repeat(np.repeat(strata, PIXELS_PER_CHUNK, axis=0), PIXELS_PER_CHUNK, axis=1)


def stratified_mae(
    predictions: Iterable[np.ndarray],
    targets: Iterable[np.ndarray],
    strata: Iterable[np.ndarray],
) -> dict:
    """Per-stratum mean absolute error over a set of tiles.

    ``predictions``, ``targets`` are per-tile (256, 256) or (257, 257) arrays; ``strata`` are
    per-tile (16, 16) chunk stratum labels (upsampled to match). Returns ``{flat: mae, relief: mae}``
    with each stratum's own count; a stratum with zero pixels reports ``mae=None``.
    """
    preds = [np.asarray(p, dtype=np.float32) for p in predictions]
    tgts = [np.asarray(t, dtype=np.float32) for t in targets]
    strs = [np.asarray(s, dtype=np.uint8) for s in strata]
    if not (len(preds) == len(tgts) == len(strs)):
        raise ReliefStratificationError("predictions/targets/strata length mismatch")
    if not preds:
        raise ReliefStratificationError("no tiles to score")

    flat_err = 0.0
    flat_n = 0
    relief_err = 0.0
    relief_n = 0
    for p, t, s in zip(preds, tgts, strs, strict=True):
        # Crop to the common 256x256 region for pixel alignment.
        pp = p[:256, :256]
        tt = t[:256, :256]
        ss = _upsample_strata_to_pixels(s[:16, :16])[:256, :256]
        err = np.abs(pp - tt)
        flat_err += float(err[ss == STRATUM_FLAT].sum())
        flat_n += int((ss == STRATUM_FLAT).sum())
        relief_err += float(err[ss == STRATUM_RELIEF].sum())
        relief_n += int((ss == STRATUM_RELIEF).sum())

    return {
        "flat": {"mae": (flat_err / flat_n) if flat_n else None, "pixels": flat_n},
        "relief": {"mae": (relief_err / relief_n) if relief_n else None, "pixels": relief_n},
    }


def tile_mean_baseline_mae(targets: Iterable[np.ndarray]) -> float:
    """Trivial baseline: predict each tile's own mean -> mean absolute error across all tiles.

    This is the no-structure baseline no model in this project has beaten on relief-bearing
    regions (SC-007). Reported alongside each stratum so the comparison is explicit.
    """
    tgts = [np.asarray(t, dtype=np.float32) for t in targets]
    if not tgts:
        raise ReliefStratificationError("no targets for tile-mean baseline")
    total_err = 0.0
    total_n = 0
    for t in tgts:
        tt = t[:256, :256]
        mean = tt.mean()
        total_err += float(np.abs(tt - mean).sum())
        total_n += tt.size
    return total_err / total_n


def count_reused_piece_overlap(
    train_rgb: Iterable[np.ndarray],
    held_out_rgb: Iterable[np.ndarray],
    *,
    block: int = DEFAULT_BLOCK,
    ncc_threshold: float = DEFAULT_NCC_THRESHOLD,
    max_held_blocks: int = 64,
) -> dict:
    """Count held-out minimap blocks that match a train block under the 8 dihedral transforms.

    For each held-out block (up to ``max_held_blocks``), a match is recorded if any train block's
    highest NCC over the 8 transforms is at/above ``ncc_threshold``. Returns the count and the
    fraction of held-out blocks that matched (FR-013 / SC-008).
    """
    train = [np.asarray(r, dtype=np.float32) for r in train_rgb]
    held = [np.asarray(r, dtype=np.float32) for r in held_out_rgb]
    if not train or not held:
        raise ReliefStratificationError("need both train and held-out minimaps to measure overlap")

    def _luma(rgb: np.ndarray) -> np.ndarray:
        return rgb.mean(axis=2) if rgb.ndim == 3 else rgb

    def _ncc(a: np.ndarray, b: np.ndarray) -> float:
        a = a - a.mean()
        b = b - b.mean()
        denom = np.sqrt((a * a).sum() * (b * b).sum())
        if denom <= 1e-12:
            return 0.0
        return float((a * b).sum() / denom)

    # Pre-extract train blocks (luma) once.
    train_blocks: list[np.ndarray] = []
    for img in train:
        lm = _luma(img)
        h, w = lm.shape
        for y in range(0, h - block + 1, block):
            for x in range(0, w - block + 1, block):
                train_blocks.append(lm[y : y + block, x : x + block])
    if not train_blocks:
        return {"matched_blocks": 0, "checked_blocks": 0, "match_fraction": 0.0}

    matched = 0
    checked = 0
    for img in held:
        lm = _luma(img)
        h, w = lm.shape
        for y in range(0, h - block + 1, block):
            for x in range(0, w - block + 1, block):
                if checked >= max_held_blocks:
                    break
                hb = lm[y : y + block, x : x + block]
                best = -1.0
                for tb in train_blocks:
                    for k, f in _DIHEDRAL:
                        cand = _apply_dihedral(tb, k, f)
                        ncc = _ncc(hb, cand)
                        if ncc > best:
                            best = ncc
                            if best >= ncc_threshold:
                                break
                    if best >= ncc_threshold:
                        break
                checked += 1
                if best >= ncc_threshold:
                    matched += 1
            if checked >= max_held_blocks:
                break
        if checked >= max_held_blocks:
            break

    return {
        "matched_blocks": matched,
        "checked_blocks": checked,
        "match_fraction": (matched / checked) if checked else 0.0,
    }


__all__ = [
    "ReliefStratificationError",
    "chunk_strata",
    "stratified_mae",
    "tile_mean_baseline_mae",
    "count_reused_piece_overlap",
    "RELIEF_STD_THRESHOLD",
    "STRATUM_FLAT",
    "STRATUM_RELIEF",
]
