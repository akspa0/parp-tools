"""Spec 077 Phase 5 (US4) asset-candidate matching contract.

For each connected component in the predicted object mask, the
asset-candidate lane compares the corresponding minimap crop against the
entries in the spec 077 per-object capture library and emits a ranked
list of candidate matches.

This is the ADT-free runtime path for development-map and PM4-only tiles.
It is intentionally small and explicit so the first proof can be
validated against the object library built in spec 077 Phase 2; later
slices can swap in learned embeddings (DINOv2, etc.) without changing
the public surface here.
"""

from __future__ import annotations

import io
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
from PIL import Image

from harvester.inference_object import (
    AssetCandidate,
    InferenceObjectHypothesis,
)


# ---------------------------------------------------------------------------
# Library I/O
# ---------------------------------------------------------------------------

def load_library_assets(
    library_path: str | Path,
) -> list[dict]:
    """Read ``assets.parquet`` from a built object library Zarr store."""
    path = Path(library_path) / "assets.parquet"
    if not path.exists():
        return []
    table = pq.read_table(str(path))
    return [
        {col: table.column(col)[idx].as_py() for col in table.column_names}
        for idx in range(table.num_rows)
    ]


def load_library_index(library_path: str | Path) -> list[dict]:
    """Read ``index.parquet`` from a built object library Zarr store."""
    path = Path(library_path) / "index.parquet"
    if not path.exists():
        return []
    table = pq.read_table(str(path))
    return [
        {col: table.column(col)[idx].as_py() for col in table.column_names}
        for idx in range(table.num_rows)
    ]


# ---------------------------------------------------------------------------
# Library capture access
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class LibraryEntryThumbnail:
    """In-memory thumbnail of a single library entry for matching.

    ``image`` is an ``(H, W, 3)`` uint8 RGB array, ``mask`` is the
    matching ``(H, W)`` uint8 mask, ``fingerprint`` is the perceptual
    hash, ``library_id`` and ``asset_path`` round-trip to the source
    ``assets.parquet`` row.
    """

    library_id: str
    asset_path: str
    image: np.ndarray
    mask: np.ndarray
    fingerprint: str
    normalized_asset_path: str

    def mask_coverage(self) -> float:
        if self.mask.size == 0:
            return 0.0
        return float(self.mask.mean() / 255.0)


def _decode_png_thumbnail(png_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png_bytes)) as img:
        return np.asarray(img.convert("RGB"))


def _decode_png_mask(png_bytes: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(png_bytes)) as img:
        return np.asarray(img.convert("L"))


def _phash(rgb: np.ndarray) -> str:
    """Differential perceptual hash (16-bit) — deterministic, fast.

    Lifted from ``object_roof.variant_fingerprint_from_rgb`` so the
    candidate ranker can use the same hash family as the existing roof
    library. Returns a hex string of the 16-bit hash.
    """
    if rgb.ndim != 3:
        raise ValueError("Expected RGB array for fingerprint")
    luminance = rgb.astype(np.float32).mean(axis=2)
    target_w = 17
    target_h = 16
    ys = np.linspace(0, luminance.shape[0] - 1, target_h).astype(np.int32)
    xs = np.linspace(0, luminance.shape[1] - 1, target_w).astype(np.int32)
    sampled = luminance[np.ix_(ys, xs)]
    diff = sampled[:, 1:] > sampled[:, :-1]
    bits = "".join("1" if flag else "0" for flag in diff.reshape(-1).tolist())
    return f"{int(bits, 2):04x}"


def _hamming_hex(a: str, b: str) -> int:
    if len(a) != len(b):
        return max(len(a), len(b)) * 4
    ai = int(a, 16)
    bi = int(b, 16)
    return bin(ai ^ bi).count("1")


def build_thumbnails_from_captures(
    captures_dir: str | Path,
    library_assets: list[dict],
) -> list[LibraryEntryThumbnail]:
    """Read capture PNGs from disk and turn them into matching thumbnails.

    The captures directory is expected to follow the
    ``<variant_id>_image.png`` / ``<variant_id>_mask.png`` /
    ``<variant_id>_pose.json`` layout written by the spec 077 capture
    tool. Only entries with a matching image/mask pair are returned.
    """
    captures = Path(captures_dir)
    out: list[LibraryEntryThumbnail] = []
    for asset in library_assets:
        if str(asset.get("capture_status", "")) != "captured":
            continue
        variant_id = str(asset.get("preferred_variant_id", "") or "")
        if not variant_id:
            continue
        image_path = captures / f"{variant_id}_image.png"
        mask_path = captures / f"{variant_id}_mask.png"
        if not image_path.exists() or not mask_path.exists():
            continue
        image = _decode_png_thumbnail(image_path.read_bytes())
        mask = _decode_png_mask(mask_path.read_bytes())
        out.append(
            LibraryEntryThumbnail(
                library_id=str(asset.get("library_id", "")),
                asset_path=str(asset.get("original_asset_path", "")),
                normalized_asset_path=str(asset.get("normalized_asset_path", "")),
                image=image,
                mask=mask,
                fingerprint=_phash(image),
            )
        )
    return out


# ---------------------------------------------------------------------------
# Crop + match
# ---------------------------------------------------------------------------

def _crop_minimap_region(
    minimap_rgb: np.ndarray,
    mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """Crop minimap and mask to bbox; pad with neutral gray if empty."""
    if minimap_rgb.ndim != 3 or minimap_rgb.shape[2] != 3:
        raise ValueError(f"Expected HxWx3 minimap; got {minimap_rgb.shape}")
    h, w = minimap_rgb.shape[:2]
    x0, y0, x1, y1 = (int(v) for v in bbox_xyxy)
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    crop_rgb = minimap_rgb[y0:y1, x0:x1]
    crop_mask = mask[y0:y1, x0:x1]
    return crop_rgb, crop_mask


def _resize_nearest(arr: np.ndarray, size: int) -> np.ndarray:
    if arr.ndim == 2:
        h, w = arr.shape
    else:
        h, w = arr.shape[:2]
    if h == size and w == size:
        return arr
    ys = np.linspace(0, h - 1, size).astype(np.int64)
    xs = np.linspace(0, w - 1, size).astype(np.int64)
    if arr.ndim == 2:
        return arr[np.ix_(ys, xs)]
    return arr[np.ix_(ys, xs, np.arange(arr.shape[2]))]


def _masked_correlation(
    a: np.ndarray,
    a_mask: np.ndarray,
    b: np.ndarray,
    b_mask: np.ndarray,
) -> float:
    """Color/texture similarity on the intersection of two binary masks.

    Returns 0.0 when masks do not overlap enough to compute a stable
    statistic; otherwise returns a similarity score in ``[0, 1]`` over the
    intersection of the two masks. Constant-color masked regions fall back
    to mean RGB distance so solid-color captures still rank correctly.
    """
    a_mask = (a_mask > 127).astype(np.float32)
    b_mask = (b_mask > 127).astype(np.float32)
    intersection = a_mask * b_mask
    inter_sum = float(intersection.sum())
    if inter_sum < 8.0:
        return 0.0
    a_flat = a.astype(np.float32).reshape(-1, 3)
    b_flat = b.astype(np.float32).reshape(-1, 3)
    inter_flat = intersection.reshape(-1)
    a_mean = (a_flat * inter_flat[:, None]).sum(axis=0) / inter_sum
    b_mean = (b_flat * inter_flat[:, None]).sum(axis=0) / inter_sum
    a_centered = (a_flat - a_mean[None, :]) * inter_flat[:, None]
    b_centered = (b_flat - b_mean[None, :]) * inter_flat[:, None]
    a_norm = float((a_centered ** 2).sum())
    b_norm = float((b_centered ** 2).sum())
    if a_norm < 1e-6 or b_norm < 1e-6:
        max_dist = math.sqrt(3.0 * 255.0 * 255.0)
        dist = float(np.linalg.norm(a_mean - b_mean))
        return max(0.0, min(1.0, 1.0 - dist / max_dist))
    corr = float((a_centered * b_centered).sum() / math.sqrt(a_norm * b_norm))
    return max(0.0, min(1.0, (corr + 1.0) * 0.5))


def score_candidates(
    minimap_rgb: np.ndarray,
    object_mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    thumbnails: list[LibraryEntryThumbnail],
    *,
    target_size: int = 64,
    top_k: int = 5,
) -> list[AssetCandidate]:
    """Return up to *top_k* ranked candidates for the predicted bbox.

    Score = ``0.5 * phash_similarity + 0.5 * masked_correlation``. Both
    terms are bounded in ``[0, 1]``; the score is in the same range. A
    candidate that has no image content (zero mask) returns a 0 score.
    """
    crop_rgb, crop_mask = _crop_minimap_region(minimap_rgb, object_mask, bbox_xyxy)
    crop_resized_rgb = _resize_nearest(crop_rgb, target_size)
    crop_resized_mask = _resize_nearest(crop_mask, target_size)
    crop_fingerprint = _phash(crop_resized_rgb)

    scored: list[AssetCandidate] = []
    for thumb in thumbnails:
        if thumb.mask_coverage() <= 0.0:
            continue
        thumb_resized = _resize_nearest(thumb.image, target_size)
        thumb_mask_resized = _resize_nearest(thumb.mask, target_size)
        phash_dist = _hamming_hex(crop_fingerprint, thumb.fingerprint)
        phash_sim = max(0.0, 1.0 - phash_dist / 64.0)
        corr = _masked_correlation(
            crop_resized_rgb, crop_resized_mask, thumb_resized, thumb_mask_resized
        )
        score = 0.5 * phash_sim + 0.5 * corr
        scored.append(
            AssetCandidate(
                asset_path=thumb.asset_path,
                library_id=thumb.library_id,
                score=float(score),
                pose_xy=(0.0, 0.0),  # filled in by the ranker below
                pose_yaw=0.0,
                bbox_xyxy=bbox_xyxy,
            )
        )
    scored.sort(key=lambda c: c.score, reverse=True)
    return scored[:top_k]


def build_hypothesis_from_bbox(
    *,
    tile_id: int,
    instance_id: int,
    minimap_rgb: np.ndarray,
    object_mask: np.ndarray,
    bbox_xyxy: tuple[int, int, int, int],
    thumbnails: list[LibraryEntryThumbnail],
    mask_confidence: float = 1.0,
    top_k: int = 5,
) -> InferenceObjectHypothesis:
    """One-shot helper: score candidates and emit a hypothesis.

    The hypothesis carries XY = bbox center and yaw = 0.0 — pitch, roll,
    scale, and a non-zero yaw are explicitly deferred per spec 077
    FR-018. The downstream restorer can promote yaw when the candidate
    set is rich enough to recover it (out of scope for the first
    pass).
    """
    candidates = score_candidates(
        minimap_rgb, object_mask, bbox_xyxy, thumbnails, top_k=top_k
    )
    x0, y0, x1, y1 = bbox_xyxy
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    return InferenceObjectHypothesis(
        tile_id=int(tile_id),
        instance_id=int(instance_id),
        mask_bbox=bbox_xyxy,
        mask_confidence=float(mask_confidence),
        asset_candidate_paths=tuple(c.asset_path for c in candidates),
        asset_candidate_scores=tuple(c.score for c in candidates),
        asset_candidate_library_ids=tuple(c.library_id for c in candidates),
        pose_xy=(float(cx), float(cy)),
        pose_yaw=0.0,
        pose_z_from_terrain=None,
    )
