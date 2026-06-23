"""Near-duplicate clustering for raw fractal components.

Exact alpha-shape matching is too brittle for real terrain art. This module
groups raw components by translation/mirror/rotation-invariant normalized
binary thumbnails, with an optional Hamming-radius for small variations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from harvester.fractal_raw_analysis import RawComponentFingerprint

_THUMBNAIL_SIZE = 32


def cluster_near_duplicates(
    fingerprints: list[RawComponentFingerprint],
    canvas: zarr.Group,
    *,
    threshold: float = 0.05,
    size: int = _THUMBNAIL_SIZE,
    radius: int = 0,
) -> dict[str, list[RawComponentFingerprint]]:
    """Group raw components by invariant normalized binary thumbnail hashes.

    Parameters
    ----------
    fingerprints:
        Raw component fingerprints (typically from fingerprint_raw_regions).
    canvas:
        Source canvas group containing `alpha_256`.
    threshold:
        Binarization threshold applied to the alpha crop.
    size:
        Square thumbnail edge length for the normalized crop.
    radius:
        Maximum Hamming distance (in thumbnail bits) allowed when matching.
        Radius 0 means exact normalized-hash match after transforms.
        Radius 1 is usually enough to catch quantization/antialiasing drift.

    Returns
    -------
    Mapping from cluster_id to member fingerprints. The first member is the
    representative used for the cluster id.
    """
    alpha = canvas["alpha_256"][:].astype(np.float32)
    variant_to_cluster: dict[str, str] = {}
    clusters: dict[str, list[RawComponentFingerprint]] = {}

    for fingerprint in fingerprints:
        crop = _extract_crop(alpha, fingerprint)
        if crop is None:
            continue
        thumbnail = _normalize_crop(crop, size=size)
        cluster_id: str | None = None
        variant_hashes: list[str] = []
        for transformed in _transforms(thumbnail):
            base_hash = _thumbnail_hash(transformed)
            if radius <= 0:
                candidate_hashes = [base_hash]
            else:
                candidate_hashes = _hamming_variants(base_hash, radius=radius)
            for candidate in candidate_hashes:
                if candidate in variant_to_cluster:
                    cluster_id = variant_to_cluster[candidate]
                    break
            variant_hashes.extend(candidate_hashes)
            if cluster_id is not None:
                break

        if cluster_id is None:
            cluster_id = f"near_{_thumbnail_hash(thumbnail)[:20]}"
            clusters[cluster_id] = []
            for variant in variant_hashes:
                variant_to_cluster[variant] = cluster_id

        clusters[cluster_id].append(fingerprint)

    return clusters


def _extract_crop(
    alpha: np.ndarray,
    fingerprint: RawComponentFingerprint,
) -> np.ndarray | None:
    x, y, w, h = fingerprint.bbox_xywh
    if w <= 0 or h <= 0:
        return None
    layer_slot = int(fingerprint.layer_slot)
    if layer_slot >= alpha.shape[2]:
        return None
    return alpha[y : y + h, x : x + w, layer_slot]


def _normalize_crop(crop: np.ndarray, *, size: int) -> np.ndarray:
    """Resize/pad a binary or float crop to a square boolean thumbnail."""
    binary = np.asarray(crop, dtype=np.float32) > 0.0
    h, w = binary.shape[:2]
    if h == 0 or w == 0:
        return np.zeros((size, size), dtype=bool)

    scale = size / max(h, w)
    new_h = max(1, int(round(h * scale)))
    new_w = max(1, int(round(w * scale)))

    y_indices = (np.linspace(0, h - 1, new_h) + 0.5).astype(np.int64)
    x_indices = (np.linspace(0, w - 1, new_w) + 0.5).astype(np.int64)
    scaled = binary[y_indices[:, None], x_indices[None, :]]

    thumb = np.zeros((size, size), dtype=bool)
    off_y = (size - new_h) // 2
    off_x = (size - new_w) // 2
    thumb[off_y : off_y + new_h, off_x : off_x + new_w] = scaled
    return thumb


def _transforms(thumbnail: np.ndarray) -> list[np.ndarray]:
    """Return dihedral group transforms: identity, flips, rotations."""
    transforms: list[np.ndarray] = [thumbnail]
    transforms.append(np.flipud(thumbnail))
    transforms.append(np.fliplr(thumbnail))
    transforms.append(np.flipud(np.fliplr(thumbnail)))
    transforms.append(np.rot90(thumbnail, k=1))
    transforms.append(np.rot90(thumbnail, k=2))
    transforms.append(np.rot90(thumbnail, k=3))
    transforms.append(np.flipud(np.rot90(thumbnail, k=1)))
    return transforms


def _thumbnail_hash(thumbnail: np.ndarray) -> str:
    packed = np.packbits(thumbnail.reshape(-1).astype(np.uint8))
    return hashlib.sha256(packed.tobytes()).hexdigest()[:32]


def _hamming_variants(hash_str: str, *, radius: int) -> list[str]:
    """Generate hashes within a small Hamming radius of a base hash.

    Only the first 64 bits of the hex hash are varied. This is intentionally
    approximate; large radii explode combinatorially.
    """
    if radius < 0:
        return []
    value = int(hash_str, 16)
    variants: list[str] = [hash_str]
    if radius == 0:
        return variants

    bits = 64
    for bit in range(bits):
        variants.append(f"{(value ^ (1 << bit)):064x}")
    if radius >= 2:
        for b1 in range(bits):
            for b2 in range(b1 + 1, bits):
                variants.append(f"{(value ^ (1 << b1) ^ (1 << b2)):064x}")
    return variants


def write_near_dedupe_outputs(
    output_dir: str | Path,
    clusters: dict[str, list[RawComponentFingerprint]],
) -> dict[str, Any]:
    """Write near-duplicate cluster catalog and member rows."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    member_rows: list[dict[str, Any]] = []
    catalog_rows: list[dict[str, Any]] = []
    for cluster_id, members in clusters.items():
        members_sorted = sorted(members, key=lambda item: (item.build, item.map_name, item.layer_idx, item.region_id))
        first = members_sorted[0]
        builds = sorted({item.build for item in members_sorted})
        maps = sorted({item.map_name for item in members_sorted})
        layers = sorted({int(item.layer_idx) for item in members_sorted})
        catalog_rows.append(
            {
                "cluster_id": cluster_id,
                "member_count": int(len(members_sorted)),
                "build_count": int(len(builds)),
                "map_count": int(len(maps)),
                "layer_count": int(len(layers)),
                "builds": builds,
                "maps": maps,
                "layer_indices": layers,
                "crop_w": int(first.crop_w),
                "crop_h": int(first.crop_h),
                "area": int(first.area),
                "example_region_id": first.region_id,
                "example_bbox_xywh": list(first.bbox_xywh),
                "region_ids": [item.region_id for item in members_sorted[:128]],
                "mcly_texture_ids": sorted({texture for item in members_sorted for texture in item.mcly_texture_ids})[:64],
                "mcly_active_layers": sorted({layer for item in members_sorted for layer in item.mcly_active_layers}),
            }
        )
        for member in members_sorted:
            row = {
                "cluster_id": cluster_id,
                **_json_ready(asdict(member)),
            }
            member_rows.append(row)

    catalog_rows.sort(key=lambda row: (-int(row["member_count"]), -int(row["area"]), str(row["cluster_id"])))
    _write_table(out / "near_patterns.parquet", catalog_rows)
    _write_jsonl(out / "near_patterns.jsonl", catalog_rows)
    _write_table(out / "near_pattern_members.parquet", member_rows)
    _write_jsonl(out / "near_pattern_members.jsonl", member_rows)

    counts = [int(len(members)) for members in clusters.values()]
    summary = {
        "cluster_count": int(len(clusters)),
        "member_count": int(sum(counts)),
        "duplicate_cluster_count": int(sum(1 for count in counts if count > 1)),
        "max_cluster_size": int(max(counts, default=0)),
        "outputs": {
            "near_patterns_parquet": str(out / "near_patterns.parquet"),
            "near_pattern_members_parquet": str(out / "near_pattern_members.parquet"),
        },
    }
    (out / "near_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    table = pa.Table.from_pylist(rows) if rows else pa.Table.from_pylist([])
    pq.write_table(table, path)


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value
