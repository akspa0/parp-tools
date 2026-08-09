"""Validation contract for the v60 synthetic terrain control corpus.

The C# harvest tool is the synthesis authority. This module only validates and
indexes its small NPZ corpus; it does not recreate lighting or terrain signals.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

CONTROL_SCHEMA = "v60-control-corpus-v1"
INPUT_SIGNAL = "terrain_shadow_256"
TARGET_SIGNAL = "height_257"
INPUT_SHAPE = (256, 256)
TARGET_SHAPE = (257, 257)
VALID_SPLITS = frozenset({"train", "validation", "test"})
COMPLEXITY_BUCKETS = ("easy", "medium", "hard", "pathological")
CONTROL_FAMILY_BUCKETS = {
    "flat": "easy",
    "slope": "easy",
    "dome": "medium",
    "basin": "medium",
    "plateau": "medium",
    "rolling": "medium",
    "ridge": "hard",
    "valley": "hard",
    "terrace": "hard",
    "cliff": "hard",
    "mountainous": "hard",
    "sheer_dropoff": "pathological",
    "zone_style_blend": "pathological",
    "chunk_grid": "hard",
    "chunk_grid_mixed": "hard",
    "island_sea": "hard",
    "archipelago": "hard",
    "crater_field": "hard",
    "canyon_fan": "hard",
    "fractal_fbm": "hard",
    "fractal_ridged": "pathological",
    "lightning_burn": "pathological",
    "cross_tile_lightning": "pathological",
    "cross_tile_burn": "pathological",
    "noise": "pathological",
    "mixed": "pathological",
    "pathological": "pathological",
}
EXPECTED_CONTROL_FAMILIES = tuple(CONTROL_FAMILY_BUCKETS)
CROSS_TILE_FAMILIES = ("cross_tile_lightning", "cross_tile_burn")
ALIGNMENT_POLICY = "subcell_shifted_except_explicit_chunk_grid"


def _sha256_array(array: np.ndarray) -> str:
    """Hash the canonical little-endian C-contiguous float32 bytes."""
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def load_control_manifest(corpus_root: str | Path) -> dict[str, Any]:
    root = Path(corpus_root)
    manifest_path = root / "control_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"control manifest not found: {manifest_path}")
    with manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != CONTROL_SCHEMA:
        raise ValueError(f"expected schema {CONTROL_SCHEMA!r}, got {manifest.get('schema')!r}")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("control manifest rows must be a list")
    if int(manifest.get("row_count", -1)) != len(rows):
        raise ValueError("control manifest row_count does not match rows")
    return manifest


def validate_control_corpus(corpus_root: str | Path) -> dict[str, Any]:
    """Validate every control row and return a JSON-safe report."""
    root = Path(corpus_root)
    manifest = load_control_manifest(root)
    seen_ids: set[str] = set()
    family_splits: dict[str, str] = {}
    family_buckets: dict[str, str] = {}
    cross_tile_positions: dict[str, set[tuple[int, int]]] = {}
    cross_tile_pattern_ids: dict[str, set[str]] = {}
    alignment_modes: dict[str, set[str]] = {}
    field_offsets: dict[str, list[tuple[float, float]]] = {}
    split_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    failures: list[str] = []

    for position, row in enumerate(manifest["rows"]):
        prefix = f"row[{position}]"
        row_id = str(row.get("row_id", ""))
        if not row_id:
            failures.append(f"{prefix}: missing row_id")
        elif row_id in seen_ids:
            failures.append(f"{prefix}: duplicate row_id {row_id}")
        seen_ids.add(row_id)

        family = str(row.get("control_family", ""))
        split = str(row.get("split", ""))
        if not family:
            failures.append(f"{prefix}: missing control_family")
        if split not in VALID_SPLITS:
            failures.append(f"{prefix}: invalid split {split!r}")
        elif family:
            prior_split = family_splits.setdefault(family, split)
            if prior_split != split:
                failures.append(f"{prefix}: family {family!r} crosses {prior_split!r}/{split!r}")
            split_counts[split] = split_counts.get(split, 0) + 1

        bucket = str(row.get("complexity_bucket", ""))
        if bucket not in COMPLEXITY_BUCKETS:
            failures.append(f"{prefix}: invalid or missing complexity_bucket {bucket!r}")
        elif family:
            expected_bucket = CONTROL_FAMILY_BUCKETS.get(family)
            if expected_bucket and bucket != expected_bucket:
                failures.append(
                    f"{prefix}: family {family!r} has bucket {bucket!r}, expected {expected_bucket!r}"
                )
            prior_bucket = family_buckets.setdefault(family, bucket)
            if prior_bucket != bucket:
                failures.append(f"{prefix}: family {family!r} crosses complexity buckets")
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1

        if family in CROSS_TILE_FAMILIES:
            try:
                tile_span = int(row.get("pattern_tile_span", 0))
                tile_x = int(row.get("pattern_tile_x", -1))
                tile_y = int(row.get("pattern_tile_y", -1))
                pattern_id = str(row.get("pattern_id", ""))
            except (TypeError, ValueError):
                tile_span, tile_x, tile_y, pattern_id = 0, -1, -1, ""
            if tile_span != 2 or tile_x not in (0, 1) or tile_y not in (0, 1) or not pattern_id:
                failures.append(f"{prefix}: invalid cross-tile pattern metadata")
            position_key = (tile_x, tile_y)
            positions = cross_tile_positions.setdefault(family, set())
            if position_key in positions:
                failures.append(f"{prefix}: duplicate cross-tile position {position_key}")
            positions.add(position_key)
            cross_tile_pattern_ids.setdefault(family, set()).add(pattern_id)

        if manifest.get("alignment_policy") == ALIGNMENT_POLICY:
            alignment = str(row.get("cell_alignment", ""))
            try:
                offset_x = float(row.get("field_offset_x"))
                offset_y = float(row.get("field_offset_y"))
            except (TypeError, ValueError):
                alignment, offset_x, offset_y = "", float("nan"), float("nan")
            if alignment not in {"chunk_aligned", "subcell_shifted", "mixed_alignment"}:
                failures.append(f"{prefix}: missing or invalid cell_alignment")
            if not np.isfinite((offset_x, offset_y)).all():
                failures.append(f"{prefix}: field offsets are not finite")
            if family == "chunk_grid" and (alignment != "chunk_aligned" or offset_x != 0.0 or offset_y != 0.0):
                failures.append(f"{prefix}: chunk_grid must remain explicitly chunk_aligned")
            if family != "chunk_grid" and alignment == "chunk_aligned":
                failures.append(f"{prefix}: non-grid family is incorrectly chunk_aligned")
            alignment_modes.setdefault(family, set()).add(alignment)
            field_offsets.setdefault(family, []).append((offset_x, offset_y))

        npz_name = row.get("npz")
        if not isinstance(npz_name, str) or not npz_name:
            failures.append(f"{prefix}: missing npz path")
            continue
        npz_path = root / npz_name
        if not npz_path.is_file():
            failures.append(f"{prefix}: NPZ not found: {npz_name}")
            continue

        try:
            with np.load(npz_path, allow_pickle=False) as payload:
                if INPUT_SIGNAL not in payload:
                    failures.append(f"{prefix}: missing {INPUT_SIGNAL}")
                else:
                    shadow = np.asarray(payload[INPUT_SIGNAL])
                    if shadow.shape != INPUT_SHAPE:
                        failures.append(f"{prefix}: shadow shape {shadow.shape} != {INPUT_SHAPE}")
                    if not np.isfinite(shadow).all():
                        failures.append(f"{prefix}: shadow contains non-finite values")
                    if shadow.size and (float(shadow.min()) < -1e-6 or float(shadow.max()) > 1.000001):
                        failures.append(f"{prefix}: shadow is outside [0, 1]")
                    expected = str(row.get("input_sha256", ""))
                    if expected and _sha256_array(shadow) != expected:
                        failures.append(f"{prefix}: shadow hash mismatch")

                if TARGET_SIGNAL not in payload:
                    failures.append(f"{prefix}: missing {TARGET_SIGNAL}")
                else:
                    height = np.asarray(payload[TARGET_SIGNAL])
                    if height.shape != TARGET_SHAPE:
                        failures.append(f"{prefix}: height shape {height.shape} != {TARGET_SHAPE}")
                    if not np.isfinite(height).all():
                        failures.append(f"{prefix}: height contains non-finite values")
                    expected = str(row.get("target_sha256", ""))
                    if expected and _sha256_array(height) != expected:
                        failures.append(f"{prefix}: height hash mismatch")
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f"{prefix}: unable to read {npz_name}: {exc}")

    report = {
        "schema": "v60-control-validation-v1",
        "corpus_root": str(root),
        "manifest_schema": manifest["schema"],
        "row_count": len(manifest["rows"]),
        "family_count": len(family_splits),
        "family_splits": dict(sorted(family_splits.items())),
        "split_counts": dict(sorted(split_counts.items())),
        "family_buckets": dict(sorted(family_buckets.items())),
        "complexity_bucket_counts": dict(sorted(bucket_counts.items())),
        "cross_tile_positions": {
            family: [list(position) for position in sorted(positions)]
            for family, positions in sorted(cross_tile_positions.items())
        },
        "cross_tile_pattern_ids": {
            family: sorted(pattern_ids)
            for family, pattern_ids in sorted(cross_tile_pattern_ids.items())
        },
        "alignment_policy": manifest.get("alignment_policy"),
        "alignment_modes": {
            family: sorted(modes) for family, modes in sorted(alignment_modes.items())
        },
        "field_offset_ranges": {
            family: {
                "x": [min(offset[0] for offset in offsets), max(offset[0] for offset in offsets)],
                "y": [min(offset[1] for offset in offsets), max(offset[1] for offset in offsets)],
            }
            for family, offsets in sorted(field_offsets.items())
        },
        "failures": failures,
        "valid": not failures,
    }
    declared_buckets = manifest.get("complexity_bucket_counts")
    if isinstance(declared_buckets, dict):
        normalized_declared = {
            str(key): int(value) for key, value in declared_buckets.items()
        }
        if normalized_declared != dict(sorted(bucket_counts.items())):
            failures.append(
                "manifest complexity_bucket_counts does not match row complexity_bucket values"
            )
    declared_vocabulary = manifest.get("complexity_bucket_vocabulary")
    if declared_vocabulary is not None and sorted(str(value) for value in declared_vocabulary) != sorted(COMPLEXITY_BUCKETS):
        failures.append("manifest complexity_bucket_vocabulary does not match the v60 vocabulary")
    for family, positions in cross_tile_positions.items():
        missing_positions = sorted({(0, 0), (0, 1), (1, 0), (1, 1)} - positions)
        if missing_positions:
            failures.append(f"cross-tile family {family!r} is missing positions {missing_positions}")
        pattern_ids = cross_tile_pattern_ids.get(family, set())
        if len(pattern_ids) != 1:
            failures.append(
                f"cross-tile family {family!r} must use one pattern_id, got {sorted(pattern_ids)}"
            )
    if manifest.get("alignment_policy") == ALIGNMENT_POLICY:
        for family, offsets in field_offsets.items():
            distinct_offsets = {(round(offset_x, 5), round(offset_y, 5)) for offset_x, offset_y in offsets}
            if family != "chunk_grid" and len(distinct_offsets) < 2 and len(offsets) > 1:
                failures.append(f"family {family!r} does not vary its sub-cell field offset")
    report["failures"] = failures
    report["valid"] = not failures
    return report
