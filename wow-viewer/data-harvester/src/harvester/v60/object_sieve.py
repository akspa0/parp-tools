"""Validation contract for the v60 synthetic object-sieve corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

OBJECT_SIEVE_SCHEMA = "v60-object-sieve-control-v1"
INPUT_SIGNAL = "objectified_terrain_shadow_256"
CLEAN_SIGNAL = "terrain_shadow_256"
MASK_SIGNAL = "object_contamination_mask_256"
PLACEMENT_REGIMES = ("none", "sparse", "dense", "overlap", "boundary_crossing")
OBJECT_FAMILIES = ("tree", "rock", "building", "bridge")
SIGNAL_SHAPE = (256, 256)


def _sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def load_object_sieve_manifest(corpus_root: str | Path) -> dict[str, Any]:
    root = Path(corpus_root)
    path = root / "object_sieve_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"object sieve manifest not found: {path}")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("schema") != OBJECT_SIEVE_SCHEMA:
        raise ValueError(f"expected schema {OBJECT_SIEVE_SCHEMA!r}, got {manifest.get('schema')!r}")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("object sieve manifest rows must be a list")
    if int(manifest.get("row_count", -1)) != len(rows):
        raise ValueError("object sieve manifest row_count does not match rows")
    return manifest


def validate_object_sieve_corpus(corpus_root: str | Path) -> dict[str, Any]:
    root = Path(corpus_root)
    manifest = load_object_sieve_manifest(root)
    failures: list[str] = []
    seen_ids: set[str] = set()
    regime_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    split_counts: dict[str, int] = {}
    terrain_row_regimes: dict[str, set[str]] = {}
    coverage_by_regime: dict[str, list[float]] = {}
    boundary_touch_by_regime: dict[str, int] = {}

    for index, row in enumerate(manifest["rows"]):
        prefix = f"row[{index}]"
        row_id = str(row.get("row_id", ""))
        if not row_id:
            failures.append(f"{prefix}: missing row_id")
        elif row_id in seen_ids:
            failures.append(f"{prefix}: duplicate row_id {row_id}")
        seen_ids.add(row_id)

        regime = str(row.get("placement_regime", ""))
        object_family = str(row.get("object_family", ""))
        terrain_row_id = str(row.get("terrain_control_row_id", ""))
        split = str(row.get("split", ""))
        if regime not in PLACEMENT_REGIMES:
            failures.append(f"{prefix}: invalid placement_regime {regime!r}")
        if object_family not in OBJECT_FAMILIES:
            failures.append(f"{prefix}: invalid object_family {object_family!r}")
        if split not in {"train", "validation", "test"}:
            failures.append(f"{prefix}: invalid split {split!r}")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1
        family_counts[object_family] = family_counts.get(object_family, 0) + 1
        split_counts[split] = split_counts.get(split, 0) + 1
        if terrain_row_id:
            terrain_row_regimes.setdefault(terrain_row_id, set()).add(regime)

        npz_name = row.get("npz")
        if not isinstance(npz_name, str) or not npz_name:
            failures.append(f"{prefix}: missing npz path")
            continue
        path = root / npz_name
        if not path.is_file():
            failures.append(f"{prefix}: NPZ not found: {npz_name}")
            continue
        try:
            with np.load(path, allow_pickle=False) as payload:
                missing = [name for name in (INPUT_SIGNAL, CLEAN_SIGNAL, MASK_SIGNAL) if name not in payload]
                if missing:
                    failures.append(f"{prefix}: missing signals {missing}")
                    continue
                contaminated = np.asarray(payload[INPUT_SIGNAL], dtype=np.float32)
                clean = np.asarray(payload[CLEAN_SIGNAL], dtype=np.float32)
                mask = np.asarray(payload[MASK_SIGNAL], dtype=np.float32)
        except (OSError, ValueError) as exc:
            failures.append(f"{prefix}: unable to read {npz_name}: {exc}")
            continue

        for name, array in ((INPUT_SIGNAL, contaminated), (CLEAN_SIGNAL, clean), (MASK_SIGNAL, mask)):
            if array.shape != SIGNAL_SHAPE:
                failures.append(f"{prefix}: {name} shape {array.shape} != {SIGNAL_SHAPE}")
            if not np.isfinite(array).all():
                failures.append(f"{prefix}: {name} contains non-finite values")
        if contaminated.size and (contaminated.min() < -1e-6 or contaminated.max() > 1.000001):
            failures.append(f"{prefix}: {INPUT_SIGNAL} is outside [0, 1]")
        if clean.size and (clean.min() < -1e-6 or clean.max() > 1.000001):
            failures.append(f"{prefix}: {CLEAN_SIGNAL} is outside [0, 1]")
        if mask.size and (mask.min() < -1e-6 or mask.max() > 1.000001):
            failures.append(f"{prefix}: {MASK_SIGNAL} is outside [0, 1]")

        coverage = float((mask >= 0.5).mean())
        coverage_by_regime.setdefault(regime, []).append(coverage)
        boundary_touch = bool(
            (mask[:8] >= 0.5).any()
            or (mask[-8:] >= 0.5).any()
            or (mask[:, :8] >= 0.5).any()
            or (mask[:, -8:] >= 0.5).any()
        )
        if regime == "none":
            if coverage != 0.0:
                failures.append(f"{prefix}: none regime has contamination pixels")
            if not np.array_equal(contaminated, clean):
                failures.append(f"{prefix}: none regime input differs from clean target")
        else:
            if coverage <= 0.0:
                failures.append(f"{prefix}: {regime} regime has no contamination pixels")
            if regime == "boundary_crossing" and not boundary_touch:
                failures.append(f"{prefix}: boundary_crossing mask does not touch a tile boundary")
        if boundary_touch:
            boundary_touch_by_regime[regime] = boundary_touch_by_regime.get(regime, 0) + 1

        for field, array, row_key in (
            (INPUT_SIGNAL, contaminated, "input_sha256"),
            (CLEAN_SIGNAL, clean, "terrain_target_sha256"),
            (MASK_SIGNAL, mask, "contamination_target_sha256"),
        ):
            expected = str(row.get(row_key, ""))
            if expected and _sha256_array(array) != expected:
                failures.append(f"{prefix}: {field} hash mismatch")

    missing_regimes = sorted(set(PLACEMENT_REGIMES) - set(regime_counts))
    if missing_regimes:
        failures.append(f"missing placement regimes {missing_regimes}")
    declared_families = manifest.get("object_families")
    if isinstance(declared_families, list):
        missing_families = sorted({str(value) for value in declared_families} - set(family_counts))
        if missing_families:
            failures.append(f"missing object families {missing_families}")
    declared_regimes = manifest.get("placement_regimes")
    if isinstance(declared_regimes, list):
        missing_declared_regimes = sorted(
            {str(value) for value in declared_regimes} - set(regime_counts)
        )
        if missing_declared_regimes:
            failures.append(f"missing declared placement regimes {missing_declared_regimes}")
    declared_terrain_rows = manifest.get("terrain_row_count")
    if declared_terrain_rows is not None:
        try:
            expected_terrain_rows = int(declared_terrain_rows)
        except (TypeError, ValueError):
            expected_terrain_rows = -1
        if expected_terrain_rows != len(terrain_row_regimes):
            failures.append(
                f"terrain_row_count {declared_terrain_rows!r} does not match "
                f"observed {len(terrain_row_regimes)}"
            )
        for terrain_row_id, row_regimes in sorted(terrain_row_regimes.items()):
            missing_row_regimes = sorted(set(PLACEMENT_REGIMES) - row_regimes)
            if missing_row_regimes:
                failures.append(
                    f"terrain control row {terrain_row_id!r} is missing regimes {missing_row_regimes}"
                )
    report = {
        "schema": "v60-object-sieve-validation-v1",
        "corpus_root": str(root),
        "manifest_schema": manifest["schema"],
        "row_count": len(manifest["rows"]),
        "regime_counts": dict(sorted(regime_counts.items())),
        "object_family_counts": dict(sorted(family_counts.items())),
        "terrain_control_row_count": len(terrain_row_regimes),
        "split_counts": dict(sorted(split_counts.items())),
        "mean_mask_coverage_by_regime": {
            regime: float(np.mean(values)) for regime, values in sorted(coverage_by_regime.items())
        },
        "boundary_touch_counts": dict(sorted(boundary_touch_by_regime.items())),
        "failures": failures,
        "valid": not failures,
    }
    return report


__all__ = [
    "INPUT_SIGNAL",
    "CLEAN_SIGNAL",
    "MASK_SIGNAL",
    "OBJECT_FAMILIES",
    "PLACEMENT_REGIMES",
    "load_object_sieve_manifest",
    "validate_object_sieve_corpus",
]
