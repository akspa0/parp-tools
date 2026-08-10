"""Manifest, hash, split, and array validation for the clean-signal v60 corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from harvester.v60.clean_signal_inputs import (
    IMAGE_SHAPE,
    validate_clean_observation,
)
from harvester.v60.clean_signal_targets import TARGET_SHAPE

CORPUS_SCHEMA = "v7-clean-signal-corpus-v1"
VALID_SOURCE_KINDS = frozenset({"synthetic_control", "accepted_real"})
VALID_SPLITS = frozenset({"train", "validation", "test"})
VALID_SPLIT_MODES = frozenset({"within_family", "complete_family"})
LUMA_SIGNAL = "clean_observation_luma_256"
GRADIENT_SIGNAL = "clean_observation_gradient_256"
CONFIDENCE_SIGNAL = "clean_observation_confidence_256"
HEIGHT_SIGNAL = "height_257"
RELATIVE_HEIGHT_SIGNAL = "relative_height_257"
COARSE_SIGNAL = "coarse_relief_257"
DETAIL_SIGNAL = "detail_residual_257"
REQUIRED_ARRAYS = (
    LUMA_SIGNAL,
    GRADIENT_SIGNAL,
    CONFIDENCE_SIGNAL,
    HEIGHT_SIGNAL,
    RELATIVE_HEIGHT_SIGNAL,
    COARSE_SIGNAL,
    DETAIL_SIGNAL,
)


def _sha256_array(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def load_clean_signal_manifest(corpus_root: str | Path) -> dict[str, Any]:
    root = Path(corpus_root)
    path = root / "clean_signal_manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"clean-signal manifest not found: {path}")
    with path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)
    if manifest.get("schema") != CORPUS_SCHEMA:
        raise ValueError(f"expected schema {CORPUS_SCHEMA!r}, got {manifest.get('schema')!r}")
    rows = manifest.get("rows")
    if not isinstance(rows, list):
        raise ValueError("clean-signal manifest rows must be a list")
    if int(manifest.get("row_count", -1)) != len(rows):
        raise ValueError("clean-signal manifest row_count does not match rows")
    return manifest


def _read_array(payload: Any, name: str, expected_shape: tuple[int, ...], failures: list[str], prefix: str) -> np.ndarray | None:
    if name not in payload:
        failures.append(f"{prefix}: missing {name}")
        return None
    array = np.asarray(payload[name], dtype=np.float32)
    if array.shape != expected_shape:
        failures.append(f"{prefix}: {name} shape {array.shape} != {expected_shape}")
    elif not np.isfinite(array).all():
        failures.append(f"{prefix}: {name} contains non-finite values")
    return array


def validate_clean_signal_corpus(corpus_root: str | Path) -> dict[str, Any]:
    """Validate a corpus without touching any client data or model code."""

    root = Path(corpus_root)
    manifest = load_clean_signal_manifest(root)
    failures: list[str] = []
    seen_ids: set[str] = set()
    group_splits: dict[str, str] = {}
    family_splits: dict[str, str] = {}
    split_counts: dict[str, int] = {}
    family_counts: dict[str, int] = {}
    forbidden_seen: set[str] = set()
    source_counts: dict[str, int] = {}
    split_mode = str(manifest.get("split_mode", "complete_family"))
    if split_mode not in VALID_SPLIT_MODES:
        failures.append(f"invalid split_mode {split_mode!r}")

    for position, row in enumerate(manifest["rows"]):
        prefix = f"row[{position}]"
        if not isinstance(row, dict):
            failures.append(f"{prefix}: row must be an object")
            continue
        row_id = str(row.get("row_id", ""))
        source_group_id = str(row.get("source_group_id", ""))
        family = str(row.get("family", ""))
        split = str(row.get("split", ""))
        source_kind = str(row.get("source_kind", ""))
        if not row_id:
            failures.append(f"{prefix}: missing row_id")
        elif row_id in seen_ids:
            failures.append(f"{prefix}: duplicate row_id {row_id}")
        seen_ids.add(row_id)
        if not source_group_id:
            failures.append(f"{prefix}: missing source_group_id")
        if not family:
            failures.append(f"{prefix}: missing family")
        if split not in VALID_SPLITS:
            failures.append(f"{prefix}: invalid split {split!r}")
        else:
            split_counts[split] = split_counts.get(split, 0) + 1
        if source_kind not in VALID_SOURCE_KINDS:
            failures.append(f"{prefix}: invalid source_kind {source_kind!r}")
        else:
            source_counts[source_kind] = source_counts.get(source_kind, 0) + 1
        if source_group_id:
            prior = group_splits.setdefault(source_group_id, split)
            if prior != split:
                failures.append(f"{prefix}: source_group_id {source_group_id!r} crosses {prior!r}/{split!r}")
        if family:
            family_counts[family] = family_counts.get(family, 0) + 1
            prior = family_splits.setdefault(family, split)
            if split_mode == "complete_family" and prior != split:
                failures.append(f"{prefix}: family {family!r} crosses {prior!r}/{split!r}")

        row_forbidden = row.get("forbidden_signals", [])
        if isinstance(row_forbidden, dict):
            row_forbidden = row_forbidden.keys()
        if not isinstance(row_forbidden, (list, tuple, set)):
            failures.append(f"{prefix}: forbidden_signals must be a list")
            row_forbidden = []
        forbidden_seen.update(str(value) for value in row_forbidden)

        confidence_status = str(row.get("confidence_status", ""))
        gate_status = str(row.get("observation_status", "accepted"))
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
                arrays = {
                    LUMA_SIGNAL: _read_array(payload, LUMA_SIGNAL, IMAGE_SHAPE, failures, prefix),
                    GRADIENT_SIGNAL: _read_array(payload, GRADIENT_SIGNAL, (2, *IMAGE_SHAPE), failures, prefix),
                    CONFIDENCE_SIGNAL: _read_array(payload, CONFIDENCE_SIGNAL, IMAGE_SHAPE, failures, prefix),
                    HEIGHT_SIGNAL: _read_array(payload, HEIGHT_SIGNAL, TARGET_SHAPE, failures, prefix),
                    RELATIVE_HEIGHT_SIGNAL: _read_array(payload, RELATIVE_HEIGHT_SIGNAL, TARGET_SHAPE, failures, prefix),
                    COARSE_SIGNAL: _read_array(payload, COARSE_SIGNAL, TARGET_SHAPE, failures, prefix),
                    DETAIL_SIGNAL: _read_array(payload, DETAIL_SIGNAL, TARGET_SHAPE, failures, prefix),
                }
                if all(arrays[name] is not None for name in (LUMA_SIGNAL, GRADIENT_SIGNAL, CONFIDENCE_SIGNAL)):
                    input_report = validate_clean_observation(
                        arrays[LUMA_SIGNAL],
                        arrays[GRADIENT_SIGNAL],
                        arrays[CONFIDENCE_SIGNAL],
                        confidence_status,
                        observation_status=gate_status,
                        provenance=row.get("observation_provenance"),
                        forbidden_signals=row_forbidden,
                    )
                    failures.extend(f"{prefix}: {failure}" for failure in input_report["failures"])
                relative = arrays[RELATIVE_HEIGHT_SIGNAL]
                coarse = arrays[COARSE_SIGNAL]
                detail = arrays[DETAIL_SIGNAL]
                if relative is not None and (float(relative.min()) < -1e-6 or float(relative.max()) > 1.000001):
                    failures.append(f"{prefix}: relative_height_257 is outside [0, 1]")
                if coarse is not None and (float(coarse.min()) < -1e-6 or float(coarse.max()) > 1.000001):
                    failures.append(f"{prefix}: coarse_relief_257 is outside [0, 1]")
                if relative is not None and coarse is not None and detail is not None:
                    if not np.allclose(relative, coarse + detail, atol=2e-6, rtol=0.0):
                        failures.append(f"{prefix}: coarse/detail recomposition mismatch")
                declared_hashes = row.get("array_hashes", {})
                if not isinstance(declared_hashes, dict):
                    failures.append(f"{prefix}: array_hashes must be an object")
                    declared_hashes = {}
                for name, array in arrays.items():
                    if array is None or name not in declared_hashes:
                        if array is not None:
                            failures.append(f"{prefix}: missing array hash for {name}")
                        continue
                    if str(declared_hashes[name]) != _sha256_array(array):
                        failures.append(f"{prefix}: hash mismatch for {name}")
        except (OSError, ValueError, KeyError) as exc:
            failures.append(f"{prefix}: unable to read {npz_name}: {exc}")

    required_families = {str(value) for value in manifest.get("required_families", [])}
    missing_families = sorted(required_families - set(family_counts))
    if missing_families:
        failures.append(f"missing required families: {missing_families}")
    declared_forbidden = {str(value) for value in manifest.get("forbidden_signals_seen", [])}
    if forbidden_seen != declared_forbidden:
        failures.append("manifest forbidden_signals_seen does not match row metadata")
    if forbidden_seen:
        failures.append(f"forbidden inference signals present: {sorted(forbidden_seen)}")

    return {
        "schema": "v7-clean-signal-validation-v1",
        "corpus_root": str(root),
        "manifest_schema": manifest["schema"],
        "row_count": len(manifest["rows"]),
        "split_mode": split_mode,
        "split_counts": dict(sorted(split_counts.items())),
        "family_counts": dict(sorted(family_counts.items())),
        "family_splits": dict(sorted(family_splits.items())),
        "source_counts": dict(sorted(source_counts.items())),
        "forbidden_signals_seen": sorted(forbidden_seen),
        "failures": failures,
        "valid": not failures,
    }


__all__ = [
    "COARSE_SIGNAL",
    "CONFIDENCE_SIGNAL",
    "CORPUS_SCHEMA",
    "DETAIL_SIGNAL",
    "GRADIENT_SIGNAL",
    "HEIGHT_SIGNAL",
    "LUMA_SIGNAL",
    "RELATIVE_HEIGHT_SIGNAL",
    "REQUIRED_ARRAYS",
    "load_clean_signal_manifest",
    "validate_clean_signal_corpus",
]
