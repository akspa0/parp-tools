"""Manifest, hash, split, and array validation for the clean-signal v60 corpus."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from harvester.v60.clean_signal_inputs import (
    IMAGE_SHAPE,
    build_clean_observation,
    validate_clean_observation,
)
from harvester.v60.clean_signal_targets import TARGET_SHAPE, decompose_relative_height
from harvester.v60.control_corpus import load_control_manifest, validate_control_corpus

CORPUS_SCHEMA = "v7-clean-signal-corpus-v1"
VALID_SOURCE_KINDS = frozenset(
    {"synthetic_control", "accepted_real", "real_terrain_synthetic", "real_minimap_diagnostic"}
)
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


def array_sha256(array: np.ndarray) -> str:
    canonical = np.ascontiguousarray(np.asarray(array, dtype="<f4"))
    return hashlib.sha256(canonical.tobytes(order="C")).hexdigest()


def _sha256_array(array: np.ndarray) -> str:
    """Backward-compatible private alias for validators written during the contract slice."""

    return array_sha256(array)


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


def clean_signal_build_plan(
    control_root: str | Path,
    *,
    confidence_value: float = 1.0,
) -> dict[str, Any]:
    """Validate the source control manifest and return a no-write corpus build plan."""

    root = Path(control_root)
    validation = validate_control_corpus(root)
    if not validation["valid"]:
        failures = "; ".join(str(value) for value in validation["failures"][:8])
        raise ValueError(f"invalid control corpus: {failures}")
    if not np.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise ValueError("confidence_value must be finite and within [0, 1]")
    manifest = load_control_manifest(root)
    families = sorted({str(row["control_family"]) for row in manifest["rows"]})
    return {
        "schema": "v7-clean-signal-build-plan-v1",
        "source_root": str(root.resolve()),
        "source_schema": manifest["schema"],
        "source_manifest": str((root / "control_manifest.json").resolve()),
        "source_row_count": len(manifest["rows"]),
        "row_count": len(manifest["rows"]),
        "families": families,
        "family_count": len(families),
        "split_mode": "complete_family",
        "confidence_status": "measured",
        "confidence_value": float(confidence_value),
        "input_observation": "terrain_shadow_256_as_synthetic_luma",
        "forbidden_signals_seen": [],
        "dry_run": True,
    }


def build_clean_signal_corpus(
    control_root: str | Path,
    output_root: str | Path,
    *,
    confidence_value: float = 1.0,
) -> dict[str, Any]:
    """Materialize a clean-signal corpus atomically from an already validated control corpus."""

    plan = clean_signal_build_plan(control_root, confidence_value=confidence_value)
    source_root = Path(control_root)
    output = Path(output_root)
    partial = output.with_name(f"{output.name}.partial")
    if output.exists():
        raise FileExistsError(f"refusing to overwrite existing clean-signal corpus: {output}")
    if partial.exists():
        raise FileExistsError(f"refusing to reuse existing partial clean-signal corpus: {partial}")
    partial.mkdir(parents=True)
    rows: list[dict[str, Any]] = []
    source_manifest_bytes = (source_root / "control_manifest.json").read_bytes()
    source_manifest_sha256 = hashlib.sha256(source_manifest_bytes).hexdigest()
    source_manifest = load_control_manifest(source_root)
    confidence = np.full(IMAGE_SHAPE, confidence_value, dtype=np.float32)
    try:
        for position, source_row in enumerate(source_manifest["rows"]):
            source_npz = source_root / str(source_row["npz"])
            with np.load(source_npz, allow_pickle=False) as payload:
                if "terrain_shadow_256" not in payload or "height_257" not in payload:
                    raise ValueError(f"source row {source_row['row_id']!r} is missing the control pair")
                shadow = np.asarray(payload["terrain_shadow_256"], dtype=np.float32)
                height = np.asarray(payload["height_257"], dtype=np.float32)
            provenance = {
                "operation": "synthetic_control_observation_v1",
                "source_signal": "terrain_shadow_256",
                "source_manifest_sha256": source_manifest_sha256,
                "source_row_id": str(source_row["row_id"]),
                "artifact_status": "fresh",
            }
            package = build_clean_observation(
                shadow,
                confidence,
                "measured",
                provenance=provenance,
            )
            target = decompose_relative_height(height)
            arrays = {**package.arrays(), **target.arrays}
            slug = "".join(character if character.isalnum() or character in "-_" else "_" for character in str(source_row["row_id"]))
            relative_npz = Path("rows") / f"{position:06d}-{slug}.npz"
            npz_path = partial / relative_npz
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(npz_path, **arrays)
            output_row = {
                    "row_id": str(source_row["row_id"]),
                    "source_kind": "synthetic_control",
                    "source_group_id": str(source_row.get("source_group_id", source_row["control_family"])),
                    "family": str(source_row["control_family"]),
                    "complexity_bucket": str(source_row.get("complexity_bucket", "")),
                    "variant": int(source_row.get("variant", 0)),
                    "split": str(source_row["split"]),
                    "npz": relative_npz.as_posix(),
                    "confidence_status": "measured",
                    "observation_status": "accepted",
                    "observation_provenance": provenance,
                    "forbidden_signals": [],
                    "array_hashes": {name: array_sha256(array) for name, array in arrays.items()},
                }
            for metadata_key in (
                "pattern_id",
                "pattern_tile_x",
                "pattern_tile_y",
                "pattern_tile_span",
                "pattern_continuity",
                "cell_alignment",
                "field_offset_x",
                "field_offset_y",
            ):
                if metadata_key in source_row:
                    output_row[metadata_key] = source_row[metadata_key]
            rows.append(output_row)
        manifest = {
            "schema": CORPUS_SCHEMA,
            "row_count": len(rows),
            "split_mode": "complete_family",
            "source_control_manifest": str((source_root / "control_manifest.json").resolve()),
            "source_control_manifest_sha256": source_manifest_sha256,
            "source_control_schema": source_manifest["schema"],
            "required_families": sorted({str(row["family"]) for row in rows}),
            "forbidden_signals_seen": [],
            "builder": "harvester.v60.clean_signal_corpus.build_clean_signal_corpus",
            "confidence_status": "measured",
            "confidence_value": float(confidence_value),
            "rows": rows,
        }
        (partial / "clean_signal_manifest.json").write_text(
            json.dumps(manifest, indent=2),
            encoding="utf-8",
        )
        partial.replace(output)
    except Exception:
        # Leave the exact partial root for inspection; it has no manifest-bearing valid corpus.
        raise
    return {
        **plan,
        "dry_run": False,
        "output_root": str(output.resolve()),
        "manifest": str((output / "clean_signal_manifest.json").resolve()),
        "row_count": len(rows),
    }


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
    "array_sha256",
    "build_clean_signal_corpus",
    "clean_signal_build_plan",
    "load_clean_signal_manifest",
    "validate_clean_signal_corpus",
]
