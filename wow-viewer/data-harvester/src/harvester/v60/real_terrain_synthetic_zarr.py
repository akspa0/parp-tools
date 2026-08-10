"""Build the real-terrain synthetic bridge from the complete v50.1 Zarr store."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr

from harvester.v60.clean_signal_corpus import (
    CORPUS_SCHEMA,
    array_sha256,
    validate_clean_signal_corpus,
)
from harvester.v60.clean_signal_inputs import IMAGE_SHAPE, build_clean_observation
from harvester.v60.clean_signal_targets import decompose_relative_height

ZARR_BRIDGE_SCHEMA = "v7-clean-signal-real-terrain-synthetic-zarr-v1"
SOURCE_KIND = "real_terrain_synthetic"


def _sha256_bytes(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _load_rows(store: Path, source_filter: str) -> list[tuple[int, dict[str, Any]]]:
    index_path = store / "index.parquet"
    if not store.is_dir() or not index_path.is_file():
        raise FileNotFoundError(f"v50.1 Zarr store or index not found: {store}")
    if source_filter not in {"synthetic"}:
        raise ValueError("source_filter must be 'synthetic'; authored RGB is not this bridge")
    rows = pq.read_table(index_path).to_pylist()
    selected = [(index, row) for index, row in enumerate(rows) if str(row.get("minimap_source", "")) == source_filter]
    if not selected:
        raise ValueError(f"Zarr store has no rows for minimap_source={source_filter!r}")
    groups = [str(row.get("source_group_id", "")) for _, row in selected]
    if any(not group for group in groups):
        raise ValueError("every selected row requires source_group_id")
    if len(set(groups)) != len(groups):
        raise ValueError("synthetic source_group_id values must be unique")
    for _, row in selected:
        for key in ("map", "tile_x", "tile_y"):
            if key not in row:
                raise ValueError(f"selected Zarr row is missing {key!r}")
    return sorted(selected, key=lambda item: (str(item[1]["map"]), int(item[1]["tile_y"]), int(item[1]["tile_x"])))


def zarr_real_terrain_synthetic_build_plan(
    store: str | Path,
    *,
    validation_map: str,
    source_filter: str = "synthetic",
    confidence_value: float = 1.0,
) -> dict[str, Any]:
    """Return a no-write map-held-out build plan for the complete v50.1 synthetic side."""

    source = Path(store)
    rows = _load_rows(source, source_filter)
    if not validation_map:
        raise ValueError("validation_map is required")
    if not np.isfinite(confidence_value) or not 0.0 <= confidence_value <= 1.0:
        raise ValueError("confidence_value must be finite and within [0, 1]")
    maps = sorted({str(row["map"]) for _, row in rows})
    if validation_map not in maps:
        raise ValueError(f"validation_map {validation_map!r} is not present; available maps: {maps}")
    counts = {name: sum(str(row["map"]) == name for _, row in rows) for name in maps}
    return {
        "schema": ZARR_BRIDGE_SCHEMA,
        "corpus_schema": CORPUS_SCHEMA,
        "source_store": str(source.resolve()),
        "source_index_sha256": _sha256_bytes(source / "index.parquet"),
        "source_filter": source_filter,
        "source_kind": SOURCE_KIND,
        "source_row_count": len(rows),
        "map_counts": counts,
        "validation_map": validation_map,
        "train_row_count": len(rows) - counts[validation_map],
        "validation_row_count": counts[validation_map],
        "split_mode": "complete_family",
        "input_signal": "terrain_shadow_256",
        "target_signal": "height_257",
        "confidence_status": "measured",
        "confidence_value": float(confidence_value),
        "forbidden_signals_seen": [],
        "dry_run": True,
    }


def build_zarr_real_terrain_synthetic_corpus(
    store: str | Path,
    output: str | Path,
    *,
    validation_map: str,
    source_filter: str = "synthetic",
    confidence_value: float = 1.0,
) -> dict[str, Any]:
    """Materialize a complete-family bridge corpus without mutating the source Zarr."""

    source = Path(store)
    plan = zarr_real_terrain_synthetic_build_plan(
        source,
        validation_map=validation_map,
        source_filter=source_filter,
        confidence_value=confidence_value,
    )
    rows = _load_rows(source, source_filter)
    output_path = Path(output)
    partial = output_path.with_name(f"{output_path.name}.partial")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite clean-signal corpus: {output_path}")
    if partial.exists():
        raise FileExistsError(f"refusing to reuse partial clean-signal corpus: {partial}")
    group = zarr.open_group(str(source), mode="r")
    for name in ("terrain_shadow_256", "height_257"):
        if name not in group:
            raise ValueError(f"Zarr store is missing required array {name!r}")
    partial.mkdir(parents=True)
    confidence = np.full(IMAGE_SHAPE, confidence_value, dtype=np.float32)
    manifest_rows: list[dict[str, Any]] = []
    try:
        for source_index, source_row in rows:
            shadow = np.asarray(group["terrain_shadow_256"][source_index], dtype=np.float32)
            height = np.asarray(group["height_257"][source_index], dtype=np.float32)
            if shadow.shape != IMAGE_SHAPE:
                raise ValueError(f"source row {source_index}: terrain_shadow_256 shape {shadow.shape}")
            if height.shape != (257, 257):
                raise ValueError(f"source row {source_index}: height_257 shape {height.shape}")
            if not np.isfinite(shadow).all() or not np.isfinite(height).all():
                raise ValueError(f"source row {source_index}: non-finite source array")
            if float(shadow.min()) < 0.0 or float(shadow.max()) > 1.0:
                raise ValueError(f"source row {source_index}: terrain_shadow_256 outside [0, 1]")
            map_name = str(source_row["map"])
            build = str(source_row.get("build") or "unknown")
            row_id = (
                f"{SOURCE_KIND}-{_slug(build)}-{_slug(map_name)}-"
                f"{int(source_row['tile_x']):02d}-{int(source_row['tile_y']):02d}"
            )
            family = f"{build}:{map_name}"
            provenance = {
                "operation": "real_terrain_synthetic_zarr_observation_v1",
                "source_signal": "terrain_shadow_256",
                "target_signal": "height_257",
                "source_store": str(source.resolve()),
                "source_index_sha256": plan["source_index_sha256"],
                "source_row_index": source_index,
                "source_group_id": str(source_row["source_group_id"]),
                "source_build": build,
                "source_map": map_name,
                "source_minimap_filter": source_filter,
                "artifact_status": "fresh",
                "inference_target_reads": [],
            }
            package = build_clean_observation(shadow, confidence, "measured", provenance=provenance)
            targets = decompose_relative_height(height)
            arrays = {**package.arrays(), **targets.arrays}
            relative_npz = Path("rows") / f"{len(manifest_rows):06d}-{_slug(row_id)}.npz"
            npz_path = partial / relative_npz
            npz_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(npz_path, **arrays)
            manifest_rows.append(
                {
                    "row_id": row_id,
                    "source_kind": SOURCE_KIND,
                    "source_group_id": str(source_row["source_group_id"]),
                    "family": family,
                    "complexity_bucket": str(source_row.get("height_regime") or "real_observation"),
                    "variant": source_index,
                    "split": "validation" if map_name == validation_map else "train",
                    "npz": relative_npz.as_posix(),
                    "confidence_status": "measured",
                    "observation_status": "accepted",
                    "observation_provenance": provenance,
                    "forbidden_signals": [],
                    "array_hashes": {name: array_sha256(array) for name, array in arrays.items()},
                    "map": map_name,
                    "tile_x": int(source_row["tile_x"]),
                    "tile_y": int(source_row["tile_y"]),
                }
            )
        manifest = {
            "schema": CORPUS_SCHEMA,
            "row_count": len(manifest_rows),
            "split_mode": "complete_family",
            "source_real_terrain_synthetic_store": str(source.resolve()),
            "source_real_terrain_synthetic_index_sha256": plan["source_index_sha256"],
            "source_schema": ZARR_BRIDGE_SCHEMA,
            "source_filter": source_filter,
            "validation_map": validation_map,
            "required_families": sorted({str(row["family"]) for row in manifest_rows}),
            "forbidden_signals_seen": [],
            "builder": "harvester.v60.real_terrain_synthetic_zarr.build_zarr_real_terrain_synthetic_corpus",
            "confidence_status": "measured",
            "confidence_value": float(confidence_value),
            "rows": manifest_rows,
        }
        (partial / "clean_signal_manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n",
            encoding="utf-8",
        )
        partial.replace(output_path)
    except Exception:
        raise
    validation = validate_clean_signal_corpus(output_path)
    if not validation["valid"]:
        raise ValueError(f"published Zarr bridge failed validation: {validation['failures'][:8]}")
    return {
        **plan,
        "dry_run": False,
        "output_root": str(output_path.resolve()),
        "manifest": str((output_path / "clean_signal_manifest.json").resolve()),
        "validation": validation,
    }


__all__ = [
    "SOURCE_KIND",
    "ZARR_BRIDGE_SCHEMA",
    "build_zarr_real_terrain_synthetic_corpus",
    "zarr_real_terrain_synthetic_build_plan",
]
