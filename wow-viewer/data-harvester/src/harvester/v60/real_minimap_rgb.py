"""Build a minimap-observable RGB baseline corpus from a v50.1 Zarr store."""

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

RGB_BASELINE_SCHEMA = "v7-clean-signal-real-minimap-rgb-v1"
SOURCE_KIND = "real_minimap_diagnostic"
PREPARATION = "raw_luma_v1"
LUMA_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


def _sha256_bytes(path: Path) -> str:
    return f"sha256:{hashlib.sha256(path.read_bytes()).hexdigest()}"


def _slug(value: str) -> str:
    return "".join(character if character.isalnum() or character in "-_" else "_" for character in value)


def _load_rows(store: Path, source_filter: str) -> list[tuple[int, dict[str, Any]]]:
    index_path = store / "index.parquet"
    if not store.is_dir() or not index_path.is_file():
        raise FileNotFoundError(f"v50.1 Zarr store or index not found: {store}")
    if source_filter not in {"synthetic", "authored"}:
        raise ValueError("source_filter must be 'synthetic' or 'authored'")
    rows = pq.read_table(index_path).to_pylist()
    selected = [(index, row) for index, row in enumerate(rows) if str(row.get("minimap_source", "")) == source_filter]
    if not selected:
        raise ValueError(f"Zarr store has no rows for minimap_source={source_filter!r}")
    groups = [str(row.get("source_group_id", "")) for _, row in selected]
    if any(not group for group in groups):
        raise ValueError("every selected row requires source_group_id")
    if len(set(groups)) != len(groups):
        raise ValueError("selected minimap source_group_id values must be unique")
    for _, row in selected:
        for key in ("map", "tile_x", "tile_y"):
            if key not in row:
                raise ValueError(f"selected Zarr row is missing {key!r}")
    return sorted(selected, key=lambda item: (str(item[1]["map"]), int(item[1]["tile_y"]), int(item[1]["tile_x"])))


def real_minimap_rgb_build_plan(
    store: str | Path,
    *,
    source_filter: str,
    validation_map: str,
) -> dict[str, Any]:
    """Return a no-write plan for raw minimap RGB -> luma preparation."""

    source = Path(store)
    rows = _load_rows(source, source_filter)
    maps = sorted({str(row["map"]) for _, row in rows})
    if validation_map not in maps:
        raise ValueError(f"validation_map {validation_map!r} is not present; available maps: {maps}")
    counts = {name: sum(str(row["map"]) == name for _, row in rows) for name in maps}
    return {
        "schema": RGB_BASELINE_SCHEMA,
        "corpus_schema": CORPUS_SCHEMA,
        "source_store": str(source.resolve()),
        "source_index_sha256": _sha256_bytes(source / "index.parquet"),
        "source_filter": source_filter,
        "source_kind": SOURCE_KIND,
        "preparation": PREPARATION,
        "source_row_count": len(rows),
        "map_counts": counts,
        "validation_map": validation_map,
        "train_row_count": len(rows) - counts[validation_map],
        "validation_row_count": counts[validation_map],
        "split_mode": "complete_family",
        "input_signal": "minimap_rgb",
        "output_signal": "clean_observation_luma_256",
        "confidence_status": "absent_explicit",
        "input_contract": "minimap_rgb_to_raw_luma_diagnostic_v1",
        "albedo_gate_status": "not_run",
        "forbidden_signals_seen": [],
        "dry_run": True,
    }


def build_real_minimap_rgb_corpus(
    store: str | Path,
    output: str | Path,
    *,
    source_filter: str,
    validation_map: str,
) -> dict[str, Any]:
    """Materialize raw RGB observations without claiming albedo normalization."""

    source = Path(store)
    plan = real_minimap_rgb_build_plan(source, source_filter=source_filter, validation_map=validation_map)
    output_path = Path(output)
    partial = output_path.with_name(f"{output_path.name}.partial")
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite clean-signal corpus: {output_path}")
    if partial.exists():
        raise FileExistsError(f"refusing to reuse partial clean-signal corpus: {partial}")
    group = zarr.open_group(str(source), mode="r")
    for name in ("minimap_rgb", "height_257"):
        if name not in group:
            raise ValueError(f"Zarr store is missing required array {name!r}")
    rgb_array = group["minimap_rgb"]
    if len(rgb_array.shape) != 4 or tuple(rgb_array.shape[1:]) != (256, 256, 3):
        raise ValueError(f"minimap_rgb shape {rgb_array.shape} != (N, 256, 256, 3)")
    confidence = np.zeros(IMAGE_SHAPE, dtype=np.float32)
    manifest_rows: list[dict[str, Any]] = []
    partial.mkdir(parents=True)
    rows = _load_rows(source, source_filter)
    try:
        for source_index, source_row in rows:
            rgb = np.asarray(rgb_array[source_index], dtype=np.float32)
            height = np.asarray(group["height_257"][source_index], dtype=np.float32)
            if rgb.shape != (256, 256, 3):
                raise ValueError(f"source row {source_index}: minimap_rgb shape {rgb.shape}")
            if height.shape != (257, 257):
                raise ValueError(f"source row {source_index}: height_257 shape {height.shape}")
            if not np.isfinite(rgb).all() or not np.isfinite(height).all():
                raise ValueError(f"source row {source_index}: non-finite source array")
            if float(rgb.min()) < 0.0 or float(rgb.max()) > 255.0:
                raise ValueError(f"source row {source_index}: minimap_rgb outside [0, 255]")
            luma = np.tensordot(rgb / 255.0, LUMA_WEIGHTS, axes=([-1], [0])).astype(np.float32)
            map_name = str(source_row["map"])
            build = str(source_row.get("build") or "unknown")
            row_id = (
                f"{SOURCE_KIND}-{source_filter}-{_slug(build)}-{_slug(map_name)}-"
                f"{int(source_row['tile_x']):02d}-{int(source_row['tile_y']):02d}"
            )
            provenance = {
                "operation": "real_minimap_rgb_raw_luma_v1",
                "source_signal": "minimap_rgb",
                "preparation": PREPARATION,
                "source_store": str(source.resolve()),
                "source_index_sha256": plan["source_index_sha256"],
                "source_row_index": source_index,
                "source_group_id": str(source_row["source_group_id"]),
                "source_build": build,
                "source_map": map_name,
                "source_minimap_filter": source_filter,
                "albedo_gate_status": "not_run",
                "artifact_status": "fresh",
                "inference_target_reads": [],
            }
            package = build_clean_observation(
                luma,
                confidence,
                "absent_explicit",
                provenance=provenance,
            )
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
                    "family": f"{build}:{map_name}",
                    "complexity_bucket": str(source_row.get("height_regime") or "real_observation"),
                    "variant": source_index,
                    "split": "validation" if map_name == validation_map else "train",
                    "npz": relative_npz.as_posix(),
                    "confidence_status": "absent_explicit",
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
            "source_real_minimap_store": str(source.resolve()),
            "source_real_minimap_index_sha256": plan["source_index_sha256"],
            "source_schema": RGB_BASELINE_SCHEMA,
            "source_filter": source_filter,
            "preparation": PREPARATION,
            "validation_map": validation_map,
            "input_contract": plan["input_contract"],
            "albedo_gate_status": "not_run",
            "required_families": sorted({str(row["family"]) for row in manifest_rows}),
            "forbidden_signals_seen": [],
            "builder": "harvester.v60.real_minimap_rgb.build_real_minimap_rgb_corpus",
            "confidence_status": "absent_explicit",
            "confidence_value": 0.0,
            "rows": manifest_rows,
        }
        (partial / "clean_signal_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        partial.replace(output_path)
    except Exception:
        raise
    validation = validate_clean_signal_corpus(output_path)
    if not validation["valid"]:
        raise ValueError(f"published raw-RGB corpus failed validation: {validation['failures'][:8]}")
    return {
        **plan,
        "dry_run": False,
        "output_root": str(output_path.resolve()),
        "manifest": str((output_path / "clean_signal_manifest.json").resolve()),
        "validation": validation,
    }


__all__ = [
    "PREPARATION",
    "RGB_BASELINE_SCHEMA",
    "SOURCE_KIND",
    "build_real_minimap_rgb_corpus",
    "real_minimap_rgb_build_plan",
]
