"""Minimal, identity-checked numeric store for Spec 102."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from harvester.spec102.m0 import STRICT_OBJECT_TARGET_KEY
from harvester.spec102.strict_target_contract import (
    REQUIRED_STRICT_OBJECT_TARGET_VERSION,
    STRICT_FRAGMENT_COUNT_FIELD,
    STRICT_FRAGMENT_SHA256_FIELD,
    STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD,
    STRICT_FRAGMENT_TRACE_SCHEMA_FIELD,
    STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY,
    STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD,
    STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD,
    STRICT_FRAGMENT_TRACE_VALIDATED_FIELD,
    STRICT_OBJECT_TARGET_VERSION_FIELD,
    STRICT_TARGET_ASSETS_FIELD,
    STRICT_TARGET_UNRESOLVED_PLACEMENTS_FIELD,
    sha256_tree,
    validate_fragment_trace_sidecar,
)
from harvester.v25.dataset import DEFAULT_CODEC

CURATED_SELECTION_MODE = "curated_selection"
RAW_V18_IDENTITY_SELECTION_MODE = "raw_v18_identity"
SELECTION_MODES = (CURATED_SELECTION_MODE, RAW_V18_IDENTITY_SELECTION_MODE)

STRICT_LIQUID_EVIDENCE_STATUS_FIELD = "object_geometry_target_liquid_evidence_status"
STRICT_LIQUID_COUNTER_FIELDS = (
    "object_geometry_target_liquid_covered_pixel_count",
    "object_geometry_target_liquid_surface_unknown_pixel_count",
    "object_geometry_target_liquid_covered_fragment_count",
    "object_geometry_target_liquid_hidden_fragment_count",
    "object_geometry_target_liquid_above_surface_fragment_count",
    "object_geometry_target_liquid_unknown_fragment_count",
)

SPECS = {
    "minimap_rgb": (np.uint8, (256, 256, 3)),
    STRICT_OBJECT_TARGET_KEY: (np.float32, (257, 257)),
    "object_geometry_visible_top_elevation_257": (np.float32, (257, 257)),
    "object_geometry_visible_terrain_elevation_257": (np.float32, (257, 257)),
    "object_geometry_visible_source_257": (np.uint8, (257, 257)),
    "liquid_mask_256": (np.uint8, (256, 256)),
    "liquid_height_256": (np.float32, (256, 256)),
    "mcnk_flags_16": (np.int32, (16, 16)),
    "normal_xyz_257": (np.int8, (257, 257, 3)),
    "height_257": (np.float32, (257, 257)),
}
SOURCE_ARRAYS = {
    "minimap_rgb": "minimap_rgb",
    STRICT_OBJECT_TARGET_KEY: "object_geometry_visible_mask",
    "object_geometry_visible_top_elevation_257": "object_geometry_visible_top_elevation",
    "object_geometry_visible_terrain_elevation_257": "object_geometry_visible_terrain_elevation",
    "object_geometry_visible_source_257": "object_geometry_visible_source",
    "liquid_mask_256": "liquid_mask",
    "liquid_height_256": "liquid_height",
    "mcnk_flags_16": "mcnk_flags_16",
    "normal_xyz_257": "normal_xyz",
    "height_257": "height_257",
}


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _u8_unit(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value)
    if data.dtype == np.uint8:
        return data
    data = data.astype(np.float32)
    if data.max(initial=0.0) <= 1.5:
        data *= 255.0
    return np.clip(np.rint(data), 0, 255).astype(np.uint8)


def _i8_normals(value: np.ndarray) -> np.ndarray:
    data = np.asarray(value)
    if data.dtype == np.int8:
        return data
    data = data.astype(np.float32)
    if np.abs(data).max(initial=0.0) <= 1.5:
        data *= 127.0
    return np.clip(np.rint(data), -127, 127).astype(np.int8)


def _source_validation(path: Path) -> dict:
    """Record source-wide validation honestly; M0 later judges its audited signals."""
    validation = path / "signal_validation.json"
    if not validation.is_file():
        return {"present": False, "passed": None, "sha256": None}
    try:
        payload = json.loads(validation.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"present": True, "passed": None, "sha256": _sha256_file(validation), "read_error": str(error)}
    return {
        "present": True,
        "passed": payload.get("passed"),
        "sha256": _sha256_file(validation),
        "failures": payload.get("failures", []),
        "warnings": payload.get("warnings", []),
    }


def _validate_v18_source(path: Path, group: zarr.Group, rows: list[dict]) -> str:
    if not rows:
        raise RuntimeError(f"V18 source index is empty: {path}")
    for output_name, (dtype, shape) in SPECS.items():
        source_name = SOURCE_ARRAYS[output_name]
        if source_name not in group:
            raise RuntimeError(f"V18 source is missing required array {source_name}: {path}")
        array = group[source_name]
        expected_shape = (len(rows), *shape)
        if tuple(array.shape) != expected_shape:
            raise RuntimeError(
                f"V18 source array {source_name} has shape {tuple(array.shape)}, expected {expected_shape}"
            )
        documented_conversion = output_name in {"liquid_mask_256", "normal_xyz_257"}
        if documented_conversion and not np.issubdtype(array.dtype, np.number):
            raise RuntimeError(f"V18 source array {source_name} must be numeric for its registered conversion")
        if not documented_conversion and not np.can_cast(array.dtype, np.dtype(dtype), casting="same_kind"):
            raise RuntimeError(f"V18 source array {source_name} cannot safely convert to {np.dtype(dtype)}")
    strict_geometry_count_fields = (
        "object_geometry_target_geometry_unresolved_placement_count",
        "object_geometry_target_fallback_required_placement_count",
        "object_geometry_target_terrain_unknown_pixel_count",
    )
    provenance_fields = (
        STRICT_OBJECT_TARGET_VERSION_FIELD,
        "object_geometry_target_status",
        "object_geometry_target_materialized",
        "object_geometry_target_arrays_present",
        *strict_geometry_count_fields,
        STRICT_LIQUID_EVIDENCE_STATUS_FIELD,
        *STRICT_LIQUID_COUNTER_FIELDS,
        STRICT_FRAGMENT_TRACE_SCHEMA_FIELD,
        STRICT_FRAGMENT_COUNT_FIELD,
        STRICT_FRAGMENT_SHA256_FIELD,
        "object_geometry_target_assets_json",
        "object_geometry_target_unresolved_placements_json",
        STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD,
        STRICT_FRAGMENT_TRACE_VALIDATED_FIELD,
        STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD,
        STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD,
    )
    complete_statuses = {"CompleteEmpty", "CompleteVisible"}
    known_statuses = complete_statuses | {
        "PlacementCatalogUnavailable", "TerrainSurfaceUnavailable", "IncompleteGeometry",
    }
    for row_number, row in enumerate(rows):
        missing = [field for field in provenance_fields if field not in row]
        if missing:
            raise RuntimeError(
                "V18 source lacks strict object-target provenance at row "
                f"{row_number}: {', '.join(missing)}"
            )
        version = str(row.get(STRICT_OBJECT_TARGET_VERSION_FIELD, "") or "")
        status = str(row.get("object_geometry_target_status", ""))
        liquid_evidence_status = str(row.get(STRICT_LIQUID_EVIDENCE_STATUS_FIELD, "") or "")
        materialized = bool(row.get("object_geometry_target_materialized", False))
        arrays_present = bool(row.get("object_geometry_target_arrays_present", False))
        if version != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
            raise RuntimeError(
                f"V18 row {row_number} has unsupported strict object-target version {version!r}; "
                f"expected {REQUIRED_STRICT_OBJECT_TARGET_VERSION!r}"
            )
        if status not in known_statuses:
            raise RuntimeError(f"V18 row {row_number} has unknown strict object-target status {status!r}")
        if liquid_evidence_status in {"", "missing"}:
            raise RuntimeError(f"V18 row {row_number} lacks liquid-evidence status")
        counts: dict[str, int] = {}
        for field in strict_geometry_count_fields:
            try:
                counts[field] = int(row.get(field, 0) or 0)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"V18 row {row_number} has invalid strict object-target count {field}"
                ) from error
            if counts[field] < 0:
                raise RuntimeError(f"V18 row {row_number} has negative strict object-target count {field}")
        if materialized and (status not in complete_statuses or not arrays_present):
            raise RuntimeError(
                f"V18 row {row_number} claims a materialized strict target without complete provenance"
            )
        if status in complete_statuses and not materialized:
            raise RuntimeError(
                f"V18 row {row_number} has complete strict target status but did not materialize its arrays"
            )
        if status not in complete_statuses and materialized:
            raise RuntimeError(f"V18 row {row_number} materialized an incomplete strict target")
        if materialized and any(counts.values()):
            raise RuntimeError(
                f"V18 row {row_number} materialized a strict target with unresolved geometry, fallback, or unknown terrain"
            )
        liquid_counts: dict[str, int] = {}
        for field in STRICT_LIQUID_COUNTER_FIELDS:
            try:
                liquid_counts[field] = int(row.get(field, 0) or 0)
            except (TypeError, ValueError) as error:
                raise RuntimeError(
                    f"V18 row {row_number} has invalid liquid-evidence count {field}"
                ) from error
            if liquid_counts[field] < 0:
                raise RuntimeError(f"V18 row {row_number} has negative liquid-evidence count {field}")
        if liquid_evidence_status == "Dry" and any(liquid_counts.values()):
            raise RuntimeError(
                f"V18 row {row_number} declares Dry liquid evidence with nonzero liquid counters"
            )
    try:
        return validate_fragment_trace_sidecar(path, rows)
    except Exception as error:
        raise RuntimeError(f"V18 source has invalid strict fragment-trace sidecar: {error}") from error


def _validate_raw_v18_identity_selection(selection_store: Path, v18_stores: list[Path], rows: list[dict]) -> None:
    if len(v18_stores) != 1:
        raise RuntimeError("raw_v18_identity selection requires exactly one V18 source")
    if selection_store.resolve() != v18_stores[0].resolve():
        raise RuntimeError("raw_v18_identity selection must read the exact same V18 store it copies")
    for ordinal, row in enumerate(rows):
        try:
            tile_id = int(row["tile_id"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError(f"raw_v18_identity row {ordinal} has no integer tile_id") from error
        if tile_id != ordinal:
            raise RuntimeError(
                "raw_v18_identity requires contiguous ordinal V18 tile_id values; "
                f"row {ordinal} has tile_id {tile_id}"
            )


def _resolve_origin(
    *,
    selected: dict,
    out_row: int,
    sources: dict[str, tuple[Path, zarr.Group, list[dict]]],
    selection_mode: str,
) -> tuple[Path, zarr.Group, int, dict]:
    build = str(selected["build"])
    if build not in sources:
        raise RuntimeError(f"missing V18 source for selected build {build}")
    source_path, source, source_index = sources[build]
    source_row = out_row if selection_mode == RAW_V18_IDENTITY_SELECTION_MODE else int(selected["v18_row"])
    if not 0 <= source_row < len(source_index):
        raise RuntimeError(f"selected V18 row is outside source range: {source_row}")
    origin = source_index[source_row]
    identity_fields = ("build", "map", "tile_id", "tile_x", "tile_y")
    mismatches = [field for field in identity_fields if str(selected[field]) != str(origin[field])]
    if mismatches:
        raise RuntimeError(
            f"signal identity mismatch at output row {out_row}, V18 row {source_row}: {mismatches}"
        )
    return source_path, source, source_row, origin


def _metadata_row(
    *,
    selected: dict,
    out_row: int,
    source_path: Path,
    source_row: int,
    origin: dict,
    selection_mode: str,
    strict_fragment_trace_sidecar_sha256: str,
) -> dict:
    row = dict(selected)
    row.update({key: value for key, value in origin.items() if key.startswith("has_")})
    row["row"] = out_row
    row["v18_row"] = source_row
    row["source_v18_row"] = source_row
    row["source_v18_tile_id"] = int(origin["tile_id"])
    row["selection_mode"] = selection_mode
    row["height_repaired"] = False
    row["identity_verified"] = True
    row["strict_target_version"] = str(
        origin.get(STRICT_OBJECT_TARGET_VERSION_FIELD, "missing") or "missing"
    )
    row["strict_target_materialized"] = bool(origin.get("object_geometry_target_materialized", False))
    row["strict_target_status"] = str(origin.get("object_geometry_target_status", "missing"))
    row["strict_target_arrays_present"] = bool(origin.get("object_geometry_target_arrays_present", False))
    row["strict_target_geometry_unresolved_placement_count"] = int(
        origin.get("object_geometry_target_geometry_unresolved_placement_count", 0) or 0
    )
    row["strict_target_fallback_required_placement_count"] = int(
        origin.get("object_geometry_target_fallback_required_placement_count", 0) or 0
    )
    row["strict_target_terrain_unknown_pixel_count"] = int(
        origin.get("object_geometry_target_terrain_unknown_pixel_count", 0) or 0
    )
    row["strict_target_liquid_evidence_status"] = str(
        origin.get(STRICT_LIQUID_EVIDENCE_STATUS_FIELD, "missing") or "missing"
    )
    for source_field in STRICT_LIQUID_COUNTER_FIELDS:
        numeric_field = source_field.removeprefix("object_geometry_target_")
        row[f"strict_target_{numeric_field}"] = int(origin.get(source_field, 0) or 0)
    row["strict_target_fragment_trace_schema"] = str(
        origin.get(STRICT_FRAGMENT_TRACE_SCHEMA_FIELD, "missing") or "missing"
    )
    row["strict_target_fragment_count"] = int(origin.get(STRICT_FRAGMENT_COUNT_FIELD, 0) or 0)
    row["strict_target_fragment_sha256"] = str(
        origin.get(STRICT_FRAGMENT_SHA256_FIELD, "missing") or "missing"
    )
    row["strict_target_assets_json"] = str(
        origin.get("object_geometry_target_assets_json", "[]") or "[]"
    )
    row["strict_target_unresolved_placements_json"] = str(
        origin.get("object_geometry_target_unresolved_placements_json", "[]") or "[]"
    )
    row["strict_target_fragment_trace_arrays_present"] = bool(
        origin.get(STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD, False)
    )
    row["strict_target_fragment_trace_validated"] = bool(
        origin.get(STRICT_FRAGMENT_TRACE_VALIDATED_FIELD, False)
    )
    row["strict_target_fragment_trace_sidecar_start"] = int(
        origin.get(STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD, -1)
    )
    row["strict_target_fragment_trace_sidecar_end"] = int(
        origin.get(STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD, -1)
    )
    row["strict_target_fragment_trace_source_sidecar_sha256"] = strict_fragment_trace_sidecar_sha256
    liquid_sources = [
        name for name in ("mcnk", "mh2o", "mclq", "unified", "wl")
        if bool(origin.get(f"has_liquid_source_{name}", False))
    ]
    row["liquid_source"] = liquid_sources[0] if len(liquid_sources) == 1 else None
    row["source_v18_store"] = str(source_path.resolve())
    return row


def _progress_binding(
    *,
    selection_store: Path,
    v18_stores: list[Path],
    selection_rows: list[dict],
    selection_mode: str,
) -> dict:
    return {
        "schema": "spec102-numeric-store-progress-v1",
        "selection_mode": selection_mode,
        "selection_store": str(selection_store.resolve()),
        "selection_index_sha256": _sha256_file(selection_store / "index.parquet"),
        "v18_stores": [str(path.resolve()) for path in v18_stores],
        "v18_index_sha256": [_sha256_file(path / "index.parquet") for path in v18_stores],
        "v18_fragment_trace_sidecar_sha256": [
            sha256_tree(path / STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY) for path in v18_stores
        ],
        "total_rows": len(selection_rows),
    }


def _write_progress(output: Path, binding: dict, *, next_row: int, complete: bool) -> None:
    progress = {**binding, "next_row": next_row, "complete": complete}
    target = output / "numeric_build_progress.json"
    temporary = output / "numeric_build_progress.json.tmp"
    temporary.write_text(json.dumps(progress, indent=2), encoding="utf-8")
    temporary.replace(target)


def _open_output(
    *,
    output: Path,
    selection_rows: list[dict],
    selection_mode: str,
    resume: bool,
    binding: dict,
) -> tuple[zarr.Group, dict[str, zarr.Array], int]:
    if output.exists():
        if not resume:
            raise FileExistsError(f"refusing to overwrite numeric store: {output}")
        if selection_mode != RAW_V18_IDENTITY_SELECTION_MODE:
            raise RuntimeError("only raw_v18_identity mode may resume a partial numeric store")
        if (output / "index.parquet").exists() or (output / "contract.json").exists():
            raise RuntimeError("numeric store is already complete; refusing to resume or overwrite it")
        out = zarr.open_group(str(output), mode="a")
        arrays: dict[str, zarr.Array] = {}
        for name, (dtype, shape) in SPECS.items():
            if name not in out:
                raise RuntimeError(f"partial numeric store lacks array {name}")
            array = out[name]
            if tuple(array.shape) != (len(selection_rows), *shape) or np.dtype(array.dtype) != np.dtype(dtype):
                raise RuntimeError(f"partial numeric store has incompatible array {name}")
            arrays[name] = array
        progress_path = output / "numeric_build_progress.json"
        if not progress_path.is_file():
            # The first interrupted pre-resume build has no trustworthy checkpoint.
            # Recopy from zero rather than guessing which chunks are complete.
            return out, arrays, 0
        progress = json.loads(progress_path.read_text(encoding="utf-8"))
        expected = {**binding}
        actual = {key: progress.get(key) for key in expected}
        if actual != expected or progress.get("complete") is True:
            raise RuntimeError("partial numeric store progress is stale, mismatched, or already complete")
        try:
            next_row = int(progress["next_row"])
        except (KeyError, TypeError, ValueError) as error:
            raise RuntimeError("partial numeric store progress has no valid next_row") from error
        if not 0 <= next_row < len(selection_rows):
            raise RuntimeError("partial numeric store progress next_row is outside the source range")
        return out, arrays, next_row

    output.parent.mkdir(parents=True, exist_ok=True)
    out = zarr.open_group(str(output), mode="w")
    arrays = {
        name: out.create_array(
            name, shape=(len(selection_rows), *shape), chunks=(1, *shape),
            dtype=dtype, compressors=DEFAULT_CODEC,
        )
        for name, (dtype, shape) in SPECS.items()
    }
    return out, arrays, 0


def build_numeric_store(
    *,
    selection_store: Path,
    v18_stores: list[Path],
    output: Path,
    selection_mode: str = CURATED_SELECTION_MODE,
    resume: bool = False,
    max_rows: int | None = None,
) -> Path:
    """Copy a selected source exactly; raw mode may resume with a hash-bound checkpoint."""
    if selection_mode not in SELECTION_MODES:
        raise ValueError(f"selection_mode must be one of {SELECTION_MODES}")
    if max_rows is not None and max_rows < 1:
        raise ValueError("max_rows must be positive when supplied")
    selection_rows = pq.read_table(selection_store / "index.parquet").to_pylist()
    if selection_mode == RAW_V18_IDENTITY_SELECTION_MODE:
        _validate_raw_v18_identity_selection(selection_store, v18_stores, selection_rows)
    elif any("v18_row" not in row for row in selection_rows):
        raise RuntimeError("curated_selection requires an explicit v18_row on every selection row")

    sources: dict[str, tuple[Path, zarr.Group, list[dict]]] = {}
    source_fragment_trace_hashes: dict[str, str] = {}
    for path in v18_stores:
        group = zarr.open_group(str(path), mode="r")
        rows = pq.read_table(path / "index.parquet").to_pylist()
        builds = {str(row["build"]) for row in rows}
        if len(builds) != 1:
            raise RuntimeError(f"V18 source must contain exactly one build: {path}")
        fragment_trace_hash = _validate_v18_source(path, group, rows)
        build = next(iter(builds))
        if build in sources:
            raise RuntimeError(f"duplicate V18 source build {build}")
        sources[build] = (path, group, rows)
        source_fragment_trace_hashes[build] = fragment_trace_hash

    binding = _progress_binding(
        selection_store=selection_store,
        v18_stores=v18_stores,
        selection_rows=selection_rows,
        selection_mode=selection_mode,
    )
    out, arrays, start_row = _open_output(
        output=output,
        selection_rows=selection_rows,
        selection_mode=selection_mode,
        resume=resume,
        binding=binding,
    )
    stop_row = min(len(selection_rows), start_row + max_rows) if max_rows is not None else len(selection_rows)
    for out_row in range(start_row, stop_row):
        selected = selection_rows[out_row]
        _source_path, source, source_row, _origin = _resolve_origin(
            selected=selected,
            out_row=out_row,
            sources=sources,
            selection_mode=selection_mode,
        )
        arrays["minimap_rgb"][out_row] = np.asarray(source["minimap_rgb"][source_row], dtype=np.uint8)
        arrays[STRICT_OBJECT_TARGET_KEY][out_row] = np.asarray(
            source["object_geometry_visible_mask"][source_row], dtype=np.float32
        )
        arrays["object_geometry_visible_top_elevation_257"][out_row] = np.asarray(
            source["object_geometry_visible_top_elevation"][source_row], dtype=np.float32
        )
        arrays["object_geometry_visible_terrain_elevation_257"][out_row] = np.asarray(
            source["object_geometry_visible_terrain_elevation"][source_row], dtype=np.float32
        )
        arrays["object_geometry_visible_source_257"][out_row] = np.asarray(
            source["object_geometry_visible_source"][source_row], dtype=np.uint8
        )
        arrays["liquid_mask_256"][out_row] = _u8_unit(source["liquid_mask"][source_row])
        arrays["liquid_height_256"][out_row] = np.asarray(source["liquid_height"][source_row], dtype=np.float32)
        arrays["mcnk_flags_16"][out_row] = np.asarray(source["mcnk_flags_16"][source_row], dtype=np.int32)
        arrays["normal_xyz_257"][out_row] = _i8_normals(source["normal_xyz"][source_row])
        arrays["height_257"][out_row] = np.asarray(source["height_257"][source_row], dtype=np.float32)
    if stop_row < len(selection_rows):
        _write_progress(output, binding, next_row=stop_row, complete=False)
        return output

    copied_rows = []
    for out_row, selected in enumerate(selection_rows):
        source_path, _source, source_row, origin = _resolve_origin(
            selected=selected,
            out_row=out_row,
            sources=sources,
            selection_mode=selection_mode,
        )
        copied_rows.append(
            _metadata_row(
                selected=selected,
                out_row=out_row,
                source_path=source_path,
                source_row=source_row,
                origin=origin,
                selection_mode=selection_mode,
                strict_fragment_trace_sidecar_sha256=source_fragment_trace_hashes[str(origin["build"])],
            )
        )
    pq.write_table(pa.Table.from_pylist(copied_rows), output / "index.parquet")
    source_contracts = {
        build: {
            "store": str(path.resolve()),
            "index_sha256": _sha256_file(path / "index.parquet"),
            "row_count": len(rows),
            "map_count": len({str(row["map"]) for row in rows}),
            "signal_validation": _source_validation(path),
            "strict_fragment_trace_sidecar": {
                "path": str((path / STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY).resolve()),
                "sha256": source_fragment_trace_hashes[build],
            },
        }
        for build, (path, _group, rows) in sources.items()
    }
    out.attrs.update({
        "schema": "spec102-numeric-store-v4",
        "created_utc": datetime.now(UTC).isoformat(),
        "tile_count": len(copied_rows),
        "selection_mode": selection_mode,
        "source_selection_store": str(selection_store.resolve()),
        "source_selection_index_sha256": _sha256_file(selection_store / "index.parquet"),
        "source_v18_stores": [str(path.resolve()) for path in v18_stores],
        "source_v18_contracts": source_contracts,
        "signals": list(SPECS),
        "object_target_provenance": {
            "target_array": STRICT_OBJECT_TARGET_KEY,
            "target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
            "target_kind": "strict_transformed_geometry_terrain_visible",
            "legacy_precise_mask_policy": "prohibited",
            "terrain_occlusion_clipped": True,
            "per_pixel_object_top_elevation": True,
            "top_elevation_array": "object_geometry_visible_top_elevation_257",
            "source_top_elevation_array": "object_geometry_visible_top_elevation",
            "terrain_elevation_array": "object_geometry_visible_terrain_elevation_257",
            "source_terrain_elevation_array": "object_geometry_visible_terrain_elevation",
            "source_array": "object_geometry_visible_source_257",
            "source_source_array": "object_geometry_visible_source",
            "clip_rule": "retain only transformed M2/WMO triangle raster fragments with object_elevation > raw_mcvt_terrain_elevation + 0.25",
            "complete_statuses": ["CompleteEmpty", "CompleteVisible"],
            "liquid_evidence_status_field": "strict_target_liquid_evidence_status",
            "liquid_evidence_required_for_m0": "Dry with every liquid counter equal to zero",
            "liquid_evidence_source_status_field": STRICT_LIQUID_EVIDENCE_STATUS_FIELD,
            "liquid_evidence_source_counter_fields": list(STRICT_LIQUID_COUNTER_FIELDS),
            "fragment_trace": {
                "schema": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
                "sidecar": STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY,
                "arrays": [
                    "object_geometry_fragment_raster_xy",
                    "object_geometry_fragment_world_xyze",
                    "object_geometry_fragment_source_ids",
                    "object_geometry_fragment_source_classification",
                    "object_geometry_fragment_terrain_vertex_dense_xy",
                    "object_geometry_fragment_terrain_vertex_z",
                    "object_geometry_fragment_terrain_vertex_present",
                    "object_geometry_fragment_terrain_weights",
                    "object_geometry_fragment_terrain_liquid_elevation",
                ],
                "required_for_materialized_target": True,
                "model_input_policy": "audit_sidecar_only_not_a_numeric_model_input",
                "metadata_fields": {
                    "schema": STRICT_FRAGMENT_TRACE_SCHEMA_FIELD,
                    "count": STRICT_FRAGMENT_COUNT_FIELD,
                    "sha256": STRICT_FRAGMENT_SHA256_FIELD,
                    "assets": STRICT_TARGET_ASSETS_FIELD,
                    "unresolved_placements": STRICT_TARGET_UNRESOLVED_PLACEMENTS_FIELD,
                },
            },
            "unmaterialized_target_policy": "reject_from_m0_until_complete_empty_or_complete_visible",
            "whole_instance_erasure": "prohibited",
        },
        "prohibited_absent_signals": [
            "clean_minimap_256", "object_mask_256", "object_visibility_256",
            "object_precise_mask_257",
            "wdl_height_33", "placements", "height_repair",
        ],
    })
    (output / "contract.json").write_text(json.dumps(dict(out.attrs), indent=2), encoding="utf-8")
    _write_progress(output, binding, next_row=len(selection_rows), complete=True)
    return output
