"""Exact, fail-closed contract for the strict Spec 102 geometry target.

The fragment trace is real transformed-mesh / terrain-interpolation evidence.
It deliberately lives in a variable-length audit sidecar, never in an image
tensor or an M0 forward-input batch.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import zarr

# This is intentionally one exact, editable contract knob. Do not infer a
# version from target pixels or accept an older target as an equivalent label.
# Advance it only with the corresponding C# emitter, provenance schema, and
# focused contract tests.
REQUIRED_STRICT_OBJECT_TARGET_VERSION = "strict-geometry-terrain-liquid-fragment-trace-v3"
STRICT_OBJECT_TARGET_VERSION_FIELD = "object_geometry_target_version"

STRICT_FRAGMENT_TRACE_SCHEMA_FIELD = "object_geometry_fragment_trace_schema"
STRICT_FRAGMENT_COUNT_FIELD = "object_geometry_fragment_count"
STRICT_FRAGMENT_SHA256_FIELD = "object_geometry_fragment_sha256"
STRICT_TARGET_ASSETS_FIELD = "object_geometry_target_assets"
STRICT_TARGET_UNRESOLVED_PLACEMENTS_FIELD = "object_geometry_target_unresolved_placements"
# The tile-decode path (`build_v18_dataset`) validates the structured fields
# above straight from the C# metadata blob.  The persisted V18 parquet index,
# however, can only carry scalar columns, so it stores the same data as these
# canonical JSON strings.  Sidecar re-validation reads the index rows, so the
# trace validator must accept either shape for the same fact.
STRICT_TARGET_ASSETS_JSON_FIELD = "object_geometry_target_assets_json"
STRICT_TARGET_UNRESOLVED_PLACEMENTS_JSON_FIELD = "object_geometry_target_unresolved_placements_json"
STRICT_FRAGMENT_TRACE_VALIDATED_FIELD = "object_geometry_fragment_trace_validated"
STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD = "object_geometry_fragment_trace_arrays_present"
STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD = "object_geometry_fragment_trace_sidecar_start"
STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD = "object_geometry_fragment_trace_sidecar_end"

STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY = "object_geometry_fragment_trace.zarr"
STRICT_FRAGMENT_TRACE_SIDECAR_SCHEMA = "spec102-object-geometry-fragment-trace-sidecar-v1"
STRICT_FRAGMENT_TRACE_OFFSETS_KEY = "tile_offsets"

# These names, dtypes, dimensions, metadata names, and the canonical SHA-256
# layout mirror WowViewer.Core ObjectGeometryFragmentTrace exactly. They are a
# proof sidecar, not an array list to add to `numeric_store.SPECS`.
STRICT_FRAGMENT_TRACE_ARRAY_SPECS: tuple[tuple[str, np.dtype[Any], int], ...] = (
    ("object_geometry_fragment_raster_xy", np.dtype("<i4"), 2),
    ("object_geometry_fragment_world_xyze", np.dtype("<f4"), 4),
    ("object_geometry_fragment_source_ids", np.dtype("<i4"), 3),
    ("object_geometry_fragment_source_classification", np.dtype("u1"), 2),
    ("object_geometry_fragment_terrain_vertex_dense_xy", np.dtype("<i4"), 6),
    ("object_geometry_fragment_terrain_vertex_z", np.dtype("<f4"), 3),
    ("object_geometry_fragment_terrain_vertex_present", np.dtype("u1"), 3),
    ("object_geometry_fragment_terrain_weights", np.dtype("<f4"), 3),
    ("object_geometry_fragment_terrain_liquid_elevation", np.dtype("<f4"), 2),
)
STRICT_FRAGMENT_TRACE_ARRAY_NAMES = tuple(spec[0] for spec in STRICT_FRAGMENT_TRACE_ARRAY_SPECS)
STRICT_FRAGMENT_TRACE_COLUMNS = {name: columns for name, _dtype, columns in STRICT_FRAGMENT_TRACE_ARRAY_SPECS}
STRICT_FRAGMENT_TRACE_DTYPES = {name: dtype for name, dtype, _columns in STRICT_FRAGMENT_TRACE_ARRAY_SPECS}

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SOURCE_KINDS = {"M2Triangle": 1, "WmoTriangle": 2}
_FRAGMENT_CLASSIFICATIONS = frozenset({1, 2, 3, 4, 5})
_COMPLETE_STATUSES = frozenset({"CompleteEmpty", "CompleteVisible"})


class StrictFragmentTraceError(RuntimeError):
    """Raised when a v3 strict target lacks lossless mesh-trace proof."""


@dataclass(frozen=True)
class StrictFragmentTrace:
    """Validated materialized-tile trace retained for an audit sidecar write."""

    arrays: dict[str, np.ndarray]
    count: int
    sha256: str
    assets: tuple[dict[str, Any], ...]
    unresolved_placements: tuple[dict[str, Any], ...]

    @property
    def assets_json(self) -> str:
        return canonical_json(self.assets)

    @property
    def unresolved_placements_json(self) -> str:
        return canonical_json(self.unresolved_placements)


def canonical_json(value: Any) -> str:
    """Return one deterministic representation for index provenance fields."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def sha256_tree(path: Path) -> str:
    """Hash a sidecar directory as a stable source-binding artifact."""
    if not path.is_dir():
        raise StrictFragmentTraceError(f"strict fragment-trace sidecar is missing: {path}")
    digest = hashlib.sha256()
    for child in sorted((item for item in path.rglob("*") if item.is_file()), key=lambda item: item.relative_to(path).as_posix()):
        relative = child.relative_to(path).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, byteorder="little", signed=False))
        digest.update(relative)
        with child.open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _metadata_int(metadata: Mapping[str, Any], field: str) -> int:
    value = metadata.get(field)
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise StrictFragmentTraceError(f"strict fragment trace metadata {field} must be an integer")
    result = int(value)
    if result < 0:
        raise StrictFragmentTraceError(f"strict fragment trace metadata {field} cannot be negative")
    return result


def _metadata_list(metadata: Mapping[str, Any], *fields: str) -> list[dict[str, Any]]:
    """Read a list-of-objects fact from the first present of one or more fields.

    The same fact is a structured array in the C# tile blob and a canonical JSON
    string in the persisted parquet index, so callers pass both names.
    """
    value: Any = None
    chosen = fields[0]
    for field in fields:
        candidate = metadata.get(field)
        if candidate is not None:
            value = candidate
            chosen = field
            break
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise StrictFragmentTraceError(
                f"strict fragment trace metadata {chosen} is not valid JSON"
            ) from error
    if not isinstance(value, list) or not all(isinstance(item, dict) for item in value):
        raise StrictFragmentTraceError(
            f"strict fragment trace metadata {fields[0]} must be an array of objects"
        )
    return [dict(item) for item in value]


def _validate_assets(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    assets = _metadata_list(metadata, STRICT_TARGET_ASSETS_FIELD, STRICT_TARGET_ASSETS_JSON_FIELD)
    normalized: list[dict[str, Any]] = []
    for expected_index, asset in enumerate(sorted(assets, key=lambda item: item.get("asset_index", -1))):
        asset_index = asset.get("asset_index")
        source = asset.get("source")
        path = asset.get("normalized_asset_path")
        if isinstance(asset_index, bool) or not isinstance(asset_index, (int, np.integer)):
            raise StrictFragmentTraceError("strict fragment trace asset_index must be an integer")
        if int(asset_index) != expected_index:
            raise StrictFragmentTraceError(
                "strict fragment trace assets must have contiguous canonical asset_index values"
            )
        if source not in _SOURCE_KINDS:
            raise StrictFragmentTraceError(f"strict fragment trace asset {expected_index} has unknown source {source!r}")
        if not isinstance(path, str):
            raise StrictFragmentTraceError(
                f"strict fragment trace asset {expected_index} has no normalized_asset_path"
            )
        normalized.append(
            {
                "asset_index": expected_index,
                "source": source,
                "normalized_asset_path": path,
            }
        )
    return tuple(normalized)


def _validate_unresolved_placements(metadata: Mapping[str, Any]) -> tuple[dict[str, Any], ...]:
    placements = _metadata_list(
        metadata,
        STRICT_TARGET_UNRESOLVED_PLACEMENTS_FIELD,
        STRICT_TARGET_UNRESOLVED_PLACEMENTS_JSON_FIELD,
    )
    normalized: list[dict[str, Any]] = []
    identities: set[tuple[int, str, str, str]] = set()
    for placement in placements:
        unique_id = placement.get("placement_unique_id")
        source = placement.get("source")
        path = placement.get("normalized_asset_path")
        reason = placement.get("reason")
        if isinstance(unique_id, bool) or not isinstance(unique_id, (int, np.integer)):
            raise StrictFragmentTraceError("strict unresolved placement unique id must be an integer")
        if source not in _SOURCE_KINDS:
            raise StrictFragmentTraceError(f"strict unresolved placement has unknown source {source!r}")
        if not isinstance(path, str) or not isinstance(reason, str) or not reason:
            raise StrictFragmentTraceError("strict unresolved placement must retain path and nonempty reason")
        identity = (int(unique_id), source, path, reason)
        if identity in identities:
            raise StrictFragmentTraceError("strict unresolved placements contain a duplicate identity")
        identities.add(identity)
        normalized.append(
            {
                "placement_unique_id": int(unique_id),
                "source": source,
                "normalized_asset_path": path,
                "reason": reason,
            }
        )
    return tuple(normalized)


def _require_array(data: Mapping[str, Any], name: str, dtype: np.dtype[Any], columns: int, count: int) -> np.ndarray:
    if name not in data:
        raise StrictFragmentTraceError(f"materialized strict target is missing trace array {name}")
    array = np.asarray(data[name])
    if array.dtype != dtype:
        raise StrictFragmentTraceError(
            f"strict fragment trace array {name} has dtype {array.dtype}, expected {dtype}"
        )
    if array.ndim != 2 or tuple(array.shape) != (count, columns):
        raise StrictFragmentTraceError(
            f"strict fragment trace array {name} has shape {tuple(array.shape)}, expected {(count, columns)}"
        )
    return np.ascontiguousarray(array)


def _add_i32(digest: hashlib._Hash, value: int) -> None:
    digest.update(np.asarray(value, dtype="<i4").tobytes())


def _add_f32(digest: hashlib._Hash, value: np.float32) -> None:
    digest.update(np.asarray(value, dtype="<f4").tobytes())


def _add_u8(digest: hashlib._Hash, value: np.uint8) -> None:
    digest.update(bytes((int(value),)))


def compute_strict_fragment_trace_sha256(
    arrays: Mapping[str, np.ndarray],
    assets: tuple[dict[str, Any], ...],
) -> str:
    """Reproduce C# ObjectGeometryFragmentTrace.ComputeSha256 byte-for-byte."""
    raster_xy = arrays["object_geometry_fragment_raster_xy"]
    world_xyze = arrays["object_geometry_fragment_world_xyze"]
    source_ids = arrays["object_geometry_fragment_source_ids"]
    source_classification = arrays["object_geometry_fragment_source_classification"]
    terrain_dense_xy = arrays["object_geometry_fragment_terrain_vertex_dense_xy"]
    terrain_z = arrays["object_geometry_fragment_terrain_vertex_z"]
    terrain_present = arrays["object_geometry_fragment_terrain_vertex_present"]
    terrain_weights = arrays["object_geometry_fragment_terrain_weights"]
    terrain_liquid = arrays["object_geometry_fragment_terrain_liquid_elevation"]
    digest = hashlib.sha256()
    _add_i32(digest, len(assets))
    for asset in assets:
        _add_i32(digest, int(asset["asset_index"]))
        _add_u8(digest, np.uint8(_SOURCE_KINDS[str(asset["source"])]))
        path = str(asset["normalized_asset_path"]).encode("utf-8")
        _add_i32(digest, len(path))
        digest.update(path)
    count = int(raster_xy.shape[0])
    _add_i32(digest, count)
    for row in range(count):
        _add_i32(digest, int(raster_xy[row, 0]))
        _add_i32(digest, int(raster_xy[row, 1]))
        for column in range(4):
            _add_f32(digest, world_xyze[row, column])
        for column in range(3):
            _add_i32(digest, int(source_ids[row, column]))
        _add_u8(digest, source_classification[row, 0])
        _add_u8(digest, source_classification[row, 1])
        for column in range(6):
            _add_i32(digest, int(terrain_dense_xy[row, column]))
        for column in range(3):
            _add_f32(digest, terrain_z[row, column])
        for column in range(3):
            _add_u8(digest, terrain_present[row, column])
        for column in range(3):
            _add_f32(digest, terrain_weights[row, column])
        for column in range(2):
            _add_f32(digest, terrain_liquid[row, column])
    return digest.hexdigest()


def _validate_trace_values(arrays: Mapping[str, np.ndarray], assets: tuple[dict[str, Any], ...]) -> None:
    raster_xy = arrays["object_geometry_fragment_raster_xy"]
    world_xyze = arrays["object_geometry_fragment_world_xyze"]
    source_ids = arrays["object_geometry_fragment_source_ids"]
    source_classification = arrays["object_geometry_fragment_source_classification"]
    terrain_dense_xy = arrays["object_geometry_fragment_terrain_vertex_dense_xy"]
    terrain_present = arrays["object_geometry_fragment_terrain_vertex_present"]
    if np.any(raster_xy < 0) or np.any(raster_xy >= 257):
        raise StrictFragmentTraceError("strict fragment trace raster coordinates are outside the 257x257 lattice")
    if not np.isfinite(world_xyze).all():
        raise StrictFragmentTraceError("strict fragment trace has nonfinite transformed object coordinates")
    if np.any(source_ids[:, 1] < 0) or np.any(source_ids[:, 1] >= len(assets)):
        raise StrictFragmentTraceError("strict fragment trace references an asset outside the canonical asset table")
    if np.any(source_ids[:, 2] < 0):
        raise StrictFragmentTraceError("strict fragment trace has a negative source triangle index")
    if not set(np.unique(source_classification[:, 0]).tolist()).issubset(set(_SOURCE_KINDS.values())):
        raise StrictFragmentTraceError("strict fragment trace has an unknown M2/WMO source kind")
    if not set(np.unique(source_classification[:, 1]).tolist()).issubset(_FRAGMENT_CLASSIFICATIONS):
        raise StrictFragmentTraceError("strict fragment trace has an unknown fragment classification")
    if not np.isin(terrain_present, (0, 1)).all():
        raise StrictFragmentTraceError("strict fragment trace terrain-vertex presence flags must be binary")
    present_coordinates = terrain_dense_xy.reshape(-1, 3, 2)[terrain_present.astype(bool)]
    if present_coordinates.size and (np.any(present_coordinates < 0) or np.any(present_coordinates >= 257)):
        raise StrictFragmentTraceError("strict fragment trace has an invalid present raw-MCVT vertex coordinate")


def validate_materialized_strict_fragment_trace(
    data: Mapping[str, Any],
    metadata: Mapping[str, Any],
    *,
    status: str,
) -> StrictFragmentTrace:
    """Validate exact C# v3 trace data for one materialized target tile."""
    if str(metadata.get(STRICT_OBJECT_TARGET_VERSION_FIELD, "") or "") != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
        raise StrictFragmentTraceError("strict fragment trace does not declare the exact v3 target version")
    if status not in _COMPLETE_STATUSES:
        raise StrictFragmentTraceError("only complete strict target statuses may carry a materialized v3 trace")
    if str(metadata.get(STRICT_FRAGMENT_TRACE_SCHEMA_FIELD, "") or "") != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
        raise StrictFragmentTraceError("strict fragment trace metadata schema does not match the exact C# v3 contract")
    count = _metadata_int(metadata, STRICT_FRAGMENT_COUNT_FIELD)
    declared_sha256 = str(metadata.get(STRICT_FRAGMENT_SHA256_FIELD, "") or "")
    if not _SHA256_RE.fullmatch(declared_sha256):
        raise StrictFragmentTraceError("strict fragment trace metadata must contain a lowercase SHA-256 hash")
    assets = _validate_assets(metadata)
    unresolved = _validate_unresolved_placements(metadata)
    unresolved_count = _metadata_int(metadata, "object_geometry_target_geometry_unresolved_placement_count")
    placement_count = _metadata_int(metadata, "object_geometry_target_placement_count")
    resolved_count = _metadata_int(metadata, "object_geometry_target_geometry_resolved_placement_count")
    fallback_count = _metadata_int(metadata, "object_geometry_target_fallback_required_placement_count")
    terrain_unknown_count = _metadata_int(metadata, "object_geometry_target_terrain_unknown_pixel_count")
    if unresolved_count != len(unresolved):
        raise StrictFragmentTraceError("strict unresolved placement metadata count does not match its retained records")
    if placement_count != resolved_count + unresolved_count:
        raise StrictFragmentTraceError("strict resolved/unresolved placement counts do not equal placement_count")
    if unresolved_count or fallback_count or terrain_unknown_count:
        raise StrictFragmentTraceError("a materialized strict target cannot retain unresolved/fallback/unknown-terrain provenance")
    arrays = {
        name: _require_array(data, name, dtype, columns, count)
        for name, dtype, columns in STRICT_FRAGMENT_TRACE_ARRAY_SPECS
    }
    _validate_trace_values(arrays, assets)
    if status == "CompleteVisible" and count == 0:
        raise StrictFragmentTraceError("CompleteVisible strict target has an empty fragment trace")
    actual_sha256 = compute_strict_fragment_trace_sha256(arrays, assets)
    if actual_sha256 != declared_sha256:
        raise StrictFragmentTraceError("strict fragment trace SHA-256 does not match its numeric arrays and asset table")
    return StrictFragmentTrace(
        arrays=arrays,
        count=count,
        sha256=declared_sha256,
        assets=assets,
        unresolved_placements=unresolved,
    )


class StrictFragmentTraceSidecar:
    """Append-only variable-length trace sidecar, indexed by V18 tile ordinal."""

    def __init__(self, path: Path, store: zarr.storage.LocalStore, group: zarr.Group):
        self.path = path
        self._store = store
        self._group = group

    @classmethod
    def open(
        cls,
        store_root: Path,
        *,
        tile_capacity: int,
        resume_tile_count: int = 0,
    ) -> "StrictFragmentTraceSidecar":
        path = store_root / STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY
        new_sidecar = not path.exists()
        store = zarr.storage.LocalStore(str(path), read_only=False)
        group = zarr.open_group(store=store, mode="w" if new_sidecar else "a")
        if new_sidecar:
            group.attrs.update(
                {
                    "schema": STRICT_FRAGMENT_TRACE_SIDECAR_SCHEMA,
                    "target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
                    "arrays": list(STRICT_FRAGMENT_TRACE_ARRAY_NAMES),
                }
            )
            group.create_array(
                STRICT_FRAGMENT_TRACE_OFFSETS_KEY,
                shape=(max(1, tile_capacity + 1),),
                chunks=(max(1, min(4096, tile_capacity + 1)),),
                dtype=np.int64,
                fill_value=-1,
            )
            group[STRICT_FRAGMENT_TRACE_OFFSETS_KEY][0] = 0
            for name, dtype, columns in STRICT_FRAGMENT_TRACE_ARRAY_SPECS:
                group.create_array(
                    name,
                    shape=(0, columns),
                    chunks=(max(1, min(4096, max(1, tile_capacity))), columns),
                    dtype=dtype,
                    fill_value=0,
                )
        sidecar = cls(path, store, group)
        sidecar._validate_layout()
        sidecar.ensure_tile_capacity(tile_capacity)
        if resume_tile_count:
            sidecar.truncate_to_tile_count(resume_tile_count)
        return sidecar

    def _validate_layout(self) -> None:
        if self._group.attrs.get("schema") != STRICT_FRAGMENT_TRACE_SIDECAR_SCHEMA:
            raise StrictFragmentTraceError("strict fragment-trace sidecar has an unsupported schema")
        if self._group.attrs.get("target_version") != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
            raise StrictFragmentTraceError("strict fragment-trace sidecar is not bound to the exact v3 target version")
        if self._group.attrs.get("arrays") != list(STRICT_FRAGMENT_TRACE_ARRAY_NAMES):
            raise StrictFragmentTraceError("strict fragment-trace sidecar does not declare the exact nine C# arrays")
        if STRICT_FRAGMENT_TRACE_OFFSETS_KEY not in self._group:
            raise StrictFragmentTraceError("strict fragment-trace sidecar lacks tile offsets")
        for name, dtype, columns in STRICT_FRAGMENT_TRACE_ARRAY_SPECS:
            if name not in self._group:
                raise StrictFragmentTraceError(f"strict fragment-trace sidecar lacks {name}")
            array = self._group[name]
            if np.dtype(array.dtype) != dtype or array.ndim != 2 or array.shape[1] != columns:
                raise StrictFragmentTraceError(f"strict fragment-trace sidecar has an invalid {name} layout")

    def ensure_tile_capacity(self, tile_capacity: int) -> None:
        offsets = self._group[STRICT_FRAGMENT_TRACE_OFFSETS_KEY]
        needed = max(1, int(tile_capacity) + 1)
        if offsets.shape[0] < needed:
            offsets.resize((needed,))

    def truncate_to_tile_count(self, tile_count: int) -> None:
        offsets = self._group[STRICT_FRAGMENT_TRACE_OFFSETS_KEY]
        if tile_count < 0 or offsets.shape[0] <= tile_count:
            raise StrictFragmentTraceError("strict fragment-trace sidecar cannot resume the requested tile count")
        values = np.asarray(offsets[: tile_count + 1], dtype=np.int64)
        if values[0] != 0 or np.any(values < 0) or np.any(values[1:] < values[:-1]):
            raise StrictFragmentTraceError("strict fragment-trace sidecar offsets are incomplete or nonmonotonic")
        end = int(values[-1])
        for name in STRICT_FRAGMENT_TRACE_ARRAY_NAMES:
            array = self._group[name]
            if array.shape[0] < end:
                raise StrictFragmentTraceError(f"strict fragment-trace sidecar {name} ends before its tile offset")
            if array.shape[0] != end:
                array.resize((end, array.shape[1]))

    def append(self, tile_id: int, trace: StrictFragmentTrace | None) -> tuple[int, int]:
        offsets = self._group[STRICT_FRAGMENT_TRACE_OFFSETS_KEY]
        self.ensure_tile_capacity(tile_id)
        if tile_id < 0 or tile_id + 1 >= offsets.shape[0]:
            raise StrictFragmentTraceError("strict fragment-trace sidecar tile id is outside the offset table")
        start = int(offsets[tile_id])
        if start < 0:
            raise StrictFragmentTraceError("strict fragment-trace sidecar tile offsets must be appended in ordinal order")
        if trace is None:
            offsets[tile_id + 1] = start
            return start, start
        end = start + trace.count
        for name in STRICT_FRAGMENT_TRACE_ARRAY_NAMES:
            array = self._group[name]
            columns = array.shape[1]
            if array.shape[0] != start:
                raise StrictFragmentTraceError("strict fragment-trace sidecar arrays do not agree with the current offset")
            array.resize((end, columns))
            if trace.count:
                array[start:end] = trace.arrays[name]
        offsets[tile_id + 1] = end
        return start, end

    def finalize(self, tile_count: int) -> None:
        self.truncate_to_tile_count(tile_count)
        offsets = self._group[STRICT_FRAGMENT_TRACE_OFFSETS_KEY]
        offsets.resize((tile_count + 1,))
        self._group.attrs.update({"tile_count": int(tile_count), "finalized": True})

    def close(self) -> None:
        self._store.close()


def validate_fragment_trace_sidecar(
    store_root: Path,
    rows: list[Mapping[str, Any]],
    *,
    selected_rows: list[int] | None = None,
) -> str:
    """Verify exact offsets and rehash materialized rows from the V18 sidecar."""
    path = store_root / STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY
    if not path.is_dir():
        raise StrictFragmentTraceError(f"strict fragment-trace sidecar is missing: {path}")
    store = zarr.storage.LocalStore(str(path), read_only=True)
    try:
        group = zarr.open_group(store=store, mode="r")
        sidecar = StrictFragmentTraceSidecar(path, store, group)
        sidecar._validate_layout()
        offsets = group[STRICT_FRAGMENT_TRACE_OFFSETS_KEY]
        if offsets.shape[0] != len(rows) + 1:
            raise StrictFragmentTraceError("strict fragment-trace sidecar offset count does not match V18 index rows")
        actual_rows = selected_rows if selected_rows is not None else list(range(len(rows)))
        for ordinal in actual_rows:
            if ordinal < 0 or ordinal >= len(rows):
                raise StrictFragmentTraceError("strict fragment-trace sidecar selected row is outside the V18 index")
            row = rows[ordinal]
            if int(row.get("tile_id", -1)) != ordinal:
                raise StrictFragmentTraceError("strict fragment-trace sidecar requires ordinal V18 tile_id values")
            start = int(offsets[ordinal])
            end = int(offsets[ordinal + 1])
            if start < 0 or end < start:
                raise StrictFragmentTraceError("strict fragment-trace sidecar has missing or nonmonotonic tile offsets")
            materialized = bool(row.get("object_geometry_target_materialized", False))
            if not materialized:
                continue
            if row.get(STRICT_FRAGMENT_TRACE_VALIDATED_FIELD) is not True:
                raise StrictFragmentTraceError("materialized strict target lacks validated fragment-trace provenance")
            if row.get(STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD) is not True:
                raise StrictFragmentTraceError("materialized strict target lacks all nine fragment-trace arrays")
            if int(row.get(STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD, -1)) != start:
                raise StrictFragmentTraceError("strict fragment-trace sidecar start offset differs from V18 index provenance")
            if int(row.get(STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD, -1)) != end:
                raise StrictFragmentTraceError("strict fragment-trace sidecar end offset differs from V18 index provenance")
            if int(row.get(STRICT_FRAGMENT_COUNT_FIELD, -1)) != end - start:
                raise StrictFragmentTraceError("strict fragment-trace sidecar count differs from V18 index provenance")
            arrays = {
                name: np.asarray(group[name][start:end])
                for name in STRICT_FRAGMENT_TRACE_ARRAY_NAMES
            }
            validate_materialized_strict_fragment_trace(arrays, row, status=str(row.get("object_geometry_target_status", "")))
        return sha256_tree(path)
    finally:
        store.close()
