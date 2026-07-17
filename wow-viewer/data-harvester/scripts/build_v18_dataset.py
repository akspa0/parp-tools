"""Build V18 consolidated Zarr dataset directly from game client archives.

Single-pass pipeline: C# harvester streams NPZ blobs -> Python reads from pipe -> Zarr.
NO intermediate files on disk. The Zarr store IS the dataset.

Now carries ALL available NPZ signals including per-instance object mask and placement data.

Usage:
    cd wow-viewer/data-harvester

    # Build one build (auto-discovered terrain maps):
    uv run python scripts/build_v18_dataset.py build --build 3_3_5_12340

    # Build multiple builds:
    uv run python scripts/build_v18_dataset.py build --builds 3_3_5_12340 4_0_0_11927

    # Limit tiles (for testing):
    uv run python scripts/build_v18_dataset.py build --build 3_3_5_12340 --limit 100

    # Only specific maps:
    uv run python scripts/build_v18_dataset.py build --build 3_3_5_12340 --maps Azeroth Northrend

    # Check stats:
    uv run python scripts/build_v18_dataset.py stats --build 3_3_5_12340
"""

from __future__ import annotations

import argparse
import json
import os
import PIL.Image
import tempfile
import shutil
import struct
import subprocess
import sys
import threading
import time
import warnings
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq
import zarr
import zarr.codecs
import zarr.storage

from harvester.spec102.strict_target_contract import (
    REQUIRED_STRICT_OBJECT_TARGET_VERSION,
    STRICT_FRAGMENT_COUNT_FIELD,
    STRICT_FRAGMENT_SHA256_FIELD,
    STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD,
    STRICT_FRAGMENT_TRACE_ARRAY_NAMES,
    STRICT_FRAGMENT_TRACE_SCHEMA_FIELD,
    STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY,
    STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD,
    STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD,
    STRICT_FRAGMENT_TRACE_VALIDATED_FIELD,
    STRICT_OBJECT_TARGET_VERSION_FIELD,
    STRICT_TARGET_ASSETS_FIELD,
    STRICT_TARGET_UNRESOLVED_PLACEMENTS_FIELD,
    StrictFragmentTrace,
    StrictFragmentTraceError,
    StrictFragmentTraceSidecar,
    validate_fragment_trace_sidecar,
    validate_materialized_strict_fragment_trace,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_HARVEST_TOOL_DIR = _PROJECT_ROOT / "tools" / "harvest" / "WowViewer.Tool.Harvest" / "bin" / "Debug" / "net10.0"
_VALIDATION_CAPTURE_TOOL_DIR = _PROJECT_ROOT / "tools" / "validation-capture" / "WowViewer.Tool.ValidationCapture" / "bin" / "Debug" / "net10.0"
_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v18"
_CLIENT_ROOTS = _PROJECT_ROOT.parent / "output" / "tmp" / "wowarchive-clients"
DEFAULT_MAP_WORKERS = max(1, min(4, os.cpu_count() or 1))
DEFAULT_TILE_WORKERS = max(1, min(16, os.cpu_count() or 1))

# V18 is the direct successor to the V16 dataset builder. The fixed-shape arrays
# below remain the canonical streamed build surface copied forward from V16.
# Signals patched onto V16 during later workflows are tracked separately as V18
# promoted arrays so the canonical contract is explicit.
V18_STREAMED_ARRAY_NAMES = {
    "height_257",
    "normal_xyz",
    "normal_mask",
    "alpha_256",
    "holes_16",
    "liquid_mask",
    "liquid_height",
    "object_mask",
    "object_precise_mask",
    "object_geometry_visible_mask",
    "object_geometry_visible_top_elevation",
    "object_geometry_visible_terrain_elevation",
    "object_geometry_visible_source",
    "object_instance_mask",
    "mcnk_flags_16",
    "mddf_mask",
    "modf_mask",
    "object_filtered_mask",
    "object_roof_mask",
    "object_roof_confidence",
    "minimap_rgb",
    "shadow_mask",
    "mcly_texture_ids",
    "mcly_layer_mask",
}

# Initial promoted V18 arrays. `object_visibility_mask` is the canonical
# renderer-truth object-loss signal for the focused spec 047 Plan A lane.
# `no_object_minimap` remains supported as an optional legacy QA sidecar when a
# capture variant emits it, but it is not required for a valid focused-corpus
# store.
V18_PROMOTED_ARRAY_NAMES = {
    "object_visibility_mask",
    "no_object_minimap",
}

EXPERIMENTAL_RENDERER_TRUTH_SIGNAL_KEYS = (
    "object_visibility_mask",
    "no_object_minimap",
)

# Canonical V18 artifacts required before a store can be treated as finalized.
V18_REQUIRED_ARTIFACTS = (
    "index.parquet",
    "decoded_metadata.parquet",
    "harvest_metrics.json",
    "signal_validation.json",
    "decoded_metadata_validation.json",
    STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY,
    "_resume_state.json",
)

V18_REQUIRED_ARTIFACTS_MERGED = V18_REQUIRED_ARTIFACTS + (
    "merge_manifest.json",
)

# ── NPZ key → Zarr array name mapping ──────────────────────────────────
# All signals from the C# harvester that map to fixed-shape Zarr arrays.
OUTPUT_ARRAY_NAMES = {
    "height_257": "height_257",
    "mcnr_normal_xyz": "normal_xyz",
    "mcal_alpha_pack_256": "alpha_256",
    "hole_mask_16": "holes_16",
    "unified_liquid_mask": "liquid_mask",
    "unified_liquid_height": "liquid_height",
    "object_mask_257": "object_mask",
    "object_precise_mask_257": "object_precise_mask",
    "object_geometry_visible_mask_257": "object_geometry_visible_mask",
    "object_geometry_visible_top_elevation_257": "object_geometry_visible_top_elevation",
    "object_geometry_visible_terrain_elevation_257": "object_geometry_visible_terrain_elevation",
    "object_geometry_visible_source_257": "object_geometry_visible_source",
    "object_instance_mask_257": "object_instance_mask",
    "mcnk_flags_16": "mcnk_flags_16",
    "mddf_mask_257": "mddf_mask",
    "modf_mask_257": "modf_mask",
    "object_filtered_mask_257": "object_filtered_mask",
    "minimap_rgb_256": "minimap_rgb",
    "mcsh_shadow_mask_256": "shadow_mask",
    "object_roof_mask_256": "object_roof_mask",
    "object_roof_confidence_256": "object_roof_confidence",
    "mcly_texture_ids": "mcly_texture_ids",
    "mcly_layer_mask": "mcly_layer_mask",
}

DTYPES = {
    "height_257": np.float32, "normal_xyz": np.float32, "normal_mask": np.bool_,
    "alpha_256": np.float32, "holes_16": np.bool_, "liquid_mask": np.float32,
    "liquid_height": np.float32, "object_mask": np.bool_, "object_precise_mask": np.float32,
    "object_geometry_visible_mask": np.float32, "object_geometry_visible_top_elevation": np.float32,
    "object_geometry_visible_terrain_elevation": np.float32,
    "object_geometry_visible_source": np.uint8,
    "object_instance_mask": np.int32, "mcnk_flags_16": np.int32,
    "mddf_mask": np.float32, "modf_mask": np.float32, "object_filtered_mask": np.float32,
    "object_roof_mask": np.float32, "object_roof_confidence": np.float32,
    "minimap_rgb": np.uint8,
    "shadow_mask": np.float32, "mcly_texture_ids": np.int32, "mcly_layer_mask": np.float32,
}

FILL_VALUES = {
    "height_257": 0.0, "normal_xyz": 0.0, "normal_mask": False,
    "alpha_256": 0.0, "holes_16": False, "liquid_mask": 0.0,
    "liquid_height": 0.0, "object_mask": False, "object_precise_mask": 0.0,
    "object_geometry_visible_mask": 0.0, "object_geometry_visible_top_elevation": 0.0,
    "object_geometry_visible_terrain_elevation": 0.0,
    "object_geometry_visible_source": 0,
    "object_instance_mask": 0, "mcnk_flags_16": 0,
    "mddf_mask": 0.0, "modf_mask": 0.0, "object_filtered_mask": 0.0,
    "object_roof_mask": 0.0, "object_roof_confidence": 0.0,
    "minimap_rgb": 0,
    "shadow_mask": 0.0, "mcly_texture_ids": -1, "mcly_layer_mask": 0.0,
}

SHAPES = {
    "height_257": (257, 257), "normal_xyz": (257, 257, 3), "normal_mask": (257, 257),
    "alpha_256": (256, 256, 4), "holes_16": (16, 16), "liquid_mask": (256, 256),
    "liquid_height": (256, 256), "object_mask": (257, 257), "object_precise_mask": (257, 257),
    "object_geometry_visible_mask": (257, 257), "object_geometry_visible_top_elevation": (257, 257),
    "object_geometry_visible_terrain_elevation": (257, 257),
    "object_geometry_visible_source": (257, 257),
    "object_instance_mask": (257, 257), "mcnk_flags_16": (16, 16),
    "mddf_mask": (257, 257), "modf_mask": (257, 257), "object_filtered_mask": (257, 257),
    "object_roof_mask": (256, 256), "object_roof_confidence": (256, 256),
    "minimap_rgb": (256, 256, 3),
    "shadow_mask": (256, 256), "mcly_texture_ids": (16, 16, 4), "mcly_layer_mask": (16, 16, 4),
}

CHUNK_SIZES = {
    "height_257": (64, 257, 257), "normal_xyz": (64, 257, 257, 3),
    "normal_mask": (256, 257, 257), "alpha_256": (64, 256, 256, 4),
    "holes_16": (1024, 16, 16), "liquid_mask": (64, 256, 256),
    "liquid_height": (64, 256, 256), "object_mask": (256, 257, 257),
    "object_precise_mask": (256, 257, 257), "object_geometry_visible_mask": (256, 257, 257),
    "object_geometry_visible_top_elevation": (64, 257, 257), "object_geometry_visible_terrain_elevation": (64, 257, 257),
    "object_geometry_visible_source": (256, 257, 257),
    "object_instance_mask": (256, 257, 257),
    "mcnk_flags_16": (256, 16, 16),
    "mddf_mask": (256, 257, 257), "modf_mask": (256, 257, 257),
    "object_filtered_mask": (256, 257, 257),
    "object_roof_mask": (64, 256, 256), "object_roof_confidence": (64, 256, 256),
    "minimap_rgb": (64, 256, 256, 3), "shadow_mask": (64, 256, 256),
    "mcly_texture_ids": (1024, 16, 16, 4), "mcly_layer_mask": (256, 16, 16, 4),
}


def _load_manifest_keep_keys(manifest_path: Path) -> set[tuple[str, int]]:
    if not manifest_path.exists():
        raise FileNotFoundError(f"Curation manifest not found: {manifest_path}")

    path = manifest_path
    if path.is_dir():
        kept = path / "kept_tiles.parquet"
        tiles = path / "tiles.parquet"
        if kept.exists():
            path = kept
        elif tiles.exists():
            path = tiles
        else:
            raise FileNotFoundError(f"No kept_tiles.parquet or tiles.parquet under {manifest_path}")

    if path.suffix.lower() == ".parquet":
        table = pq.read_table(str(path))
        rows = [{col: table.column(col)[i].as_py() for col in table.column_names} for i in range(table.num_rows)]
    elif path.suffix.lower() in {".jsonl", ".ndjson"}:
        rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    elif path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload if isinstance(payload, list) else payload.get("rows", [])
    else:
        raise RuntimeError(f"Unsupported curation manifest format: {path}")

    out: set[tuple[str, int]] = set()
    for row in rows:
        if not bool(row.get("keep", True)):
            continue
        build = str(row.get("build", "")).strip()
        tile_id = int(row.get("tile_id", -1))
        if build and tile_id >= 0:
            out.add((build, tile_id))
    return out


def _build_tile_pose_metadata_from_placements(zarr_path: Path) -> dict[int, dict[str, object]]:
    """Build a per-tile pose metadata map from placements.parquet.

    Includes *all* placement rows per tile under `object_instances`, plus
    a representative top-level entry for backwards-compatible consumers.
    """
    placements_path = zarr_path / "placements.parquet"
    if not placements_path.exists():
        return {}

    table = pq.read_table(str(placements_path))
    rows = [{col: table.column(col)[i].as_py() for col in table.column_names} for i in range(table.num_rows)]

    by_tile: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        try:
            tile_id = int(row.get("tile_id", -1))
        except Exception:
            continue
        if tile_id < 0:
            continue
        by_tile.setdefault(tile_id, []).append(row)

    resolved: dict[int, dict[str, object]] = {}

    def _to_int(value: object) -> int | None:
        try:
            return int(float(value))
        except Exception:
            return None

    def _to_float(value: object) -> float | None:
        try:
            return float(value)
        except Exception:
            return None

    for tile_id, tile_rows in by_tile.items():
        if not tile_rows:
            continue

        modf_rows = [r for r in tile_rows if str(r.get("instance_type", "")).lower() == "modf"]
        mddf_rows = [r for r in tile_rows if str(r.get("instance_type", "")).lower() == "mddf"]
        chosen = modf_rows[0] if modf_rows else (mddf_rows[0] if mddf_rows else tile_rows[0])

        instances: list[dict[str, object]] = []
        for row in tile_rows:
            instances.append(
                {
                    "asset_path": str(row.get("asset_path", "") or "") or None,
                    "instance_type": str(row.get("instance_type", "") or "") or None,
                    "instance_idx": _to_int(row.get("instance_idx")),
                    "unique_id": _to_int(row.get("uniqueId")),
                    "rot_x": _to_float(row.get("rotX")),
                    "rot_y": _to_float(row.get("rotY")),
                    "rot_z": _to_float(row.get("rotZ")),
                    "scale": _to_float(row.get("scale")),
                    "pos_x": _to_float(row.get("posX")),
                    "pos_y": _to_float(row.get("posY")),
                    "pos_z": _to_float(row.get("posZ")),
                }
            )

        resolved[tile_id] = {
            "object_instance_count": len(instances),
            "object_instances": instances,
            "asset_path": str(chosen.get("asset_path", "") or "") or None,
            "instance_type": str(chosen.get("instance_type", "") or "") or None,
            "unique_id": _to_int(chosen.get("uniqueId")),
            "rot_x": _to_float(chosen.get("rotX")),
            "rot_y": _to_float(chosen.get("rotY")),
            "rot_z": _to_float(chosen.get("rotZ")),
            "scale": _to_float(chosen.get("scale")),
        }

    return resolved

ALL_ARRAY_KEYS = list(V18_STREAMED_ARRAY_NAMES)

LIQUID_SOURCE_KEYS = ("mcnk", "mh2o", "mclq", "unified", "wl")
OBJECT_SIGNAL_KEYS = ("object_mask", "object_precise_mask", "object_instance_mask")
STRICT_OBJECT_TARGET_COMPLETE_STATUSES = {"CompleteEmpty", "CompleteVisible"}

V18_INDEX_STRING_FIELDS = (
    "object_roof_mask_source",
    STRICT_OBJECT_TARGET_VERSION_FIELD,
    "object_geometry_target_status",
    "object_geometry_target_liquid_evidence_status",
    STRICT_FRAGMENT_TRACE_SCHEMA_FIELD,
    STRICT_FRAGMENT_SHA256_FIELD,
    "object_geometry_target_assets_json",
    "object_geometry_target_unresolved_placements_json",
)
V18_INDEX_BOOL_FIELDS = (
    "object_geometry_target_materialized",
    "object_geometry_target_arrays_present",
    STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD,
    STRICT_FRAGMENT_TRACE_VALIDATED_FIELD,
)
V18_INDEX_INT_FIELDS = (
    "n_mddf",
    "n_modf",
    "object_geometry_target_placement_count",
    "object_geometry_target_geometry_resolved_placement_count",
    "object_geometry_target_geometry_unresolved_placement_count",
    "object_geometry_target_fallback_required_placement_count",
    "object_geometry_target_triangle_count",
    "object_geometry_target_visible_pixel_count",
    "object_geometry_target_occluded_pixel_count",
    "object_geometry_target_terrain_unknown_pixel_count",
    "object_geometry_target_liquid_covered_pixel_count",
    "object_geometry_target_liquid_surface_unknown_pixel_count",
    "object_geometry_target_liquid_covered_fragment_count",
    "object_geometry_target_liquid_hidden_fragment_count",
    "object_geometry_target_liquid_above_surface_fragment_count",
    "object_geometry_target_liquid_unknown_fragment_count",
    STRICT_FRAGMENT_COUNT_FIELD,
)
V18_INDEX_INT64_FIELDS = (
    STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD,
    STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD,
)

# Integration keys: derive has_* flags for these signals in the Parquet index.
# Include all fixed-shape arrays plus explicit liquid-source provenance flags.
SIGNAL_FLAG_KEYS = [
    *ALL_ARRAY_KEYS,
    *sorted(V18_PROMOTED_ARRAY_NAMES),
    *(f"liquid_source_{name}" for name in LIQUID_SOURCE_KEYS),
]

REQUIRED_KEYS = {"minimap_rgb_256", "height_257"}
DEFAULT_CODEC = "lz4"
DEFAULT_CLEVEL = 1
DEFAULT_SHUFFLE = "shuffle"
WRITE_RETRY_ATTEMPTS = 8
WRITE_RETRY_BASE_DELAY_SECONDS = 0.15
WRITE_BATCH_SIZE = 16

LK_CATA_BUILD_PREFIXES = ("3_", "4_")
WL_LIQUID_SURFACE_QUADS_SIGNAL = "wl_liquid_surface_quads_v1"
WL_LIQUID_ABOVE_TERRAIN_SIGNAL = "wl_liquid_above_terrain_v1"
WL_LIQUID_BASIC_TYPE_SIGNAL = "wl_liquid_basic_type_header_v1"
WL_LIQUID_REQUIRED_PROVENANCE = frozenset(
    {
        WL_LIQUID_SURFACE_QUADS_SIGNAL,
        WL_LIQUID_ABOVE_TERRAIN_SIGNAL,
        WL_LIQUID_BASIC_TYPE_SIGNAL,
    }
)


def _decode_metadata_json(tile_blob: dict[str, np.ndarray]) -> dict[str, object]:
    payload = tile_blob.get("metadata.json")
    if payload is None:
        return {}
    try:
        if hasattr(payload, "tobytes"):
            raw = payload.tobytes()
        elif isinstance(payload, bytes):
            raw = payload
        else:
            raw = bytes(payload)
        decoded = json.loads(raw.decode("utf-8"))
    except Exception:
        return {}
    return decoded if isinstance(decoded, dict) else {}


def _has_unverified_wl_liquid_fallback(tile_blob: dict[str, np.ndarray]) -> bool:
    """Reject WL* masks without contiguous, terrain-gated, typed provenance."""
    if "wl_liquid_mask" not in tile_blob and "wl_liquid_height" not in tile_blob:
        return False
    signals = _decode_metadata_json(tile_blob).get("available_signals", [])
    return not isinstance(signals, list) or not WL_LIQUID_REQUIRED_PROVENANCE.issubset(
        {str(signal) for signal in signals}
    )


def _tile_rejection_report_path(output_path: Path, build_name: str) -> Path:
    return output_path.parent / f"{build_name}.rejected_tiles.jsonl"


def _resume_state_path(output_path: Path) -> Path:
    return output_path / "_resume_state.json"


def _load_resume_state(output_path: Path) -> dict[str, object] | None:
    path = _resume_state_path(output_path)
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _collect_missing_store_components(
    output_path: Path,
    *,
    required_files: tuple[str, ...] = V18_REQUIRED_ARTIFACTS,
    required_arrays: list[str] | None = None,
) -> list[str]:
    missing: list[str] = []
    for required_file in required_files:
        if not (output_path / required_file).exists():
            missing.append(required_file)

    required_arrays = ALL_ARRAY_KEYS if required_arrays is None else required_arrays
    for key in required_arrays:
        if not (output_path / key).exists():
            missing.append(key)

    return missing


def _open_zarr_group_readonly(zarr_path: Path):
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
            category=UserWarning,
        )
        store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
        root = zarr.open_group(store=store, mode="r")
    return store, root


def _load_completed_final_store_state(output_path: Path) -> dict[str, object] | None:
    if not output_path.exists():
        return None

    missing_components = _collect_missing_store_components(output_path)
    if missing_components:
        return None

    idx_path = output_path / "index.parquet"
    if not idx_path.exists():
        return None

    table = pq.read_table(str(idx_path), columns=["map"])
    store, root = _open_zarr_group_readonly(output_path)
    try:
        array_length = int(root["height_257"].shape[0])
    finally:
        store.close()

    if table.num_rows != array_length:
        return None

    state = _load_resume_state(output_path) or {}
    if not state:
        maps: list[str] = []
        seen: set[str] = set()
        for value in table.column("map"):
            map_name = str(value.as_py())
            if map_name in seen:
                continue
            seen.add(map_name)
            maps.append(map_name)
        state = {
            "build": output_path.stem.replace(".zarr", ""),
            "requested_maps": maps,
            "completed_maps": maps,
            "valid_tiles": table.num_rows,
            "skipped_zero_usable_maps": 0,
            "rejected_tile_count": 0,
            "codec": "unknown-final-store",
            "clevel": -1,
            "shuffle": "unknown-final-store",
            "capacity": table.num_rows,
            "finalized": True,
            "inferred_from_final_store": True,
        }
    else:
        state = dict(state)
        state["finalized"] = True

    return state


def _write_resume_state(
    output_path: Path,
    *,
    build: str,
    requested_maps: list[str],
    completed_maps: list[str],
    valid: int,
    skipped_zero_usable_maps: int,
    rejected_tile_count: int,
    codec_name: str,
    codec_level: int,
    codec_shuffle: str,
    capacity: int,
    finalized: bool = False,
) -> None:
    state = {
        "build": build,
        "requested_maps": requested_maps,
        "completed_maps": completed_maps,
        "valid_tiles": valid,
        "skipped_zero_usable_maps": skipped_zero_usable_maps,
        "rejected_tile_count": rejected_tile_count,
        "codec": codec_name,
        "clevel": codec_level,
        "shuffle": codec_shuffle,
        "capacity": capacity,
        "finalized": finalized,
    }
    _resume_state_path(output_path).write_text(json.dumps(state, indent=2), encoding="utf-8")


def _write_finalization_state(
    output_path: Path,
    *,
    build: str,
    finalized: bool,
    signal_validation_path: Path | None,
    decoded_metadata_validation_path: Path | None,
    required_files: tuple[str, ...],
    required_arrays: list[str] | None = None,
) -> Path:
    missing_components = _collect_missing_store_components(
        output_path,
        required_files=required_files,
        required_arrays=required_arrays,
    )
    payload = {
        "build": build,
        "finalized": bool(finalized),
        "checked_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "required_files": list(required_files),
        "required_arrays": list(ALL_ARRAY_KEYS if required_arrays is None else required_arrays),
        "missing_components": missing_components,
        "signal_validation_path": str(signal_validation_path) if signal_validation_path is not None else None,
        "decoded_metadata_validation_path": str(decoded_metadata_validation_path) if decoded_metadata_validation_path is not None else None,
    }
    out_path = output_path / "finalization.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path

# ── Placement data columns for the companion Parquet table ────────────
PLACEMENT_COLUMNS_MDDF = [
    "nameId", "uniqueId", "posX", "posY", "posZ", "rotX", "rotY", "rotZ", "scale",
]
PLACEMENT_COLUMNS_MODF = [
    "nameId", "uniqueId", "posX", "posY", "posZ", "rotX", "rotY", "rotZ",
    "bbMinX", "bbMinY", "bbMinZ", "bbMaxX", "bbMaxY", "bbMaxZ",
]

NPZB_MAGIC = b"NPZB"
ARRY_MAGIC = b"ARRY"
ENDS_MAGIC = b"ENDS"


def _decode_blob(blob: bytes) -> dict[str, np.ndarray]:
    """Decode a tile blob — supports both raw binary (ARRY) and legacy NPZ formats."""
    if blob[:4] == ARRY_MAGIC:
        return _read_raw_blob(blob)
    return dict(np.load(BytesIO(blob), allow_pickle=False))


def _read_raw_blob(blob: bytes) -> dict[str, np.ndarray]:
    """Read one raw binary tile blob (zero-compression ARRY format)."""
    stream = BytesIO(blob)
    magic = stream.read(4)
    if magic != ARRY_MAGIC:
        raise ValueError(f"Expected ARRY magic, got {magic!r}")
    meta_len = struct.unpack("<I", stream.read(4))[0]
    meta_bytes = stream.read(meta_len)
    metadata = json.loads(meta_bytes.decode("utf-8"))
    result: dict[str, np.ndarray] = {}
    result["metadata.json"] = meta_bytes
    result["_metadata"] = metadata
    while stream.tell() < len(blob):
        pos = stream.tell()
        peek = stream.read(4)
        if not peek or len(peek) < 4:
            break
        if peek == b"ENDS":
            break
        stream.seek(pos)
        name_len = struct.unpack("<I", stream.read(4))[0]
        name = stream.read(name_len).decode("utf-8")
        ndim = struct.unpack("<I", stream.read(4))[0]
        shape = struct.unpack(f"<{ndim}I", stream.read(ndim * 4))
        dtype_raw = stream.read(8).rstrip(b"\x00")
        try:
            from harvester.raw_reader import _DTYPE_MAP
            dtype = _DTYPE_MAP.get(dtype_raw, np.dtype("float32"))
        except ImportError:
            dt_str = dtype_raw.decode()
            if dt_str in ("<f4", "<f8", "<i4", "<u4", "<i2", "<u2"):
                dtype = np.dtype(dt_str)
            elif dt_str == "|u1":
                dtype = np.dtype("uint8")
            elif dt_str == "|b1":
                dtype = np.dtype("bool")
            else:
                dtype = np.dtype("float32")
        data_len = struct.unpack("<Q", stream.read(8))[0]
        data = stream.read(data_len)
        arr = np.frombuffer(data, dtype=dtype).reshape(shape).copy()
        result[name] = arr
    return result


def _dir_size_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file())


def _is_retryable_windows_file_error(ex: BaseException) -> bool:
    if isinstance(ex, PermissionError):
        return True
    if isinstance(ex, OSError) and getattr(ex, "winerror", None) in {5, 32}:
        return True
    return False


def _flush_tile_batch_with_retry(
    arrays: dict[str, zarr.Array],
    start_index: int,
    pending_arrays: dict[str, list[np.ndarray]],
    pending_count: int,
    *,
    map_name: str,
) -> int:
    if pending_count <= 0:
        return start_index

    for key in ALL_ARRAY_KEYS:
        batch_value = np.stack(pending_arrays[key], axis=0)
        for attempt in range(1, WRITE_RETRY_ATTEMPTS + 1):
            try:
                arrays[key][start_index:start_index + pending_count] = batch_value
                break
            except Exception as ex:
                if not _is_retryable_windows_file_error(ex) or attempt == WRITE_RETRY_ATTEMPTS:
                    raise RuntimeError(
                        f"Failed writing tile batch start={start_index} count={pending_count} "
                        f"for map {map_name} array={key} after {attempt} attempts: {ex}"
                    ) from ex
                delay = WRITE_RETRY_BASE_DELAY_SECONDS * attempt
                print(
                    f"    Warning: retrying Zarr batch write for map {map_name} "
                    f"array={key} start={start_index} count={pending_count} after filesystem error: {ex}",
                    file=sys.stderr,
                    flush=True,
                )
                time.sleep(delay)

    for key in ALL_ARRAY_KEYS:
        pending_arrays[key].clear()

    return start_index + pending_count


def _tail_text(lines: deque[str]) -> str:
    if not lines:
        return "(no stderr output)"
    return "\n".join(lines)


def _pump_stderr(stderr_pipe, map_name: str, tail: deque[str]) -> None:
    try:
        for raw in iter(stderr_pipe.readline, b""):
            if not raw:
                break
            line = raw.decode("utf-8", errors="replace").rstrip()
            if not line:
                continue
            tail.append(line)
            print(f"    [harvest:{map_name}] {line}", file=sys.stderr, flush=True)
    finally:
        try:
            stderr_pipe.close()
        except Exception:
            pass


def _find_harvest_tool() -> Path:
    exe = _HARVEST_TOOL_DIR / "WowViewer.Tool.Harvest.exe"
    if exe.exists():
        return exe
    for p in sorted((_PROJECT_ROOT / "tools" / "harvest" / "WowViewer.Tool.Harvest" / "bin" / "Debug").glob("*/WowViewer.Tool.Harvest.exe")):
        if p.exists():
            return p
    raise FileNotFoundError("Harvest tool not found. Build it first.")


def _find_validation_capture_tool() -> Path:
    exe = _VALIDATION_CAPTURE_TOOL_DIR / "WowViewer.Tool.ValidationCapture.exe"
    if exe.exists():
        return exe
    for p in sorted((_PROJECT_ROOT / "tools" / "validation-capture" / "WowViewer.Tool.ValidationCapture" / "bin" / "Debug").glob("*/WowViewer.Tool.ValidationCapture.exe")):
        if p.exists():
            return p
    raise FileNotFoundError("Validation capture tool not found. Build wow-viewer/tools/validation-capture/WowViewer.Tool.ValidationCapture first.")


def _find_client_root(build: str) -> Path | None:
    parent = _CLIENT_ROOTS / build
    if not parent.exists():
        return None
    for child in parent.iterdir():
        if child.is_dir() and ((child / "WoW.exe").exists() or (child / "Data").exists()):
            return child
    return None


def _process_tile_data(data: dict[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, bool]] | None:
    if "minimap_rgb_256" not in data or "height_257" not in data:
        return None
    if _has_unverified_wl_liquid_fallback(data):
        return None

    tile_arrays: dict[str, np.ndarray] = {}
    has_signals: dict[str, bool] = {}

    for src_key, dst_key in OUTPUT_ARRAY_NAMES.items():
        if src_key in data:
            tile_arrays[dst_key] = _normalize_array(data[src_key], dst_key)
            has_signals[dst_key] = True
        else:
            shape = SHAPES[dst_key]
            dtype = DTYPES[dst_key]
            fill = FILL_VALUES[dst_key]
            tile_arrays[dst_key] = np.full(shape, fill, dtype=dtype)
            has_signals[dst_key] = False

    # Rebuild liquid supervision from raw liquid signals with explicit priority.
    # This avoids over-trusting legacy unified_liquid_* payloads when they only carry WL* fallback.
    liquid_mask, liquid_height, liquid_has_signal, liquid_source = _derive_liquid_supervision(data)
    tile_arrays["liquid_mask"] = liquid_mask
    tile_arrays["liquid_height"] = liquid_height
    has_signals["liquid_mask"] = liquid_has_signal
    has_signals["liquid_height"] = liquid_has_signal
    for source_name in LIQUID_SOURCE_KEYS:
        has_signals[f"liquid_source_{source_name}"] = liquid_source == source_name

    if "mcnr_normal_xyz" in data:
        nrm = data["mcnr_normal_xyz"].astype(np.float32)
        normal_mask = (np.abs(nrm).sum(axis=-1) > 1e-6)
        zero_mask = ~normal_mask
        nrm[zero_mask] = [0.0, 0.0, 1.0]
        norms = np.linalg.norm(nrm, axis=-1, keepdims=True)
        norms = np.where(norms < 1e-6, 1.0, norms)
        nrm = nrm / norms
        tile_arrays["normal_xyz"] = nrm.astype(np.float32)
        has_signals["normal_xyz"] = True
    else:
        tile_arrays["normal_xyz"] = np.zeros((257, 257, 3), dtype=np.float32)
        normal_mask = np.zeros((257, 257), dtype=np.bool_)
        has_signals["normal_xyz"] = False

    tile_arrays["normal_mask"] = normal_mask.astype(np.bool_)
    has_signals["normal_mask"] = True

    return tile_arrays, has_signals


def _derive_mcnk_liquid_flags(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray | None, np.ndarray | None]:
    # Fast path: read directly from mcnk_flags_16 NPZ key (new C# extraction)
    raw_flags = data.get("mcnk_flags_16")
    if raw_flags is not None:
        flags_arr = np.asarray(raw_flags, dtype=np.int32)
        if flags_arr.shape == (16, 16):
            flag_grid = flags_arr.astype(np.uint32, copy=False)
            type_grid = np.full((16, 16), -1, dtype=np.int32)
            any_liquid = False
            for cy in range(16):
                for cx in range(16):
                    f = flag_grid[cy, cx]
                    if (f & 0x3C) == 0:
                        continue
                    any_liquid = True
                    if (f & 0x20) != 0:
                        type_grid[cy, cx] = 3
                    elif (f & 0x10) != 0:
                        type_grid[cy, cx] = 2
                    else:
                        type_grid[cy, cx] = 1
            if any_liquid:
                return flag_grid, type_grid

    # Legacy fallback: parse from raw_chunks metadata (pre-existing shards)
    meta = _decode_metadata_json(data)
    raw_chunks = meta.get("raw_chunks")
    if not isinstance(raw_chunks, list):
        return None, None

    flag_grid = np.zeros((16, 16), dtype=np.uint32)
    type_grid = np.full((16, 16), -1, dtype=np.int32)
    any_liquid = False

    for raw_chunk in raw_chunks:
        if not isinstance(raw_chunk, dict):
            continue
        if str(raw_chunk.get("scope", "")).lower() != "mcnk":
            continue
        if str(raw_chunk.get("chunk_id", "")).upper() != "MCNK":
            continue

        chunk_x = raw_chunk.get("chunk_x")
        chunk_y = raw_chunk.get("chunk_y")
        entry_name = raw_chunk.get("entry_name")
        if not isinstance(chunk_x, int) or not isinstance(chunk_y, int) or not isinstance(entry_name, str):
            continue
        if not (0 <= chunk_x < 16 and 0 <= chunk_y < 16):
            continue

        payload = data.get(entry_name)
        if payload is None:
            continue
        raw = np.asarray(payload)
        if raw.ndim != 1 or raw.size < 4:
            continue

        flags = struct.unpack_from("<I", raw.astype(np.uint8, copy=False).tobytes(), 0)[0]
        flag_grid[chunk_y, chunk_x] = flags
        if (flags & 0x3C) == 0:
            continue

        any_liquid = True
        if (flags & 0x20) != 0:
            liquid_type = 3
        elif (flags & 0x10) != 0:
            liquid_type = 2
        else:
            liquid_type = 1
        type_grid[chunk_y, chunk_x] = liquid_type

    if not any_liquid:
        return None, None

    return flag_grid, type_grid


def _derive_liquid_supervision(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, bool, str | None]:
    def _to_2d(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        out = np.asarray(arr)
        if out.ndim < 2:
            return None
        if out.ndim > 2:
            out = np.squeeze(out)
            if out.ndim != 2:
                return None
        return out

    def _resize_liquid_grid(arr: np.ndarray) -> np.ndarray:
        src = np.asarray(arr)
        if src.ndim != 2:
            return src
        target_h, target_w = SHAPES["liquid_mask"]
        if src.shape == (target_h, target_w):
            return src
        y_idx = np.rint(np.linspace(0, src.shape[0] - 1, num=target_h)).astype(np.int32)
        x_idx = np.rint(np.linspace(0, src.shape[1] - 1, num=target_w)).astype(np.int32)
        return src[np.ix_(y_idx, x_idx)]

    def _reorient_non_wl_liquid(arr: np.ndarray | None) -> np.ndarray | None:
        if arr is None:
            return None
        # Trust the harvester's terrain-space orientation directly.
        # Dataset curation should not apply extra liquid rotations on top.
        return arr

    def _coerce_liq_mask(arr: np.ndarray) -> np.ndarray:
        out = _resize_liquid_grid(arr.astype(np.float32, copy=False))
        if out.max() > 1.5:
            out = out / 255.0
        return np.clip(out, 0.0, 1.0).astype(np.float32, copy=False)

    def _coerce_liq_height(arr: np.ndarray) -> np.ndarray:
        return _resize_liquid_grid(arr.astype(np.float32, copy=False)).astype(np.float32, copy=False)

    def _coerce_liq_type(arr: np.ndarray) -> np.ndarray:
        return _resize_liquid_grid(arr.astype(np.float32, copy=False)).astype(np.float32, copy=False)

    def _normalize_binary_mask(mask: np.ndarray) -> np.ndarray:
        out = mask.astype(np.float32)
        if out.max() > 1.5:
            out = out / 255.0
        return np.clip(out, 0.0, 1.0)

    mcnk_flags, mcnk_type_16 = _derive_mcnk_liquid_flags(data)
    mcnk_type = _coerce_liq_type(mcnk_type_16) if mcnk_type_16 is not None else None
    mcnk_mask = None
    if mcnk_flags is not None:
        mcnk_mask = _coerce_liq_mask(((mcnk_flags & 0x3C) != 0).astype(np.float32))

    mh2o_type = _reorient_non_wl_liquid(_to_2d(data.get("mh2o_type_mask")))
    mh2o_height = _reorient_non_wl_liquid(_to_2d(data.get("mh2o_surface_height")))
    mh2o_depth = _reorient_non_wl_liquid(_to_2d(data.get("mh2o_depth")))
    mclq_type = _reorient_non_wl_liquid(_to_2d(data.get("mclq_type_mask")))
    mclq_height = _reorient_non_wl_liquid(_to_2d(data.get("mclq_surface_height")))
    unified_height = _reorient_non_wl_liquid(_to_2d(data.get("unified_liquid_height")))
    wl_height = _to_2d(data.get("wl_liquid_height"))

    if mcnk_mask is not None:
        height_source = (
            mh2o_height
            if mh2o_height is not None
            else mclq_height
            if mclq_height is not None
            else unified_height
            if unified_height is not None
            else wl_height
        )
        if height_source is None:
            height_source = np.zeros_like(mcnk_mask, dtype=np.float32)
        return (
            _coerce_liq_mask(_normalize_binary_mask(mcnk_mask)),
            _coerce_liq_height(height_source.astype(np.float32)),
            True,
            "mcnk",
        )

    def _presence_mask(name: str, *, reorient_non_wl: bool = False) -> np.ndarray | None:
        raw = _to_2d(data.get(name))
        if raw is None:
            return None
        if reorient_non_wl:
            raw = _reorient_non_wl_liquid(raw)
        raw_liq = _coerce_liq_mask(raw.astype(np.float32))
        if raw.dtype == np.bool_:
            mask = raw_liq.astype(np.float32)
        else:
            mask = (raw_liq > 0.5).astype(np.float32)
        if float(mask.max()) <= 0.0:
            return None
        return mask

    # 1) MH2O (preferred)
    mh2o_mask = _presence_mask("mh2o_presence_mask", reorient_non_wl=True)
    if mh2o_mask is None and (mh2o_type is not None or mh2o_height is not None or mh2o_depth is not None):
        inferred = np.zeros(SHAPES["liquid_mask"], dtype=np.bool_)
        if mh2o_height is not None:
            inferred |= (np.abs(_coerce_liq_height(mh2o_height)) > 1e-6)
        if mh2o_depth is not None:
            inferred |= (np.abs(_coerce_liq_height(mh2o_depth)) > 1e-6)
        if mh2o_type is not None:
            # Preserve legacy behavior as a weak fallback for shards without explicit presence masks.
            inferred |= (_coerce_liq_type(mh2o_type) > 0.0)
        mh2o_mask = inferred.astype(np.float32)
        if float(mh2o_mask.max()) <= 0.0:
            mh2o_mask = None
    if mh2o_mask is not None:
        if mh2o_height is None:
            mh2o_height = np.zeros_like(mh2o_mask, dtype=np.float32)
        return (
            _coerce_liq_mask(_normalize_binary_mask(mh2o_mask)),
            _coerce_liq_height(mh2o_height.astype(np.float32)),
            True,
            "mh2o",
        )

    # 2) MCLQ (next priority)
    mclq_mask = _presence_mask("mclq_presence_mask", reorient_non_wl=True)
    if mcnk_type is not None:
        mclq_type = mcnk_type
    if mclq_mask is None and mclq_type is not None:
        mclq_type_i = _coerce_liq_type(mclq_type).astype(np.int32, copy=False)
        if int(mclq_type_i.min()) < 0:
            # Alpha-derived shards use -1 for "not present"; 0 is valid water.
            mclq_mask = (mclq_type_i >= 0).astype(np.float32)
        elif int(mclq_type_i.max()) > 0:
            # When a coarse type grid exists, prefer it over height-based inference:
            # liquid heights can legitimately sit at 0.0f for sea-level water.
            mclq_mask = (mclq_type_i > 0).astype(np.float32)
        elif mclq_height is not None:
            # Legacy fallback when no explicit mask exists: infer from non-zero heights.
            mclq_mask = (np.abs(_coerce_liq_height(mclq_height)) > 1e-6).astype(np.float32)
        else:
            mclq_mask = (mclq_type_i > 0).astype(np.float32)
        if float(mclq_mask.max()) <= 0.0:
            mclq_mask = None
    if mclq_mask is not None:
        if mclq_height is None:
            mclq_height = np.zeros_like(mclq_mask, dtype=np.float32)
        return (
            _coerce_liq_mask(_normalize_binary_mask(mclq_mask)),
            _coerce_liq_height(mclq_height.astype(np.float32)),
            True,
            "mclq",
        )

    # 3) Existing unified signal
    unified_mask = _reorient_non_wl_liquid(_to_2d(data.get("unified_liquid_mask")))
    if unified_mask is not None:
        unified_mask = _normalize_binary_mask(unified_mask)
        if float(unified_mask.max()) > 0.0:
            unified_height = _reorient_non_wl_liquid(_to_2d(data.get("unified_liquid_height")))
            if unified_height is None:
                unified_height = np.zeros_like(unified_mask, dtype=np.float32)
            return (
                _coerce_liq_mask(unified_mask),
                _coerce_liq_height(unified_height.astype(np.float32)),
                True,
                "unified",
            )

    # 4) WL* last-resort fallback
    wl_mask = _to_2d(data.get("wl_liquid_mask"))
    if wl_mask is not None:
        wl_mask = _normalize_binary_mask(wl_mask.astype(np.float32))
        if float(wl_mask.max()) > 0.0:
            wl_height = _to_2d(data.get("wl_liquid_height"))
            if wl_height is None:
                wl_height = np.zeros_like(wl_mask, dtype=np.float32)
            return (
                _coerce_liq_mask(wl_mask),
                _coerce_liq_height(wl_height.astype(np.float32)),
                True,
                "wl",
            )

    # No usable liquid signal
    return (
        np.zeros(SHAPES["liquid_mask"], dtype=np.float32),
        np.zeros(SHAPES["liquid_height"], dtype=np.float32),
        False,
        None,
    )


def _derive_object_supervision(
    data: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, bool, bool, bool, np.ndarray, np.ndarray, np.ndarray, bool, bool, bool]:
    raw_mask = data.get("object_mask_257")
    raw_precise = data.get("object_precise_mask_257")
    raw_instance = data.get("object_instance_mask_257")
    raw_mddf = data.get("mddf_mask_257")
    raw_modf = data.get("modf_mask_257")
    raw_filtered = data.get("object_filtered_mask_257")

    if raw_mask is not None:
        object_mask = _normalize_array(raw_mask, "object_mask")
    else:
        object_mask = np.zeros(SHAPES["object_mask"], dtype=DTYPES["object_mask"])

    if raw_precise is not None:
        object_precise = _normalize_array(raw_precise, "object_precise_mask")
    else:
        object_precise = np.zeros(SHAPES["object_precise_mask"], dtype=DTYPES["object_precise_mask"])

    if raw_instance is not None:
        object_instance = _normalize_array(raw_instance, "object_instance_mask")
    else:
        object_instance = np.zeros(SHAPES["object_instance_mask"], dtype=DTYPES["object_instance_mask"])

    if raw_mddf is not None:
        mddf_mask = _normalize_array(raw_mddf, "mddf_mask").astype(np.float32)
    else:
        mddf_mask = np.zeros(SHAPES["mddf_mask"], dtype=np.float32)

    if raw_modf is not None:
        modf_mask = _normalize_array(raw_modf, "modf_mask").astype(np.float32)
    else:
        modf_mask = np.zeros(SHAPES["modf_mask"], dtype=np.float32)

    if raw_filtered is not None:
        object_filtered = _normalize_array(raw_filtered, "object_filtered_mask").astype(np.float32)
    else:
        # Fallback: use merged mask if filtered not available (legacy shards)
        object_filtered = object_mask.astype(np.float32)

    has_object_mask = bool(np.any(object_mask))
    has_object_precise = bool(np.any(object_precise > 0.0))
    has_object_instance = bool(np.any(object_instance > 0))
    has_mddf = bool(np.any(mddf_mask > 0.0))
    has_modf = bool(np.any(modf_mask > 0.0))
    has_filtered = bool(np.any(object_filtered > 0.0))
    return (object_mask, object_precise, object_instance,
            has_object_mask, has_object_precise, has_object_instance,
            mddf_mask, modf_mask, object_filtered,
            has_mddf, has_modf, has_filtered)


def _extract_metadata(data: dict[str, np.ndarray]) -> dict:
    meta_raw = data.get("metadata.json")
    if meta_raw is None:
        return {}
    try:
        if isinstance(meta_raw, str):
            return json.loads(meta_raw)
        if isinstance(meta_raw, bytes):
            return json.loads(meta_raw.decode())
        return json.loads(meta_raw.tobytes().decode())
    except Exception:
        return {}


def _extract_strict_object_target_provenance(
    data: dict[str, np.ndarray],
    metadata: dict,
) -> tuple[dict[str, object], StrictFragmentTrace | None]:
    """Keep explicit strict-target status; never infer it from mask pixels."""
    status = str(metadata.get("object_geometry_target_status", "") or "")
    declared_materialized = metadata.get("object_geometry_target_materialized") is True
    required_arrays = (
        "object_geometry_visible_mask_257",
        "object_geometry_visible_top_elevation_257",
        "object_geometry_visible_terrain_elevation_257",
        "object_geometry_visible_source_257",
    )
    arrays_present = all(name in data for name in required_arrays)
    materialized = (
        declared_materialized
        and status in STRICT_OBJECT_TARGET_COMPLETE_STATUSES
        and arrays_present
    )

    def as_int(name: str) -> int:
        try:
            return int(metadata.get(name, 0) or 0)
        except (TypeError, ValueError):
            return 0

    provenance: dict[str, object] = {
        STRICT_OBJECT_TARGET_VERSION_FIELD: str(
            metadata.get(STRICT_OBJECT_TARGET_VERSION_FIELD, "") or "missing"
        ),
        "object_geometry_target_status": status or "missing",
        "object_geometry_target_materialized": materialized,
        "object_geometry_target_arrays_present": arrays_present,
        "object_geometry_target_liquid_evidence_status": str(
            metadata.get("object_geometry_target_liquid_evidence_status", "") or "missing"
        ),
        "object_geometry_target_liquid_covered_pixel_count": as_int(
            "object_geometry_target_liquid_covered_pixel_count"
        ),
        "object_geometry_target_liquid_surface_unknown_pixel_count": as_int(
            "object_geometry_target_liquid_surface_unknown_pixel_count"
        ),
        "object_geometry_target_liquid_covered_fragment_count": as_int(
            "object_geometry_target_liquid_covered_fragment_count"
        ),
        "object_geometry_target_liquid_hidden_fragment_count": as_int(
            "object_geometry_target_liquid_hidden_fragment_count"
        ),
        "object_geometry_target_liquid_above_surface_fragment_count": as_int(
            "object_geometry_target_liquid_above_surface_fragment_count"
        ),
        "object_geometry_target_liquid_unknown_fragment_count": as_int(
            "object_geometry_target_liquid_unknown_fragment_count"
        ),
        "object_geometry_target_placement_count": as_int("object_geometry_target_placement_count"),
        "object_geometry_target_geometry_resolved_placement_count": as_int(
            "object_geometry_target_geometry_resolved_placement_count"
        ),
        "object_geometry_target_geometry_unresolved_placement_count": as_int(
            "object_geometry_target_geometry_unresolved_placement_count"
        ),
        "object_geometry_target_fallback_required_placement_count": as_int(
            "object_geometry_target_fallback_required_placement_count"
        ),
        "object_geometry_target_triangle_count": as_int("object_geometry_target_triangle_count"),
        "object_geometry_target_visible_pixel_count": as_int(
            "object_geometry_target_visible_pixel_count"
        ),
        "object_geometry_target_occluded_pixel_count": as_int(
            "object_geometry_target_occluded_pixel_count"
        ),
        "object_geometry_target_terrain_unknown_pixel_count": as_int(
            "object_geometry_target_terrain_unknown_pixel_count"
        ),
    }
    trace: StrictFragmentTrace | None = None
    if materialized:
        try:
            trace = validate_materialized_strict_fragment_trace(data, metadata, status=status)
        except StrictFragmentTraceError as error:
            raise RuntimeError(f"materialized strict target has invalid v3 fragment trace: {error}") from error
        provenance.update(
            {
                STRICT_FRAGMENT_TRACE_SCHEMA_FIELD: REQUIRED_STRICT_OBJECT_TARGET_VERSION,
                STRICT_FRAGMENT_COUNT_FIELD: trace.count,
                STRICT_FRAGMENT_SHA256_FIELD: trace.sha256,
                "object_geometry_target_assets_json": trace.assets_json,
                "object_geometry_target_unresolved_placements_json": trace.unresolved_placements_json,
                STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD: True,
                STRICT_FRAGMENT_TRACE_VALIDATED_FIELD: True,
                STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD: -1,
                STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD: -1,
            }
        )
    else:
        provenance.update(
            {
                STRICT_FRAGMENT_TRACE_SCHEMA_FIELD: str(
                    metadata.get(STRICT_FRAGMENT_TRACE_SCHEMA_FIELD, "") or "missing"
                ),
                STRICT_FRAGMENT_COUNT_FIELD: as_int(STRICT_FRAGMENT_COUNT_FIELD),
                STRICT_FRAGMENT_SHA256_FIELD: str(
                    metadata.get(STRICT_FRAGMENT_SHA256_FIELD, "") or "missing"
                ),
                "object_geometry_target_assets_json": "[]",
                "object_geometry_target_unresolved_placements_json": "[]",
                STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD: False,
                STRICT_FRAGMENT_TRACE_VALIDATED_FIELD: False,
                STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD: -1,
                STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD: -1,
            }
        )
    return provenance, trace


def _try_parse_tile_coords_from_stem(stem: str) -> tuple[int | None, int | None]:
    parts = stem.rsplit("_", 2)
    if len(parts) < 3:
        return None, None
    try:
        return int(parts[-2]), int(parts[-1])
    except (TypeError, ValueError):
        return None, None


def _try_parse_alpha_tile_coords(value: str) -> tuple[int | None, int | None]:
    marker = "alpha-tile("
    idx = value.lower().find(marker)
    if idx < 0:
        return None, None
    inside = value[idx + len(marker):]
    end = inside.find(")")
    if end < 0:
        return None, None
    pair = inside[:end].split(",", 1)
    if len(pair) != 2:
        return None, None
    try:
        tx = int(pair[0].strip())
        ty = int(pair[1].strip())
    except (TypeError, ValueError):
        return None, None
    if tx < 0 or tx > 63 or ty < 0 or ty > 63:
        return None, None
    return tx, ty


def _extract_tile_coords_from_metadata(meta: dict[str, object]) -> tuple[int, int]:
    tx = meta.get("tile_x")
    ty = meta.get("tile_y")
    if tx is not None and ty is not None:
        try:
            return int(tx), int(ty)
        except (TypeError, ValueError):
            pass

    tile_name = str(meta.get("tile_name", "") or "")
    if tile_name:
        alpha_tx, alpha_ty = _try_parse_alpha_tile_coords(tile_name)
        if alpha_tx is not None and alpha_ty is not None:
            return alpha_tx, alpha_ty
        parsed_tx, parsed_ty = _try_parse_tile_coords_from_stem(tile_name)
        if parsed_tx is not None and parsed_ty is not None:
            return parsed_tx, parsed_ty

    source = str(meta.get("source_adt_path", "") or "")
    if source:
        alpha_tx, alpha_ty = _try_parse_alpha_tile_coords(source)
        if alpha_tx is not None and alpha_ty is not None:
            return alpha_tx, alpha_ty
        parsed_tx, parsed_ty = _try_parse_tile_coords_from_stem(Path(source).stem)
        if parsed_tx is not None and parsed_ty is not None:
            return parsed_tx, parsed_ty

    return 0, 0


def _extract_placements(data: dict[str, np.ndarray], meta: dict) -> tuple[list[dict], list[dict], list[str], list[str]]:
    mddf_rows = []
    modf_rows = []
    mddf_names = meta.get("placement_mddf_names", [])
    modf_names = meta.get("placement_modf_names", [])

    mddf_data = data.get("placement_mddf_data")
    if mddf_data is not None and mddf_data.ndim == 2 and mddf_data.shape[0] > 0:
        for i in range(mddf_data.shape[0]):
            row = {col: float(mddf_data[i, j]) for j, col in enumerate(PLACEMENT_COLUMNS_MDDF) if j < mddf_data.shape[1]}
            row["instance_type"] = "mddf"
            row["instance_idx"] = i
            name_id = int(row.get("nameId", -1))
            row["asset_path"] = mddf_names[name_id] if 0 <= name_id < len(mddf_names) else ""
            mddf_rows.append(row)

    modf_data = data.get("placement_modf_data")
    if modf_data is not None and modf_data.ndim == 2 and modf_data.shape[0] > 0:
        for i in range(modf_data.shape[0]):
            row = {col: float(modf_data[i, j]) for j, col in enumerate(PLACEMENT_COLUMNS_MODF) if j < modf_data.shape[1]}
            row["instance_type"] = "modf"
            row["instance_idx"] = i
            name_id = int(row.get("nameId", -1))
            row["asset_path"] = modf_names[name_id] if 0 <= name_id < len(modf_names) else ""
            modf_rows.append(row)

    return mddf_rows, modf_rows, mddf_names, modf_names


def _normalize_array(arr: np.ndarray, dst_key: str) -> np.ndarray:
    arr = arr.astype(DTYPES.get(dst_key, np.float32))
    if dst_key == "alpha_256":
        if arr.max() > 1.5:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
    elif dst_key == "liquid_mask":
        if arr.max() > 1.5:
            arr = arr / 255.0
        arr = np.clip(arr, 0.0, 1.0)
    elif dst_key in ("holes_16", "object_mask"):
        arr = arr.astype(np.bool_)
    elif dst_key == "object_instance_mask":
        arr = arr.astype(np.int32)
    return _coerce_array_shape(arr, dst_key)


def _coerce_array_shape(arr: np.ndarray, dst_key: str) -> np.ndarray:
    target_shape = SHAPES[dst_key]
    if arr.shape == target_shape:
        return arr

    # Common case for terrain signals with variable layer counts:
    # squeeze accidental singleton axes, then restore missing trailing axes.
    arr = np.squeeze(arr)
    while arr.ndim < len(target_shape):
        arr = np.expand_dims(arr, axis=-1)

    fill = FILL_VALUES.get(dst_key, 0)
    coerced = np.full(target_shape, fill, dtype=DTYPES[dst_key])

    copy_rank = min(arr.ndim, len(target_shape))
    src_slices = []
    dst_slices = []
    for axis in range(copy_rank):
        extent = min(arr.shape[axis], target_shape[axis])
        src_slices.append(slice(0, extent))
        dst_slices.append(slice(0, extent))

    if src_slices:
        coerced[tuple(dst_slices)] = arr[tuple(src_slices)].astype(DTYPES[dst_key], copy=False)

    return coerced


def _discover_maps_for_build(harvest_tool: Path, client_root: Path) -> list[str]:
    def getv(row: dict, key: str, default=None):
        return row.get(key, row.get(key[:1].upper() + key[1:], default))

    cmd = [
        str(harvest_tool),
        "discover-maps",
        "--client-root",
        str(client_root),
    ]
    proc = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        check=False,
    )
    if proc.stderr:
        print(proc.stderr, end="", file=sys.stderr, flush=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"discover-maps failed for {client_root} with exit code {proc.returncode}"
        )

    try:
        payload = json.loads(proc.stdout)
    except json.JSONDecodeError as ex:
        raise RuntimeError(
            f"discover-maps returned invalid JSON for {client_root}: {ex}"
        ) from ex

    if not isinstance(payload, list):
        raise RuntimeError(f"discover-maps returned unexpected payload type: {type(payload)!r}")

    included = [getv(row, "map") for row in payload if getv(row, "include")]
    print(f"Discovered {len(included)} trainable maps from WDT summaries")
    for row in payload:
        map_name = getv(row, "map", "<unknown>")
        reason = getv(row, "reason", "unknown")
        tiles = getv(row, "tilesWithData", 0)
        has_wmo = getv(row, "hasWorldModelAsset", False)
        has_tile = getv(row, "hasReadableTile", False)
        status = "include" if getv(row, "include") else "skip"
        print(
            f"  [{status}] {map_name}: reason={reason}, "
            f"tiles={tiles}, wmo={has_wmo}, readable_tile={has_tile}"
        )

    if not included:
        raise RuntimeError(f"No trainable maps discovered for {client_root}")

    return included


def _write_index(rows: list[dict], output_path: Path) -> None:
    schema_fields = [
        pa.field("tile_id", pa.int64()),
        pa.field("build", pa.string()),
        pa.field("map", pa.string()),
        pa.field("tile_x", pa.int32()),
        pa.field("tile_y", pa.int32()),
        pa.field("height_mean", pa.float32()),
        pa.field("height_std", pa.float32()),
    ]
    schema_fields.extend(pa.field(name, pa.string()) for name in V18_INDEX_STRING_FIELDS)
    schema_fields.extend(pa.field(name, pa.bool_()) for name in V18_INDEX_BOOL_FIELDS)
    schema_fields.extend(pa.field(name, pa.int32()) for name in V18_INDEX_INT_FIELDS)
    schema_fields.extend(pa.field(name, pa.int64()) for name in V18_INDEX_INT64_FIELDS)
    bool_fields = sorted({
        k
        for row in rows
        for k in row.keys()
        if k.startswith("has_")
    }) if rows else []
    for bf in bool_fields:
        schema_fields.append(pa.field(bf, pa.bool_()))

    schema = pa.schema(schema_fields)
    col_data = {k: [] for k in schema.names}
    for row in rows:
        for k in schema.names:
            if k.startswith("has_") or k in V18_INDEX_BOOL_FIELDS:
                value = bool(row.get(k, False))
            elif k in V18_INDEX_STRING_FIELDS:
                value = str(row.get(k, "missing") or "missing")
            elif k in V18_INDEX_INT64_FIELDS:
                raw_value = row.get(k, -1)
                value = -1 if raw_value is None else int(raw_value)
            else:
                value = row.get(k, 0)
            col_data[k].append(value)

    table = pa.table(col_data, schema=schema)
    pq.write_table(table, str(output_path / "index.parquet"))


def _write_harvest_metrics(
    *,
    build: str,
    output_path: Path,
    index_rows: list[dict],
    placements_total: int,
    skipped_zero_usable_maps: int,
    rejected_tile_count: int,
    elapsed_seconds: float,
) -> Path:
    tile_count = len(index_rows)
    signal_counts: dict[str, int] = {}
    for row in index_rows:
        for key, value in row.items():
            if not key.startswith("has_"):
                continue
            signal_counts[key] = signal_counts.get(key, 0) + (1 if bool(value) else 0)

    coverage = {
        key: {
            "count": int(count),
            "fraction": float(count / tile_count) if tile_count > 0 else 0.0,
        }
        for key, count in sorted(signal_counts.items())
    }

    map_counts: dict[str, int] = {}
    for row in index_rows:
        map_name = str(row.get("map", ""))
        map_counts[map_name] = map_counts.get(map_name, 0) + 1

    payload = {
        "build": build,
        "tile_count": tile_count,
        "placements_total": int(placements_total),
        "skipped_zero_usable_maps": int(skipped_zero_usable_maps),
        "rejected_missing_required_tiles": int(rejected_tile_count),
        "elapsed_seconds": float(elapsed_seconds),
        "tiles_per_second": float(tile_count / max(elapsed_seconds, 0.01)),
        "signal_coverage": coverage,
        "map_tile_counts": dict(sorted(map_counts.items())),
    }
    out_path = output_path / "harvest_metrics.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out_path


def _count_signal_coverage(index_rows: list[dict]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in index_rows:
        for key, value in row.items():
            if not key.startswith("has_"):
                continue
            counts[key] = counts.get(key, 0) + (1 if bool(value) else 0)
    return counts


def _validate_build_signals(build: str, output_path: Path, strict: bool = True) -> Path:
    index_rows = _read_index_rows(output_path)
    tile_count = len(index_rows)
    expected_has_cols = sorted(f"has_{key}" for key in SIGNAL_FLAG_KEYS)
    present_cols = sorted({
        key
        for row in index_rows
        for key in row.keys()
        if key.startswith("has_")
    })
    coverage = _count_signal_coverage(index_rows)

    failures: list[str] = []
    warnings_list: list[str] = []

    if tile_count <= 0:
        failures.append("index has zero rows")

    strict_v3_rows = sum(
        int(row.get(STRICT_OBJECT_TARGET_VERSION_FIELD) == REQUIRED_STRICT_OBJECT_TARGET_VERSION)
        for row in index_rows
    )
    if strict_v3_rows != tile_count:
        failures.append(
            "strict target version coverage must be exact v3 for every V18 row "
            f"({strict_v3_rows}/{tile_count})"
        )
    materialized_strict_rows = [
        row for row in index_rows if row.get("object_geometry_target_materialized") is True
    ]
    strict_trace_sidecar_sha256: str | None = None
    try:
        strict_trace_sidecar_sha256 = validate_fragment_trace_sidecar(output_path, index_rows)
    except StrictFragmentTraceError as error:
        failures.append(f"strict fragment-trace sidecar invalid: {error}")
    for row in materialized_strict_rows:
        if row.get(STRICT_FRAGMENT_TRACE_VALIDATED_FIELD) is not True:
            failures.append("materialized strict target lacks validated v3 fragment trace")
            break
        if row.get(STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD) is not True:
            failures.append("materialized strict target lacks the nine v3 fragment-trace arrays")
            break
        if row.get(STRICT_FRAGMENT_TRACE_SCHEMA_FIELD) != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
            failures.append("materialized strict target trace metadata schema is not exact C# v3")
            break

    missing_cols = sorted(set(expected_has_cols) - set(present_cols))
    if missing_cols:
        warnings_list.append(f"missing has_* columns (store predates these signals): {missing_cols}")

    required_all_tiles = [
        "has_height_257",
        "has_minimap_rgb",
        "has_normal_mask",
    ]
    for key in required_all_tiles:
        count = int(coverage.get(key, 0))
        if count != tile_count:
            failures.append(f"{key} expected {tile_count}/{tile_count}, got {count}/{tile_count}")

    required_nonzero = [
        "has_normal_xyz",
        "has_alpha_256",
        "has_mcly_texture_ids",
        "has_mcly_layer_mask",
    ]
    for key in required_nonzero:
        count = int(coverage.get(key, 0))
        if count <= 0:
            failures.append(f"{key} expected >0, got {count}")

    holes_count = int(coverage.get("has_holes_16", 0))
    if holes_count <= 0:
        warnings_list.append(
            "has_holes_16 coverage is 0; allowed for some early-client builds but check expected format coverage."
        )

    liquid_count = int(coverage.get("has_liquid_mask", 0))
    liquid_source_keys = [
        "has_liquid_source_mcnk",
        "has_liquid_source_mh2o",
        "has_liquid_source_mclq",
        "has_liquid_source_unified",
        "has_liquid_source_wl",
    ]
    liquid_source_total = sum(int(coverage.get(key, 0)) for key in liquid_source_keys)
    if liquid_source_total != liquid_count:
        failures.append(
            f"liquid source total mismatch: sources={liquid_source_total}, has_liquid_mask={liquid_count}"
        )

    mcnk_count = int(coverage.get("has_liquid_source_mcnk", 0))
    mh2o_count = int(coverage.get("has_liquid_source_mh2o", 0))
    mclq_count = int(coverage.get("has_liquid_source_mclq", 0))
    unified_count = int(coverage.get("has_liquid_source_unified", 0))
    wl_count = int(coverage.get("has_liquid_source_wl", 0))

    if build.startswith("0_5_"):
        if mclq_count <= 0 and mcnk_count <= 0 and liquid_count > 0:
            failures.append("0.5.x build has liquid tiles but zero MCNK/MCLQ-derived coverage; expected chunk-flag or MCLQ liquid provenance for alpha-era data.")
        if wl_count > max(mclq_count, mcnk_count):
            warnings_list.append(
                f"WL* fallback dominates 0.5.x liquid labels (wl={wl_count}, mcnk={mcnk_count}, mclq={mclq_count})."
            )
    elif build.startswith("0_7_"):
        if liquid_count > 0 and mcnk_count <= 0 and mclq_count <= 0 and mh2o_count <= 0 and unified_count > 0:
            warnings_list.append(
                "0.7.x liquid supervision is unified-only (no explicit MCNK/MCLQ/MH2O provenance in stream). "
                "This is allowed but indicates source granularity limits."
            )
    elif build.startswith(LK_CATA_BUILD_PREFIXES):
        if liquid_count > 0 and mcnk_count <= 0 and mh2o_count <= 0 and unified_count > 0:
            warnings_list.append(
                "LK/Cata liquid supervision is unified-only (no explicit MCNK/MH2O provenance). "
                "Allowed, but verify source extraction if MH2O-native supervision is expected."
            )

    # V18 promoted renderer-truth signal validation
    object_visibility_count = int(coverage.get("has_object_visibility_mask", 0))
    no_object_count = int(coverage.get("has_no_object_minimap", 0))
    if object_visibility_count > 0 or no_object_count > 0:
        warnings_list.append(
            f"Renderer-truth signals present: object_visibility_mask={object_visibility_count}/{tile_count}, "
            f"no_object_minimap={no_object_count}/{tile_count}. "
            "object_visibility_mask is the canonical focused-corpus signal; "
            "no_object_minimap is an optional legacy QA sidecar when the chosen "
            "capture variant emits it. Coverage is expected to be partial until "
            "all focused-build tiles are captured."
        )

    object_roof_mask_count = int(coverage.get("has_object_roof_mask", 0))
    object_roof_confidence_count = int(coverage.get("has_object_roof_confidence", 0))
    if object_roof_mask_count != object_roof_confidence_count:
        failures.append(
            f"object_roof_mask ({object_roof_mask_count}) and object_roof_confidence ({object_roof_confidence_count}) tile counts must match"
        )

    roof_source_rows = 0
    roof_source_nonempty = 0
    for row in index_rows:
        if "object_roof_mask_source" not in row:
            continue
        roof_source_rows += 1
        if str(row.get("object_roof_mask_source", "") or "").strip():
            roof_source_nonempty += 1

    if object_roof_mask_count > 0:
        if roof_source_rows != tile_count:
            warnings_list.append(
                f"object_roof_mask coverage is present but object_roof_mask_source exists on only {roof_source_rows}/{tile_count} index rows"
            )
        elif roof_source_nonempty < object_roof_mask_count:
            warnings_list.append(
                f"object_roof_mask coverage is {object_roof_mask_count}/{tile_count}, but only {roof_source_nonempty}/{tile_count} rows have non-empty object_roof_mask_source"
            )

    payload = {
        "build": build,
        "tile_count": tile_count,
        "strict": bool(strict),
        "passed": len(failures) == 0,
        "failures": failures,
        "warnings": warnings_list,
        "strict_fragment_trace": {
            "target_version": REQUIRED_STRICT_OBJECT_TARGET_VERSION,
            "sidecar": STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY,
            "sidecar_sha256": strict_trace_sidecar_sha256,
            "materialized_rows": len(materialized_strict_rows),
            "validated_materialized_rows": sum(
                int(row.get(STRICT_FRAGMENT_TRACE_VALIDATED_FIELD) is True)
                for row in materialized_strict_rows
            ),
            "array_names": list(STRICT_FRAGMENT_TRACE_ARRAY_NAMES),
        },
        "signal_coverage": {
            key: {
                "count": int(coverage.get(key, 0)),
                "fraction": float(coverage.get(key, 0) / tile_count) if tile_count > 0 else 0.0,
            }
            for key in sorted(set(expected_has_cols) | set(coverage.keys()))
        },
    }
    out_path = output_path / "signal_validation.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if failures and strict:
        raise RuntimeError(
            f"Signal validation failed for {build}. See {out_path}.\n"
            + "\n".join(f"  - {item}" for item in failures)
        )

    return out_path


def _validate_decoded_metadata_table(build: str, output_path: Path, strict: bool = True) -> Path:
    index_rows = _read_index_rows(output_path)
    expected_tile_count = len(index_rows)
    expected_tile_ids = {int(row.get("tile_id", -1)) for row in index_rows}

    metadata_path = output_path / "decoded_metadata.parquet"
    failures: list[str] = []
    warnings_list: list[str] = []

    metadata_rows: list[dict[str, object]] = []
    if not metadata_path.exists():
        failures.append("decoded_metadata.parquet is missing")
    else:
        metadata_rows = _read_parquet_rows(metadata_path)

    if metadata_rows:
        row_count = len(metadata_rows)
        if row_count != expected_tile_count:
            failures.append(
                f"decoded metadata row count mismatch: expected {expected_tile_count}, got {row_count}"
            )

        observed_ids = [int(row.get("tile_id", -1)) for row in metadata_rows]
        observed_set = set(observed_ids)
        if len(observed_set) != len(observed_ids):
            failures.append("decoded metadata contains duplicate tile_id rows")

        missing_ids = sorted(expected_tile_ids - observed_set)
        extra_ids = sorted(observed_set - expected_tile_ids)
        if missing_ids:
            failures.append(
                f"decoded metadata missing tile_ids (sample): {missing_ids[:16]}"
            )
        if extra_ids:
            failures.append(
                f"decoded metadata has unexpected tile_ids (sample): {extra_ids[:16]}"
            )

        bad_json_rows = 0
        for row in metadata_rows:
            raw = row.get("decoded_metadata_json", "{}")
            text = str(raw if raw is not None else "{}")
            if not text:
                bad_json_rows += 1
                continue
            try:
                parsed = json.loads(text)
                if not isinstance(parsed, dict):
                    bad_json_rows += 1
            except Exception:
                bad_json_rows += 1
        if bad_json_rows > 0:
            failures.append(
                f"decoded metadata has {bad_json_rows} rows with invalid decoded_metadata_json payloads"
            )

    payload = {
        "build": build,
        "strict": bool(strict),
        "expected_tile_count": expected_tile_count,
        "metadata_row_count": len(metadata_rows),
        "passed": len(failures) == 0,
        "failures": failures,
        "warnings": warnings_list,
        "metadata_path": str(metadata_path),
    }
    out_path = output_path / "decoded_metadata_validation.json"
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    if failures and strict:
        raise RuntimeError(
            f"Decoded metadata validation failed for {build}. See {out_path}.\n"
            + "\n".join(f"  - {item}" for item in failures)
        )

    return out_path


def _read_index_rows(output_path: Path) -> list[dict]:
    idx_path = output_path / "index.parquet"
    table = pq.read_table(str(idx_path))
    return [
        {col: table.column(col)[i].as_py() for col in table.column_names}
        for i in range(table.num_rows)
    ]


def _tile_name_from_entry(entry: dict) -> str:
    map_name = str(entry.get("map", "")).strip()
    tile_x = entry.get("tile_x")
    tile_y = entry.get("tile_y")
    if tile_x is not None and tile_y is not None and map_name:
        return f"{map_name}_{int(tile_x)}_{int(tile_y)}"
    return ""


def _load_png_as_grayscale(path: Path, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    img = PIL.Image.open(str(path)).convert("L")
    img = img.resize(target_size, PIL.Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.float32) / 255.0


def _load_png_as_rgb(path: Path, target_size: tuple[int, int] = (256, 256)) -> np.ndarray:
    img = PIL.Image.open(str(path)).convert("RGB")
    img = img.resize(target_size, PIL.Image.Resampling.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _discover_capture_tiles(capture_dir: Path) -> dict[str, dict[str, Path]]:
    images_dir = capture_dir / "images"
    if not images_dir.exists():
        raise RuntimeError(f"No images directory found at {images_dir}")

    tiles: dict[str, dict[str, Path]] = {}
    for png in sorted(images_dir.glob("*.png")):
        name = png.stem
        if name.endswith("_object_visibility_mask"):
            tile_name = name[: -len("_object_visibility_mask")]
            tiles.setdefault(tile_name, {})["visibility_mask"] = png
        elif name.endswith("_no_objects"):
            tile_name = name[: -len("_no_objects")]
            tiles.setdefault(tile_name, {})["no_objects"] = png

    return tiles


def _apply_renderer_truth_patch(
    *,
    build: str,
    store_path: Path,
    capture_dir: Path,
    curation_manifest: Path | None,
    no_backup: bool,
) -> Path:
    if not store_path.exists():
        raise RuntimeError(f"No store at {store_path}")
    if not capture_dir.exists():
        raise RuntimeError(f"No capture directory at {capture_dir}")

    print(f"Patching renderer-truth signals for {build}")
    print(f"Store: {store_path}")
    print(f"Captures: {capture_dir}")

    manifest_keys: set[tuple[str, int]] | None = None
    if curation_manifest is not None:
        manifest_keys = _load_manifest_keep_keys(curation_manifest)
        print(f"Curation manifest: {curation_manifest} (keep keys={len(manifest_keys)})")

    tiles = _discover_capture_tiles(capture_dir)
    print(f"Found {len(tiles)} capture tiles: {sorted(tiles.keys())[:10]}...")
    if not tiles:
        raise RuntimeError("No capture tiles found. Check capture directory structure.")

    index_rows = _read_index_rows(store_path)
    print(f"Store has {len(index_rows)} tiles in index")

    capture_to_tile_id: dict[str, int] = {}
    manifest_expected: list[tuple[int, str]] = []
    skipped_by_manifest = 0
    for i, entry in enumerate(index_rows):
        tile_id = int(entry.get("tile_id", i))
        if manifest_keys is not None and (build, tile_id) not in manifest_keys:
            skipped_by_manifest += 1
            continue
        tile_name = _tile_name_from_entry(entry)
        if manifest_keys is not None:
            manifest_expected.append((tile_id, tile_name))
        if tile_name in tiles:
            capture_to_tile_id[tile_name] = i

    print(f"Matched {len(capture_to_tile_id)} capture tiles to store index")
    if not capture_to_tile_id:
        raise RuntimeError(
            "No capture tiles matched store index. "
            "Check that capture tile names match index map/tile_x/tile_y."
        )

    n_tiles = len(index_rows)
    visibility_masks = np.zeros((n_tiles, 256, 256), dtype=np.float32)
    no_object_minimaps = np.zeros((n_tiles, 256, 256, 3), dtype=np.uint8)
    has_visibility = np.zeros(n_tiles, dtype=bool)
    has_no_object = np.zeros(n_tiles, dtype=bool)

    matched_count = 0
    captured_complete = 0
    captured_partial = 0
    missing_capture = 0
    tile_status: dict[str, str] = {}
    if manifest_keys is not None:
        for _tile_id, tile_name in manifest_expected:
            caps = tiles.get(tile_name, {})
            has_mask = "visibility_mask" in caps
            has_noobj = "no_objects" in caps
            if has_mask and has_noobj:
                tile_status[tile_name] = "captured_complete"
                captured_complete += 1
            elif has_mask or has_noobj:
                tile_status[tile_name] = "captured_partial"
                captured_partial += 1
            else:
                tile_status[tile_name] = "missing"
                missing_capture += 1

    for tile_name, tile_id in capture_to_tile_id.items():
        caps = tiles[tile_name]
        if "visibility_mask" in caps:
            visibility_masks[tile_id] = _load_png_as_grayscale(caps["visibility_mask"])
            has_visibility[tile_id] = True
        if "no_objects" in caps:
            no_object_minimaps[tile_id] = _load_png_as_rgb(caps["no_objects"])
            has_no_object[tile_id] = True
        matched_count += 1

    print(f"Loaded renderer-truth data for {matched_count} tiles")
    print(f"  visibility_mask coverage: {has_visibility.sum()}/{n_tiles}")
    print(f"  no_object_minimap coverage: {has_no_object.sum()}/{n_tiles}")

    idx_path = store_path / "index.parquet"
    if not no_backup:
        backup_path = store_path / "index.parquet.bak.renderer_truth"
        if not backup_path.exists():
            shutil.copy2(idx_path, backup_path)
            print(f"Backed up {idx_path} -> {backup_path}")

    codec = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")
    store = zarr.storage.LocalStore(str(store_path), read_only=False)
    root = zarr.open_group(store=store, mode="a")
    try:
        root.create_array(
            "object_visibility_mask",
            data=visibility_masks,
            chunks=(1, 256, 256),
            compressors=codec,
            overwrite=True,
        )
        if int(has_no_object.sum()) > 0:
            root.create_array(
                "no_object_minimap",
                data=no_object_minimaps,
                chunks=(1, 256, 256, 3),
                compressors=codec,
                overwrite=True,
            )
    finally:
        store.close()

    for i, row in enumerate(index_rows):
        row["has_object_visibility_mask"] = bool(has_visibility[i])
        row["has_no_object_minimap"] = bool(has_no_object[i])

    _write_index(index_rows, store_path)

    report = {
        "build": build,
        "patched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tiles": n_tiles,
        "matched_capture_tiles": matched_count,
        "visibility_mask_coverage": int(has_visibility.sum()),
        "no_object_minimap_coverage": int(has_no_object.sum()),
        "capture_dir": str(capture_dir),
        "manifest_path": str(curation_manifest) if curation_manifest else None,
        "manifest_scope_tile_count": len(manifest_expected) if manifest_keys is not None else None,
        "manifest_skipped_index_tiles": skipped_by_manifest if manifest_keys is not None else None,
        "manifest_captured_complete": captured_complete if manifest_keys is not None else None,
        "manifest_captured_partial": captured_partial if manifest_keys is not None else None,
        "manifest_missing_capture": missing_capture if manifest_keys is not None else None,
        "manifest_tile_status": tile_status if manifest_keys is not None else None,
        "matched_tiles": sorted(capture_to_tile_id.keys()),
    }
    report_path = store_path / "renderer_truth_patch_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _clear_renderer_truth_signals(
    *,
    build: str,
    store_path: Path,
    no_backup: bool,
    reason: str,
) -> Path:
    if not store_path.exists():
        raise RuntimeError(f"No store at {store_path}")

    print(f"Clearing renderer-truth signals for {build}")
    print(f"Store: {store_path}")

    index_rows = _read_index_rows(store_path)
    n_tiles = len(index_rows)
    if n_tiles <= 0:
        raise RuntimeError(f"Store index is empty for {build}")

    idx_path = store_path / "index.parquet"
    if not no_backup:
        backup_path = store_path / "index.parquet.bak.renderer_truth_clear"
        if not backup_path.exists():
            shutil.copy2(idx_path, backup_path)
            print(f"Backed up {idx_path} -> {backup_path}")

    store = zarr.storage.LocalStore(str(store_path), read_only=False)
    root = zarr.open_group(store=store, mode="a")
    try:
        if "object_visibility_mask" in root:
            array = root["object_visibility_mask"]
            shape = array.shape
            root.create_array(
                "object_visibility_mask",
                data=np.zeros(shape, dtype=np.float32),
                chunks=array.chunks,
                compressors=array.compressors,
                overwrite=True,
            )
        if "no_object_minimap" in root:
            array = root["no_object_minimap"]
            shape = array.shape
            root.create_array(
                "no_object_minimap",
                data=np.zeros(shape, dtype=np.uint8),
                chunks=array.chunks,
                compressors=array.compressors,
                overwrite=True,
            )
    finally:
        store.close()

    for row in index_rows:
        row["has_object_visibility_mask"] = False
        row["has_no_object_minimap"] = False

    _write_index(index_rows, store_path)

    patch_report_path = store_path / "renderer_truth_patch_report.json"
    if patch_report_path.exists():
        patch_report_path.unlink()

    report = {
        "build": build,
        "cleared_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "total_tiles": n_tiles,
        "reason": reason,
        "cleared_signals": [
            "object_visibility_mask",
            "no_object_minimap",
        ],
    }
    report_path = store_path / "renderer_truth_reset_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def _read_parquet_rows(parquet_path: Path) -> list[dict]:
    table = pq.read_table(str(parquet_path))
    return [
        {col: table.column(col)[i].as_py() for col in table.column_names}
        for i in range(table.num_rows)
    ]


def _ordered_maps_from_index_rows(index_rows: list[dict]) -> list[str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for row in index_rows:
        map_name = str(row.get("map", ""))
        if not map_name or map_name in seen:
            continue
        seen.add(map_name)
        ordered.append(map_name)
    return ordered


def _collect_map_rows_parallel(
    map_names: list[str],
    worker_fn,
    *,
    map_workers: int,
    label: str,
) -> dict[str, list[dict[str, object]]]:
    ordered_maps = [str(name) for name in map_names]
    if not ordered_maps:
        return {}

    worker_count = max(1, min(int(map_workers), len(ordered_maps)))
    if worker_count == 1:
        return {map_name: worker_fn(map_name) for map_name in ordered_maps}

    print(f"{label}: using map_workers={worker_count}")
    results: dict[str, list[dict[str, object]]] = {}
    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        future_to_map = {
            executor.submit(worker_fn, map_name): map_name
            for map_name in ordered_maps
        }
        for future in as_completed(future_to_map):
            map_name = future_to_map[future]
            rows = future.result()
            results[map_name] = rows
            print(f"  finished {label} stream for {map_name}: {len(rows)} rows")

    return results


def _normalize_map_name(meta_map: object, requested_map: str) -> str:
    raw = str(meta_map or "").strip()
    if not raw:
        return requested_map
    low = raw.lower()
    if low in {"memory", "<memory>", "unknown", "<unknown>"}:
        return requested_map
    return raw


def _is_placeholder_map_name(name: str) -> bool:
    low = str(name or "").strip().lower()
    return low in {"", "memory", "<memory>", "unknown", "<unknown>"}


def _stream_valid_tile_metadata(
    harvest_tool: Path,
    client_root: Path,
    map_name: str,
    build_version: str | None,
) -> list[dict[str, object]]:
    cmd = [
        str(harvest_tool),
        "harvest-stream",
        "--client-root",
        str(client_root),
        "--map",
        map_name,
    ]
    if build_version:
        cmd.extend(["--build", build_version])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    if proc.stdout is None or proc.stderr is None:
        proc.terminate()
        raise RuntimeError(f"Failed to open harvest-stream pipes for map {map_name}.")

    stderr_tail: deque[str] = deque(maxlen=40)
    stderr_thread = threading.Thread(
        target=_pump_stderr,
        args=(proc.stderr, map_name, stderr_tail),
        daemon=True,
    )
    stderr_thread.start()

    rows: list[dict[str, object]] = []
    saw_end_marker = False
    stream_error: str | None = None

    while True:
        header = proc.stdout.read(8)
        if not header:
            stream_error = "stdout closed before ENDS sentinel"
            break
        if len(header) < 8:
            stream_error = f"truncated stream header ({len(header)}/8 bytes)"
            break

        magic = header[:4]
        if magic == ENDS_MAGIC:
            saw_end_marker = True
            break
        if magic not in (NPZB_MAGIC, ARRY_MAGIC):
            stream_error = f"unexpected stream magic {magic!r}"
            break

        length = struct.unpack("<I", header[4:8])[0]
        if length == 0 or length > 50_000_000:
            stream_error = f"invalid NPZ blob length {length}"
            break

        blob = proc.stdout.read(length)
        if not blob or len(blob) < length:
            stream_error = f"truncated NPZ blob ({len(blob) if blob else 0}/{length} bytes)"
            break

        try:
            data = _decode_blob(blob)
        except Exception as ex:
            stream_error = f"failed to decode streamed NPZ blob: {ex}"
            break

        if not REQUIRED_KEYS.issubset(data.keys()):
            continue

        meta = _decode_metadata_json(data)
        tx, ty = _extract_tile_coords_from_metadata(meta)
        actual_map = _normalize_map_name(meta.get("map_name", map_name), map_name)
        rows.append({"map": actual_map, "tile_x": tx, "tile_y": ty})

    if proc.poll() is None and (stream_error is not None or not saw_end_marker):
        proc.terminate()

    return_code = proc.wait()
    stderr_thread.join(timeout=2.0)

    if stream_error is not None:
        raise RuntimeError(
            f"Harvest stream failed for map {map_name}: {stream_error}\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )
    if not saw_end_marker:
        raise RuntimeError(
            f"Harvest stream ended without ENDS sentinel for map {map_name}.\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )
    if return_code != 0:
        raise RuntimeError(
            f"Harvest stream exited with code {return_code} for map {map_name}.\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )

    return rows


def _stream_valid_tile_liquid_rows(
    harvest_tool: Path,
    client_root: Path,
    map_name: str,
    build_version: str | None,
) -> list[dict[str, object]]:
    cmd = [
        str(harvest_tool),
        "harvest-stream",
        "--client-root",
        str(client_root),
        "--map",
        map_name,
    ]
    if build_version:
        cmd.extend(["--build", build_version])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    if proc.stdout is None or proc.stderr is None:
        proc.terminate()
        raise RuntimeError(f"Failed to open harvest-stream pipes for map {map_name}.")

    stderr_tail: deque[str] = deque(maxlen=40)
    stderr_thread = threading.Thread(
        target=_pump_stderr,
        args=(proc.stderr, map_name, stderr_tail),
        daemon=True,
    )
    stderr_thread.start()

    rows: list[dict[str, object]] = []
    saw_end_marker = False
    stream_error: str | None = None

    while True:
        header = proc.stdout.read(8)
        if not header:
            stream_error = "stdout closed before ENDS sentinel"
            break
        if len(header) < 8:
            stream_error = f"truncated stream header ({len(header)}/8 bytes)"
            break

        magic = header[:4]
        if magic == ENDS_MAGIC:
            saw_end_marker = True
            break
        if magic not in (NPZB_MAGIC, ARRY_MAGIC):
            stream_error = f"unexpected stream magic {magic!r}"
            break

        length = struct.unpack("<I", header[4:8])[0]
        if length == 0 or length > 50_000_000:
            stream_error = f"invalid NPZ blob length {length}"
            break

        blob = proc.stdout.read(length)
        if not blob or len(blob) < length:
            stream_error = f"truncated NPZ blob ({len(blob) if blob else 0}/{length} bytes)"
            break

        try:
            data = _decode_blob(blob)
        except Exception as ex:
            stream_error = f"failed to decode streamed NPZ blob: {ex}"
            break

        if not REQUIRED_KEYS.issubset(data.keys()):
            continue

        liquid_mask, liquid_height, liquid_has_signal, liquid_source = _derive_liquid_supervision(data)
        meta = _decode_metadata_json(data)
        tx, ty = _extract_tile_coords_from_metadata(meta)
        actual_map = _normalize_map_name(meta.get("map_name", map_name), map_name)
        row = {
            "map": actual_map,
            "tile_x": int(tx),
            "tile_y": int(ty),
            "liquid_mask": liquid_mask.astype(np.float32, copy=False),
            "liquid_height": liquid_height.astype(np.float32, copy=False),
            "has_liquid_mask": bool(liquid_has_signal),
            "has_liquid_height": bool(liquid_has_signal),
        }
        for source_name in LIQUID_SOURCE_KEYS:
            row[f"has_liquid_source_{source_name}"] = liquid_source == source_name
        rows.append(row)

    if proc.poll() is None and (stream_error is not None or not saw_end_marker):
        proc.terminate()

    return_code = proc.wait()
    stderr_thread.join(timeout=2.0)

    if stream_error is not None:
        raise RuntimeError(
            f"Harvest stream failed for map {map_name}: {stream_error}\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )
    if not saw_end_marker:
        raise RuntimeError(
            f"Harvest stream ended without ENDS sentinel for map {map_name}.\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )
    if return_code != 0:
        raise RuntimeError(
            f"Harvest stream exited with code {return_code} for map {map_name}.\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )

    return rows


def _stream_valid_tile_object_rows(
    harvest_tool: Path,
    client_root: Path,
    map_name: str,
    build_version: str | None,
) -> list[dict[str, object]]:
    cmd = [
        str(harvest_tool),
        "harvest-stream",
        "--client-root",
        str(client_root),
        "--map",
        map_name,
    ]
    if build_version:
        cmd.extend(["--build", build_version])

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=0,
    )
    if proc.stdout is None or proc.stderr is None:
        proc.terminate()
        raise RuntimeError(f"Failed to open harvest-stream pipes for map {map_name}.")

    stderr_tail: deque[str] = deque(maxlen=40)
    stderr_thread = threading.Thread(
        target=_pump_stderr,
        args=(proc.stderr, map_name, stderr_tail),
        daemon=True,
    )
    stderr_thread.start()

    rows: list[dict[str, object]] = []
    saw_end_marker = False
    stream_error: str | None = None

    while True:
        header = proc.stdout.read(8)
        if not header:
            stream_error = "stdout closed before ENDS sentinel"
            break
        if len(header) < 8:
            stream_error = f"truncated stream header ({len(header)}/8 bytes)"
            break

        magic = header[:4]
        if magic == ENDS_MAGIC:
            saw_end_marker = True
            break
        if magic not in (NPZB_MAGIC, ARRY_MAGIC):
            stream_error = f"unexpected stream magic {magic!r}"
            break

        length = struct.unpack("<I", header[4:8])[0]
        if length == 0 or length > 50_000_000:
            stream_error = f"invalid NPZ blob length {length}"
            break

        blob = proc.stdout.read(length)
        if not blob or len(blob) < length:
            stream_error = f"truncated NPZ blob ({len(blob) if blob else 0}/{length} bytes)"
            break

        try:
            data = _decode_blob(blob)
        except Exception as ex:
            stream_error = f"failed to decode streamed NPZ blob: {ex}"
            break

        if not REQUIRED_KEYS.issubset(data.keys()):
            continue

        object_mask, object_precise, object_instance, has_mask, has_precise, has_instance, mddf_mask, modf_mask, object_filtered, has_mddf, has_modf, has_filtered = _derive_object_supervision(data)
        meta = _decode_metadata_json(data)
        tx, ty = _extract_tile_coords_from_metadata(meta)
        actual_map = _normalize_map_name(meta.get("map_name", map_name), map_name)
        rows.append(
            {
                "map": actual_map,
                "tile_x": int(tx),
                "tile_y": int(ty),
                "object_mask": object_mask,
                "object_precise_mask": object_precise,
                "object_instance_mask": object_instance,
                "has_object_mask": bool(has_mask),
                "has_object_precise_mask": bool(has_precise),
                "has_object_instance_mask": bool(has_instance),
            }
        )

    if proc.poll() is None and (stream_error is not None or not saw_end_marker):
        proc.terminate()

    return_code = proc.wait()
    stderr_thread.join(timeout=2.0)

    if stream_error is not None:
        raise RuntimeError(
            f"Harvest stream failed for map {map_name}: {stream_error}\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )
    if not saw_end_marker:
        raise RuntimeError(
            f"Harvest stream ended without ENDS sentinel for map {map_name}.\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )
    if return_code != 0:
        raise RuntimeError(
            f"Harvest stream exited with code {return_code} for map {map_name}.\n"
            f"stderr tail:\n{_tail_text(stderr_tail)}"
        )

    return rows


def _maybe_apply_v18_promoted_signals(
    *,
    build: str,
    output_path: Path,
    capture_root: Path | None,
    curation_manifest: Path | None,
    no_backup: bool,
    experimental_renderer_truth_promotion: bool,
) -> list[Path]:
    reports: list[Path] = []

    if capture_root is not None and experimental_renderer_truth_promotion:
        capture_dir = capture_root / build
        if capture_dir.exists():
            report_path = _apply_renderer_truth_patch(
                build=build,
                store_path=output_path,
                capture_dir=capture_dir,
                curation_manifest=curation_manifest,
                no_backup=no_backup,
            )
            reports.append(report_path)
        else:
            print(f"Promoted renderer-truth capture root not found for {build}: {capture_dir}")

    return reports


def _write_placements(all_placements: list[dict], output_path: Path) -> None:
    if not all_placements:
        return
    fields = [
        pa.field("tile_id", pa.int64()),
        pa.field("instance_type", pa.string()),
        pa.field("instance_idx", pa.int32()),
        pa.field("asset_path", pa.string()),
    ]
    for col in PLACEMENT_COLUMNS_MDDF:
        fields.append(pa.field(col, pa.float32()))
    for col in PLACEMENT_COLUMNS_MODF:
        if col not in [f.name for f in fields]:
            fields.append(pa.field(col, pa.float32()))

    schema = pa.schema(fields)
    col_data = {f.name: [] for f in fields}
    for row in all_placements:
        for f in fields:
            val = row.get(f.name, 0.0 if f.type == pa.float32() else "")
            col_data[f.name].append(val)

    table = pa.table(col_data, schema=schema)
    pq.write_table(table, str(output_path / "placements.parquet"))


def _write_decoded_metadata(rows: list[dict[str, object]], output_path: Path) -> None:
    if not rows:
        return

    schema = pa.schema(
        [
            pa.field("tile_id", pa.int64()),
            pa.field("build", pa.string()),
            pa.field("map", pa.string()),
            pa.field("tile_x", pa.int32()),
            pa.field("tile_y", pa.int32()),
            pa.field("tile_name", pa.string()),
            pa.field("source_adt_path", pa.string()),
            pa.field("source_wdt_path", pa.string()),
            pa.field("raw_chunks_count", pa.int32()),
            pa.field("decoded_metadata_json", pa.large_string()),
            pa.field("decoded_metadata_keys_json", pa.large_string()),
        ]
    )

    col_data = {f.name: [] for f in schema}
    for row in rows:
        col_data["tile_id"].append(int(row.get("tile_id", -1)))
        col_data["build"].append(str(row.get("build", "")))
        col_data["map"].append(str(row.get("map", "")))
        col_data["tile_x"].append(int(row.get("tile_x", 0)))
        col_data["tile_y"].append(int(row.get("tile_y", 0)))
        col_data["tile_name"].append(str(row.get("tile_name", "")))
        col_data["source_adt_path"].append(str(row.get("source_adt_path", "")))
        col_data["source_wdt_path"].append(str(row.get("source_wdt_path", "")))
        col_data["raw_chunks_count"].append(int(row.get("raw_chunks_count", 0)))
        col_data["decoded_metadata_json"].append(str(row.get("decoded_metadata_json", "{}")))
        col_data["decoded_metadata_keys_json"].append(str(row.get("decoded_metadata_keys_json", "[]")))

    table = pa.table(col_data, schema=schema)
    pq.write_table(table, str(output_path / "decoded_metadata.parquet"))


def _format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{num_bytes} B"


def _format_ratio(raw_bytes: int, stored_bytes: int) -> str:
    if stored_bytes <= 0:
        return "n/a"
    return f"{raw_bytes / stored_bytes:.2f}x"


def _require_explicit_zarr_write(args: argparse.Namespace, operation: str) -> None:
    if getattr(args, "allow_zarr_write", False):
        return

    raise SystemExit(
            "Refusing to write or mutate a Zarr store without --allow-zarr-write. "
            f"This {operation} command changes ../output/datasets/v18/*.zarr directly. "
            "Inspect raw harvest output first with scripts/inspect_v16_harvest_samples.py, "
            "then rerun the write command with --allow-zarr-write once the raw masks/signals look correct."
        )


def cmd_build(args: argparse.Namespace) -> None:
    _require_explicit_zarr_write(args, "build")

    builds = args.builds or [args.build]
    harvest_tool = _find_harvest_tool()
    print(f"Harvest tool: {harvest_tool}")

    maps_override = getattr(args, "maps", None)
    limit = args.limit
    resume = args.resume
    codec_name = args.codec
    codec_level = args.clevel
    codec_shuffle = args.shuffle
    rebuild_existing = args.rebuild_existing
    signal_validation = args.signal_validation
    signal_validation_strict = args.signal_validation_strict
    decoded_metadata_validation = args.decoded_metadata_validation
    decoded_metadata_validation_strict = args.decoded_metadata_validation_strict
    capture_root = Path(args.capture_root) if getattr(args, "capture_root", None) else None
    curation_manifest = Path(args.curation_manifest) if getattr(args, "curation_manifest", None) else None
    no_backup = bool(getattr(args, "no_backup", False))
    experimental_renderer_truth_promotion = bool(getattr(args, "experimental_renderer_truth_promotion", False))
    tile_workers = max(1, int(args.tile_workers))
    if capture_root is not None and not experimental_renderer_truth_promotion:
        raise RuntimeError(
            "`--capture-root` now requires `--experimental-renderer-truth-promotion`. "
            "The image-derived renderer-truth lane is still bounded-proof only and must not be treated as a closed canonical signal path yet."
        )
    if codec_name == "none" and (codec_level != 0 or codec_shuffle != "noshuffle"):
        print(
            "Warning: --codec none ignores --clevel/--shuffle; storing arrays uncompressed.",
            file=sys.stderr,
            flush=True,
        )

    for build in builds:
        client_root = _find_client_root(build)
        if client_root is None:
            print(f"SKIP build {build}: no client root found at {_CLIENT_ROOTS / build}")
            continue

        output_path = _DATASET_ROOT / f"{build}.zarr"
        staging_path = _DATASET_ROOT / f"{build}.zarr.partial"
        completed_state = None if rebuild_existing else _load_completed_final_store_state(output_path)
        if completed_state is not None and not staging_path.exists():
            completed_maps = completed_state.get("completed_maps", [])
            print(
                f"SKIP build {build}: final store already complete at {output_path} "
                f"({len(completed_maps)} maps, {completed_state.get('valid_tiles', 'unknown')} tiles)"
            )
            continue
        if staging_path.exists() and not resume:
            shutil.rmtree(staging_path)
        if not staging_path.exists():
            staging_path.mkdir(parents=True, exist_ok=True)

        build_version = build.replace("_", ".")
        map_names = maps_override or _discover_maps_for_build(harvest_tool, client_root)

        print(f"\n{'='*60}")
        print(f"Building V18 dataset for {build}")
        print(f"Client: {client_root}")
        print(f"Maps: {map_names}")
        print(f"Output: {output_path}")
        print(f"Staging: {staging_path}")
        print(f"Rejected tiles report: {_tile_rejection_report_path(output_path, build)}")
        print(f"Resume: {resume}")
        print(f"Tile workers: {tile_workers}")
        print(f"Codec: {codec_name} clevel={codec_level} shuffle={codec_shuffle}")
        print(f"Signal validation: enabled={signal_validation} strict={signal_validation_strict}")
        print(
            "Decoded metadata validation: "
            f"enabled={decoded_metadata_validation} strict={decoded_metadata_validation_strict}"
        )
        if experimental_renderer_truth_promotion:
            print(
                "Experimental renderer-truth promotion: enabled "
                f"for {EXPERIMENTAL_RENDERER_TRUTH_SIGNAL_KEYS} via capture root {capture_root}"
            )

        try:
            _build_zarr_streaming(
                harvest_tool=harvest_tool,
                client_root=client_root,
                build=build,
                build_version=build_version,
                map_names=map_names,
                output_path=staging_path,
                limit=limit,
                rejected_tiles_report_path=_tile_rejection_report_path(output_path, build),
                resume=resume,
                codec_name=codec_name,
                codec_level=codec_level,
                codec_shuffle=codec_shuffle,
                tile_workers=tile_workers,
            )
            if output_path.exists():
                shutil.rmtree(output_path)
            staging_path.replace(output_path)
            print(f"Promoted staged dataset -> {output_path}")
            promoted_reports = _maybe_apply_v18_promoted_signals(
                build=build,
                output_path=output_path,
                capture_root=capture_root,
                curation_manifest=curation_manifest,
                no_backup=no_backup,
                experimental_renderer_truth_promotion=experimental_renderer_truth_promotion,
            )
            for report_path in promoted_reports:
                print(f"Promoted signal report: {report_path}")
            validation_path: Path | None = None
            metadata_validation_path: Path | None = None
            if signal_validation:
                validation_path = _validate_build_signals(
                    build=build,
                    output_path=output_path,
                    strict=signal_validation_strict,
                )
                print(f"Signal validation report: {validation_path}")
            if decoded_metadata_validation:
                metadata_validation_path = _validate_decoded_metadata_table(
                    build=build,
                    output_path=output_path,
                    strict=decoded_metadata_validation_strict,
                )
                print(f"Decoded metadata validation report: {metadata_validation_path}")
            finalization_path = _write_finalization_state(
                output_path,
                build=build,
                finalized=True,
                signal_validation_path=validation_path,
                decoded_metadata_validation_path=metadata_validation_path,
                required_files=V18_REQUIRED_ARTIFACTS,
            )
            print(f"Finalization report: {finalization_path}")
        except Exception:
            print(
                f"Build failed for {build}. Partial output preserved at {staging_path}",
                file=sys.stderr,
                flush=True,
            )
            raise


def _build_zarr_streaming(
    harvest_tool: Path,
    client_root: Path,
    build: str,
    build_version: str,
    map_names: list[str],
    output_path: Path,
    limit: int | None,
    rejected_tiles_report_path: Path | None = None,
    resume: bool = False,
    codec_name: str = DEFAULT_CODEC,
    codec_level: int = DEFAULT_CLEVEL,
    codec_shuffle: str = DEFAULT_SHUFFLE,
    tile_workers: int = DEFAULT_TILE_WORKERS,
) -> None:
    compressors = None
    if codec_name != "none":
        compressors = [zarr.codecs.BloscCodec(cname=codec_name, clevel=codec_level, shuffle=codec_shuffle)]
    resume_state = _load_resume_state(output_path) if resume else None

    if resume and resume_state is None:
        has_meaningful_partial = output_path.exists() and any(output_path.iterdir())
        if has_meaningful_partial:
            substantive_entries = [
                path.name
                for path in output_path.iterdir()
                if path.name not in {"zarr.json", ".zgroup", ".zattrs"}
            ]
            if substantive_entries:
                raise RuntimeError(
                    f"Resume requested for {output_path}, but no {_resume_state_path(output_path).name} was found."
                )
        print(
            f"  Resume requested for {build}, but no {_resume_state_path(output_path).name} exists yet. "
            f"Starting a fresh staged build at {output_path}.",
            flush=True,
        )

    store = zarr.storage.LocalStore(str(output_path), read_only=False)
    root = zarr.open_group(store=store, mode="a" if resume_state is not None else "w")

    arrays: dict[str, zarr.Array] = {}
    index_rows: list[dict] = []
    all_placements: list[dict] = []
    decoded_metadata_rows: list[dict[str, object]] = []
    valid = 0
    skipped_zero_usable_maps = 0
    rejected_tile_count = 0
    t0 = time.perf_counter()
    capacity = 50000
    completed_maps: list[str] = []
    pending_arrays: dict[str, list[np.ndarray]] = {key: [] for key in ALL_ARRAY_KEYS}
    pending_count = 0

    if resume_state is not None:
        expected_maps = resume_state.get("requested_maps", [])
        if expected_maps != map_names:
            raise RuntimeError(
                f"Resume map list mismatch for {build}. Existing partial requested_maps={expected_maps} "
                f"but current maps={map_names}."
            )
        if resume_state.get("codec") != codec_name or int(resume_state.get("clevel", -1)) != codec_level or resume_state.get("shuffle") != codec_shuffle:
            raise RuntimeError(
                f"Resume codec mismatch for {build}. Existing partial uses "
                f"{resume_state.get('codec')} clevel={resume_state.get('clevel')} shuffle={resume_state.get('shuffle')}, "
                f"current request is {codec_name} clevel={codec_level} shuffle={codec_shuffle}."
            )
        capacity = int(resume_state.get("capacity", capacity))
        completed_maps = [str(name) for name in resume_state.get("completed_maps", [])]
        skipped_zero_usable_maps = int(resume_state.get("skipped_zero_usable_maps", 0))
        rejected_tile_count = int(resume_state.get("rejected_tile_count", 0))
        idx_path = output_path / "index.parquet"
        if idx_path.exists():
            table = pq.read_table(str(idx_path))
            index_rows = [
                {col: table.column(col)[i].as_py() for col in table.column_names}
                for i in range(table.num_rows)
            ]
            valid = len(index_rows)
        else:
            valid = int(resume_state.get("valid_tiles", 0))
        pl_path = output_path / "placements.parquet"
        if pl_path.exists():
            pl_table = pq.read_table(str(pl_path))
            all_placements = [
                {col: pl_table.column(col)[i].as_py() for col in pl_table.column_names}
                for i in range(pl_table.num_rows)
            ]
        dm_path = output_path / "decoded_metadata.parquet"
        if dm_path.exists():
            dm_table = pq.read_table(str(dm_path))
            decoded_metadata_rows = [
                {col: dm_table.column(col)[i].as_py() for col in dm_table.column_names}
                for i in range(dm_table.num_rows)
            ]

    rejected_tiles_report = None
    if rejected_tiles_report_path is not None:
        rejected_tiles_report_path.parent.mkdir(parents=True, exist_ok=True)
        if rejected_tiles_report_path.exists() and resume_state is None:
            rejected_tiles_report_path.unlink()
        rejected_tiles_report = rejected_tiles_report_path.open("a" if resume_state is not None else "w", encoding="utf-8")

    for key in ALL_ARRAY_KEYS:
        if key in root:
            arrays[key] = root[key]
        else:
            shape = (capacity,) + SHAPES[key]
            chunks = CHUNK_SIZES.get(key, (64,) + SHAPES[key])
            arrays[key] = root.create_array(
                key, shape=shape, chunks=chunks, dtype=DTYPES[key],
                compressors=compressors, fill_value=FILL_VALUES.get(key, 0),
            )

    trace_sidecar = StrictFragmentTraceSidecar.open(
        output_path,
        tile_capacity=capacity,
        resume_tile_count=valid,
    )

    for map_name in map_names:
        if map_name in completed_maps:
            print(f"\n  Skipping completed map: {map_name}")
            continue

        print(f"\n  Streaming map: {map_name}")

        cmd = [
            str(harvest_tool), "harvest-stream",
            "--client-root", str(client_root),
            "--map", map_name,
            "--stream-profile", "v16",
            "--tile-workers", str(tile_workers),
        ]
        if build_version:
            cmd.extend(["--build", build_version])

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, bufsize=0,
        )
        if proc.stdout is None or proc.stderr is None:
            proc.terminate()
            raise RuntimeError(f"Failed to open harvest-stream pipes for map {map_name}.")

        stderr_tail: deque[str] = deque(maxlen=40)
        stderr_thread = threading.Thread(
            target=_pump_stderr,
            args=(proc.stderr, map_name, stderr_tail),
            daemon=True,
        )
        stderr_thread.start()

        tile_count = 0
        dropped_missing_required = 0
        map_placements = 0
        map_blob_bytes = 0
        saw_end_marker = False
        terminated_for_limit = False
        stream_error: str | None = None
        while True:
            header = proc.stdout.read(8)
            if not header:
                stream_error = "stdout closed before ENDS sentinel"
                break
            if len(header) < 8:
                stream_error = f"truncated stream header ({len(header)}/8 bytes)"
                break

            magic = header[:4]
            if magic == ENDS_MAGIC:
                saw_end_marker = True
                break
            if magic not in (NPZB_MAGIC, ARRY_MAGIC):
                stream_error = f"unexpected stream magic {magic!r}"
                break

            length = struct.unpack("<I", header[4:8])[0]
            if length == 0 or length > 50_000_000:
                stream_error = f"invalid NPZ blob length {length}"
                break

            blob = proc.stdout.read(length)
            if not blob or len(blob) < length:
                stream_error = f"truncated NPZ blob ({len(blob) if blob else 0}/{length} bytes)"
                break

            try:
                data = _decode_blob(blob)
            except Exception as ex:
                stream_error = f"failed to decode streamed NPZ blob: {ex}"
                break

            result = _process_tile_data(data)
            if result is None:
                dropped_missing_required += 1
                rejected_tile_count += 1
                missing = sorted(REQUIRED_KEYS - set(data.keys()))
                rejection_reason = (
                    "unverified_wl_liquid_provenance"
                    if _has_unverified_wl_liquid_fallback(data)
                    else "missing_required_keys"
                )
                meta = _decode_metadata_json(data)
                source_adt_path = str(meta.get("source_adt_path", ""))
                tx, ty = _extract_tile_coords_from_metadata(meta)
                if rejected_tiles_report is not None:
                    rejected_map = _normalize_map_name(meta.get("map_name", map_name), map_name)
                    rejected_tiles_report.write(
                        json.dumps(
                            {
                                "build": build,
                                "map_name": rejected_map,
                                "source_adt_path": source_adt_path,
                                "tile_x": tx,
                                "tile_y": ty,
                                "rejection_reason": rejection_reason,
                                "missing_required_keys": missing,
                                "available_keys": sorted(data.keys()),
                            },
                            sort_keys=True,
                        )
                        + "\n"
                    )
                    rejected_tiles_report.flush()
                if dropped_missing_required <= 5:
                    print(
                        f"    Warning: dropped tile blob ({rejection_reason}); missing required keys {missing}; "
                        f"available keys: {sorted(data.keys())}",
                        file=sys.stderr,
                        flush=True,
                    )
                continue

            tile_arrays, has_signals = result

            h_mean = float(np.mean(tile_arrays["height_257"]))
            h_std = float(np.std(tile_arrays["height_257"])) + 1e-8

            meta = _decode_metadata_json(data)
            tx, ty = 0, 0
            actual_map = map_name
            if meta:
                tx, ty = _extract_tile_coords_from_metadata(meta)
                actual_map = _normalize_map_name(meta.get("map_name", map_name), map_name)

            # Extract placement data
            tile_id = valid + pending_count
            mddf_rows, modf_rows, mddf_names, modf_names = _extract_placements(data, meta)
            for row in mddf_rows:
                row["tile_id"] = tile_id
                all_placements.append(row)
            for row in modf_rows:
                row["tile_id"] = tile_id
                all_placements.append(row)
            map_placements += len(mddf_rows) + len(modf_rows)

            row = {
                "tile_id": tile_id, "build": build, "map": actual_map,
                "tile_x": tx, "tile_y": ty,
                "height_mean": h_mean, "height_std": h_std,
                "n_mddf": len(mddf_rows), "n_modf": len(modf_rows),
                "object_roof_mask_source": str(meta.get("object_roof_mask_source", "") if isinstance(meta, dict) else ""),
            }
            strict_provenance, strict_trace = _extract_strict_object_target_provenance(
                data,
                meta if isinstance(meta, dict) else {},
            )
            row.update(strict_provenance)
            trace_start, trace_end = trace_sidecar.append(tile_id, strict_trace)
            if strict_trace is not None:
                row[STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD] = trace_start
                row[STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD] = trace_end
            for key in SIGNAL_FLAG_KEYS:
                row[f"has_{key}"] = bool(has_signals.get(key, False))
            index_rows.append(row)

            raw_chunks = meta.get("raw_chunks") if isinstance(meta, dict) else None
            raw_chunks_count = len(raw_chunks) if isinstance(raw_chunks, list) else 0
            metadata_keys = sorted(meta.keys()) if isinstance(meta, dict) else []
            decoded_metadata_rows.append(
                {
                    "tile_id": tile_id,
                    "build": build,
                    "map": actual_map,
                    "tile_x": int(tx),
                    "tile_y": int(ty),
                    "tile_name": str(meta.get("tile_name", "") if isinstance(meta, dict) else ""),
                    "source_adt_path": str(meta.get("source_adt_path", "") if isinstance(meta, dict) else ""),
                    "source_wdt_path": str(meta.get("source_wdt_path", "") if isinstance(meta, dict) else ""),
                    "raw_chunks_count": int(raw_chunks_count),
                    "decoded_metadata_json": json.dumps(meta if isinstance(meta, dict) else {}, sort_keys=True, separators=(",", ":")),
                    "decoded_metadata_keys_json": json.dumps(metadata_keys, separators=(",", ":")),
                }
            )

            needed = valid + pending_count + 1
            while needed >= capacity:
                capacity += 50000
                for key in ALL_ARRAY_KEYS:
                    arrays[key].resize((capacity,) + SHAPES[key])
                trace_sidecar.ensure_tile_capacity(capacity)

            for key in ALL_ARRAY_KEYS:
                pending_arrays[key].append(tile_arrays[key])
            pending_count += 1

            if pending_count >= WRITE_BATCH_SIZE:
                valid = _flush_tile_batch_with_retry(
                    arrays,
                    valid,
                    pending_arrays,
                    pending_count,
                    map_name=actual_map,
                )
                pending_count = 0

            tile_count += 1
            map_blob_bytes += length
            if tile_count == 1 or tile_count % 10 == 0:
                elapsed = time.perf_counter() - t0
                total_written = valid + pending_count
                rate = total_written / max(elapsed, 0.01)
                store_mb = _dir_size_bytes(output_path) / 1024 / 1024
                print(
                    f"    Progress {map_name}: map_tiles={tile_count} total_tiles={total_written} "
                    f"placements={map_placements} raw_npz={map_blob_bytes / 1024 / 1024:.1f} MB "
                    f"store={store_mb:.1f} MB rate={rate:.1f} tiles/s",
                    flush=True,
                )

            if limit is not None and valid + pending_count >= limit:
                terminated_for_limit = True
                proc.terminate()
                break

        if proc.poll() is None and (stream_error is not None or (not saw_end_marker and not terminated_for_limit)):
            proc.terminate()

        return_code = proc.wait()
        stderr_thread.join(timeout=2.0)

        if stream_error is not None:
            raise RuntimeError(
                f"Harvest stream failed for map {map_name}: {stream_error}\n"
                f"stderr tail:\n{_tail_text(stderr_tail)}"
            )
        if not saw_end_marker and not terminated_for_limit:
            raise RuntimeError(
                f"Harvest stream ended without ENDS sentinel for map {map_name}.\n"
                f"stderr tail:\n{_tail_text(stderr_tail)}"
            )
        if return_code != 0 and not terminated_for_limit:
            raise RuntimeError(
                f"Harvest stream exited with code {return_code} for map {map_name}.\n"
                f"stderr tail:\n{_tail_text(stderr_tail)}"
            )
        if pending_count > 0:
            valid = _flush_tile_batch_with_retry(
                arrays,
                valid,
                pending_arrays,
                pending_count,
                map_name=map_name,
            )
            pending_count = 0
        if tile_count == 0:
            skipped_zero_usable_maps += 1
            completed_maps.append(map_name)
            _write_resume_state(
                output_path,
                build=build,
                requested_maps=map_names,
                completed_maps=completed_maps,
                valid=valid,
                skipped_zero_usable_maps=skipped_zero_usable_maps,
                rejected_tile_count=rejected_tile_count,
                codec_name=codec_name,
                codec_level=codec_level,
                codec_shuffle=codec_shuffle,
                capacity=capacity,
            )
            print(
                f"    Warning: skipping map {map_name} because harvest produced zero usable V18 tiles. "
                f"Dropped missing-required blobs: {dropped_missing_required}. "
                f"Report: {rejected_tiles_report_path}",
                file=sys.stderr,
                flush=True,
            )
            continue

        if dropped_missing_required > 0:
            print(
                f"    Warning: dropped {dropped_missing_required} blobs for map {map_name} "
                f"because required dataset keys were missing. "
                f"Report: {rejected_tiles_report_path}",
                file=sys.stderr,
                flush=True,
            )

        print(
            f"    Map {map_name}: {tile_count} tiles streamed, placements={map_placements}, "
            f"raw_npz={map_blob_bytes / 1024 / 1024:.1f} MB, "
            f"dropped_missing_required={dropped_missing_required}",
            flush=True,
        )

        completed_maps.append(map_name)
        if index_rows:
            _write_index(index_rows, output_path)
        if all_placements:
            _write_placements(all_placements, output_path)
        if decoded_metadata_rows:
            _write_decoded_metadata(decoded_metadata_rows, output_path)
        _write_resume_state(
            output_path,
            build=build,
            requested_maps=map_names,
            completed_maps=completed_maps,
            valid=valid,
            skipped_zero_usable_maps=skipped_zero_usable_maps,
            rejected_tile_count=rejected_tile_count,
            codec_name=codec_name,
            codec_level=codec_level,
            codec_shuffle=codec_shuffle,
            capacity=capacity,
        )

        if limit is not None and valid >= limit:
            break

    for key in ALL_ARRAY_KEYS:
        arrays[key].resize((valid,) + SHAPES[key])

    if valid == 0:
        raise RuntimeError(
            "Harvest stream produced zero usable tiles across all requested maps."
        )

    if index_rows:
        _write_index(index_rows, output_path)

    if all_placements:
        _write_placements(all_placements, output_path)

    if decoded_metadata_rows:
        _write_decoded_metadata(decoded_metadata_rows, output_path)

    trace_sidecar.finalize(valid)
    trace_sidecar.close()
    store.close()
    _write_resume_state(
        output_path,
        build=build,
        requested_maps=map_names,
        completed_maps=completed_maps,
        valid=valid,
        skipped_zero_usable_maps=skipped_zero_usable_maps,
        rejected_tile_count=rejected_tile_count,
        codec_name=codec_name,
        codec_level=codec_level,
        codec_shuffle=codec_shuffle,
        capacity=valid,
        finalized=True,
    )

    total_bytes = _dir_size_bytes(output_path)
    liq_count = sum(1 for r in index_rows if r.get("has_liquid_mask", False))
    inst_count = sum(1 for r in index_rows if r.get("has_object_instance_mask", False))
    elapsed = time.perf_counter() - t0
    if rejected_tiles_report is not None:
        rejected_tiles_report.close()
    print(f"\nDone. {valid} tiles -> {output_path}")
    print(f"Size: {total_bytes / 1024 / 1024:.1f} MB")
    print(f"Liquid: {liq_count}/{valid}, Instance mask: {inst_count}/{valid}")
    print(f"Placements: {len(all_placements)} total")
    print(f"Skipped zero-usable maps: {skipped_zero_usable_maps}")
    print(f"Rejected missing-required tiles: {rejected_tile_count}")
    if rejected_tiles_report_path is not None:
        print(f"Rejected tiles report: {rejected_tiles_report_path}")
    metrics_path = _write_harvest_metrics(
        build=build,
        output_path=output_path,
        index_rows=index_rows,
        placements_total=len(all_placements),
        skipped_zero_usable_maps=skipped_zero_usable_maps,
        rejected_tile_count=rejected_tile_count,
        elapsed_seconds=elapsed,
    )
    print(f"Harvest metrics: {metrics_path}")
    print(f"Time: {elapsed:.0f}s ({valid / max(elapsed, 0.01):.1f} tiles/s)")


def cmd_stats(args: argparse.Namespace) -> None:
    builds = args.builds or [args.build]
    aggregate_total_rows = 0
    aggregate_unique_tiles: set[tuple[str, int, int]] = set()
    aggregate_per_build_unique: dict[str, int] = {}
    for build in builds:
        zarr_path = _DATASET_ROOT / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"SKIP {build}: no Zarr store at {zarr_path}")
            continue
        store_bytes = _dir_size_bytes(zarr_path)
        total_raw_array_bytes = 0
        total_array_disk_bytes = 0
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=r"Object at .* is not recognized as a component of a Zarr hierarchy\.",
                category=UserWarning,
            )
            store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
            root = zarr.open_group(store=store, mode="r")
            n = root["height_257"].shape[0]
            print(f"\n{build}: {n} tiles")
            for k in sorted(root.array_keys()):
                a = root[k]
                raw_bytes = int(np.prod(a.shape, dtype=np.int64) * np.dtype(a.dtype).itemsize)
                disk_bytes = _dir_size_bytes(zarr_path / k) if (zarr_path / k).exists() else 0
                total_raw_array_bytes += raw_bytes
                total_array_disk_bytes += disk_bytes
                print(
                    f"  {k}: shape={a.shape} dtype={a.dtype} "
                    f"raw={_format_bytes(raw_bytes)} disk={_format_bytes(disk_bytes)} "
                    f"ratio={_format_ratio(raw_bytes, disk_bytes)}"
                )
            store.close()

        print(
            f"  arrays total: raw={_format_bytes(total_raw_array_bytes)} "
            f"disk={_format_bytes(total_array_disk_bytes)} "
            f"ratio={_format_ratio(total_raw_array_bytes, total_array_disk_bytes)}"
        )
        print(
            f"  full store: disk={_format_bytes(store_bytes)} "
            f"savings_vs_raw_arrays={_format_bytes(max(total_raw_array_bytes - store_bytes, 0))}"
        )

        idx_path = zarr_path / "index.parquet"
        if idx_path.exists():
            table = pq.read_table(str(idx_path))
            idx_bytes = idx_path.stat().st_size
            print(
                f"  index.parquet: {table.num_rows} rows, {table.num_columns} cols, "
                f"disk={_format_bytes(idx_bytes)}"
            )
            for col in table.column_names:
                if col.startswith("has_"):
                    count_scalar = pc.sum(table.column(col))
                    count = 0 if count_scalar is None else int(count_scalar.as_py() or 0)
                    print(f"    {col}: {count}/{table.num_rows}")
            local_unique: set[tuple[str, int, int]] = set()
            if {"map", "tile_x", "tile_y"}.issubset(set(table.column_names)):
                for row in table.select(["map", "tile_x", "tile_y"]).to_pylist():
                    map_name = str(row.get("map") or "")
                    tx = int(row.get("tile_x") or 0)
                    ty = int(row.get("tile_y") or 0)
                    key = (map_name, tx, ty)
                    local_unique.add(key)
                    aggregate_unique_tiles.add(key)
                aggregate_total_rows += int(table.num_rows)
                aggregate_per_build_unique[build] = len(local_unique)
            if table.num_rows != n:
                print(
                    f"  WARNING: array length ({n}) does not match index rows ({table.num_rows}). "
                    f"Build may be incomplete or corrupted."
                )
        else:
            print("  WARNING: index.parquet missing. This store looks incomplete or failed before finalization.")

        metrics_path = zarr_path / "harvest_metrics.json"
        if metrics_path.exists():
            print(f"  harvest_metrics.json: {metrics_path}")
        signal_validation_path = zarr_path / "signal_validation.json"
        if signal_validation_path.exists():
            print(f"  signal_validation.json: {signal_validation_path}")

        pl_path = zarr_path / "placements.parquet"
        if pl_path.exists():
            pl_table = pq.read_table(str(pl_path))
            pl_bytes = pl_path.stat().st_size
            print(f"  placements.parquet: {pl_table.num_rows} placements, disk={_format_bytes(pl_bytes)}")
        partial_path = _DATASET_ROOT / f"{build}.zarr.partial"
        if partial_path.exists():
            print(f"  WARNING: staged partial output still exists at {partial_path}")

    if len(builds) > 1 and aggregate_total_rows > 0:
        unique_count = len(aggregate_unique_tiles)
        duplication_factor = aggregate_total_rows / max(unique_count, 1)
        print("\nCross-build overlap summary:")
        print(f"  total rows across builds: {aggregate_total_rows}")
        print(f"  unique (map,tile_x,tile_y): {unique_count}")
        print(f"  duplication factor: {duplication_factor:.2f}x")
        for build in sorted(aggregate_per_build_unique):
            print(f"  {build}: unique tile coords={aggregate_per_build_unique[build]}")


def cmd_validate_signals(args: argparse.Namespace) -> None:
    builds = args.builds or [args.build]
    strict = bool(args.strict)
    for build in builds:
        zarr_path = _DATASET_ROOT / f"{build}.zarr"
        if not zarr_path.exists():
            raise RuntimeError(f"Build store not found: {zarr_path}")
        report_path = _validate_build_signals(build=build, output_path=zarr_path, strict=strict)
        print(f"{build}: signal validation report -> {report_path}")
        metadata_report_path = _validate_decoded_metadata_table(build=build, output_path=zarr_path, strict=strict)
        print(f"{build}: decoded metadata validation report -> {metadata_report_path}")
        finalization_path = _write_finalization_state(
            zarr_path,
            build=build,
            finalized=True,
            signal_validation_path=report_path,
            decoded_metadata_validation_path=metadata_report_path,
            required_files=V18_REQUIRED_ARTIFACTS,
        )
        print(f"{build}: finalization report -> {finalization_path}")


def cmd_merge_builds(args: argparse.Namespace) -> None:
    _require_explicit_zarr_write(args, "merge-builds")

    builds = args.builds or sorted(
        d.stem.replace(".zarr", "")
        for d in _DATASET_ROOT.glob("*.zarr")
        if d.is_dir()
    )
    if not builds:
        raise RuntimeError("No source build stores found under output/datasets/v18.")

    output_name = args.output_name or "merged_all"
    output_path = _DATASET_ROOT / f"{output_name}.zarr"
    output_partial = _DATASET_ROOT / f"{output_name}.zarr.partial"
    dedupe_mode = str(args.dedupe_mode or "coords_height")
    if dedupe_mode not in {"none", "coords", "coords_height"}:
        raise RuntimeError("--dedupe-mode must be one of: none, coords, coords_height")

    if output_path.exists() and not args.rebuild_existing:
        raise RuntimeError(
            f"Output merged store already exists at {output_path}. Use --rebuild-existing to overwrite."
        )

    if output_partial.exists():
        shutil.rmtree(output_partial)
    if output_path.exists() and args.rebuild_existing:
        shutil.rmtree(output_path)

    source_handles: dict[str, tuple[zarr.storage.LocalStore, zarr.Group]] = {}
    source_index_rows: dict[str, list[dict]] = {}
    source_placements_rows: dict[str, list[dict]] = {}
    source_decoded_metadata_rows: dict[str, list[dict]] = {}
    t0 = time.perf_counter()

    for build in builds:
        zarr_path = _DATASET_ROOT / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"SKIP {build}: no store at {zarr_path}")
            continue
        store, root = _open_zarr_group_readonly(zarr_path)
        source_handles[build] = (store, root)
        idx_path = zarr_path / "index.parquet"
        if not idx_path.exists():
            store.close()
            raise RuntimeError(f"Source build {build} has no index.parquet.")
        source_index_rows[build] = _read_parquet_rows(idx_path)
        pl_path = zarr_path / "placements.parquet"
        source_placements_rows[build] = _read_parquet_rows(pl_path) if pl_path.exists() else []
        dm_path = zarr_path / "decoded_metadata.parquet"
        source_decoded_metadata_rows[build] = _read_parquet_rows(dm_path) if dm_path.exists() else []

    try:
        records: list[tuple[str, int, dict]] = []
        if dedupe_mode == "none":
            for build in builds:
                rows = source_index_rows.get(build, [])
                for row in rows:
                    records.append((build, int(row["tile_id"]), row))
        else:
            selected: dict[tuple, tuple[str, int, dict]] = {}
            ordered_keys: list[tuple] = []
            for build in builds:
                rows = source_index_rows.get(build, [])
                for row in rows:
                    map_name = str(row.get("map", ""))
                    tx = int(row.get("tile_x") or 0)
                    ty = int(row.get("tile_y") or 0)
                    if dedupe_mode == "coords":
                        key = (map_name, tx, ty)
                    else:
                        key = (
                            map_name,
                            tx,
                            ty,
                            round(float(row.get("height_mean", 0.0)), 4),
                            round(float(row.get("height_std", 0.0)), 4),
                        )
                    if key not in selected:
                        ordered_keys.append(key)
                    # Keep latest source in requested build order.
                    selected[key] = (build, int(row["tile_id"]), row)
            for key in ordered_keys:
                records.append(selected[key])

        if not records:
            raise RuntimeError("No records selected for merge.")

        capacity = len(records)
        compressors = None
        if DEFAULT_CODEC != "none":
            compressors = [
                zarr.codecs.BloscCodec(
                    cname=DEFAULT_CODEC,
                    clevel=DEFAULT_CLEVEL,
                    shuffle=DEFAULT_SHUFFLE,
                )
            ]
        out_store = zarr.storage.LocalStore(str(output_partial), read_only=False)
        out_root = zarr.open_group(store=out_store, mode="w")
        out_arrays: dict[str, zarr.Array] = {}
        for key in ALL_ARRAY_KEYS:
            chunks = CHUNK_SIZES.get(key, (64,) + SHAPES[key])
            out_arrays[key] = out_root.create_array(
                key,
                shape=(capacity,) + SHAPES[key],
                chunks=chunks,
                dtype=DTYPES[key],
                compressors=compressors,
                fill_value=FILL_VALUES.get(key, 0),
            )

        merged_index_rows: list[dict] = []
        source_to_merged_tile: dict[tuple[str, int], int] = {}
        batch_size = int(args.batch_size or 64)
        pending_arrays: dict[str, list[np.ndarray]] = {key: [] for key in ALL_ARRAY_KEYS}
        write_pos = 0

        for build, src_tile_id, row in records:
            _, src_root = source_handles[build]
            for key in ALL_ARRAY_KEYS:
                pending_arrays[key].append(src_root[key][src_tile_id].astype(DTYPES[key], copy=False))

            merged_row = dict(row)
            merged_row["tile_id"] = write_pos
            merged_row["build"] = str(row.get("build") or build)
            merged_row["source_build"] = build
            merged_row["source_tile_id"] = src_tile_id
            merged_index_rows.append(merged_row)
            source_to_merged_tile[(build, src_tile_id)] = write_pos
            write_pos += 1

            if len(pending_arrays["height_257"]) >= batch_size:
                start = write_pos - len(pending_arrays["height_257"])
                for key in ALL_ARRAY_KEYS:
                    out_arrays[key][start:write_pos] = np.stack(pending_arrays[key], axis=0)
                    pending_arrays[key].clear()

        remaining = len(pending_arrays["height_257"])
        if remaining > 0:
            start = write_pos - remaining
            for key in ALL_ARRAY_KEYS:
                out_arrays[key][start:write_pos] = np.stack(pending_arrays[key], axis=0)
                pending_arrays[key].clear()

        _write_index(merged_index_rows, output_partial)

        merged_placements: list[dict] = []
        for build in builds:
            for row in source_placements_rows.get(build, []):
                old_tile_id = int(row.get("tile_id", -1))
                new_tile_id = source_to_merged_tile.get((build, old_tile_id))
                if new_tile_id is None:
                    continue
                new_row = dict(row)
                new_row["tile_id"] = int(new_tile_id)
                new_row["source_build"] = build
                new_row["source_tile_id"] = old_tile_id
                merged_placements.append(new_row)
        if merged_placements:
            _write_placements(merged_placements, output_partial)

        merged_decoded_metadata: list[dict[str, object]] = []
        for build in builds:
            metadata_rows = source_decoded_metadata_rows.get(build, [])
            if metadata_rows:
                for row in metadata_rows:
                    old_tile_id = int(row.get("tile_id", -1))
                    new_tile_id = source_to_merged_tile.get((build, old_tile_id))
                    if new_tile_id is None:
                        continue
                    new_row = dict(row)
                    new_row["tile_id"] = int(new_tile_id)
                    new_row["build"] = str(new_row.get("build") or build)
                    merged_decoded_metadata.append(new_row)
                continue

            # Fallback for older stores: preserve complete tile coverage.
            for src_build, src_tile_id, row in records:
                if src_build != build:
                    continue
                new_tile_id = source_to_merged_tile.get((build, src_tile_id))
                if new_tile_id is None:
                    continue
                merged_decoded_metadata.append(
                    {
                        "tile_id": int(new_tile_id),
                        "build": str(row.get("build") or build),
                        "map": str(row.get("map", "")),
                        "tile_x": int(row.get("tile_x", 0)),
                        "tile_y": int(row.get("tile_y", 0)),
                        "tile_name": "",
                        "source_adt_path": "",
                        "source_wdt_path": "",
                        "raw_chunks_count": 0,
                        "decoded_metadata_json": "{}",
                        "decoded_metadata_keys_json": "[]",
                    }
                )

        if merged_decoded_metadata:
            _write_decoded_metadata(merged_decoded_metadata, output_partial)

        elapsed = time.perf_counter() - t0
        metrics_path = _write_harvest_metrics(
            build=output_name,
            output_path=output_partial,
            index_rows=merged_index_rows,
            placements_total=len(merged_placements),
            skipped_zero_usable_maps=0,
            rejected_tile_count=0,
            elapsed_seconds=elapsed,
        )
        merge_manifest = {
            "output_name": output_name,
            "output_path": str(output_path),
            "source_builds": builds,
            "dedupe_mode": dedupe_mode,
            "selected_tiles": len(records),
            "source_tiles_total": int(sum(len(source_index_rows.get(b, [])) for b in builds)),
            "batch_size": batch_size,
            "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "harvest_metrics_path": str(metrics_path),
        }
        (output_partial / "merge_manifest.json").write_text(json.dumps(merge_manifest, indent=2), encoding="utf-8")
        _write_resume_state(
            output_partial,
            build=output_name,
            requested_maps=sorted({str(r.get("map", "")) for r in merged_index_rows}),
            completed_maps=sorted({str(r.get("map", "")) for r in merged_index_rows}),
            valid=len(merged_index_rows),
            skipped_zero_usable_maps=0,
            rejected_tile_count=0,
            codec_name=DEFAULT_CODEC,
            codec_level=DEFAULT_CLEVEL,
            codec_shuffle=DEFAULT_SHUFFLE,
            capacity=len(merged_index_rows),
            finalized=True,
        )
        out_store.close()

        # Safety gate: never publish an incomplete merged store.
        missing_paths = _collect_missing_store_components(
            output_partial,
            required_files=V18_REQUIRED_ARTIFACTS_MERGED,
        )
        if missing_paths:
            raise RuntimeError(
                "Refusing to publish incomplete merged store. Missing: "
                + ", ".join(missing_paths)
            )

        os_replace_target = output_path
        output_partial.replace(os_replace_target)
        print(f"Merged store: {output_path}")
        print(f"Tiles: {len(merged_index_rows)} (from {sum(len(source_index_rows.get(b, [])) for b in builds)} source rows)")
        print(f"Placements: {len(merged_placements)}")
        print(f"Dedupe mode: {dedupe_mode}")
        print(f"Metrics: {output_path / 'harvest_metrics.json'}")
        metadata_validation_path = _validate_decoded_metadata_table(
            build=output_name,
            output_path=output_path,
            strict=True,
        )
        print(f"Decoded metadata validation report: {metadata_validation_path}")
        finalization_path = _write_finalization_state(
            output_path,
            build=output_name,
            finalized=True,
            signal_validation_path=None,
            decoded_metadata_validation_path=metadata_validation_path,
            required_files=V18_REQUIRED_ARTIFACTS_MERGED,
        )
        print(f"Finalization report: {finalization_path}")
    finally:
        for store, _root in source_handles.values():
            store.close()


def cmd_repair_index(args: argparse.Namespace) -> None:
    builds = args.builds or [args.build]
    harvest_tool = _find_harvest_tool()
    map_workers = max(1, int(args.map_workers))

    for build in builds:
        output_path = _DATASET_ROOT / f"{build}.zarr"
        if not output_path.exists():
            print(f"SKIP {build}: no final store at {output_path}")
            continue

        client_root = _find_client_root(build)
        if client_root is None:
            raise RuntimeError(f"Could not find staged client root for build {build}.")

        idx_path = output_path / "index.parquet"
        if not idx_path.exists():
            raise RuntimeError(f"Build {build} has no index.parquet to repair.")

        if not args.no_backup:
            backup_path = output_path / "index.parquet.bak"
            if not backup_path.exists():
                shutil.copy2(idx_path, backup_path)
                print(f"Backed up {idx_path} -> {backup_path}")

        index_rows = _read_index_rows(output_path)
        ordered_maps = _ordered_maps_from_index_rows(index_rows)
        build_version = build.replace("_", ".")

        print(f"Repairing index coordinates for {build}")
        print(f"Client: {client_root}")
        print(f"Maps: {ordered_maps}")

        if ordered_maps and all(_is_placeholder_map_name(m) for m in ordered_maps):
            discovered_maps = _discover_maps_for_build(harvest_tool, client_root)
            print("Index map labels are placeholder-only; attempting full map relabel from fresh stream order.")
            print(f"Discovered maps: {discovered_maps}")
            streamed_by_map = _collect_map_rows_parallel(
                discovered_maps,
                lambda map_name: _stream_valid_tile_metadata(harvest_tool, client_root, map_name, build_version),
                map_workers=map_workers,
                label=f"repair-index {build}",
            )
            streamed_all: list[dict[str, object]] = []
            for map_name in discovered_maps:
                streamed_all.extend(streamed_by_map[map_name])

            if len(streamed_all) != len(index_rows):
                raise RuntimeError(
                    f"Repair relabel count mismatch for {build}: "
                    f"index has {len(index_rows)} rows, discovered stream produced {len(streamed_all)} valid tiles."
                )

            for idx, streamed in enumerate(streamed_all):
                index_rows[idx]["map"] = str(streamed["map"])
                index_rows[idx]["tile_x"] = int(streamed["tile_x"])
                index_rows[idx]["tile_y"] = int(streamed["tile_y"])

            _write_index(index_rows, output_path)
            print(f"Wrote repaired index: {idx_path}")
            continue

        streamed_by_map = _collect_map_rows_parallel(
            ordered_maps,
            lambda map_name: _stream_valid_tile_metadata(harvest_tool, client_root, map_name, build_version),
            map_workers=map_workers,
            label=f"repair-index {build}",
        )

        for map_name in ordered_maps:
            row_indices = [i for i, row in enumerate(index_rows) if str(row.get("map")) == map_name]
            streamed_rows = streamed_by_map[map_name]

            if len(streamed_rows) != len(row_indices):
                raise RuntimeError(
                    f"Coordinate repair count mismatch for {build}/{map_name}: "
                    f"index has {len(row_indices)} rows, stream produced {len(streamed_rows)} valid tiles."
                )

            for row_idx, streamed in zip(row_indices, streamed_rows):
                index_rows[row_idx]["map"] = str(streamed["map"])
                index_rows[row_idx]["tile_x"] = int(streamed["tile_x"])
                index_rows[row_idx]["tile_y"] = int(streamed["tile_y"])

            print(f"  repaired {map_name}: {len(row_indices)} rows")

        _write_index(index_rows, output_path)
        print(f"Wrote repaired index: {idx_path}")


def cmd_patch_liquids(args: argparse.Namespace) -> None:
    _require_explicit_zarr_write(args, "patch-liquids")

    builds = args.builds or [args.build]
    harvest_tool = _find_harvest_tool()
    batch_size = max(1, int(args.batch_size))
    map_workers = max(1, int(args.map_workers))

    for build in builds:
        output_path = _DATASET_ROOT / f"{build}.zarr"
        if not output_path.exists():
            print(f"SKIP {build}: no final store at {output_path}")
            continue

        client_root = _find_client_root(build)
        if client_root is None:
            raise RuntimeError(f"Could not find staged client root for build {build}.")

        idx_path = output_path / "index.parquet"
        if not idx_path.exists():
            raise RuntimeError(f"Build {build} has no index.parquet to patch.")

        if not args.no_backup:
            backup_path = output_path / "index.parquet.bak.liquids"
            if not backup_path.exists():
                shutil.copy2(idx_path, backup_path)
                print(f"Backed up {idx_path} -> {backup_path}")

        index_rows = _read_index_rows(output_path)
        build_version = build.replace("_", ".")
        old_counts = _count_signal_coverage(index_rows)

        print(f"Patching liquid supervision for {build}")
        print(f"Client: {client_root}")
        print(f"Store: {output_path}")

        patch_rows: list[dict[str, object]] = [None] * len(index_rows)  # type: ignore[list-item]
        ordered_maps = _ordered_maps_from_index_rows(index_rows)

        if ordered_maps and all(_is_placeholder_map_name(m) for m in ordered_maps):
            discovered_maps = _discover_maps_for_build(harvest_tool, client_root)
            print("Index map labels are placeholder-only; patching by full discovered stream order.")
            streamed_by_map = _collect_map_rows_parallel(
                discovered_maps,
                lambda map_name: _stream_valid_tile_liquid_rows(harvest_tool, client_root, map_name, build_version),
                map_workers=map_workers,
                label=f"patch-liquids {build}",
            )
            streamed_all: list[dict[str, object]] = []
            for map_name in discovered_maps:
                streamed_all.extend(streamed_by_map[map_name])

            if len(streamed_all) != len(index_rows):
                raise RuntimeError(
                    f"Liquid patch count mismatch for {build}: "
                    f"index has {len(index_rows)} rows, stream produced {len(streamed_all)} valid tiles."
                )
            for i, streamed in enumerate(streamed_all):
                patch_rows[i] = streamed
        else:
            streamed_by_map = _collect_map_rows_parallel(
                ordered_maps,
                lambda map_name: _stream_valid_tile_liquid_rows(harvest_tool, client_root, map_name, build_version),
                map_workers=map_workers,
                label=f"patch-liquids {build}",
            )
            for map_name in ordered_maps:
                row_indices = [i for i, row in enumerate(index_rows) if str(row.get("map")) == map_name]
                streamed_rows = streamed_by_map[map_name]
                if len(streamed_rows) != len(row_indices):
                    raise RuntimeError(
                        f"Liquid patch count mismatch for {build}/{map_name}: "
                        f"index has {len(row_indices)} rows, stream produced {len(streamed_rows)} valid tiles."
                    )
                for row_idx, streamed in zip(row_indices, streamed_rows):
                    patch_rows[row_idx] = streamed
                print(f"  patched stream rows for {map_name}: {len(row_indices)}")

        if any(row is None for row in patch_rows):
            missing = sum(1 for row in patch_rows if row is None)
            raise RuntimeError(f"Internal patch error for {build}: {missing} rows were not assigned streamed liquid data.")

        store = zarr.storage.LocalStore(str(output_path), read_only=False)
        root = zarr.open_group(store=store, mode="a")
        try:
            if "liquid_mask" not in root or "liquid_height" not in root:
                raise RuntimeError(f"Build {build} store missing liquid arrays.")
            arr_mask = root["liquid_mask"]
            arr_height = root["liquid_height"]
            if arr_mask.shape[0] != len(index_rows) or arr_height.shape[0] != len(index_rows):
                raise RuntimeError(
                    f"Array/index length mismatch for {build}: "
                    f"liquid_mask={arr_mask.shape[0]} liquid_height={arr_height.shape[0]} index={len(index_rows)}"
                )

            tile_ids = np.asarray([int(row.get("tile_id", i)) for i, row in enumerate(index_rows)], dtype=np.int64)
            order = np.argsort(tile_ids)
            sorted_ids = tile_ids[order]
            if len(sorted_ids) > 1 and np.any(np.diff(sorted_ids) != 1):
                raise RuntimeError(f"Build {build} tile_id sequence is non-contiguous; aborting in-place liquid patch.")

            for start in range(0, len(order), batch_size):
                end = min(start + batch_size, len(order))
                chunk_order = order[start:end]
                chunk_ids = sorted_ids[start:end]
                masks = np.stack([patch_rows[i]["liquid_mask"] for i in chunk_order], axis=0).astype(np.float32, copy=False)
                heights = np.stack([patch_rows[i]["liquid_height"] for i in chunk_order], axis=0).astype(np.float32, copy=False)
                arr_mask[int(chunk_ids[0]): int(chunk_ids[-1]) + 1] = masks
                arr_height[int(chunk_ids[0]): int(chunk_ids[-1]) + 1] = heights
        finally:
            store.close()

        for i, row in enumerate(index_rows):
            patch = patch_rows[i]
            row["has_liquid_mask"] = bool(patch["has_liquid_mask"])
            row["has_liquid_height"] = bool(patch["has_liquid_height"])
            for source_name in LIQUID_SOURCE_KEYS:
                row[f"has_liquid_source_{source_name}"] = bool(patch[f"has_liquid_source_{source_name}"])

        _write_index(index_rows, output_path)
        new_counts = _count_signal_coverage(index_rows)

        patch_report = {
            "build": build,
            "patched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tile_count": len(index_rows),
            "batch_size": batch_size,
            "before": {k: int(v) for k, v in sorted(old_counts.items()) if k.startswith("has_liquid")},
            "after": {k: int(v) for k, v in sorted(new_counts.items()) if k.startswith("has_liquid")},
            "client_root": str(client_root),
        }
        report_path = output_path / "liquid_patch_report.json"
        report_path.write_text(json.dumps(patch_report, indent=2), encoding="utf-8")
        print(f"Wrote patched liquid arrays + index flags for {build}")
        print(f"Liquid patch report: {report_path}")

        if args.signal_validation:
            validation_path = _validate_build_signals(build=build, output_path=output_path, strict=args.signal_validation_strict)
            print(f"Signal validation report: {validation_path}")


def cmd_patch_objects(args: argparse.Namespace) -> None:
    _require_explicit_zarr_write(args, "patch-objects")

    builds = args.builds or [args.build]
    harvest_tool = _find_harvest_tool()
    batch_size = max(1, int(args.batch_size))
    map_workers = max(1, int(args.map_workers))

    for build in builds:
        output_path = _DATASET_ROOT / f"{build}.zarr"
        if not output_path.exists():
            print(f"SKIP {build}: no final store at {output_path}")
            continue

        client_root = _find_client_root(build)
        if client_root is None:
            raise RuntimeError(f"Could not find staged client root for build {build}.")

        idx_path = output_path / "index.parquet"
        if not idx_path.exists():
            raise RuntimeError(f"Build {build} has no index.parquet to patch.")

        if not args.no_backup:
            backup_path = output_path / "index.parquet.bak.objects"
            if not backup_path.exists():
                shutil.copy2(idx_path, backup_path)
                print(f"Backed up {idx_path} -> {backup_path}")

        index_rows = _read_index_rows(output_path)
        build_version = build.replace("_", ".")
        old_counts = _count_signal_coverage(index_rows)

        print(f"Patching object supervision for {build}")
        print(f"Client: {client_root}")
        print(f"Store: {output_path}")

        patch_rows: list[dict[str, object]] = [None] * len(index_rows)  # type: ignore[list-item]
        ordered_maps = _ordered_maps_from_index_rows(index_rows)

        if ordered_maps and all(_is_placeholder_map_name(m) for m in ordered_maps):
            discovered_maps = _discover_maps_for_build(harvest_tool, client_root)
            print("Index map labels are placeholder-only; patching by full discovered stream order.")
            streamed_by_map = _collect_map_rows_parallel(
                discovered_maps,
                lambda map_name: _stream_valid_tile_object_rows(harvest_tool, client_root, map_name, build_version),
                map_workers=map_workers,
                label=f"patch-objects {build}",
            )
            streamed_all: list[dict[str, object]] = []
            for map_name in discovered_maps:
                streamed_all.extend(streamed_by_map[map_name])

            if len(streamed_all) != len(index_rows):
                raise RuntimeError(
                    f"Object patch count mismatch for {build}: "
                    f"index has {len(index_rows)} rows, stream produced {len(streamed_all)} valid tiles."
                )
            for i, streamed in enumerate(streamed_all):
                patch_rows[i] = streamed
        else:
            streamed_by_map = _collect_map_rows_parallel(
                ordered_maps,
                lambda map_name: _stream_valid_tile_object_rows(harvest_tool, client_root, map_name, build_version),
                map_workers=map_workers,
                label=f"patch-objects {build}",
            )
            for map_name in ordered_maps:
                row_indices = [i for i, row in enumerate(index_rows) if str(row.get("map")) == map_name]
                streamed_rows = streamed_by_map[map_name]
                if len(streamed_rows) != len(row_indices):
                    raise RuntimeError(
                        f"Object patch count mismatch for {build}/{map_name}: "
                        f"index has {len(row_indices)} rows, stream produced {len(streamed_rows)} valid tiles."
                    )
                for row_idx, streamed in zip(row_indices, streamed_rows):
                    patch_rows[row_idx] = streamed
                print(f"  patched stream rows for {map_name}: {len(row_indices)}")

        if any(row is None for row in patch_rows):
            missing = sum(1 for row in patch_rows if row is None)
            raise RuntimeError(f"Internal patch error for {build}: {missing} rows were not assigned streamed object data.")

        store = zarr.storage.LocalStore(str(output_path), read_only=False)
        root = zarr.open_group(store=store, mode="a")
        try:
            required_arrays = ["object_mask", "object_precise_mask", "object_instance_mask"]
            for key in required_arrays:
                if key not in root:
                    raise RuntimeError(f"Build {build} store missing {key} array.")

            arr_mask = root["object_mask"]
            arr_precise = root["object_precise_mask"]
            arr_instance = root["object_instance_mask"]
            if (
                arr_mask.shape[0] != len(index_rows)
                or arr_precise.shape[0] != len(index_rows)
                or arr_instance.shape[0] != len(index_rows)
            ):
                raise RuntimeError(
                    f"Array/index length mismatch for {build}: "
                    f"object_mask={arr_mask.shape[0]} object_precise_mask={arr_precise.shape[0]} "
                    f"object_instance_mask={arr_instance.shape[0]} index={len(index_rows)}"
                )

            tile_ids = np.asarray([int(row.get("tile_id", i)) for i, row in enumerate(index_rows)], dtype=np.int64)
            order = np.argsort(tile_ids)
            sorted_ids = tile_ids[order]
            if len(sorted_ids) > 1 and np.any(np.diff(sorted_ids) != 1):
                raise RuntimeError(f"Build {build} tile_id sequence is non-contiguous; aborting in-place object patch.")

            for start in range(0, len(order), batch_size):
                end = min(start + batch_size, len(order))
                chunk_order = order[start:end]
                chunk_ids = sorted_ids[start:end]
                masks = np.stack([patch_rows[i]["object_mask"] for i in chunk_order], axis=0).astype(np.bool_, copy=False)
                precise = np.stack([patch_rows[i]["object_precise_mask"] for i in chunk_order], axis=0).astype(np.float32, copy=False)
                instances = np.stack([patch_rows[i]["object_instance_mask"] for i in chunk_order], axis=0).astype(np.int32, copy=False)
                arr_mask[int(chunk_ids[0]): int(chunk_ids[-1]) + 1] = masks
                arr_precise[int(chunk_ids[0]): int(chunk_ids[-1]) + 1] = precise
                arr_instance[int(chunk_ids[0]): int(chunk_ids[-1]) + 1] = instances
        finally:
            store.close()

        for i, row in enumerate(index_rows):
            patch = patch_rows[i]
            row["has_object_mask"] = bool(patch["has_object_mask"])
            row["has_object_precise_mask"] = bool(patch["has_object_precise_mask"])
            row["has_object_instance_mask"] = bool(patch["has_object_instance_mask"])

        _write_index(index_rows, output_path)
        new_counts = _count_signal_coverage(index_rows)

        patch_report = {
            "build": build,
            "patched_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "tile_count": len(index_rows),
            "batch_size": batch_size,
            "before": {k: int(v) for k, v in sorted(old_counts.items()) if k in {f"has_{name}" for name in OBJECT_SIGNAL_KEYS}},
            "after": {k: int(v) for k, v in sorted(new_counts.items()) if k in {f"has_{name}" for name in OBJECT_SIGNAL_KEYS}},
            "client_root": str(client_root),
        }
        report_path = output_path / "object_patch_report.json"
        report_path.write_text(json.dumps(patch_report, indent=2), encoding="utf-8")
        print(f"Wrote patched object arrays + index flags for {build}")
        print(f"Object patch report: {report_path}")

        if args.signal_validation:
            validation_path = _validate_build_signals(build=build, output_path=output_path, strict=args.signal_validation_strict)
            print(f"Signal validation report: {validation_path}")


def cmd_generate_viewer_stubs(args: argparse.Namespace) -> None:
    """Generate per-tile JSON stubs + capture ledger from V18 index for renderer-truth capture."""
    builds = args.builds or [args.build]
    if not builds[0]:
        print("ERROR: specify --build or --builds")
        sys.exit(1)
    output_root = _DATASET_ROOT

    capture_root = Path(args.capture_root) if args.capture_root else _PROJECT_ROOT.parent / "output" / "tmp" / "mdxviewer_validation_smoke"
    manifest_keys: set[tuple[str, int]] | None = None
    if args.curation_manifest:
        manifest_keys = _load_manifest_keep_keys(Path(args.curation_manifest))
        print(f"Using curation manifest: {args.curation_manifest} (keep keys={len(manifest_keys)})")

    total = 0
    for build in builds:
        zarr_path = _DATASET_ROOT / f"{build}.zarr"
        if not zarr_path.exists():
            print(f"  SKIP: no store at {zarr_path}")
            continue

        index_path = zarr_path / "index.parquet"
        if not index_path.exists():
            print(f"  SKIP: no index.parquet at {index_path}")
            continue

        table = pq.read_table(str(index_path))
        pose_by_tile_id = _build_tile_pose_metadata_from_placements(zarr_path)
        build_capture_dir = capture_root / build
        dataset_dir = build_capture_dir / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)

        written = 0
        skipped_manifest = 0
        requested_tiles: list[dict[str, int | str]] = []
        for i in range(table.num_rows):
            tile_id = int(table.column("tile_id")[i].as_py()) if "tile_id" in table.column_names else i
            if manifest_keys is not None and (build, tile_id) not in manifest_keys:
                skipped_manifest += 1
                continue
            map_name = str(table.column("map")[i].as_py())
            tile_x = int(table.column("tile_x")[i].as_py())
            tile_y = int(table.column("tile_y")[i].as_py())
            if not map_name:
                continue
            tile_name = f"{map_name}_{tile_x}_{tile_y}"
            json_path = dataset_dir / f"{tile_name}.json"
            stub = {
                "image": f"images/{tile_name}.png",
                "depth": None,
                "terrain_data": {
                    "adt_tile": tile_name,
                    "heights": [],
                    "chunk_positions": None,
                    "holes": None,
                    "heightmap": None,
                    "heightmap_local": None,
                    "heightmap_global": None,
                    "normalmap": None,
                    "mccv_map": None,
                    "shadow_maps": None,
                    "shadow_bits": None,
                    "shadow_analysis": None,
                    "alpha_masks": None,
                    "alpha_atlas": None,
                    "liquid_mask": None,
                    "liquid_height": None,
                    "liquid_min": 0.0,
                    "liquid_max": 0.0,
                    "no_liquid_minimap": None,
                    "no_mccv_minimap": None,
                    "object_visibility_mask": None,
                    "pm4_mask": None,
                    "no_object_minimap": None,
                    "terrain_only_minimap": None,
                    "holes_mask": None,
                    "area_id_map": None,
                    "chunk_flags_map": None,
                    "liquid_type_map": None,
                    "dominant_effect_id_map": None,
                    "textures": [],
                    "chunk_layers": None,
                    "liquids": None,
                    "objects": [],
                    "wdl_heights": None,
                    "height_min": 0.0,
                    "height_max": 0.0,
                    "height_global_min": 0.0,
                    "height_global_max": 0.0,
                    "is_interleaved": False,
                },
            }
            json_path.write_text(json.dumps(stub, indent=2), encoding="utf-8")
            written += 1
            requested_tiles.append({
                "build": build,
                "tile_id": tile_id,
                "map": map_name,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "tile_name": tile_name,
                "status": "pending_capture",
                **pose_by_tile_id.get(tile_id, {}),
            })

        ledger = {
            "build": build,
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "manifest_path": str(args.curation_manifest) if args.curation_manifest else None,
            "requested_tile_count": written,
            "skipped_by_manifest": skipped_manifest,
            "tiles": requested_tiles,
        }
        ledger_path = build_capture_dir / "manifest_capture_ledger.json"
        ledger_path.write_text(json.dumps(ledger, indent=2), encoding="utf-8")

        print(f"  {build}: wrote {written} stubs to {dataset_dir} (skipped_by_manifest={skipped_manifest})")
        print(f"  {build}: ledger {ledger_path}")
        total += written

    print(f"\nTotal: {total} stubs across {len(builds)} build(s)")
    print(f"Capture root: {capture_root}")
    print("Next: run WowViewer.Tool.ValidationCapture capture-batch using each build ledger,")
    print("      or run legacy MdxViewer batch flows if you need compatibility comparison.")


def cmd_patch_renderer_truth(args: argparse.Namespace) -> None:
    """Patch renderer-truth PNGs into V18 Zarr stores."""
    builds = args.builds or [args.build]
    if not builds[0]:
        print("ERROR: specify --build or --builds")
        sys.exit(1)
    capture_root = Path(args.capture_root) if args.capture_root else _PROJECT_ROOT.parent / "output" / "tmp" / "mdxviewer_validation_smoke"
    patch_script = Path(__file__).parent / "patch_v16_renderer_truth.py"

    if not patch_script.exists():
        print(f"ERROR: patch script not found at {patch_script}")
        sys.exit(1)

    for build in builds:
        capture_dir = capture_root / build
        if not capture_dir.exists():
            print(f"  SKIP: no capture directory at {capture_dir}")
            continue
        print(f"  Patching {build} from {capture_dir}")
        report_path = _apply_renderer_truth_patch(
            build=build,
            store_path=_DATASET_ROOT / f"{build}.zarr",
            capture_dir=capture_dir,
            curation_manifest=Path(args.curation_manifest) if args.curation_manifest else None,
            no_backup=bool(args.no_backup),
        )
        print(f"  OK: {build} patched -> {report_path}")


def cmd_clear_renderer_truth(args: argparse.Namespace) -> None:
    """Clear renderer-truth signals from V18 Zarr stores when the source is not trusted."""
    builds = args.builds or [args.build]
    if not builds[0]:
        print("ERROR: specify --build or --builds")
        sys.exit(1)

    for build in builds:
        report_path = _clear_renderer_truth_signals(
            build=build,
            store_path=_DATASET_ROOT / f"{build}.zarr",
            no_backup=bool(args.no_backup),
            reason=str(args.reason).strip(),
        )
        print(f"  OK: {build} cleared -> {report_path}")


def cmd_capture_renderer_truth(args: argparse.Namespace) -> None:
    """Run wow-viewer validation capture-batch from generated ledgers."""
    builds = args.builds or [args.build]
    if not builds[0]:
        print("ERROR: specify --build or --builds")
        sys.exit(1)

    capture_root = Path(args.capture_root) if args.capture_root else _PROJECT_ROOT.parent / "output" / "tmp" / "mdxviewer_validation_smoke"
    validation_tool = _find_validation_capture_tool()
    print(f"Validation capture tool: {validation_tool}")
    print(f"Capture root: {capture_root}")

    requested_modes = [
        flag
        for enabled, flag in [
            (bool(args.dry_run), "--dry-run"),
            (bool(args.real_scene_dry_run), "--real-scene-dry-run"),
            (bool(args.renderer), "--renderer"),
            (bool(args.gpu_viewer_style), "--gpu-viewer-style"),
            (bool(args.native_renderer), "--native-renderer"),
            (bool(args.stub_scene), "--stub-scene"),
        ]
        if enabled
    ]
    if len(requested_modes) > 1:
        raise SystemExit(
            "capture-renderer-truth accepts exactly one run mode: "
            "--dry-run OR --real-scene-dry-run OR --renderer OR --native-renderer OR --stub-scene"
        )
    mode_flags = requested_modes or ["--dry-run"]

    total_groups = 0
    failures = 0
    for build in builds:
        build_capture_dir = capture_root / build
        ledger_path = build_capture_dir / "manifest_capture_ledger.json"
        if not ledger_path.exists():
            print(f"  SKIP {build}: no ledger at {ledger_path}")
            continue

        client_root = _find_client_root(build)
        if client_root is None:
            print(f"  SKIP {build}: no staged client root found at {_CLIENT_ROOTS / build}")
            continue

        try:
            ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
        except Exception as ex:
            print(f"  ERROR {build}: failed to parse ledger {ledger_path}: {ex}")
            failures += 1
            continue

        tiles = ledger.get("tiles", [])
        map_groups: dict[str, list[dict]] = {}
        for row in tiles:
            if not isinstance(row, dict):
                continue
            if str(row.get("status", "")).lower() == "captured_complete":
                continue
            map_name = str(row.get("map", "")).strip()
            if not map_name:
                continue
            map_groups.setdefault(map_name, []).append(row)

        if not map_groups:
            print(f"  {build}: ledger has no pending map groups (all captured_complete or no valid rows)")
            continue

        print(f"  {build}: map groups={len(map_groups)} client={client_root}")
        for map_name in sorted(map_groups):
            map_rows = map_groups[map_name]
            map_input = f"World\\Maps\\{map_name}\\{map_name}.wdt"
            map_ledger = {
                "build": build,
                "generated_at": ledger.get("generated_at"),
                "manifest_path": ledger.get("manifest_path"),
                "requested_tile_count": len(map_rows),
                "tiles": map_rows,
            }

            with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as tmp:
                json.dump(map_ledger, tmp, indent=2)
                tmp_path = Path(tmp.name)

            try:
                variants_flag = args.variants or "objects-only"
                cmd = [
                    str(validation_tool),
                    "capture-batch",
                    "--client-root", str(client_root),
                    "--map-input", map_input,
                    "--map-name", map_name,
                    "--dataset-root", str(build_capture_dir),
                    "--output-root", str(build_capture_dir),
                    "--ledger-path", str(tmp_path),
                    "--build", build.replace("_", "."),
                    "--resolution", str(args.resolution),
                    "--variants", variants_flag,
                    *mode_flags,
                ]
                print(f"    {build}/{map_name}: running capture-batch ({len(map_rows)} pending tiles)")
                result = subprocess.run(cmd, capture_output=False)
                total_groups += 1
                if result.returncode != 0:
                    failures += 1
                    print(f"    ERROR {build}/{map_name}: capture-batch exited {result.returncode}")
                else:
                    print(f"    OK {build}/{map_name}: capture-batch completed")
            finally:
                try:
                    tmp_path.unlink(missing_ok=True)
                except Exception:
                    pass

    print(f"Capture-batch groups attempted: {total_groups}")
    if failures:
        print(f"Capture-batch failures: {failures}")
        sys.exit(1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build V18 consolidated Zarr dataset")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--build", type=str, help="Single build key (e.g. 3_3_5_12340)")
    common.add_argument("--builds", nargs="+", help="Multiple build keys")

    build_p = sub.add_parser("build", parents=[common])
    build_p.add_argument("--limit", type=int, default=None, help="Max tiles to extract")
    build_p.add_argument("--maps", nargs="+", default=None, help="Specific maps to extract")
    build_p.add_argument("--resume", action="store_true", help="Resume from <build>.zarr.partial if a compatible resume state exists")
    build_p.add_argument("--rebuild-existing", action="store_true", help="Rebuild even if a final <build>.zarr already looks complete")
    build_p.add_argument("--allow-zarr-write", action="store_true", help="Required confirmation flag before creating or replacing any V18 Zarr store")
    build_p.add_argument("--tile-workers", type=int, default=DEFAULT_TILE_WORKERS, help="harvest-stream workers per map during build")
    build_p.add_argument("--codec", choices=["none", "lz4", "zstd"], default=DEFAULT_CODEC, help="Compression codec (none disables compression)")
    build_p.add_argument("--clevel", type=int, default=DEFAULT_CLEVEL, help="Blosc compression level")
    build_p.add_argument("--shuffle", choices=["noshuffle", "shuffle", "bitshuffle"], default=DEFAULT_SHUFFLE, help="Blosc shuffle mode")
    build_p.add_argument("--capture-root", type=str, default=None, help="Optional root containing per-build renderer-truth captures to promote during the canonical V18 build")
    build_p.add_argument("--curation-manifest", type=str, default=None, help="Optional curation manifest (dir/file) limiting promoted-signal scope during the V18 build")
    build_p.add_argument("--no-backup", action="store_true", help="Skip creating index.parquet backups before promoted in-place signal updates during the V18 build")
    build_p.add_argument("--experimental-renderer-truth-promotion", action="store_true", help="Explicitly enable bounded-proof-only promotion of image-derived renderer-truth signals during the V18 build")
    build_p.add_argument("--signal-validation", action=argparse.BooleanOptionalAction, default=True, help="Run post-build signal validation checks")
    build_p.add_argument("--signal-validation-strict", action=argparse.BooleanOptionalAction, default=True, help="Fail build when signal validation fails")
    build_p.add_argument("--decoded-metadata-validation", action=argparse.BooleanOptionalAction, default=True, help="Run post-build decoded metadata table validation checks")
    build_p.add_argument("--decoded-metadata-validation-strict", action=argparse.BooleanOptionalAction, default=True, help="Fail build when decoded metadata validation fails")

    stats_p = sub.add_parser("stats", parents=[common])
    validate_p = sub.add_parser("validate-signals", parents=[common], help="Validate has_* signal coverage for finalized stores")
    validate_p.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True, help="Fail when validation checks fail")
    repair_p = sub.add_parser("repair-index", parents=[common])
    repair_p.add_argument("--no-backup", action="store_true", help="Skip creating index.parquet.bak before rewriting index.parquet")
    repair_p.add_argument("--map-workers", type=int, default=DEFAULT_MAP_WORKERS, help="Parallel harvest-stream workers across maps during index repair")
    patch_liquids_p = sub.add_parser("patch-liquids", parents=[common], help="Patch liquid_mask/liquid_height arrays and liquid has_* flags in-place")
    patch_liquids_p.add_argument("--batch-size", type=int, default=128, help="Tile batch size for array writes")
    patch_liquids_p.add_argument("--map-workers", type=int, default=DEFAULT_MAP_WORKERS, help="Parallel harvest-stream workers across maps during liquid patching")
    patch_liquids_p.add_argument("--allow-zarr-write", action="store_true", help="Required confirmation flag before mutating any V18 Zarr store")
    patch_liquids_p.add_argument("--no-backup", action="store_true", help="Skip creating index.parquet.bak.liquids before rewriting index.parquet")
    patch_liquids_p.add_argument("--signal-validation", action=argparse.BooleanOptionalAction, default=True, help="Run post-patch signal validation checks")
    patch_liquids_p.add_argument("--signal-validation-strict", action=argparse.BooleanOptionalAction, default=True, help="Fail when post-patch signal validation fails")
    patch_objects_p = sub.add_parser("patch-objects", parents=[common], help="Patch object mask arrays and object has_* flags in-place")
    patch_objects_p.add_argument("--batch-size", type=int, default=128, help="Tile batch size for array writes")
    patch_objects_p.add_argument("--map-workers", type=int, default=DEFAULT_MAP_WORKERS, help="Parallel harvest-stream workers across maps during object patching")
    patch_objects_p.add_argument("--allow-zarr-write", action="store_true", help="Required confirmation flag before mutating any V18 Zarr store")
    patch_objects_p.add_argument("--no-backup", action="store_true", help="Skip creating index.parquet.bak.objects before rewriting index.parquet")
    patch_objects_p.add_argument("--signal-validation", action=argparse.BooleanOptionalAction, default=True, help="Run post-patch signal validation checks")
    patch_objects_p.add_argument("--signal-validation-strict", action=argparse.BooleanOptionalAction, default=True, help="Fail when post-patch signal validation fails")
    merge_p = sub.add_parser("merge-builds", parents=[common], help="Merge per-build stores into one combined Zarr store")
    merge_p.add_argument("--output-name", type=str, default="merged_all", help="Output store name (without .zarr)")
    merge_p.add_argument("--allow-zarr-write", action="store_true", help="Required confirmation flag before creating any merged V18 Zarr store")
    merge_p.add_argument(
        "--dedupe-mode",
        choices=["none", "coords", "coords_height"],
        default="coords_height",
        help="Deduplication key: none, map+coords, or map+coords+height stats",
    )
    merge_p.add_argument("--batch-size", type=int, default=64, help="Array copy batch size during merge")
    merge_p.add_argument("--rebuild-existing", action="store_true", help="Overwrite existing merged output store")

    stubs_p = sub.add_parser("generate-viewer-stubs", parents=[common], help="Generate per-tile JSON stubs + manifest_capture_ledger.json from index.parquet")
    stubs_p.add_argument("--capture-root", type=str, default=None, help="Output root for per-build dataset directories (default: output/tmp/mdxviewer_validation_smoke)")
    stubs_p.add_argument("--curation-manifest", type=str, default=None, help="Optional curation manifest (dir/file). When provided, generate stubs only for keep=true tiles.")

    patch_rt_p = sub.add_parser("patch-renderer-truth", parents=[common], help="Patch MdxViewer renderer-truth PNGs into V18 Zarr stores")
    patch_rt_p.add_argument("--capture-root", type=str, default=None, help="Root directory containing per-build capture output (default: output/tmp/mdxviewer_validation_smoke)")
    patch_rt_p.add_argument("--no-backup", action="store_true", help="Skip backing up index.parquet before patching")
    patch_rt_p.add_argument("--curation-manifest", type=str, default=None, help="Optional curation manifest (dir/file). When provided, patch only keep=true tiles in this manifest.")

    clear_rt_p = sub.add_parser("clear-renderer-truth", parents=[common], help="Clear renderer-truth signals from V18 Zarr stores when the capture source is not trusted")
    clear_rt_p.add_argument("--no-backup", action="store_true", help="Skip backing up index.parquet before clearing")
    clear_rt_p.add_argument("--reason", type=str, default="untrusted renderer-truth source", help="Reason recorded in renderer_truth_reset_report.json")

    capture_rt_p = sub.add_parser("capture-renderer-truth", parents=[common], help="Run wow-viewer validation capture-batch using generated manifest_capture_ledger.json files")
    capture_rt_p.add_argument("--capture-root", type=str, default=None, help="Root directory containing per-build capture output + ledgers (default: output/tmp/mdxviewer_validation_smoke)")
    capture_rt_p.add_argument("--resolution", type=int, default=512, help="Capture resolution forwarded to validation-capture")
    capture_rt_p.add_argument("--dry-run", action=argparse.BooleanOptionalAction, default=False, help="Use validation-capture dry-run mode")
    capture_rt_p.add_argument("--real-scene-dry-run", action=argparse.BooleanOptionalAction, default=False, help="Use validation-capture real-scene dry-run mode")
    capture_rt_p.add_argument("--renderer", action=argparse.BooleanOptionalAction, default=False, help="Use the existing WoWViewer renderer path")
    capture_rt_p.add_argument("--gpu-viewer-style", action=argparse.BooleanOptionalAction, default=False, help="Back-compat alias for --renderer")
    capture_rt_p.add_argument("--native-renderer", action=argparse.BooleanOptionalAction, default=False, help="Use the stripped native terrain-render path")
    capture_rt_p.add_argument("--stub-scene", action=argparse.BooleanOptionalAction, default=False, help="Use validation-capture stub-scene mode")
    capture_rt_p.add_argument("--variants", type=str, default="primary,no-objects,objects-only", help="Capture variants to render (default: primary,no-objects,objects-only). Pass 'all' for the full QA set, or a comma-separated list like 'primary,no-objects,objects-only'.")

    args = parser.parse_args()

    if args.command == "build":
        cmd_build(args)
    elif args.command == "stats":
        cmd_stats(args)
    elif args.command == "validate-signals":
        cmd_validate_signals(args)
    elif args.command == "repair-index":
        cmd_repair_index(args)
    elif args.command == "patch-liquids":
        cmd_patch_liquids(args)
    elif args.command == "patch-objects":
        cmd_patch_objects(args)
    elif args.command == "merge-builds":
        cmd_merge_builds(args)
    elif args.command == "generate-viewer-stubs":
        cmd_generate_viewer_stubs(args)
    elif args.command == "patch-renderer-truth":
        cmd_patch_renderer_truth(args)
    elif args.command == "clear-renderer-truth":
        cmd_clear_renderer_truth(args)
    elif args.command == "capture-renderer-truth":
        cmd_capture_renderer_truth(args)


if __name__ == "__main__":
    main()
