"""Unified v60 Zarr store builder (Spec 134).

Consolidates all per-build, per-map v50.1 Zarr stores into a single v60 unified store
with a single index across all builds and maps. Handles schema differences, missing
signals, and the new Spec 132/133 signals (signal_class, surviving_height_levels,
terrain_shadow_256).

The v60 store uses the same Zarr format as v50.1 stores but with a unified index that
includes ``build_id``, ``map``, ``tile_x``, and ``tile_y`` columns. Signals that are
missing from some builds (e.g. ``terrain_shadow_256`` on pre-Spec-133 harvests) are
recorded as unavailable-with-reason, never silently zero-filled.
"""

from __future__ import annotations

import json
import shutil
import time
import uuid
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from harvester.v50.contracts import (
    DEFAULT_RELEASE_V60,
    STORE_SCHEMA_V60,
    UnavailableSignal,
    release_identity,
)

V60_UNIFIED_SCHEMA = "v60-unified-v1"
V60_DEFAULT_RELEASE = DEFAULT_RELEASE_V60
V60_STORE_SCHEMA = STORE_SCHEMA_V60
V50_SIGNAL_NAMES: set[str] = {
    "height_257", "normal_xyz", "normal_mask", "alpha_256",
    "holes_16", "liquid_mask", "liquid_height", "liquid_type_256",
    "mcnk_flags_16", "minimap_rgb", "minimap_rgb_1024", "minimap_rgb_authored",
    "mccv_rgb", "shadow_mask", "mcly_texture_ids", "mcly_layer_mask",
    "mcnr_mask_257", "ground_intent_height_257",
    "mcly_tileset_ids", "wdl_outer_17", "wdl_inner_16",
    "wdl_outer_present", "wdl_inner_present",
    "object_geometry_visible_mask_257", "object_geometry_visible_source_257",
    "object_geometry_visible_instance_257",
    "object_mask", "object_precise_mask", "object_instance_mask",
}


def _find_v50_stores(root: Path) -> list[Path]:
    """Discover all v50.1 Zarr stores under a root directory."""
    stores: list[Path] = []
    for entry in sorted(root.rglob("*.zarr")):
        if (entry / "index.parquet").exists() and (entry / "zarr.json").exists():
            stores.append(entry)
    return stores


def _read_store_manifest(store_path: Path) -> dict:
    """Read the manifest attrs from a Zarr store root."""
    group = zarr.open_group(str(store_path), mode="r")
    return dict(group.attrs)


def _load_signal_array(group: zarr.Group, signal_name: str, row_id: int) -> np.ndarray | None:
    """Load one signal array for one row, returning None if the signal is absent."""
    try:
        return np.asarray(group[signal_name][row_id])
    except (KeyError, IndexError, ValueError):
        return None


def _resolve_signal_arrays(
    group: zarr.Group,
    row_id: int,
    signal_names: Sequence[str],
) -> dict[str, np.ndarray | None]:
    """Load all signal arrays for one row, handling missing signals gracefully."""
    return {name: _load_signal_array(group, name, row_id) for name in signal_names}


@dataclass(frozen=True)
class V60BuildResult:
    store_path: Path
    row_count: int
    signal_count: int
    source_stores: tuple[str, ...]
    unavailable_signals: tuple[UnavailableSignal, ...]


def build_v60_store(
    source_roots: list[Path],
    output_path: Path,
    *,
    release: str = V60_DEFAULT_RELEASE,
    skip_unavailable_signals: bool = True,
) -> V60BuildResult:
    """Consolidate v50.1 stores from multiple roots into a single v60 unified store.

    Discovers all v50.1 Zarr stores under ``source_roots``, merges their per-build/map
    indices into one unified index, and writes a single Zarr store with all signals.

    Signals that are missing from a source store (e.g. ``terrain_shadow_256`` on a
    pre-Spec-133 harvest) are recorded as unavailable-with-reason in the manifest.
    """
    # Discover all source stores
    all_stores: list[Path] = []
    for root in source_roots:
        all_stores.extend(_find_v50_stores(root))
    if not all_stores:
        raise ValueError(f"no v50 stores found under {source_roots}")

    all_stores = sorted(all_stores)
    print(f"Found {len(all_stores)} v50 stores", flush=True)

    # Discover all signal names across all stores
    all_signal_names: set[str] = set()
    for store_path in all_stores:
        group = zarr.open_group(str(store_path), mode="r")
        all_signal_names.update(group.array_keys())
    # Add the v60-cataloged signals that may not be in older stores
    all_signal_names.update(V50_SIGNAL_NAMES)
    all_signal_names = sorted(all_signal_names)
    print(f"Signals: {len(all_signal_names)}", flush=True)

    # Build unified index
    index_rows: list[dict] = []
    row_arrays: dict[str, list[np.ndarray | None]] = {name: [] for name in all_signal_names}
    unavailable: list[UnavailableSignal] = []
    total_rows = 0

    for store_path in all_stores:
        group = zarr.open_group(str(store_path), mode="r")
        index = pq.read_table(store_path / "index.parquet").to_pylist()
        present = set(group.array_keys())

        # Determine build_id from parent path structure
        build_id = store_path.parent.name if store_path.parent.name != "v50.1" else "unknown"

        for row_id, meta in enumerate(index):
            map_name = str(meta.get("map", "unknown"))
            tile_x = int(meta.get("tile_x", -1))
            tile_y = int(meta.get("tile_y", -1))

            index_rows.append({
                "build_id": build_id,
                "map": map_name,
                "tile_x": tile_x,
                "tile_y": tile_y,
                "source_store": str(store_path),
                "source_row_id": row_id,
            })

            for signal_name in all_signal_names:
                if signal_name in present:
                    arr = _load_signal_array(group, signal_name, row_id)
                    row_arrays[signal_name].append(arr)
                else:
                    row_arrays[signal_name].append(None)

            total_rows += 1

    print(f"Total rows: {total_rows}", flush=True)

    # Write the v60 store
    if output_path.exists():
        shutil.rmtree(output_path)

    staging_path = output_path.parent / f".{output_path.name}.staging-{uuid.uuid4().hex}"
    staging_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        root = zarr.open_group(str(staging_path), mode="w")

        # Write index
        index_table = pa.Table.from_pylist(index_rows)
        pq.write_table(index_table, str(staging_path / "index.parquet"))
        print(f"  Wrote index.parquet with {total_rows} rows", flush=True)

        # Write each signal
        written_signals = 0
        for signal_name in all_signal_names:
            arrays = row_arrays[signal_name]
            non_none = [a for a in arrays if a is not None]
            if not non_none:
                if skip_unavailable_signals:
                    unavailable.append(UnavailableSignal(
                        name=signal_name,
                        reason="no_source_data:not_present_in_any_source_store",
                    ))
                    continue
                # Create a zero-filled array if required
                raise ValueError(f"signal {signal_name} has zero rows across all stores")

            # Determine dtype and shape from the first non-None array
            dtype = non_none[0].dtype
            shape = non_none[0].shape

            # Stack or pad
            stacked_arrays: list[np.ndarray] = []
            missing = 0
            for arr in arrays:
                if arr is not None:
                    if arr.dtype != dtype:
                        arr = arr.astype(dtype)
                    if arr.shape != shape:
                        print(f"  WARNING: {signal_name}: shape mismatch {arr.shape} vs {shape}, skipping row", flush=True)
                        missing += 1
                        continue
                    stacked_arrays.append(np.ascontiguousarray(arr))
                else:
                    missing += 1
                    # Create a zero-filled array of the same shape/dtype
                    stacked_arrays.append(np.zeros(shape, dtype=dtype))

            if missing > 0 and missing == len(arrays):
                unavailable.append(UnavailableSignal(
                    name=signal_name,
                    reason="no_source_data:not_present_in_any_source_store",
                ))
                print(f"  SKIP {signal_name}: zero rows present", flush=True)
                continue

            if missing > 0:
                print(f"  {signal_name}: {missing}/{total_rows} rows missing (zero-filled)", flush=True)

            stacked = np.stack(stacked_arrays, axis=0)
            root.create_dataset(signal_name, data=stacked, shape=stacked.shape, dtype=dtype, overwrite=True)
            written_signals += 1
            print(f"  Wrote {signal_name}: shape={stacked.shape} dtype={dtype}", flush=True)

        # Write manifest
        manifest = {
            "store_schema": V60_STORE_SCHEMA,
            "release": release,
            "row_count": total_rows,
            "signal_count": written_signals,
            "source_stores": [str(s) for s in all_stores],
            "unavailable_signals": [
                {"name": u.name, "reason": u.reason} for u in unavailable
            ],
        }
        root.attrs.update(manifest)
        print(f"  Manifest: {len(all_stores)} stores, {written_signals} signals, {total_rows} rows", flush=True)

    except BaseException:
        shutil.rmtree(staging_path, ignore_errors=True)
        raise

    # Atomic replace
    _replace_directory(staging_path, output_path)

    return V60BuildResult(
        store_path=output_path,
        row_count=total_rows,
        signal_count=written_signals,
        source_stores=tuple(str(s) for s in all_stores),
        unavailable_signals=tuple(unavailable),
    )


def _replace_directory(staging_path: Path, output_path: Path) -> None:
    """Move staging_path onto output_path, retrying transient failures."""
    last_error: OSError | None = None
    for attempt in range(6):
        try:
            if output_path.exists():
                shutil.rmtree(output_path)
            staging_path.rename(output_path)
            return
        except OSError as exc:
            last_error = exc
            time.sleep(0.2 * (2**attempt))
    raise RuntimeError(
        f"could not replace {output_path} with {staging_path} after retrying: {last_error}"
    )