"""Zarr signal store for PM4 segment and asset reference signals.

Stores segment signals and asset reference signals in Zarr v3 LocalStore
format.  Fixed-size fields are stored as chunked arrays; variable-length
fields (footprint hulls, histograms, typed bounds) are stored as JSON
in group attributes keyed by segment/asset ID.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import zarr
import zarr.storage

from .models import (
    Pm4AssetReferenceSignalRecord,
    Pm4Bounds3,
    Pm4SegmentAnchorSignals,
    Pm4SegmentHeightStats,
    Pm4SegmentSignalRecord,
    Pm4SegmentTopologyStats,
)

_CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=5, shuffle="bitshuffle")


def _bounds_to_array(bounds: Pm4Bounds3 | None) -> np.ndarray:
    """Convert bounds to float32 [2,3] array. Zero if None."""
    if bounds is None:
        return np.zeros((2, 3), dtype=np.float32)
    return np.array([list(bounds.min), list(bounds.max)], dtype=np.float32)


def _array_to_bounds(arr: np.ndarray) -> Pm4Bounds3 | None:
    """Convert float32 [2,3] array to bounds. None if all zeros."""
    if np.all(arr == 0):
        return None
    return Pm4Bounds3(
        min=(float(arr[0, 0]), float(arr[0, 1]), float(arr[0, 2])),
        max=(float(arr[1, 0]), float(arr[1, 1]), float(arr[1, 2])),
    )


def _vec3_to_array(v: tuple[float, float, float] | None) -> np.ndarray:
    if v is None:
        return np.zeros(3, dtype=np.float32)
    return np.array(list(v), dtype=np.float32)


def _array_to_vec3(arr: np.ndarray) -> tuple[float, float, float] | None:
    if np.all(arr == 0):
        return None
    return (float(arr[0]), float(arr[1]), float(arr[2]))


# ---------------------------------------------------------------------------
# Segment signal store
# ---------------------------------------------------------------------------


def write_segment_signals_zarr(
    store_path: str | Path,
    segments: list[Pm4SegmentSignalRecord],
    *,
    overwrite: bool = False,
) -> Path:
    """Write segment signals to a Zarr LocalStore.

    Fixed-size fields are stored as chunked arrays indexed by segment ordinal.
    Variable-length fields (footprint hull, histogram, typed bounds) are stored
    as JSON strings in group attributes keyed by segment ID.
    """
    store_path = Path(store_path)
    if overwrite and store_path.exists():
        _rmtree(store_path)

    n = len(segments)
    store = zarr.storage.LocalStore(str(store_path), read_only=False)
    root = zarr.group(store=store)

    # Fixed-size arrays
    bounds_arr = np.zeros((n, 2, 3), dtype=np.float32)
    height_arr = np.zeros((n, 3), dtype=np.float32)
    topo_arr = np.zeros((n, 4), dtype=np.int32)
    anchor_arr = np.zeros((n, 8), dtype=np.float32)
    hull_area_arr = np.zeros(n, dtype=np.float32)

    segment_ids = [s.segment_id for s in segments]

    for i, seg in enumerate(segments):
        bounds_arr[i] = _bounds_to_array(seg.bounds)
        height_arr[i] = [
            seg.height_stats.minimum_plane_distance,
            seg.height_stats.maximum_plane_distance,
            seg.height_stats.average_plane_distance,
        ]
        topo_arr[i] = [
            seg.topology_stats.surface_count,
            seg.topology_stats.total_index_count,
            seg.topology_stats.anchor_point_count,
            seg.topology_stats.anchor_normal_count,
        ]
        anchor_arr[i] = [
            seg.anchor_signals.linked_position_ref_count,
            seg.anchor_signals.normal_heading_count,
            seg.anchor_signals.terminator_count,
            seg.anchor_signals.floor_minimum,
            seg.anchor_signals.floor_maximum,
            seg.anchor_signals.heading_minimum_degrees or 0.0,
            seg.anchor_signals.heading_maximum_degrees or 0.0,
            seg.anchor_signals.heading_mean_degrees or 0.0,
        ]
        # Compute footprint area from hull
        hull = seg.footprint_hull
        if len(hull) >= 3:
            area = _compute_polygon_area(hull)
            hull_area_arr[i] = area

    root.create_array("bounds", data=bounds_arr, compressors=_CODEC)
    root.create_array("height_stats", data=height_arr, compressors=_CODEC)
    root.create_array("topology_stats", data=topo_arr, compressors=_CODEC)
    root.create_array("anchor_signals", data=anchor_arr, compressors=_CODEC)
    root.create_array("footprint_area", data=hull_area_arr, compressors=_CODEC)

    # Variable-length fields as JSON attrs
    variable: dict[str, Any] = {}
    for seg in segments:
        variable[f"hull:{seg.segment_id}"] = seg.footprint_hull
        variable[f"hist:{seg.segment_id}"] = seg.surface_family_histogram
        variable[f"tb:{seg.segment_id}"] = {
            str(k): {"min": list(v.min), "max": list(v.max)}
            for k, v in seg.typed_bounds.items()
        }
        variable[f"tiles:{seg.segment_id}"] = seg.tile_coordinates

    root.attrs["segment_ids"] = segment_ids
    root.attrs["signal_version"] = segments[0].signal_version if segments else ""
    root.attrs["segment_count"] = n
    root.attrs.update(variable)

    return store_path


def read_segment_signals_zarr(store_path: str | Path) -> list[Pm4SegmentSignalRecord]:
    """Read segment signals from a Zarr LocalStore."""
    store_path = Path(store_path)
    store = zarr.storage.LocalStore(str(store_path), read_only=True)
    root = zarr.open_group(store, mode="r")

    segment_ids: list[str] = root.attrs["segment_ids"]
    bounds_arr = root["bounds"][:]
    height_arr = root["height_stats"][:]
    topo_arr = root["topology_stats"][:]
    anchor_arr = root["anchor_signals"][:]
    signal_version = root.attrs.get("signal_version", "")

    segments: list[Pm4SegmentSignalRecord] = []
    for i, seg_id in enumerate(segment_ids):
        hull = root.attrs.get(f"hull:{seg_id}", [])
        hist = root.attrs.get(f"hist:{seg_id}", {})
        tb_raw = root.attrs.get(f"tb:{seg_id}", {})
        typed_bounds: dict[int, Pm4Bounds3] = {}
        for k, v in tb_raw.items():
            flag = int(k)
            typed_bounds[flag] = Pm4Bounds3(
                min=tuple(v["min"]),  # type: ignore[arg-type]
                max=tuple(v["max"]),  # type: ignore[arg-type]
            )

        segments.append(
            Pm4SegmentSignalRecord(
                segment_id=seg_id,
                bounds=_array_to_bounds(bounds_arr[i]),
                footprint_hull=[tuple(p) for p in hull],  # type: ignore[misc]
                height_stats=Pm4SegmentHeightStats(
                    minimum_plane_distance=float(height_arr[i, 0]),
                    maximum_plane_distance=float(height_arr[i, 1]),
                    average_plane_distance=float(height_arr[i, 2]),
                ),
                surface_family_histogram=dict(hist),
                topology_stats=Pm4SegmentTopologyStats(
                    surface_count=int(topo_arr[i, 0]),
                    total_index_count=int(topo_arr[i, 1]),
                    anchor_point_count=int(topo_arr[i, 2]),
                    anchor_normal_count=int(topo_arr[i, 3]),
                ),
                anchor_signals=Pm4SegmentAnchorSignals(
                    linked_position_ref_count=int(anchor_arr[i, 0]),
                    normal_heading_count=int(anchor_arr[i, 1]),
                    terminator_count=int(anchor_arr[i, 2]),
                    floor_minimum=int(anchor_arr[i, 3]),
                    floor_maximum=int(anchor_arr[i, 4]),
                    heading_minimum_degrees=float(anchor_arr[i, 5]) or None,
                    heading_maximum_degrees=float(anchor_arr[i, 6]) or None,
                    heading_mean_degrees=float(anchor_arr[i, 7]) or None,
                ),
                signal_version=signal_version,
                signal_store_row=None,
                typed_bounds=typed_bounds,
                tile_coordinates=list(root.attrs.get(f"tiles:{seg_id}", [])),
            )
        )

    return segments


# ---------------------------------------------------------------------------
# Asset reference signal store
# ---------------------------------------------------------------------------


def write_asset_references_zarr(
    store_path: str | Path,
    assets: list[Pm4AssetReferenceSignalRecord],
    *,
    overwrite: bool = False,
) -> Path:
    """Write asset reference signals to a Zarr LocalStore."""
    store_path = Path(store_path)
    if overwrite and store_path.exists():
        _rmtree(store_path)

    n = len(assets)
    store = zarr.storage.LocalStore(str(store_path), read_only=False)
    root = zarr.group(store=store)

    bounds_arr = np.zeros((n, 2, 3), dtype=np.float32)
    center_arr = np.zeros((n, 3), dtype=np.float32)
    ref_pos_arr = np.zeros((n, 3), dtype=np.float32)
    ref_rot_arr = np.zeros((n, 3), dtype=np.float32)
    ref_scale_arr = np.zeros(n, dtype=np.float32)
    footprint_area_arr = np.zeros(n, dtype=np.float32)

    asset_ids = [a.asset_id for a in assets]

    for i, asset in enumerate(assets):
        bounds_arr[i] = _bounds_to_array(asset.bounds)
        center_arr[i] = list(asset.center)
        ref_pos_arr[i] = _vec3_to_array(asset.reference_position)
        ref_rot_arr[i] = _vec3_to_array(asset.reference_rotation)
        ref_scale_arr[i] = asset.reference_scale or 0.0
        footprint_area_arr[i] = asset.footprint_area

    root.create_array("bounds", data=bounds_arr, compressors=_CODEC)
    root.create_array("center", data=center_arr, compressors=_CODEC)
    root.create_array("reference_position", data=ref_pos_arr, compressors=_CODEC)
    root.create_array("reference_rotation", data=ref_rot_arr, compressors=_CODEC)
    root.create_array("reference_scale", data=ref_scale_arr, compressors=_CODEC)
    root.create_array("footprint_area", data=footprint_area_arr, compressors=_CODEC)

    # Variable-length fields as JSON attrs
    variable: dict[str, Any] = {}
    for asset in assets:
        variable[f"hull:{asset.asset_id}"] = asset.footprint_hull
        variable[f"hist:{asset.asset_id}"] = asset.surface_family_histogram
        variable[f"signals:{asset.asset_id}"] = asset.render_or_collision_signals

    root.attrs["asset_ids"] = asset_ids
    root.attrs["asset_count"] = n
    root.attrs["signal_version"] = assets[0].signal_version if assets else ""
    # Store string fields
    for asset in assets:
        root.attrs[f"path:{asset.asset_id}"] = asset.asset_path
        root.attrs[f"kind:{asset.asset_id}"] = asset.asset_kind
        root.attrs[f"build:{asset.asset_id}"] = asset.client_build or ""
        root.attrs[f"tiles:{asset.asset_id}"] = asset.tile_coordinates
        root.attrs[f"tags:{asset.asset_id}"] = asset.validation_tags
        root.attrs[f"store_row:{asset.asset_id}"] = asset.signal_store_row or ""
    root.attrs.update(variable)

    return store_path


def read_asset_references_zarr(store_path: str | Path) -> list[Pm4AssetReferenceSignalRecord]:
    """Read asset reference signals from a Zarr LocalStore."""
    store_path = Path(store_path)
    store = zarr.storage.LocalStore(str(store_path), read_only=True)
    root = zarr.open_group(store, mode="r")

    asset_ids: list[str] = root.attrs["asset_ids"]
    bounds_arr = root["bounds"][:]
    center_arr = root["center"][:]
    ref_pos_arr = root["reference_position"][:]
    ref_rot_arr = root["reference_rotation"][:]
    ref_scale_arr = root["reference_scale"][:]
    footprint_area_arr = root["footprint_area"][:]
    signal_version = root.attrs.get("signal_version", "")

    assets: list[Pm4AssetReferenceSignalRecord] = []
    for i, asset_id in enumerate(asset_ids):
        hull = root.attrs.get(f"hull:{asset_id}", [])
        hist = root.attrs.get(f"hist:{asset_id}", {})
        signals = root.attrs.get(f"signals:{asset_id}", {})

        assets.append(
            Pm4AssetReferenceSignalRecord(
                asset_id=asset_id,
                asset_path=root.attrs.get(f"path:{asset_id}", ""),
                asset_kind=root.attrs.get(f"kind:{asset_id}", ""),
                client_build=root.attrs.get(f"build:{asset_id}") or None,
                tile_coordinates=list(root.attrs.get(f"tiles:{asset_id}", [])),
                bounds=_array_to_bounds(bounds_arr[i]),
                center=tuple(center_arr[i]),  # type: ignore[arg-type]
                footprint_hull=[tuple(p) for p in hull],  # type: ignore[misc]
                footprint_area=float(footprint_area_arr[i]),
                reference_position=_array_to_vec3(ref_pos_arr[i]),
                reference_rotation=_array_to_vec3(ref_rot_arr[i]),
                reference_scale=float(ref_scale_arr[i]) or None,
                surface_family_histogram=dict(hist),
                render_or_collision_signals=dict(signals),
                signal_version=signal_version,
                signal_store_row=root.attrs.get(f"store_row:{asset_id}") or None,
                validation_tags=list(root.attrs.get(f"tags:{asset_id}", [])),
            )
        )

    return assets


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _compute_polygon_area(points: list[tuple[float, float]]) -> float:
    """Shoelace formula for polygon area."""
    n = len(points)
    area = 0.0
    for i in range(n):
        j = (i + 1) % n
        area += points[i][0] * points[j][1]
        area -= points[j][0] * points[i][1]
    return abs(area) / 2.0


def _rmtree(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            _rmtree(child)
        else:
            child.unlink()
    path.rmdir()
