"""Full-map alpha/fractal segmentation helpers for spec 076 Phase 2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image, ImageDraw
from scipy.ndimage import binary_closing, binary_dilation, find_objects, label
from scipy.ndimage import sum as nd_sum


@dataclass(frozen=True, slots=True)
class FractalRegion:
    region_id: str
    build: str
    map_name: str
    layer_slot: int
    layer_idx: int
    bbox_xywh: tuple[int, int, int, int]
    area: int
    tile_coverage_count: int
    tile_coverage: list[dict[str, int]]
    alpha_mean: float
    alpha_max: float
    height_mean: float | None
    height_std: float | None
    height_range: float | None
    normal_mean_xyz: tuple[float, float, float] | None
    mcly_texture_ids: list[int]
    mcly_active_layers: list[int]
    curation_label: str
    rejection_reason: str | None
    linked_component_ids: list[str]
    provenance: dict[str, Any] = field(default_factory=dict)


def load_canvas_group(canvas_dir: str | Path) -> zarr.Group:
    path = Path(canvas_dir)
    if path.name != "canvas.zarr":
        path = path / "canvas.zarr"
    if not path.exists():
        raise FileNotFoundError(f"Canvas Zarr not found: {path}")
    return zarr.open_group(str(path), mode="r")


def segment_canvas_regions(
    canvas: zarr.Group,
    *,
    threshold: float = 0.05,
    min_area: int = 64,
    min_atomic_footprint_px: int = 8,
    curation_mode: str = "default",
    chonker_area_fraction: float = 0.18,
    one_off_min_area: int = 4096,
    max_regions_per_layer: int | None = None,
    catalog_rows: list[dict[str, Any]] | None = None,
) -> list[FractalRegion]:
    """Segment per-layer full-map alpha regions and attach provenance summaries."""
    alpha = canvas["alpha_256"][:].astype(np.float32)
    tile_id_256 = canvas["tile_id_256"][:].astype(np.int32)
    height = canvas["height_257"][:].astype(np.float32) if "height_257" in canvas else None
    normals = canvas["normal_xyz"][:].astype(np.float32) if "normal_xyz" in canvas else None
    mcly_ids = canvas["mcly_texture_ids"][:].astype(np.int32) if "mcly_texture_ids" in canvas else None
    mcly_mask = canvas["mcly_layer_mask"][:].astype(np.float32) if "mcly_layer_mask" in canvas else None
    layer_indices = canvas["alpha_layer_indices"][:].astype(np.int32).tolist() if "alpha_layer_indices" in canvas else list(range(alpha.shape[2]))
    layout = dict(canvas.attrs.get("layout", {}))
    build = str(layout.get("build", ""))
    map_name = str(layout.get("map_name", ""))
    total_pixels = int(alpha.shape[0] * alpha.shape[1])
    catalog_index = _build_catalog_index(catalog_rows or [])

    regions: list[FractalRegion] = []
    for layer_slot in range(alpha.shape[2]):
        layer_idx = int(layer_indices[layer_slot]) if layer_slot < len(layer_indices) else int(layer_slot)
        layer = alpha[:, :, layer_slot]
        binary = layer > float(threshold)
        labeled, count = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        if count == 0:
            continue
        areas = nd_sum(binary, labeled, range(1, count + 1))
        objects = find_objects(labeled)
        candidates: list[tuple[int, int, tuple[slice, slice]]] = []
        for label_idx, raw_area in enumerate(areas, start=1):
            area = int(raw_area)
            if area < int(min_area):
                continue
            bounds = objects[label_idx - 1]
            if bounds is None:
                continue
            candidates.append((label_idx, area, bounds))
        candidates.sort(key=lambda item: item[1], reverse=True)
        if max_regions_per_layer is not None:
            candidates = candidates[: max(0, int(max_regions_per_layer))]
        for label_idx, area, (slice_y, slice_x) in candidates:
            x0, x1 = int(slice_x.start), int(slice_x.stop)
            y0, y1 = int(slice_y.start), int(slice_y.stop)
            bbox = (x0, y0, x1 - x0, y1 - y0)
            region_mask = labeled[y0:y1, x0:x1] == label_idx
            alpha_crop = layer[y0:y1, x0:x1]
            tile_coverage = _tile_coverage(tile_id_256[y0:y1, x0:x1], region_mask)
            height_stats = _height_stats(height, bbox) if height is not None else (None, None, None)
            normal_mean = _normal_mean(normals, bbox) if normals is not None else None
            texture_ids, active_layers = _mcly_summary(mcly_ids, mcly_mask, bbox)
            if str(curation_mode) == "raw":
                curation_label, rejection_reason = "raw_component", None
            else:
                curation_label, rejection_reason = classify_region(
                    bbox_xywh=bbox,
                    area=area,
                    total_pixels=total_pixels,
                    tile_coverage_count=len(tile_coverage),
                    alpha_mean=float(alpha_crop[region_mask].mean()) if region_mask.any() else 0.0,
                    min_atomic_footprint_px=min_atomic_footprint_px,
                    chonker_area_fraction=chonker_area_fraction,
                    one_off_min_area=one_off_min_area,
                )
            linked_ids = _linked_component_ids(catalog_index, layer_idx, bbox, tile_coverage)
            region_id = _region_id(build, map_name, layer_idx, bbox, area)
            regions.append(
                FractalRegion(
                    region_id=region_id,
                    build=build,
                    map_name=map_name,
                    layer_slot=int(layer_slot),
                    layer_idx=layer_idx,
                    bbox_xywh=bbox,
                    area=area,
                    tile_coverage_count=len(tile_coverage),
                    tile_coverage=tile_coverage,
                    alpha_mean=float(alpha_crop[region_mask].mean()) if region_mask.any() else 0.0,
                    alpha_max=float(alpha_crop[region_mask].max()) if region_mask.any() else 0.0,
                    height_mean=height_stats[0],
                    height_std=height_stats[1],
                    height_range=height_stats[2],
                    normal_mean_xyz=normal_mean,
                    mcly_texture_ids=texture_ids,
                    mcly_active_layers=active_layers,
                    curation_label=curation_label,
                    rejection_reason=rejection_reason,
                    linked_component_ids=linked_ids,
                )
            )
    return regions


def detect_rectangle_pages(
    canvas: zarr.Group,
    *,
    threshold: float = 0.05,
    min_area: int = 256,
    min_extent: float = 0.85,
    max_aspect_ratio: float = 8.0,
    max_regions_per_layer: int | None = 1000,
) -> list[FractalRegion]:
    """Detect axis-aligned rectangular alpha pages independently of fractal segmentation.

    A rectangle page is an authored paste/boundary region whose binary alpha mask
    fills its bounding box almost completely. These are detected as separate regions
    because connected-component segmentation can split a page into many internal
    brush strokes.
    """
    alpha = canvas["alpha_256"][:].astype(np.float32)
    tile_id_256 = canvas["tile_id_256"][:].astype(np.int32)
    height = canvas["height_257"][:].astype(np.float32) if "height_257" in canvas else None
    normals = canvas["normal_xyz"][:].astype(np.float32) if "normal_xyz" in canvas else None
    mcly_ids = canvas["mcly_texture_ids"][:].astype(np.int32) if "mcly_texture_ids" in canvas else None
    mcly_mask = canvas["mcly_layer_mask"][:].astype(np.float32) if "mcly_layer_mask" in canvas else None
    layer_indices = canvas["alpha_layer_indices"][:].astype(np.int32).tolist() if "alpha_layer_indices" in canvas else list(range(alpha.shape[2]))
    layout = dict(canvas.attrs.get("layout", {}))
    build = str(layout.get("build", ""))
    map_name = str(layout.get("map_name", ""))

    regions: list[FractalRegion] = []
    for layer_slot in range(alpha.shape[2]):
        layer_idx = int(layer_indices[layer_slot]) if layer_slot < len(layer_indices) else int(layer_slot)
        binary = alpha[:, :, layer_slot] > float(threshold)
        labeled, count = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        if count == 0:
            continue
        areas = nd_sum(binary, labeled, range(1, count + 1))
        objects = find_objects(labeled)
        candidates: list[tuple[int, int, tuple[slice, slice]]] = []
        for label_idx, raw_area in enumerate(areas, start=1):
            area = int(raw_area)
            if area < int(min_area):
                continue
            bounds = objects[label_idx - 1]
            if bounds is None:
                continue
            candidates.append((label_idx, area, bounds))
        candidates.sort(key=lambda item: item[1], reverse=True)
        if max_regions_per_layer is not None:
            candidates = candidates[: max(0, int(max_regions_per_layer))]
        for label_idx, area, (slice_y, slice_x) in candidates:
            x0, x1 = int(slice_x.start), int(slice_x.stop)
            y0, y1 = int(slice_y.start), int(slice_y.stop)
            width = x1 - x0
            height_box = y1 - y0
            bbox_area = max(1, width * height_box)
            extent = float(area) / float(bbox_area)
            if extent < float(min_extent):
                continue
            aspect = max(width, height_box) / max(1, min(width, height_box))
            if aspect > float(max_aspect_ratio):
                continue
            bbox = (x0, y0, width, height_box)
            region_mask = labeled[y0:y1, x0:x1] == label_idx
            alpha_crop = alpha[y0:y1, x0:x1, layer_slot]
            tile_coverage = _tile_coverage(tile_id_256[y0:y1, x0:x1], region_mask)
            height_stats = _height_stats(height, bbox) if height is not None else (None, None, None)
            normal_mean = _normal_mean(normals, bbox) if normals is not None else None
            texture_ids, active_layers = _mcly_summary(mcly_ids, mcly_mask, bbox)
            region_id = _region_id(build, map_name, layer_idx, bbox, area)
            regions.append(
                FractalRegion(
                    region_id=region_id,
                    build=build,
                    map_name=map_name,
                    layer_slot=int(layer_slot),
                    layer_idx=layer_idx,
                    bbox_xywh=bbox,
                    area=area,
                    tile_coverage_count=len(tile_coverage),
                    tile_coverage=tile_coverage,
                    alpha_mean=float(alpha_crop[region_mask].mean()) if region_mask.any() else 0.0,
                    alpha_max=float(alpha_crop[region_mask].max()) if region_mask.any() else 0.0,
                    height_mean=height_stats[0],
                    height_std=height_stats[1],
                    height_range=height_stats[2],
                    normal_mean_xyz=normal_mean,
                    mcly_texture_ids=texture_ids,
                    mcly_active_layers=active_layers,
                    curation_label="rectangle_page",
                    rejection_reason=None,
                    linked_component_ids=[],
                    provenance={"extent": round(extent, 4), "aspect_ratio": round(aspect, 4)},
                )
            )
    return regions


def _downsample_alpha_layer_binary(
    alpha: Any,
    *,
    layer_slot: int,
    threshold: float,
    factor: int,
    block_rows: int = 1024,
) -> np.ndarray:
    """Max-pool one alpha layer into a coarse binary mask without dense full-map reads."""
    h_full, w_full = int(alpha.shape[0]), int(alpha.shape[1])
    factor = max(1, int(factor))
    h_ds = (h_full + factor - 1) // factor
    w_ds = (w_full + factor - 1) // factor
    out = np.zeros((h_ds, w_ds), dtype=bool)
    block_rows = factor * max(1, int(block_rows) // factor)
    for y0 in range(0, h_full, block_rows):
        y1 = min(h_full, y0 + block_rows)
        block = alpha[y0:y1, :, layer_slot].astype(np.float32) > float(threshold)
        pad_h = (-block.shape[0]) % factor
        pad_w = (-block.shape[1]) % factor
        if pad_h or pad_w:
            block = np.pad(block, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
        pooled = block.reshape(block.shape[0] // factor, factor, block.shape[1] // factor, factor).max(axis=(1, 3))
        ds_y0 = y0 // factor
        out[ds_y0 : ds_y0 + pooled.shape[0], : pooled.shape[1]] = pooled
    return out


def _downsample_alpha_layer_coverage(
    alpha: Any,
    *,
    layer_slot: int,
    threshold: float,
    factor: int,
    block_rows: int = 1024,
) -> np.ndarray:
    """Average painted coverage per block for middle-scale paste detection."""
    h_full, w_full = int(alpha.shape[0]), int(alpha.shape[1])
    factor = max(1, int(factor))
    h_ds = (h_full + factor - 1) // factor
    w_ds = (w_full + factor - 1) // factor
    out = np.zeros((h_ds, w_ds), dtype=np.float32)
    block_rows = factor * max(1, int(block_rows) // factor)
    for y0 in range(0, h_full, block_rows):
        y1 = min(h_full, y0 + block_rows)
        block = alpha[y0:y1, :, layer_slot].astype(np.float32) > float(threshold)
        pad_h = (-block.shape[0]) % factor
        pad_w = (-block.shape[1]) % factor
        if pad_h or pad_w:
            block = np.pad(block, ((0, pad_h), (0, pad_w)), mode="constant", constant_values=False)
        pooled = block.reshape(block.shape[0] // factor, factor, block.shape[1] // factor, factor).mean(axis=(1, 3))
        ds_y0 = y0 // factor
        out[ds_y0 : ds_y0 + pooled.shape[0], : pooled.shape[1]] = pooled
    return out


def segment_blocky_pastes(
    canvas: zarr.Group,
    *,
    threshold: float = 0.05,
    block_size: int = 16,
    min_block_coverage: float = 0.08,
    block_close_radius: int = 1,
    min_area: int = 512,
    min_footprint_px: int = 16,
    max_footprint_px: int | None = None,
    max_aspect_ratio: float = 12.0,
    max_regions_per_layer: int | None = 1000,
) -> list[FractalRegion]:
    """Segment dense blocky child paste/scar regions inside larger macro zones.

    This is the middle scale between raw brush-dot components and giant zone-sized
    parent canvases. It detects connected components on a block-coverage grid,
    then reprojects those components back to alpha-pixel coordinates.
    """
    alpha = canvas["alpha_256"]
    tile_id_256 = canvas["tile_id_256"]
    layer_indices = canvas["alpha_layer_indices"][:].astype(np.int32).tolist() if "alpha_layer_indices" in canvas else list(range(alpha.shape[2]))
    layout = dict(canvas.attrs.get("layout", {}))
    build = str(layout.get("build", ""))
    map_name = str(layout.get("map_name", ""))
    factor = max(1, int(block_size))
    close_radius = max(0, int(block_close_radius))
    struct = np.ones((close_radius * 2 + 1, close_radius * 2 + 1), dtype=np.uint8)
    h_full, w_full = int(alpha.shape[0]), int(alpha.shape[1])

    regions: list[FractalRegion] = []
    for layer_slot in range(alpha.shape[2]):
        layer_idx = int(layer_indices[layer_slot]) if layer_slot < len(layer_indices) else int(layer_slot)
        coverage = _downsample_alpha_layer_coverage(alpha, layer_slot=layer_slot, threshold=float(threshold), factor=factor)
        binary = coverage >= float(min_block_coverage)
        if close_radius > 0:
            binary = binary_closing(binary, structure=struct, border_value=0)
        if not binary.any():
            continue
        labeled, count = label(binary, structure=np.ones((3, 3), dtype=np.uint8))
        if count == 0:
            continue
        areas = nd_sum(coverage, labeled, range(1, count + 1))
        objects = find_objects(labeled)
        candidates: list[tuple[int, int, tuple[slice, slice]]] = []
        for label_idx, coverage_area in enumerate(areas, start=1):
            area = int(float(coverage_area) * factor * factor)
            if area < int(min_area):
                continue
            bounds = objects[label_idx - 1]
            if bounds is None:
                continue
            candidates.append((label_idx, area, bounds))
        candidates.sort(key=lambda item: item[1], reverse=True)
        if max_regions_per_layer is not None:
            candidates = candidates[: max(0, int(max_regions_per_layer))]
        for label_idx, _coverage_area, (slice_y, slice_x) in candidates:
            x0 = int(slice_x.start) * factor
            y0 = int(slice_y.start) * factor
            x1 = min(w_full, int(slice_x.stop) * factor)
            y1 = min(h_full, int(slice_y.stop) * factor)
            width = x1 - x0
            height_box = y1 - y0
            if width < int(min_footprint_px) or height_box < int(min_footprint_px):
                continue
            if max_footprint_px is not None and max(width, height_box) > int(max_footprint_px):
                continue
            aspect = max(width, height_box) / max(1, min(width, height_box))
            if aspect > float(max_aspect_ratio):
                continue
            bbox = (x0, y0, width, height_box)
            alpha_crop = alpha[y0:y1, x0:x1, layer_slot].astype(np.float32)
            region_mask = alpha_crop > float(threshold)
            raw_area = int(np.count_nonzero(region_mask))
            if raw_area < int(min_area):
                continue
            tile_coverage = _tile_coverage(tile_id_256[y0:y1, x0:x1].astype(np.int32), region_mask)
            texture_ids, active_layers = _mcly_summary(
                canvas["mcly_texture_ids"][:].astype(np.int32) if "mcly_texture_ids" in canvas else None,
                canvas["mcly_layer_mask"][:].astype(np.float32) if "mcly_layer_mask" in canvas else None,
                bbox,
            )
            block_mask = labeled[slice_y, slice_x] == label_idx
            block_extent = float(block_mask.mean()) if block_mask.size else 0.0
            region_id = _region_id(build, map_name, layer_idx, bbox, raw_area)
            regions.append(
                FractalRegion(
                    region_id=region_id,
                    build=build,
                    map_name=map_name,
                    layer_slot=int(layer_slot),
                    layer_idx=layer_idx,
                    bbox_xywh=bbox,
                    area=raw_area,
                    tile_coverage_count=len(tile_coverage),
                    tile_coverage=tile_coverage,
                    alpha_mean=float(alpha_crop[region_mask].mean()) if region_mask.any() else 0.0,
                    alpha_max=float(alpha_crop[region_mask].max()) if region_mask.any() else 0.0,
                    height_mean=None,
                    height_std=None,
                    height_range=None,
                    normal_mean_xyz=None,
                    mcly_texture_ids=texture_ids,
                    mcly_active_layers=active_layers,
                    curation_label="blocky_paste",
                    rejection_reason=None,
                    linked_component_ids=[],
                    provenance={
                        "block_size": int(factor),
                        "min_block_coverage": float(min_block_coverage),
                        "block_close_radius": int(close_radius),
                        "block_extent": round(block_extent, 4),
                    },
                )
            )
    return regions


def segment_macro_pastes(
    canvas: zarr.Group,
    *,
    threshold: float = 0.05,
    close_radius: int = 32,
    min_area: int = 4096,
    min_footprint_px: int = 64,
    max_aspect_ratio: float = 12.0,
    max_regions_per_layer: int | None = 500,
    downsample_factor: int = 8,
) -> list[FractalRegion]:
    """Segment macro paste/scar objects by proximity grouping of alpha.

    This merges nearby brush strokes into paste-sized blobs, producing
    zone-sized macro objects rather than individual brush strokes. The
    closing radius controls how far apart strokes can be to still merge.

    The binary alpha is max-pooled by ``downsample_factor`` before dilation
    to keep the morphological operation fast on full-map canvases. Bboxes
    are scaled back to full resolution.
    """
    alpha = canvas["alpha_256"]
    tile_id_256 = canvas["tile_id_256"]
    layer_indices = canvas["alpha_layer_indices"][:].astype(np.int32).tolist() if "alpha_layer_indices" in canvas else list(range(alpha.shape[2]))
    layout = dict(canvas.attrs.get("layout", {}))
    build = str(layout.get("build", ""))
    map_name = str(layout.get("map_name", ""))

    factor = max(1, int(downsample_factor))
    ds_close_radius = max(1, int(close_radius) // factor)
    struct = np.ones((ds_close_radius * 2 + 1, ds_close_radius * 2 + 1), dtype=np.uint8)

    regions: list[FractalRegion] = []
    for layer_slot in range(alpha.shape[2]):
        layer_idx = int(layer_indices[layer_slot]) if layer_slot < len(layer_indices) else int(layer_slot)
        ds_binary = _downsample_alpha_layer_binary(alpha, layer_slot=layer_slot, threshold=float(threshold), factor=factor)
        if not ds_binary.any():
            continue

        h_full, w_full = int(alpha.shape[0]), int(alpha.shape[1])

        grouped_binary = binary_dilation(ds_binary, structure=struct) if ds_close_radius > 0 else ds_binary

        labeled, count = label(grouped_binary, structure=np.ones((3, 3), dtype=np.uint8))
        if count == 0:
            continue
        areas = nd_sum(grouped_binary, labeled, range(1, count + 1))
        objects = find_objects(labeled)
        candidates: list[tuple[int, int, tuple[slice, slice]]] = []
        for label_idx, raw_area in enumerate(areas, start=1):
            area = int(raw_area) * factor * factor
            bounds = objects[label_idx - 1]
            if bounds is None:
                continue
            candidates.append((label_idx, area, bounds))
        candidates.sort(key=lambda item: item[1], reverse=True)
        if max_regions_per_layer is not None:
            candidates = candidates[: max(0, int(max_regions_per_layer))]
        for _label_idx, _group_area, (slice_y, slice_x) in candidates:
            x0 = int(slice_x.start) * factor
            y0 = int(slice_y.start) * factor
            x1 = min(w_full, int(slice_x.stop) * factor)
            y1 = min(h_full, int(slice_y.stop) * factor)
            width = x1 - x0
            height_box = y1 - y0
            if width < int(min_footprint_px) or height_box < int(min_footprint_px):
                continue
            aspect = max(width, height_box) / max(1, min(width, height_box))
            if aspect > float(max_aspect_ratio):
                continue
            bbox = (x0, y0, width, height_box)
            alpha_crop = alpha[y0:y1, x0:x1, layer_slot].astype(np.float32)
            region_mask = alpha_crop > float(threshold)
            raw_area = int(np.count_nonzero(region_mask))
            if raw_area < int(min_area):
                continue
            tile_coverage = _tile_coverage(tile_id_256[y0:y1, x0:x1].astype(np.int32), region_mask)
            texture_ids, active_layers = _mcly_summary(
                canvas["mcly_texture_ids"][:].astype(np.int32) if "mcly_texture_ids" in canvas else None,
                canvas["mcly_layer_mask"][:].astype(np.float32) if "mcly_layer_mask" in canvas else None,
                bbox,
            )
            region_id = _region_id(build, map_name, layer_idx, bbox, raw_area)
            regions.append(
                FractalRegion(
                    region_id=region_id,
                    build=build,
                    map_name=map_name,
                    layer_slot=int(layer_slot),
                    layer_idx=layer_idx,
                    bbox_xywh=bbox,
                    area=raw_area,
                    tile_coverage_count=len(tile_coverage),
                    tile_coverage=tile_coverage,
                    alpha_mean=float(alpha_crop[region_mask].mean()) if region_mask.any() else 0.0,
                    alpha_max=float(alpha_crop[region_mask].max()) if region_mask.any() else 0.0,
                    height_mean=None,
                    height_std=None,
                    height_range=None,
                    normal_mean_xyz=None,
                    mcly_texture_ids=texture_ids,
                    mcly_active_layers=active_layers,
                    curation_label="macro_paste",
                    rejection_reason=None,
                    linked_component_ids=[],
                    provenance={"close_radius": int(close_radius), "downsample": int(factor), "extent": round(float(raw_area) / float(max(1, width * height_box)), 4)},
                )
            )
    return regions


def classify_region(
    *,
    bbox_xywh: tuple[int, int, int, int],
    area: int,
    total_pixels: int,
    tile_coverage_count: int,
    alpha_mean: float,
    min_atomic_footprint_px: int = 8,
    chonker_area_fraction: float = 0.18,
    one_off_min_area: int = 4096,
) -> tuple[str, str | None]:
    """Assign conservative curation labels for review-safe training gates.

    Chonkers are preserved for composite-canvas harvesting, but default atomic
    samples must span a physically meaningful ADT footprint instead of tiny
    connected alpha slivers.
    """
    _x, _y, width, height = bbox_xywh
    bbox_area = max(1, int(width) * int(height))
    area_fraction = float(area) / float(max(1, total_pixels))
    fill_fraction = float(area) / float(bbox_area)
    if area < 16:
        return "too_small_unique", "area_below_minimum"
    if area_fraction >= float(chonker_area_fraction) or (width >= 768 and height >= 768):
        return "composite_chonker", "large_full_map_region"
    if width < int(min_atomic_footprint_px) or height < int(min_atomic_footprint_px):
        return "too_small_unique", "below_minimum_adt_footprint"
    if tile_coverage_count <= 1 and area >= int(one_off_min_area) and fill_fraction < 0.35:
        return "one_off_detail", "large_sparse_single_tile_region"
    if tile_coverage_count > 1:
        return "fractal_member", None
    if alpha_mean < 0.08:
        return "rejected_unknown", "weak_alpha_signal"
    return "accepted_candidate", None


def save_regions(path: str | Path, regions: list[FractalRegion]) -> None:
    rows = [_region_to_parquet_row(region) for region in regions]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)


def save_regions_jsonl(path: str | Path, regions: list[FractalRegion]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for region in regions:
            handle.write(json.dumps(_json_ready(asdict(region)), sort_keys=True) + "\n")


def _region_to_parquet_row(region: FractalRegion) -> dict[str, Any]:
    row = _json_ready(asdict(region))
    provenance = row.get("provenance")
    if isinstance(provenance, dict):
        row["provenance"] = json.dumps(provenance, sort_keys=True) if provenance else None
    return row


def render_region_overlay(
    canvas: zarr.Group,
    regions: list[FractalRegion],
    out_path: str | Path,
    *,
    max_regions: int = 200,
) -> None:
    alpha = canvas["alpha_256"][:].astype(np.float32)
    base = np.clip(alpha.max(axis=2), 0.0, 1.0)
    image = Image.fromarray((base * 255.0).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    colors = {
        "accepted_candidate": (80, 220, 80),
        "fractal_member": (80, 160, 255),
        "composite_chonker": (255, 80, 80),
        "one_off_detail": (255, 190, 60),
        "too_small_unique": (180, 180, 180),
        "rejected_unknown": (220, 80, 220),
        "raw_component": (80, 220, 80),
    }
    for region in sorted(regions, key=lambda item: item.area, reverse=True)[: max(0, int(max_regions))]:
        x, y, w, h = region.bbox_xywh
        color = colors.get(region.curation_label, (255, 255, 255))
        draw.rectangle((x, y, x + w - 1, y + h - 1), outline=color, width=2)
        draw.text((x + 2, y + 2), f"L{region.layer_idx} {region.curation_label}", fill=color)
    max_preview_side = 2048
    if max(image.size) > max_preview_side:
        image.thumbnail((max_preview_side, max_preview_side), Image.Resampling.NEAREST)
    output_path = Path(out_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)


def load_catalog_rows(path: str | Path | None) -> list[dict[str, Any]]:
    if path is None:
        return []
    catalog_path = Path(path)
    if catalog_path.is_dir():
        catalog_path = catalog_path / "catalog.jsonl"
    if not catalog_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with catalog_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _tile_coverage(tile_ids: np.ndarray, mask: np.ndarray) -> list[dict[str, int]]:
    values, counts = np.unique(tile_ids[mask], return_counts=True)
    coverage = [
        {"tile_id": int(value), "pixel_count": int(count)}
        for value, count in zip(values.tolist(), counts.tolist(), strict=True)
        if int(value) >= 0
    ]
    coverage.sort(key=lambda item: (-item["pixel_count"], item["tile_id"]))
    return coverage


def _height_stats(height: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float | None, float | None, float | None]:
    x, y, w, h = bbox
    crop = height[y : y + h + 1, x : x + w + 1]
    if crop.size == 0:
        return None, None, None
    return float(crop.mean()), float(crop.std()), float(crop.max() - crop.min())


def _normal_mean(normals: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[float, float, float] | None:
    x, y, w, h = bbox
    crop = normals[y : y + h + 1, x : x + w + 1, :]
    if crop.size == 0:
        return None
    mean = crop.reshape(-1, 3).mean(axis=0)
    return (float(mean[0]), float(mean[1]), float(mean[2]))


def _mcly_summary(mcly_ids: np.ndarray | None, mcly_mask: np.ndarray | None, bbox: tuple[int, int, int, int]) -> tuple[list[int], list[int]]:
    x, y, w, h = bbox
    mx0, my0 = x // 16, y // 16
    mx1, my1 = max(mx0 + 1, (x + w + 15) // 16), max(my0 + 1, (y + h + 15) // 16)
    texture_ids: list[int] = []
    active_layers: list[int] = []
    if mcly_ids is not None:
        ids = mcly_ids[my0:my1, mx0:mx1, :]
        values = sorted({int(value) for value in ids.reshape(-1).tolist() if int(value) >= 0})
        texture_ids = values[:32]
    if mcly_mask is not None:
        mask = mcly_mask[my0:my1, mx0:mx1, :]
        active_layers = [int(idx) for idx in np.where(mask.max(axis=(0, 1)) > 0.05)[0].tolist()]
    return texture_ids, active_layers


def _build_catalog_index(rows: list[dict[str, Any]]) -> dict[tuple[int, int], list[dict[str, Any]]]:
    out: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row in rows:
        try:
            key = (int(row.get("tile_id", -1)), int(row.get("layer_idx", -1)))
        except (TypeError, ValueError):
            continue
        out.setdefault(key, []).append(row)
    return out


def _linked_component_ids(
    catalog_index: dict[tuple[int, int], list[dict[str, Any]]],
    layer_idx: int,
    canvas_bbox: tuple[int, int, int, int],
    tile_coverage: list[dict[str, int]],
) -> list[str]:
    if not catalog_index:
        return []
    linked: list[str] = []
    cx, cy, cw, ch = canvas_bbox
    for coverage in tile_coverage:
        tile_id = int(coverage["tile_id"])
        for row in catalog_index.get((tile_id, int(layer_idx)), []):
            bbox = row.get("bbox_xywh", [])
            if not isinstance(bbox, list | tuple) or len(bbox) != 4:
                continue
            # 074 rows are tile-local; use permissive bbox size overlap because Phase 1
            # canvas_index does not yet encode every tile's alpha origin here.
            if _bbox_intersects((0, 0, cw, ch), tuple(int(v) for v in bbox)):
                linked.append(str(row.get("component_id", "")))
    return sorted({value for value in linked if value})[:64]


def _bbox_intersects(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> bool:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    return ax < bx + bw and ax + aw > bx and ay < by + bh and ay + ah > by


def _region_id(build: str, map_name: str, layer_idx: int, bbox: tuple[int, int, int, int], area: int) -> str:
    payload = f"{build}|{map_name}|{layer_idx}|{bbox}|{area}"
    return "fr_" + hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def segment_height_discontinuities(
    canvas: zarr.Group,
    *,
    threshold: float = 0.05,
    height_jump_threshold: float = 8.0,
    min_area: int = 1024,
    min_footprint_px: int = 32,
    max_aspect_ratio: float = 12.0,
    max_regions_per_layer: int | None = 500,
) -> list[FractalRegion]:
    """Segment regions based on heightmap discontinuities at tile boundaries.

    This detects prefabricated terrain boundaries by finding large height jumps
    between adjacent tiles. The height_257 array has one extra row/col per tile,
    so tile boundaries align with every 256th pixel (plus the shared edge).

    Args:
        canvas: Full-map canvas with height_257 array
        threshold: Alpha threshold for masking (only analyze painted areas)
        height_jump_threshold: Minimum height difference to flag as discontinuity
        min_area: Minimum region area in alpha pixels
        min_footprint_px: Minimum bbox width/height
        max_aspect_ratio: Maximum aspect ratio
        max_regions_per_layer: Max regions per layer

    Returns:
        List of FractalRegion objects with curation_label="height_discontinuity"
    """
    alpha = canvas["alpha_256"]
    height = canvas["height_257"][:].astype(np.float32) if "height_257" in canvas else None
    tile_id_256 = canvas["tile_id_256"]
    layer_indices = canvas["alpha_layer_indices"][:].astype(np.int32).tolist() if "alpha_layer_indices" in canvas else list(range(alpha.shape[2]))
    layout = dict(canvas.attrs.get("layout", {}))
    build = str(layout.get("build", ""))
    map_name = str(layout.get("map_name", ""))

    if height is None:
        return []

    h_alpha, w_alpha = int(alpha.shape[0]), int(alpha.shape[1])
    h_height, w_height = int(height.shape[0]), int(height.shape[1])

    # Height array is 257x257 per tile, alpha is 256x256 per tile.
    # Crop height to match alpha dimensions for gradient computation.
    # The extra row/col in height is the shared vertex edge between tiles.
    height_cropped = height[:h_alpha, :w_alpha]

    # Compute height gradients (finite differences) on cropped height
    # dy: vertical gradient (height change going down)
    # dx: horizontal gradient (height change going right)
    dy = np.abs(np.diff(height_cropped, axis=0, prepend=height_cropped[:1, :]))
    dx = np.abs(np.diff(height_cropped, axis=1, prepend=height_cropped[:, :1]))

    # Gradient magnitude
    grad_mag = np.sqrt(dx * dx + dy * dy)

    # Mask by alpha (only care about painted terrain)
    alpha_max = alpha[:].max(axis=2).astype(np.float32)
    alpha_mask = alpha_max > float(threshold)
    grad_mag = np.where(alpha_mask, grad_mag, 0.0)

    # Binary mask of significant height jumps
    jump_binary = grad_mag >= float(height_jump_threshold)

    # Dilate slightly to connect nearby jump pixels into regions
    struct = np.ones((3, 3), dtype=np.uint8)
    jump_binary = binary_dilation(jump_binary, structure=struct, border_value=0)

    regions: list[FractalRegion] = []
    for layer_slot in range(alpha.shape[2]):
        layer_idx = int(layer_indices[layer_slot]) if layer_slot < len(layer_indices) else int(layer_slot)
        layer_alpha = alpha[:, :, layer_slot].astype(np.float32)
        layer_binary = layer_alpha > float(threshold)

        # Intersect height jumps with this layer's painted area
        combined_binary = jump_binary & layer_binary
        if not combined_binary.any():
            continue

        labeled, count = label(combined_binary, structure=np.ones((3, 3), dtype=np.uint8))
        if count == 0:
            continue

        areas = nd_sum(combined_binary, labeled, range(1, count + 1))
        objects = find_objects(labeled)
        candidates: list[tuple[int, int, tuple[slice, slice]]] = []
        for label_idx, raw_area in enumerate(areas, start=1):
            area = int(raw_area)
            if area < int(min_area):
                continue
            bounds = objects[label_idx - 1]
            if bounds is None:
                continue
            candidates.append((label_idx, area, bounds))

        candidates.sort(key=lambda item: item[1], reverse=True)
        if max_regions_per_layer is not None:
            candidates = candidates[: max(0, int(max_regions_per_layer))]

        for label_idx, area, (slice_y, slice_x) in candidates:
            x0, x1 = int(slice_x.start), int(slice_x.stop)
            y0, y1 = int(slice_y.start), int(slice_y.stop)
            width = x1 - x0
            height_box = y1 - y0
            if width < int(min_footprint_px) or height_box < int(min_footprint_px):
                continue
            aspect = max(width, height_box) / max(1, min(width, height_box))
            if aspect > float(max_aspect_ratio):
                continue

            bbox = (x0, y0, width, height_box)
            alpha_crop = layer_alpha[y0:y1, x0:x1]
            region_mask = combined_binary[y0:y1, x0:x1]
            raw_area = int(np.count_nonzero(region_mask))
            if raw_area < int(min_area):
                continue

            tile_coverage = _tile_coverage(tile_id_256[y0:y1, x0:x1].astype(np.int32), region_mask)

            # Height stats for this region
            height_crop = height[y0:y1+1, x0:x1+1]
            height_mean = float(height_crop.mean()) if height_crop.size else None
            height_std = float(height_crop.std()) if height_crop.size else None
            height_range = float(height_crop.max() - height_crop.min()) if height_crop.size else None

            texture_ids, active_layers = _mcly_summary(
                canvas["mcly_texture_ids"][:].astype(np.int32) if "mcly_texture_ids" in canvas else None,
                canvas["mcly_layer_mask"][:].astype(np.float32) if "mcly_layer_mask" in canvas else None,
                bbox,
            )

            # Compute max gradient within region as provenance
            grad_crop = grad_mag[y0:y1, x0:x1]
            max_grad = float(grad_crop.max()) if grad_crop.size else 0.0
            mean_grad = float(grad_crop[region_mask].mean()) if region_mask.any() else 0.0

            region_id = _region_id(build, map_name, layer_idx, bbox, raw_area)
            regions.append(
                FractalRegion(
                    region_id=region_id,
                    build=build,
                    map_name=map_name,
                    layer_slot=int(layer_slot),
                    layer_idx=layer_idx,
                    bbox_xywh=bbox,
                    area=raw_area,
                    tile_coverage_count=len(tile_coverage),
                    tile_coverage=tile_coverage,
                    alpha_mean=float(alpha_crop[region_mask].mean()) if region_mask.any() else 0.0,
                    alpha_max=float(alpha_crop[region_mask].max()) if region_mask.any() else 0.0,
                    height_mean=height_mean,
                    height_std=height_std,
                    height_range=height_range,
                    normal_mean_xyz=None,
                    mcly_texture_ids=texture_ids,
                    mcly_active_layers=active_layers,
                    curation_label="height_discontinuity",
                    rejection_reason=None,
                    linked_component_ids=[],
                    provenance={
                        "height_jump_threshold": float(height_jump_threshold),
                        "max_gradient": round(max_grad, 4),
                        "mean_gradient": round(mean_grad, 4),
                    },
                )
            )
    return regions


def _json_ready(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    return value
