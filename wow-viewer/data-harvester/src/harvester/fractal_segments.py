"""Full-map alpha/fractal segmentation helpers for spec 076 Phase 2."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image, ImageDraw
from scipy.ndimage import find_objects, label
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
    rows = [_json_ready(asdict(region)) for region in regions]
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), output_path)


def save_regions_jsonl(path: str | Path, regions: list[FractalRegion]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for region in regions:
            handle.write(json.dumps(_json_ready(asdict(region)), sort_keys=True) + "\n")


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
