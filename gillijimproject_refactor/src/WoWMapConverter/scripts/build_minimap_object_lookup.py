from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image


TILE_SIZE = 533.33333
MAP_ORIGIN = 32.0 * TILE_SIZE
SHADOW_TILE_RESOLUTION = 1024.0


@dataclass
class LookupStats:
    tiles_scanned: int = 0
    tiles_with_objects: int = 0
    objects_seen: int = 0
    objects_exported: int = 0
    exported_visibility_masks: int = 0
    shadow_region_masks: int = 0
    fallback_masks: int = 0


def tile_uv_candidates(world_a: float, world_b: float, tile_x: int, tile_y: int) -> List[Tuple[float, float]]:
    return [
        (world_a / TILE_SIZE - float(tile_x), world_b / TILE_SIZE - float(tile_y)),
        ((MAP_ORIGIN - world_b) / TILE_SIZE - float(tile_x), (MAP_ORIGIN - world_a) / TILE_SIZE - float(tile_y)),
    ]


def parse_tile_xy(tile_name: str) -> Optional[Tuple[int, int]]:
    match = re.search(r"_(\d+)_(\d+)$", tile_name)
    if not match:
        return None
    return int(match.group(1)), int(match.group(2))


def resolve_dataset_path(dataset_root: Path, relative_or_absolute: Optional[str]) -> Optional[Path]:
    if not relative_or_absolute:
        return None
    candidate = Path(relative_or_absolute)
    if candidate.is_absolute():
        return candidate
    return dataset_root / candidate


def sanitize_name(name: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", name).strip("_")
    return cleaned or "unknown_object"


def choose_object_center_px(obj: dict, tile_x: int, tile_y: int, output_size: int) -> Optional[Tuple[int, int]]:
    pos_x = float(obj.get("x", obj.get("pos_x", 0.0)))
    pos_y = float(obj.get("y", obj.get("pos_y", 0.0)))
    pos_z = float(obj.get("z", obj.get("pos_z", pos_y)))

    candidates: List[Tuple[float, float]] = []
    candidates.extend(tile_uv_candidates(pos_x, pos_z, tile_x, tile_y))
    candidates.extend(tile_uv_candidates(pos_x, pos_y, tile_x, tile_y))

    best: Optional[Tuple[float, float]] = None
    best_overflow = float("inf")
    for uv_x, uv_y in candidates:
        overflow = max(0.0, -uv_x) + max(0.0, uv_x - 1.0) + max(0.0, -uv_y) + max(0.0, uv_y - 1.0)
        if overflow < best_overflow:
            best = (uv_x, uv_y)
            best_overflow = overflow
            if overflow <= 1e-6:
                break

    if best is None:
        return None

    uv_x, uv_y = best
    if uv_x < -0.25 or uv_x > 1.25 or uv_y < -0.25 or uv_y > 1.25:
        return None

    center_x = int(round(np.clip(uv_x, 0.0, 1.0) * (output_size - 1)))
    center_y = int(round(np.clip(uv_y, 0.0, 1.0) * (output_size - 1)))
    return center_x, center_y


def estimate_radius_pixels(obj: dict, output_size: int) -> int:
    pixels_per_world = output_size / TILE_SIZE
    scale = float(obj.get("scale", 1.0) or 1.0)
    if not np.isfinite(scale) or scale <= 0.0:
        scale = 1.0

    bounds_min = obj.get("bounds_min")
    bounds_max = obj.get("bounds_max")
    if bounds_min and bounds_max and len(bounds_min) >= 3 and len(bounds_max) >= 3:
        half_width_world = abs(float(bounds_max[0]) - float(bounds_min[0])) * 0.5 * scale
        half_depth_world = abs(float(bounds_max[2]) - float(bounds_min[2])) * 0.5 * scale
        radius = int(round(max(half_width_world, half_depth_world) * pixels_per_world))
        return max(2, radius)

    category = str(obj.get("category", "")).lower()
    base_world = 6.0 if "wmo" in category else 3.0
    radius = int(round(base_world * scale * pixels_per_world))
    return max(2, radius)


def build_fallback_mask(output_size: int, center_x: int, center_y: int, radius: int) -> np.ndarray:
    yy, xx = np.ogrid[:output_size, :output_size]
    dist2 = (xx - center_x) ** 2 + (yy - center_y) ** 2
    return (dist2 <= (radius * radius)).astype(np.uint8)


def build_shadow_region_mask(terrain: dict, unique_id: int, output_size: int) -> Optional[np.ndarray]:
    shadow_analysis = terrain.get("shadow_analysis") or []
    if not isinstance(shadow_analysis, list) or not shadow_analysis:
        return None

    mask = np.zeros((output_size, output_size), dtype=np.uint8)
    scale = output_size / SHADOW_TILE_RESOLUTION
    matched = False

    for chunk in shadow_analysis:
        if not isinstance(chunk, dict):
            continue

        chunk_idx = int(chunk.get("idx", -1))
        if chunk_idx < 0:
            continue

        chunk_x = chunk_idx % 16
        chunk_y = chunk_idx // 16
        regions = chunk.get("regions") or []
        if not isinstance(regions, list):
            continue

        for region in regions:
            if not isinstance(region, dict):
                continue
            candidate_ids = region.get("candidate_object_ids") or []
            if not isinstance(candidate_ids, list):
                continue

            if unique_id not in [int(v) for v in candidate_ids if isinstance(v, (int, float))]:
                continue

            bbox_min = region.get("bbox_min_px")
            bbox_max = region.get("bbox_max_px")
            if not (isinstance(bbox_min, list) and isinstance(bbox_max, list) and len(bbox_min) >= 2 and len(bbox_max) >= 2):
                continue

            x0_shadow = chunk_x * 64 + int(bbox_min[0])
            y0_shadow = chunk_y * 64 + int(bbox_min[1])
            x1_shadow = chunk_x * 64 + int(bbox_max[0])
            y1_shadow = chunk_y * 64 + int(bbox_max[1])

            x0 = int(np.clip(np.floor(x0_shadow * scale), 0, output_size - 1))
            y0 = int(np.clip(np.floor(y0_shadow * scale), 0, output_size - 1))
            x1 = int(np.clip(np.ceil((x1_shadow + 1) * scale), 0, output_size))
            y1 = int(np.clip(np.ceil((y1_shadow + 1) * scale), 0, output_size))

            if x1 <= x0 or y1 <= y0:
                continue

            mask[y0:y1, x0:x1] = 1
            matched = True

    return mask if matched else None


def load_tile_object_visibility_mask(terrain: dict, dataset_root: Path, output_size: int) -> Optional[np.ndarray]:
    mask_path = resolve_dataset_path(dataset_root, terrain.get("object_visibility_mask"))
    if mask_path is None or not mask_path.exists():
        return None

    with Image.open(mask_path).convert("L") as image:
        if image.size != (output_size, output_size):
            image = image.resize((output_size, output_size), Image.NEAREST)
        return (np.asarray(image, dtype=np.uint8) > 0).astype(np.uint8)


def extract_mask_from_exported_visibility(
    exported_mask: np.ndarray,
    center_x: int,
    center_y: int,
    radius: int,
) -> Optional[np.ndarray]:
    if exported_mask.size == 0 or not np.any(exported_mask > 0):
        return None

    local_radius = max(4, radius * 3)
    window = build_fallback_mask(exported_mask.shape[0], center_x, center_y, local_radius)
    isolated = np.where((exported_mask > 0) & (window > 0), 1, 0).astype(np.uint8)
    if not np.any(isolated > 0):
        return None

    return isolated


def bounding_rect(mask: np.ndarray) -> Optional[Tuple[int, int, int, int]]:
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def iter_json_files(dataset_roots: Sequence[Path], tile_limit: Optional[int]) -> Iterable[Tuple[Path, Path]]:
    yielded = 0
    for dataset_root in dataset_roots:
        dataset_dir = dataset_root / "dataset"
        if not dataset_dir.exists():
            continue
        for json_path in sorted(dataset_dir.glob("*.json")):
            yield dataset_root, json_path
            yielded += 1
            if tile_limit is not None and yielded >= tile_limit:
                return


def build_lookup_library(
    dataset_roots: Sequence[Path],
    output_dir: Path,
    categories: Sequence[str],
    tile_limit: Optional[int],
    crop_padding: int,
) -> LookupStats:
    stats = LookupStats()
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "lookup_manifest.jsonl"

    categories_normalized = {c.lower() for c in categories}

    with manifest_path.open("w", encoding="utf-8") as manifest:
        for dataset_root, json_path in iter_json_files(dataset_roots, tile_limit):
            stats.tiles_scanned += 1

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            terrain = payload.get("terrain_data", {})
            objects = terrain.get("objects") or []
            if not isinstance(objects, list) or not objects:
                continue
            stats.tiles_with_objects += 1

            minimap_path = resolve_dataset_path(dataset_root, payload.get("image"))
            if minimap_path is None or not minimap_path.exists():
                continue

            tile_name = str(terrain.get("adt_tile") or json_path.stem)
            parsed = parse_tile_xy(tile_name)
            if parsed is None:
                continue
            tile_x, tile_y = parsed

            with Image.open(minimap_path).convert("RGB") as minimap:
                output_size = minimap.width
                if minimap.width != minimap.height:
                    continue

                exported_visibility_mask = load_tile_object_visibility_mask(terrain, dataset_root, output_size)

                for obj_index, obj in enumerate(objects):
                    if not isinstance(obj, dict):
                        continue
                    stats.objects_seen += 1

                    category = str(obj.get("category", "")).lower()
                    if categories_normalized and category not in categories_normalized:
                        continue

                    center = choose_object_center_px(obj, tile_x, tile_y, output_size)
                    if center is None:
                        continue
                    center_x, center_y = center
                    unique_id = int(obj.get("unique_id", 0) or 0)

                    radius = estimate_radius_pixels(obj, output_size)
                    if exported_visibility_mask is not None:
                        exported_mask = extract_mask_from_exported_visibility(
                            exported_visibility_mask,
                            center_x,
                            center_y,
                            radius,
                        )
                        if exported_mask is not None and np.any(exported_mask > 0):
                            mask = exported_mask
                            mask_source = "exported_visibility"
                            stats.exported_visibility_masks += 1
                        else:
                            mask = None
                            mask_source = ""
                    else:
                        mask = None
                        mask_source = ""

                    if mask is None:
                        shadow_mask = build_shadow_region_mask(terrain, unique_id, output_size)
                        if shadow_mask is not None and np.any(shadow_mask > 0):
                            mask = shadow_mask
                            mask_source = "shadow_region"
                            stats.shadow_region_masks += 1
                        else:
                            mask = build_fallback_mask(output_size, center_x, center_y, radius)
                            mask_source = "placement_fallback"
                            stats.fallback_masks += 1

                    bbox = bounding_rect(mask)
                    if bbox is None:
                        continue

                    x0, y0, x1, y1 = bbox
                    x0 = max(0, x0 - crop_padding)
                    y0 = max(0, y0 - crop_padding)
                    x1 = min(output_size, x1 + crop_padding)
                    y1 = min(output_size, y1 + crop_padding)
                    if x1 <= x0 or y1 <= y0:
                        continue

                    obj_name = sanitize_name(str(obj.get("name", "unknown_object")))
                    obj_dir = output_dir / obj_name
                    obj_dir.mkdir(parents=True, exist_ok=True)

                    stem = f"{tile_name}_{unique_id}_{obj_index}"
                    crop_rel = f"{obj_name}/{stem}.png"
                    mask_rel = f"{obj_name}/{stem}_mask.png"
                    crop_path = output_dir / crop_rel
                    mask_path = output_dir / mask_rel

                    crop_img = minimap.crop((x0, y0, x1, y1))
                    crop_img.save(crop_path)

                    mask_uint8 = (mask[y0:y1, x0:x1] * 255).astype(np.uint8)
                    Image.fromarray(mask_uint8, mode="L").save(mask_path)

                    record = {
                        "object_name": obj_name,
                        "category": category,
                        "tile": tile_name,
                        "tile_x": tile_x,
                        "tile_y": tile_y,
                        "unique_id": unique_id,
                        "json_path": str(json_path),
                        "minimap_path": str(minimap_path),
                        "crop_path": crop_rel,
                        "mask_path": mask_rel,
                        "mask_source": mask_source,
                        "bbox": [x0, y0, x1, y1],
                        "center_px": [center_x, center_y],
                    }
                    manifest.write(json.dumps(record) + "\n")
                    stats.objects_exported += 1

    summary = {
        "tiles_scanned": stats.tiles_scanned,
        "tiles_with_objects": stats.tiles_with_objects,
        "objects_seen": stats.objects_seen,
        "objects_exported": stats.objects_exported,
        "exported_visibility_masks": stats.exported_visibility_masks,
        "shadow_region_masks": stats.shadow_region_masks,
        "fallback_masks": stats.fallback_masks,
        "dataset_roots": [str(p) for p in dataset_roots],
        "categories": list(categories_normalized),
    }
    (output_dir / "lookup_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a minimap object reverse-lookup library from VLM exports.")
    parser.add_argument("--dataset-root", action="append", required=True, help="VLM dataset root (contains dataset/ and images/).")
    parser.add_argument("--output-dir", required=True, help="Output directory for object crops, masks, and manifest.")
    parser.add_argument(
        "--category",
        action="append",
        default=["wmo"],
        help="Object categories to include (repeatable). Defaults to wmo.",
    )
    parser.add_argument("--tile-limit", type=int, default=None, help="Optional max number of tiles to scan.")
    parser.add_argument("--crop-padding", type=int, default=8, help="Padding around object mask bbox when exporting crops.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset_roots = [Path(p).resolve() for p in args.dataset_root]
    output_dir = Path(args.output_dir).resolve()
    stats = build_lookup_library(
        dataset_roots=dataset_roots,
        output_dir=output_dir,
        categories=args.category,
        tile_limit=args.tile_limit,
        crop_padding=args.crop_padding,
    )
    print("Lookup build complete")
    print(f"  tiles_scanned:       {stats.tiles_scanned}")
    print(f"  tiles_with_objects:  {stats.tiles_with_objects}")
    print(f"  objects_seen:        {stats.objects_seen}")
    print(f"  objects_exported:    {stats.objects_exported}")
    print(f"  exported_visibility: {stats.exported_visibility_masks}")
    print(f"  shadow_region_masks: {stats.shadow_region_masks}")
    print(f"  fallback_masks:      {stats.fallback_masks}")
    print(f"  output_dir:          {output_dir}")


if __name__ == "__main__":
    main()
