#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image

from build_prefab_library import (
    PATCHES_PER_TILE,
    TILE_SIZE,
    build_prefab_object_records,
    normalize_model_key,
    parse_tile_coords,
    project_objects,
    select_prefab_object_cluster,
    select_nearby_projected_objects,
)

PATCH_WORLD_SIZE = TILE_SIZE / float(PATCHES_PER_TILE)
DEFAULT_HEIGHT_SCALE = 64.0
DEFAULT_OCCURRENCES_PER_PREFAB = 1
DEFAULT_MAX_PREFABS = 0


def load_json(path: Path) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def safe_slug(value: str) -> str:
    chars = [char.lower() if char.isalnum() else "-" for char in value]
    text = "".join(chars)
    while "--" in text:
        text = text.replace("--", "-")
    return text.strip("-") or "item"


def short_model_label(model_path: str) -> str:
    normalized = str(model_path or "").replace("\\", "/")
    stem = Path(normalized).stem or normalized or "object"
    stem = re.sub(r"[^0-9A-Za-z_\-]", "_", stem)
    return stem[:40] or "object"


def patch_bounds_to_pixels(
    patch_min_x: int,
    patch_min_y: int,
    patch_max_x: int,
    patch_max_y: int,
    image_width: int,
    image_height: int,
    padding_patches: float = 2.0,
) -> Tuple[int, int, int, int]:
    min_patch_x_f = max(0.0, float(patch_min_x) - padding_patches)
    min_patch_y_f = max(0.0, float(patch_min_y) - padding_patches)
    max_patch_x_f = min(float(PATCHES_PER_TILE), float(patch_max_x) + 1.0 + padding_patches)
    max_patch_y_f = min(float(PATCHES_PER_TILE), float(patch_max_y) + 1.0 + padding_patches)

    left = int(math.floor((min_patch_x_f / float(PATCHES_PER_TILE)) * float(image_width)))
    top = int(math.floor((min_patch_y_f / float(PATCHES_PER_TILE)) * float(image_height)))
    right = int(math.ceil((max_patch_x_f / float(PATCHES_PER_TILE)) * float(image_width)))
    bottom = int(math.ceil((max_patch_y_f / float(PATCHES_PER_TILE)) * float(image_height)))
    right = max(left + 1, min(right, image_width))
    bottom = max(top + 1, min(bottom, image_height))
    left = max(0, min(left, image_width - 1))
    top = max(0, min(top, image_height - 1))
    return left, top, right, bottom


def crop_source_texture(dataset_root: Path, occurrence: Dict[str, Any], output_path: Path) -> Optional[Path]:
    source_rel = str(occurrence.get("source_image_path") or "").strip()
    if not source_rel:
        return None
    source_path = dataset_root / Path(source_rel)
    if not source_path.exists():
        return None

    with Image.open(source_path) as image:
        crop_box = patch_bounds_to_pixels(
            patch_min_x=int(occurrence.get("patch_min_x", 0) or 0),
            patch_min_y=int(occurrence.get("patch_min_y", 0) or 0),
            patch_max_x=int(occurrence.get("patch_max_x", 0) or 0),
            patch_max_y=int(occurrence.get("patch_max_y", 0) or 0),
            image_width=image.width,
            image_height=image.height,
        )
        crop = image.convert("RGB").crop(crop_box)
        crop.save(output_path)
    return output_path


def build_patch_presence_mask(group_payload: Dict[str, Any]) -> np.ndarray:
    patch_width = int(group_payload.get("patch_width", 0) or 0)
    patch_height = int(group_payload.get("patch_height", 0) or 0)
    mask = np.zeros((patch_height, patch_width), dtype=bool)
    for patch in group_payload.get("patches", []):
        local_x = int(patch.get("x", -1))
        local_y = int(patch.get("y", -1))
        if 0 <= local_x < patch_width and 0 <= local_y < patch_height:
            mask[local_y, local_x] = True
    return mask


def parse_bounds_triplet(value: Any) -> Optional[Tuple[float, float, float]]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    try:
        parsed = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        return None
    return parsed


def rotate_point(point: Tuple[float, float, float], rot_x_deg: float, rot_y_deg: float, rot_z_deg: float) -> Tuple[float, float, float]:
    x, y, z = point

    rot_x = math.radians(rot_x_deg)
    rot_y = math.radians(rot_y_deg)
    rot_z = math.radians(rot_z_deg)

    cos_x = math.cos(rot_x)
    sin_x = math.sin(rot_x)
    y, z = (y * cos_x - z * sin_x, y * sin_x + z * cos_x)

    cos_y = math.cos(rot_y)
    sin_y = math.sin(rot_y)
    x, z = (x * cos_y + z * sin_y, -x * sin_y + z * cos_y)

    cos_z = math.cos(rot_z)
    sin_z = math.sin(rot_z)
    x, y = (x * cos_z - y * sin_z, x * sin_z + y * cos_z)

    return x, y, z


def resolve_occurrence_prefab_objects(
    occurrence: Dict[str, Any],
    tile_payload: Dict[str, Any],
    group_payload: Dict[str, Any],
) -> List[Dict[str, Any]]:
    prefab_objects = occurrence.get("prefab_objects")
    if isinstance(prefab_objects, list) and prefab_objects:
        return [entry for entry in prefab_objects if isinstance(entry, dict)]

    tile_name = str(occurrence.get("tile_name") or "")
    tile_x, tile_y = parse_tile_coords(tile_name)
    projected_objects = project_objects(tile_payload.get("terrain_data", {}).get("objects", []), tile_x, tile_y)
    selected_cluster, _ = select_prefab_object_cluster(projected_objects, group_payload)
    return build_prefab_object_records(selected_cluster)


def fallback_object_extents(obj: Dict[str, Any]) -> Tuple[Tuple[float, float, float], Tuple[float, float, float]]:
    scale = max(float(obj.get("scale", 1.0) or 1.0), 0.05)
    footprint = PATCH_WORLD_SIZE * max(0.75, min(scale * 1.5, 6.0))
    height = PATCH_WORLD_SIZE * max(2.0, min(scale * 6.0, 12.0))
    return (-footprint, 0.0, -footprint), (footprint, height, footprint)


def append_prefab_object_bounds(
    obj_lines: List[str],
    obj: Dict[str, Any],
    object_name: str,
    start_index: int,
) -> int:
    bounds_min = parse_bounds_triplet(obj.get("bounds_min"))
    bounds_max = parse_bounds_triplet(obj.get("bounds_max"))
    scale = max(float(obj.get("scale", 1.0) or 1.0), 0.05)
    local_x = float(obj.get("local_x", 0.0) or 0.0)
    local_y = float(obj.get("local_y", 0.0) or 0.0)
    local_z = float(obj.get("local_z", 0.0) or 0.0)

    if bounds_min is None or bounds_max is None:
        bounds_min, bounds_max = fallback_object_extents(obj)
    else:
        bounds_min = tuple(component * scale for component in bounds_min)
        bounds_max = tuple(component * scale for component in bounds_max)

    corners = [
        (bounds_min[0], bounds_min[1], bounds_min[2]),
        (bounds_max[0], bounds_min[1], bounds_min[2]),
        (bounds_max[0], bounds_min[1], bounds_max[2]),
        (bounds_min[0], bounds_min[1], bounds_max[2]),
        (bounds_min[0], bounds_max[1], bounds_min[2]),
        (bounds_max[0], bounds_max[1], bounds_min[2]),
        (bounds_max[0], bounds_max[1], bounds_max[2]),
        (bounds_min[0], bounds_max[1], bounds_max[2]),
    ]

    rot_x = float(obj.get("rot_x", 0.0) or 0.0)
    rot_y = float(obj.get("rot_y", 0.0) or 0.0)
    rot_z = float(obj.get("rot_z", 0.0) or 0.0)

    material = "wmo_bounds" if str(obj.get("category") or "").strip().lower() == "wmo" else "m2_bounds"
    obj_lines.append(f"o {object_name}")
    obj_lines.append(f"usemtl {material}")

    for corner in corners:
        rx, ry, rz = rotate_point(corner, rot_x, rot_y, rot_z)
        obj_lines.append(f"v {local_x + rx:.6f} {local_y + ry:.6f} {local_z + rz:.6f}")

    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 7, 6), (4, 6, 5),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    for face in faces:
        a = start_index + face[0]
        b = start_index + face[1]
        c = start_index + face[2]
        obj_lines.append(f"f {a} {b} {c}")

    return start_index + 8


def normalized_height_grid_to_array(group_payload: Dict[str, Any]) -> np.ndarray:
    width = int(group_payload.get("height_grid_width", 0) or 0)
    height = int(group_payload.get("height_grid_height", 0) or 0)
    values = np.asarray(group_payload.get("normalized_height_grid", []), dtype=np.float32)
    if width <= 1 or height <= 1 or values.size != width * height:
        raise ValueError("Group payload does not contain a valid normalized height grid.")
    return values.reshape(height, width)


def sample_height(height_grid: np.ndarray, local_patch_x: float, local_patch_y: float, height_scale: float) -> float:
    max_x = max(float(height_grid.shape[1] - 1), 1.0)
    max_y = max(float(height_grid.shape[0] - 1), 1.0)
    x = min(max(local_patch_x, 0.0), max_x)
    y = min(max(local_patch_y, 0.0), max_y)
    x0 = int(math.floor(x))
    y0 = int(math.floor(y))
    x1 = min(x0 + 1, height_grid.shape[1] - 1)
    y1 = min(y0 + 1, height_grid.shape[0] - 1)
    tx = x - float(x0)
    ty = y - float(y0)
    top = float(height_grid[y0, x0]) * (1.0 - tx) + float(height_grid[y0, x1]) * tx
    bottom = float(height_grid[y1, x0]) * (1.0 - tx) + float(height_grid[y1, x1]) * tx
    return (top * (1.0 - ty) + bottom * ty) * height_scale


def build_terrain_mesh(
    group_payload: Dict[str, Any],
    height_scale: float,
) -> Tuple[List[Tuple[float, float, float]], List[Tuple[float, float]], List[Tuple[int, int, int]], np.ndarray]:
    height_grid = normalized_height_grid_to_array(group_payload)
    patch_mask = build_patch_presence_mask(group_payload)
    height_count, width_count = height_grid.shape
    vertices: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    vertex_indices = np.zeros((height_count, width_count), dtype=np.int32)

    world_width = max(float(width_count - 1), 1.0) * PATCH_WORLD_SIZE
    world_height = max(float(height_count - 1), 1.0) * PATCH_WORLD_SIZE

    for row in range(height_count):
        for col in range(width_count):
            x_world = float(col) * PATCH_WORLD_SIZE
            z_world = float(row) * PATCH_WORLD_SIZE
            y_world = float(height_grid[row, col]) * height_scale
            vertices.append((x_world, y_world, z_world))
            uvs.append((float(col) / max(float(width_count - 1), 1.0), 1.0 - float(row) / max(float(height_count - 1), 1.0)))
            vertex_indices[row, col] = len(vertices)

    for patch_y in range(patch_mask.shape[0]):
        for patch_x in range(patch_mask.shape[1]):
            if not patch_mask[patch_y, patch_x]:
                continue
            v00 = int(vertex_indices[patch_y, patch_x])
            v10 = int(vertex_indices[patch_y, patch_x + 1])
            v11 = int(vertex_indices[patch_y + 1, patch_x + 1])
            v01 = int(vertex_indices[patch_y + 1, patch_x])
            faces.append((v00, v01, v11))
            faces.append((v00, v11, v10))

    return vertices, uvs, faces, height_grid


def append_marker_box(
    obj_lines: List[str],
    origin_x: float,
    origin_y: float,
    origin_z: float,
    size_x: float,
    size_y: float,
    size_z: float,
    object_name: str,
    start_index: int,
) -> int:
    x0 = origin_x - size_x
    x1 = origin_x + size_x
    y0 = origin_y
    y1 = origin_y + size_y
    z0 = origin_z - size_z
    z1 = origin_z + size_z
    obj_lines.append(f"o {object_name}")
    obj_lines.append("usemtl object_marker")
    box_vertices = [
        (x0, y0, z0),
        (x1, y0, z0),
        (x1, y0, z1),
        (x0, y0, z1),
        (x0, y1, z0),
        (x1, y1, z0),
        (x1, y1, z1),
        (x0, y1, z1),
    ]
    for vertex in box_vertices:
        obj_lines.append(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")
    faces = [
        (0, 1, 2), (0, 2, 3),
        (4, 7, 6), (4, 6, 5),
        (0, 4, 5), (0, 5, 1),
        (1, 5, 6), (1, 6, 2),
        (2, 6, 7), (2, 7, 3),
        (3, 7, 4), (3, 4, 0),
    ]
    for face in faces:
        a = start_index + face[0]
        b = start_index + face[1]
        c = start_index + face[2]
        obj_lines.append(f"f {a} {b} {c}")
    return start_index + 8


def write_prefab_obj_bundle(
    prefab_output_dir: Path,
    prefab_id: str,
    occurrence: Dict[str, Any],
    group_payload: Dict[str, Any],
    tile_payload: Dict[str, Any],
    dataset_root: Path,
    height_scale: float,
    prefab_occurrence_count: int,
    include_terrain_context: bool,
) -> Dict[str, Any]:
    prefab_output_dir.mkdir(parents=True, exist_ok=True)
    obj_path = prefab_output_dir / f"{prefab_id}.obj"
    mtl_path = prefab_output_dir / f"{prefab_id}.mtl"
    texture_path = prefab_output_dir / f"{prefab_id}_texture.png"

    texture_written: Optional[Path] = None
    vertices: List[Tuple[float, float, float]] = []
    uvs: List[Tuple[float, float]] = []
    faces: List[Tuple[int, int, int]] = []
    height_grid: Optional[np.ndarray] = None
    if include_terrain_context:
        texture_written = crop_source_texture(dataset_root, occurrence, texture_path)
        vertices, uvs, faces, height_grid = build_terrain_mesh(group_payload, height_scale)

    obj_lines: List[str] = [
        f"# Prefab object assembly for {prefab_id}",
        f"mtllib {mtl_path.name}",
    ]

    if include_terrain_context:
        obj_lines.append("o terrain_context")
        for vertex in vertices:
            obj_lines.append(f"v {vertex[0]:.6f} {vertex[1]:.6f} {vertex[2]:.6f}")
        for uv in uvs:
            obj_lines.append(f"vt {uv[0]:.6f} {uv[1]:.6f}")
        obj_lines.append("usemtl terrain")
        for face in faces:
            obj_lines.append(f"f {face[0]}/{face[0]} {face[1]}/{face[1]} {face[2]}/{face[2]}")

    prefab_objects = resolve_occurrence_prefab_objects(occurrence, tile_payload, group_payload)
    tile_name = str(occurrence.get("tile_name") or "")
    next_vertex_index = len(vertices) + 1
    object_metadata: List[Dict[str, Any]] = []

    for object_index, obj in enumerate(prefab_objects, start=1):
        object_name = f"object_{object_index:03d}_{short_model_label(str(obj.get('model_path') or obj.get('model_key') or obj.get('name') or 'object'))}"
        next_vertex_index = append_prefab_object_bounds(
            obj_lines=obj_lines,
            obj=obj,
            object_name=object_name,
            start_index=next_vertex_index,
        )
        object_metadata.append(
            {
                "name": object_name,
                "model_key": str(obj.get("model_key") or ""),
                "model_path": str(obj.get("model_path") or ""),
                "category": str(obj.get("category") or ""),
                "scale": float(obj.get("scale", 1.0) or 1.0),
                "local_x": float(obj.get("local_x", 0.0) or 0.0),
                "local_y": float(obj.get("local_y", 0.0) or 0.0),
                "local_z": float(obj.get("local_z", 0.0) or 0.0),
                "rot_x": float(obj.get("rot_x", 0.0) or 0.0),
                "rot_y": float(obj.get("rot_y", 0.0) or 0.0),
                "rot_z": float(obj.get("rot_z", 0.0) or 0.0),
                "bounds_min": obj.get("bounds_min"),
                "bounds_max": obj.get("bounds_max"),
            }
        )

    with open(obj_path, "w", encoding="utf-8") as handle:
        handle.write("\n".join(obj_lines) + "\n")

    with open(mtl_path, "w", encoding="utf-8") as handle:
        handle.write("# Prefab OBJ materials\n")
        if include_terrain_context:
            handle.write("newmtl terrain\n")
            handle.write("Ka 1.0 1.0 1.0\n")
            handle.write("Kd 1.0 1.0 1.0\n")
            handle.write("Ks 0.0 0.0 0.0\n")
            handle.write("d 1.0\n")
            handle.write("illum 1\n")
            if texture_written is not None:
                handle.write(f"map_Kd {texture_written.name}\n")
            handle.write("\n")

        handle.write("newmtl m2_bounds\n")
        handle.write("Ka 0.18 0.42 0.20\n")
        handle.write("Kd 0.28 0.74 0.36\n")
        handle.write("Ks 0.0 0.0 0.0\n")
        handle.write("d 1.0\n")
        handle.write("illum 1\n\n")
        handle.write("newmtl wmo_bounds\n")
        handle.write("Ka 0.42 0.22 0.12\n")
        handle.write("Kd 0.85 0.50 0.20\n")
        handle.write("Ks 0.0 0.0 0.0\n")
        handle.write("d 1.0\n")
        handle.write("illum 1\n")

    metadata = {
        "schema_version": "wowviewer-ml-prefab-obj-bundle.v2",
        "prefab_kind": "object-assembly",
        "prefab_id": prefab_id,
        "tile_name": tile_name,
        "group_id": occurrence.get("group_id"),
        "occurrence_count": int(prefab_occurrence_count),
        "height_grid_width": int(group_payload.get("height_grid_width", 0) or 0),
        "height_grid_height": int(group_payload.get("height_grid_height", 0) or 0),
        "patch_width": int(group_payload.get("patch_width", 0) or 0),
        "patch_height": int(group_payload.get("patch_height", 0) or 0),
        "terrain_vertices": len(vertices),
        "terrain_faces": len(faces),
        "terrain_context_included": include_terrain_context,
        "object_instances": object_metadata,
        "cluster_object_count": len(object_metadata),
        "texture_path": texture_written.name if texture_written is not None else None,
        "obj_path": obj_path.name,
        "mtl_path": mtl_path.name,
        "height_scale": height_scale,
    }
    with open(prefab_output_dir / f"{prefab_id}.json", "w", encoding="utf-8") as handle:
        json.dump(metadata, handle, indent=2)
    return metadata


def export_prefab_obj_library(
    prefab_library_dir: Path,
    output_dir: Path,
    occurrences_per_prefab: int,
    max_prefabs: int,
    height_scale: float,
    include_terrain_context: bool,
) -> Dict[str, Any]:
    library_payload = load_json(prefab_library_dir / "prefab_library.json")
    dataset_root = Path(str(library_payload.get("dataset_root") or "")).resolve()
    summaries = list(library_payload.get("prefabs", []))
    if max_prefabs > 0:
        summaries = summaries[:max_prefabs]

    output_dir.mkdir(parents=True, exist_ok=True)
    exported_entries: List[Dict[str, Any]] = []
    group_cache: Dict[str, Dict[str, Any]] = {}
    tile_cache: Dict[str, Dict[str, Any]] = {}

    for summary in summaries:
        prefab_id = str(summary.get("prefab_id") or "")
        prefab_rel = str(summary.get("file") or "").strip()
        if not prefab_id or not prefab_rel:
            continue
        prefab_payload = load_json(prefab_library_dir / prefab_rel)
        occurrences = list(prefab_payload.get("occurrences", []))[:occurrences_per_prefab]
        prefab_output_dir = output_dir / prefab_id
        occurrence_exports: List[Dict[str, Any]] = []

        for occurrence_index, occurrence in enumerate(occurrences, start=1):
            group_file = str(occurrence.get("group_file") or "").strip()
            tile_name = str(occurrence.get("tile_name") or "").strip()
            if not group_file or not tile_name:
                continue
            group_payload = group_cache.setdefault(group_file, load_json(dataset_root / "brush_imprints" / group_file))
            tile_payload = tile_cache.setdefault(tile_name, load_json(dataset_root / "dataset" / f"{tile_name}.json"))
            occurrence_dir = prefab_output_dir if occurrences_per_prefab == 1 else prefab_output_dir / f"occurrence_{occurrence_index:02d}"
            occurrence_dir.mkdir(parents=True, exist_ok=True)
            metadata = write_prefab_obj_bundle(
                prefab_output_dir=occurrence_dir,
                prefab_id=prefab_id if occurrences_per_prefab == 1 else f"{prefab_id}_occ{occurrence_index:02d}",
                occurrence=occurrence,
                group_payload=group_payload,
                tile_payload=tile_payload,
                dataset_root=dataset_root,
                height_scale=height_scale,
                prefab_occurrence_count=int(summary.get("count", 0) or 0),
                include_terrain_context=include_terrain_context,
            )
            occurrence_exports.append(metadata)

        exported_entries.append(
            {
                "prefab_id": prefab_id,
                "occurrence_count": int(summary.get("count", 0) or 0),
                "exported_occurrences": len(occurrence_exports),
                "output_dir": str(prefab_output_dir),
                "exports": occurrence_exports,
            }
        )

    manifest = {
        "source_prefab_library_dir": str(prefab_library_dir),
        "dataset_root": str(dataset_root),
        "prefabs_exported": len(exported_entries),
        "occurrences_per_prefab": occurrences_per_prefab,
        "height_scale": height_scale,
        "terrain_context_included": include_terrain_context,
        "exports": exported_entries,
    }
    with open(output_dir / "prefab_obj_library_manifest.json", "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export OBJ object-assembly bundles for detected prefabs.")
    parser.add_argument("prefab_library_dir", help="Directory containing prefab_library.json and prefabs/.")
    parser.add_argument("--output-dir", help="Output directory. Defaults to <prefab_library_dir>/obj_library.")
    parser.add_argument("--occurrences-per-prefab", type=int, default=DEFAULT_OCCURRENCES_PER_PREFAB, help="Representative occurrences to export for each prefab.")
    parser.add_argument("--max-prefabs", type=int, default=DEFAULT_MAX_PREFABS, help="Maximum prefabs to export. Use 0 for all.")
    parser.add_argument("--height-scale", type=float, default=DEFAULT_HEIGHT_SCALE, help="Vertical scale applied to normalized prefab heights.")
    parser.add_argument("--include-terrain-context", action="store_true", help="Include the cropped terrain context mesh under the object assembly.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    prefab_library_dir = Path(args.prefab_library_dir).resolve()
    output_dir = Path(args.output_dir).resolve() if args.output_dir else prefab_library_dir / "obj_library"
    manifest = export_prefab_obj_library(
        prefab_library_dir=prefab_library_dir,
        output_dir=output_dir,
        occurrences_per_prefab=max(1, int(args.occurrences_per_prefab)),
        max_prefabs=max(0, int(args.max_prefabs)),
        height_scale=float(args.height_scale),
        include_terrain_context=bool(args.include_terrain_context),
    )
    print("Prefab OBJ library export complete")
    print(f"  prefab_library_dir: {prefab_library_dir}")
    print(f"  output_dir: {output_dir}")
    print(f"  prefabs_exported: {manifest['prefabs_exported']}")
    print(f"  manifest: {output_dir / 'prefab_obj_library_manifest.json'}")


if __name__ == "__main__":
    main()