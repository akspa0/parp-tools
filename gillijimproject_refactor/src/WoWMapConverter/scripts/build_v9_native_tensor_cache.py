from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from v7_object_masks import (
    MAX_FALLBACK_OBJECT_MASK_COVERAGE,
    MAX_PRECISE_OBJECT_MASK_COVERAGE,
    MAX_SEEDED_OBJECT_MASK_COVERAGE,
    PRECISE_OBJECT_MASK_KEYS,
    SEEDED_OBJECT_MASK_KEYS,
    build_object_context_mask,
)


TILE_HEIGHTMAP_SIZE = 257
HALF_STEPS_PER_CHUNK = 16
DEFAULT_OUTPUT_DIR = Path("output/ml-training/v9_native_tensor_cache")
MANIFEST_FILE = "v9_tensor_cache_manifest.json"
SUPPORTED_NATIVE_SIZES = (257, 129, 65, 33, 17)
INPUT_SIZE = 257
HEIGHT_GLOBAL_MIN = -1000.0
HEIGHT_GLOBAL_MAX = 3000.0
DEFAULT_NORMAL_RGB = (128, 128, 255)
BRUSH_MANIFEST_FILE = "brush_manifest.json"

_BRUSH_MANIFEST_CACHE: dict[Path, dict[str, object] | None] = {}


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def avg_gradient_magnitude(image: np.ndarray) -> float:
    image_float = image.astype(np.float32)
    if image_float.ndim == 3:
        image_float = image_float.mean(axis=2)
    dx = np.diff(image_float, axis=1)
    dy = np.diff(image_float, axis=0)
    dx = dx[:, :-1] if dx.shape[1] > 0 else dx
    dy = dy[:-1, :] if dy.shape[0] > 0 else dy
    if dx.size == 0 or dy.size == 0:
        return 0.0
    magnitude = np.sqrt(dx[: dy.shape[0], :] ** 2 + dy[:, : dx.shape[1]] ** 2)
    return float(np.mean(magnitude))


def compute_detail_energy(height_257: np.ndarray, height_65: np.ndarray) -> float:
    height_257_tensor = torch.from_numpy(height_257.astype(np.float32)).unsqueeze(0)
    height_65_tensor = torch.from_numpy(height_65.astype(np.float32)).unsqueeze(0)
    upsampled_65 = F.interpolate(height_65_tensor.unsqueeze(0), size=(257, 257), mode="bilinear", align_corners=True).squeeze(0)
    return float(torch.mean(torch.abs(height_257_tensor - upsampled_65)).item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build V9 native tensor shards from the legacy harvested-dataset compatibility path. The canonical long-range path is wow-viewer direct scan/audit/build-cache over game data."
    )
    parser.add_argument(
        "dataset_roots",
        nargs="*",
        help="Optional dataset roots to process. When omitted, roots are discovered under --search-root.",
    )
    parser.add_argument(
        "--search-root",
        action="append",
        default=["datasets"],
        help="Search root used when dataset roots are omitted. Repeat to add more roots.",
    )
    parser.add_argument(
        "--curated-manifest",
        default=None,
        help="Optional curated manifest path. When provided, the builder reads listed tile JSON paths directly instead of discovering dataset roots.",
    )
    parser.add_argument(
        "--allow-harvested-dataset-compat",
        action="store_true",
        help="Required to use harvested dataset roots or compatibility tile JSON manifests. Without this flag the builder stops so the legacy export-first path cannot masquerade as the canonical direct wow-viewer ML pipeline.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory where v9 tensor shards and the manifest will be written.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional global sample limit across all dataset roots.",
    )
    parser.add_argument(
        "--limit-per-root",
        type=int,
        default=None,
        help="Optional per-dataset-root sample limit applied before the global --limit cap.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Rewrite shards even when the output file already exists.",
    )
    parser.add_argument(
        "--default-interleaved",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Fallback chunk ordering when the harvested tile JSON omits terrain_data.is_interleaved.",
    )
    return parser.parse_args()


def discover_dataset_roots(search_roots: list[str]) -> list[Path]:
    discovered: list[Path] = []
    seen: set[Path] = set()

    for root_text in search_roots:
        root = Path(root_text)
        if not root.exists():
            continue

        if (root / "dataset").exists():
            resolved = root.resolve()
            if resolved not in seen:
                seen.add(resolved)
                discovered.append(root)

        for manifest_path in sorted(root.rglob("ml_dataset_manifest.json")):
            candidate = manifest_path.parent
            if not (candidate / "dataset").exists():
                continue

            resolved = candidate.resolve()
            if resolved in seen:
                continue

            seen.add(resolved)
            discovered.append(candidate)

    return discovered


def resolve_dataset_roots(args: argparse.Namespace) -> list[Path]:
    if not args.allow_harvested_dataset_compat:
        raise SystemExit(
            "build_v9_native_tensor_cache.py is a harvested-dataset compatibility builder. "
            "Pass --allow-harvested-dataset-compat only for bounded legacy compatibility use; "
            "the canonical ML path should run through wow-viewer direct game-root scan/audit/build-cache commands instead."
        )

    if args.dataset_roots:
        roots = [Path(value) for value in args.dataset_roots]
    else:
        roots = discover_dataset_roots(args.search_root)

    if not roots:
        raise SystemExit("No dataset roots were found. Pass explicit dataset roots or use a valid --search-root.")

    return roots


def infer_dataset_root_from_json_path(json_path: Path) -> Path:
    if json_path.parent.name.lower() == "dataset":
        return json_path.parent.parent
    return json_path.parent


def load_curated_manifest_entries(curated_manifest_path: Path, *, allow_harvested_dataset_compat: bool) -> list[tuple[Path, Path]]:
    with curated_manifest_path.open("r", encoding="utf-8") as handle:
        manifest = json.load(handle)

    source_manifest_kind = str(manifest.get("sourceManifestKind") or manifest.get("source_manifest_kind") or "").strip().lower()
    if source_manifest_kind == "scan":
        raise SystemExit(
            f"Curated manifest '{curated_manifest_path}' is a wow-viewer direct scan manifest. "
            "This compatibility builder cannot consume summary-only scan entries because it still expects harvested tile JSON/image artifacts. "
            "Implement or use the wow-viewer direct dataset build-cache path instead of routing scan manifests back through dataset-folder compatibility code."
        )

    if not allow_harvested_dataset_compat:
        raise SystemExit(
            f"Curated manifest '{curated_manifest_path}' resolved to the harvested-dataset compatibility builder. "
            "Pass --allow-harvested-dataset-compat only for bounded legacy use; the canonical ML path should remain direct wow-viewer game-data scan/audit/build-cache."
        )

    entries = manifest.get("entries")
    if not isinstance(entries, list):
        raise SystemExit(f"Curated manifest does not contain an 'entries' list: {curated_manifest_path}")

    resolved_entries: list[tuple[Path, Path]] = []
    for entry in entries:
        if not isinstance(entry, dict):
            continue

        source_json_text = entry.get("source_json") or entry.get("tile_json_path") or entry.get("compatibility_tile_json_path")
        if not source_json_text:
            continue

        source_json = Path(str(source_json_text))
        if not source_json.is_absolute():
            source_json = (curated_manifest_path.parent / source_json).resolve()

        if not source_json.exists():
            continue

        dataset_root_text = entry.get("dataset_root") or entry.get("source_root")
        if dataset_root_text:
            dataset_root = Path(str(dataset_root_text))
            if not dataset_root.is_absolute():
                dataset_root = (curated_manifest_path.parent / dataset_root).resolve()
        else:
            dataset_root = infer_dataset_root_from_json_path(source_json)

        resolved_entries.append((dataset_root, source_json))

    if not resolved_entries:
        raise SystemExit(f"Curated manifest did not resolve any usable tile JSON paths: {curated_manifest_path}")

    return resolved_entries


def dataset_root_key(dataset_root: Path) -> str:
    parts = [part for part in dataset_root.parts if part and part != dataset_root.anchor]
    if len(parts) >= 2:
        return f"{parts[-2]}__{parts[-1]}"
    if parts:
        return parts[-1]
    return dataset_root.name or "dataset"


def parse_tile_coordinates(tile_name: str) -> tuple[str, int, int]:
    parts = tile_name.rsplit("_", 2)
    if len(parts) != 3:
        return tile_name, 0, 0

    map_name, tile_x_text, tile_y_text = parts
    try:
        return map_name, int(tile_x_text), int(tile_y_text)
    except ValueError:
        return tile_name, 0, 0


def iter_tile_json_paths(dataset_root: Path) -> list[Path]:
    manifest_path = dataset_root / "ml_dataset_manifest.json"
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)

        paths: list[Path] = []
        for tile_entry in manifest.get("tiles", []):
            tile_path_text = tile_entry.get("tile_json_path")
            if not tile_path_text:
                continue
            tile_path = dataset_root / tile_path_text
            if tile_path.exists():
                paths.append(tile_path)
        if paths:
            return paths

    dataset_dir = dataset_root / "dataset"
    return sorted(dataset_dir.glob("*.json"))


def resolve_dataset_path(dataset_root: Path, relative_path: str | None) -> Path | None:
    if not relative_path:
        return None

    candidate = dataset_root / relative_path
    if candidate.exists():
        return candidate
    return None


def load_brush_manifest(dataset_root: Path) -> dict[str, object] | None:
    if dataset_root in _BRUSH_MANIFEST_CACHE:
        return _BRUSH_MANIFEST_CACHE[dataset_root]

    manifest_path = dataset_root / "brush_imprints" / BRUSH_MANIFEST_FILE
    if not manifest_path.exists():
        _BRUSH_MANIFEST_CACHE[dataset_root] = None
        return None

    try:
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:
        manifest = None

    _BRUSH_MANIFEST_CACHE[dataset_root] = manifest
    return manifest


def resolve_brush_mask_path(dataset_root: Path, tile_name: str) -> Path | None:
    manifest = load_brush_manifest(dataset_root)
    if manifest:
        for tile in manifest.get("tiles", []):
            if str(tile.get("tile_name", "")) != tile_name:
                continue
            brush_rel = tile.get("brush_mask_path")
            if not brush_rel:
                continue
            candidate = dataset_root / str(brush_rel)
            if candidate.exists():
                return candidate

    for candidate in (
        dataset_root / "brush_imprints" / "tile_masks" / f"{tile_name}_brush_mask.png",
        dataset_root / "brush_imprints" / f"{tile_name}_brush_mask.png",
    ):
        if candidate.exists():
            return candidate

    return None


def normalize_chunk_heights(heights: np.ndarray, is_interleaved: bool) -> np.ndarray:
    if is_interleaved or heights.shape[0] < 145:
        return heights

    interleaved = np.empty(145, dtype=np.float32)
    destination = 0
    for outer_row in range(9):
        outer_offset = outer_row * 9
        interleaved[destination:destination + 9] = heights[outer_offset:outer_offset + 9]
        destination += 9

        if outer_row >= 8:
            continue

        inner_offset = 81 + (outer_row * 8)
        interleaved[destination:destination + 8] = heights[inner_offset:inner_offset + 8]
        destination += 8

    return interleaved


def get_vertex_position(index: int) -> tuple[int, int, bool]:
    remaining = index
    for row_index in range(17):
        row_size = 9 if row_index % 2 == 0 else 8
        if remaining < row_size:
            return row_index, remaining, row_index % 2 != 0
        remaining -= row_size
    return 0, 0, False


def try_find_nearest_height(grid: np.ndarray, x: int, y: int) -> float | None:
    height, width = grid.shape
    max_radius = 24
    for radius in range(1, max_radius + 1):
        min_y = max(0, y - radius)
        max_y = min(height - 1, y + radius)
        min_x = max(0, x - radius)
        max_x = min(width - 1, x + radius)

        for sample_x in range(min_x, max_x + 1):
            top = grid[min_y, sample_x]
            if not np.isnan(top):
                return float(top)

            bottom = grid[max_y, sample_x]
            if not np.isnan(bottom):
                return float(bottom)

        for sample_y in range(min_y + 1, max_y):
            left = grid[sample_y, min_x]
            if not np.isnan(left):
                return float(left)

            right = grid[sample_y, max_x]
            if not np.isnan(right):
                return float(right)

    return None


def fill_height_gaps(grid: np.ndarray) -> None:
    height, width = grid.shape

    for y in range(height):
        for x in range(width):
            if not np.isnan(grid[y, x]):
                continue

            if (x & 1) == 1 and (y & 1) == 0 and 0 < x < width - 1:
                left = grid[y, x - 1]
                right = grid[y, x + 1]
                if not np.isnan(left) and not np.isnan(right):
                    grid[y, x] = (left + right) * 0.5
            elif (x & 1) == 0 and (y & 1) == 1 and 0 < y < height - 1:
                up = grid[y - 1, x]
                down = grid[y + 1, x]
                if not np.isnan(up) and not np.isnan(down):
                    grid[y, x] = (up + down) * 0.5

    for y in range(height):
        for x in range(width):
            if not np.isnan(grid[y, x]):
                continue

            nearest = try_find_nearest_height(grid, x, y)
            grid[y, x] = nearest if nearest is not None else 0.0


def build_tile_heightmap_257(chunk_heights: np.ndarray, is_interleaved: bool) -> np.ndarray:
    sum_grid = np.zeros((TILE_HEIGHTMAP_SIZE, TILE_HEIGHTMAP_SIZE), dtype=np.float32)
    count_grid = np.zeros((TILE_HEIGHTMAP_SIZE, TILE_HEIGHTMAP_SIZE), dtype=np.uint16)

    for chunk_index in range(256):
        heights = chunk_heights[chunk_index]
        if heights.shape[0] < 145:
            continue

        normalized = normalize_chunk_heights(heights, is_interleaved)
        chunk_x = chunk_index % 16
        chunk_y = chunk_index // 16
        base_x = chunk_x * HALF_STEPS_PER_CHUNK
        base_y = chunk_y * HALF_STEPS_PER_CHUNK

        for vertex_index in range(145):
            row, col, is_inner = get_vertex_position(vertex_index)
            sample_x = (col * 2) + 1 if is_inner else col * 2
            sample_y = ((row // 2) * 2) + 1 if is_inner else (row // 2) * 2

            px = base_x + sample_x
            py = base_y + sample_y
            if not (0 <= px < TILE_HEIGHTMAP_SIZE and 0 <= py < TILE_HEIGHTMAP_SIZE):
                continue

            sum_grid[py, px] += float(normalized[vertex_index])
            count_grid[py, px] += 1

    grid = np.full((TILE_HEIGHTMAP_SIZE, TILE_HEIGHTMAP_SIZE), np.nan, dtype=np.float32)
    populated = count_grid > 0
    grid[populated] = sum_grid[populated] / count_grid[populated]
    fill_height_gaps(grid)
    return grid


def extract_chunk_heights(terrain_data: dict) -> np.ndarray:
    chunk_tensor = np.zeros((256, 145), dtype=np.float32)
    for chunk_entry in terrain_data.get("heights", []):
        chunk_index = int(chunk_entry.get("idx", chunk_entry.get("chunkIndex", -1)))
        height_values = chunk_entry.get("h", chunk_entry.get("heights", []))
        if not (0 <= chunk_index < 256) or len(height_values) < 145:
            continue
        chunk_tensor[chunk_index, :] = np.asarray(height_values[:145], dtype=np.float32)
    return chunk_tensor


def extract_hole_mask(terrain_data: dict) -> np.ndarray:
    values = terrain_data.get("holes")
    if not isinstance(values, list):
        return np.zeros((16, 16), dtype=np.int32)

    flattened = np.zeros(256, dtype=np.int32)
    count = min(len(values), 256)
    if count:
        flattened[:count] = np.asarray(values[:count], dtype=np.int32)
    return flattened.reshape(16, 16)


def extract_wdl_17(terrain_data: dict) -> np.ndarray | None:
    wdl_heights = terrain_data.get("wdl_heights")
    if not isinstance(wdl_heights, dict):
        return None

    outer_17 = wdl_heights.get("outer_17")
    if not isinstance(outer_17, list) or len(outer_17) != 289:
        return None

    return np.asarray(outer_17, dtype=np.float32).reshape(17, 17)


def downsample_native_height(height_257: np.ndarray, size: int) -> np.ndarray:
    if size not in SUPPORTED_NATIVE_SIZES:
        raise ValueError(f"Unsupported native size {size}; expected one of {SUPPORTED_NATIVE_SIZES}.")

    if size == TILE_HEIGHTMAP_SIZE:
        return height_257.copy()

    step = (TILE_HEIGHTMAP_SIZE - 1) // (size - 1)
    return height_257[::step, ::step].copy()


def load_rgb_image(path: Path | None, size: int, fallback_rgb: tuple[int, int, int] | None = None) -> np.ndarray | None:
    if path is None or not path.exists():
        if fallback_rgb is None:
            return None
        return np.full((size, size, 3), fallback_rgb, dtype=np.uint8)

    with Image.open(path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (size, size):
            rgb = rgb.resize((size, size), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


def load_binary_mask(path: Path | None, size: int) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((size, size), dtype=np.uint8)

    with Image.open(path) as image:
        mask = image.convert("L")
        if mask.size != (size, size):
            mask = mask.resize((size, size), Image.Resampling.NEAREST)
        return (np.asarray(mask, dtype=np.uint8) > 0).astype(np.uint8)


def load_heightmap_16bit(path: Path | None, target_size: int) -> np.ndarray:
    if path is None or not path.exists():
        return np.zeros((target_size, target_size), dtype=np.float32)

    with Image.open(path) as image:
        if image.mode == "I;16":
            array = np.asarray(image, dtype=np.float32) / 65535.0
        elif image.mode == "I":
            array = np.asarray(image, dtype=np.float32)
            array = (array - array.min()) / (array.max() - array.min() + 1e-8)
        else:
            array = np.asarray(image.convert("L"), dtype=np.float32) / 255.0

    if array.shape != (target_size, target_size):
        tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0)
        tensor = F.interpolate(tensor, size=(target_size, target_size), mode="bilinear", align_corners=False)
        return tensor.squeeze(0).squeeze(0).numpy().astype(np.float32)

    return array.astype(np.float32)


def select_minimap_path(dataset_root: Path, tile_json: dict, terrain_data: dict) -> tuple[Path | None, str]:
    candidates = (
        (terrain_data.get("terrain_only_minimap"), "terrain_only_minimap"),
        (terrain_data.get("no_liquid_minimap"), "no_liquid_minimap"),
        (terrain_data.get("no_object_minimap"), "no_object_minimap"),
        (terrain_data.get("no_mccv_minimap"), "no_mccv_minimap"),
        (tile_json.get("image"), "image"),
    )
    for relative_path, source_name in candidates:
        candidate = resolve_dataset_path(dataset_root, relative_path)
        if candidate is not None:
            return candidate, source_name
    return None, "missing"


def load_minimap(dataset_root: Path, tile_json: dict) -> np.ndarray | None:
    terrain_data = tile_json.get("terrain_data", tile_json)
    minimap_path, _ = select_minimap_path(dataset_root, tile_json, terrain_data)
    return load_rgb_image(minimap_path, size=256)


def load_normalmap(dataset_root: Path, terrain_data: dict) -> tuple[np.ndarray, bool]:
    normalmap_path = resolve_dataset_path(dataset_root, terrain_data.get("normalmap"))
    normalmap = load_rgb_image(normalmap_path, size=256, fallback_rgb=DEFAULT_NORMAL_RGB)
    return normalmap, normalmap_path is not None


def build_height_hints_v7(terrain_data: dict) -> np.ndarray:
    height_min = float(terrain_data.get("height_min", 0.0) or 0.0)
    height_max = float(terrain_data.get("height_max", 100.0) or 100.0)
    global_min = float(terrain_data.get("height_global_min", HEIGHT_GLOBAL_MIN) or HEIGHT_GLOBAL_MIN)
    global_max = float(terrain_data.get("height_global_max", HEIGHT_GLOBAL_MAX) or HEIGHT_GLOBAL_MAX)
    global_range = max(global_max - global_min, 1e-6)

    return np.asarray(
        [
            float(np.clip((height_min - global_min) / global_range, 0.0, 1.0)),
            float(np.clip((height_max - global_min) / global_range, 0.0, 1.0)),
        ],
        dtype=np.float32,
    )


def build_object_mask_257(dataset_root: Path, terrain_data: dict, tile_name: str) -> np.ndarray:
    _, tile_x, tile_y = parse_tile_coordinates(tile_name)
    object_mask = build_object_context_mask(
        dataset_root=dataset_root,
        terrain=terrain_data,
        tile_x=tile_x,
        tile_y=tile_y,
        output_size=INPUT_SIZE,
        precise_keys=PRECISE_OBJECT_MASK_KEYS,
        seeded_keys=SEEDED_OBJECT_MASK_KEYS,
        max_precise_coverage=MAX_PRECISE_OBJECT_MASK_COVERAGE,
        max_seeded_coverage=MAX_SEEDED_OBJECT_MASK_COVERAGE,
        max_fallback_coverage=MAX_FALLBACK_OBJECT_MASK_COVERAGE,
    )
    return object_mask.squeeze(0).numpy().astype(np.uint8)


def build_shard_payload(dataset_root: Path, json_path: Path, default_interleaved: bool) -> tuple[dict[str, np.ndarray], dict[str, object]] | None:
    try:
        with json_path.open("r", encoding="utf-8") as handle:
            tile_json = json.load(handle)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        print(f"Skipping unreadable tile JSON: {json_path} ({exc})")
        return None

    terrain_data = tile_json.get("terrain_data", tile_json)
    chunk_heights = extract_chunk_heights(terrain_data)
    if not np.any(chunk_heights):
        return None

    tile_name = str(terrain_data.get("adt_tile") or json_path.stem)
    is_interleaved = bool(terrain_data.get("is_interleaved", default_interleaved))

    height_257 = build_tile_heightmap_257(chunk_heights, is_interleaved=is_interleaved)
    hole_mask_16 = extract_hole_mask(terrain_data)
    wdl_17 = extract_wdl_17(terrain_data)
    minimap_rgb_256 = load_minimap(dataset_root, tile_json)
    _, minimap_source = select_minimap_path(dataset_root, tile_json, terrain_data)
    normal_rgb_256, has_normal_rgb_256 = load_normalmap(dataset_root, terrain_data)
    height_hints_v7 = build_height_hints_v7(terrain_data)
    liquid_mask_257 = load_binary_mask(resolve_dataset_path(dataset_root, terrain_data.get("liquid_mask")), INPUT_SIZE)
    liquid_height_257 = load_heightmap_16bit(resolve_dataset_path(dataset_root, terrain_data.get("liquid_height")), INPUT_SIZE)
    liquid_height_257 *= liquid_mask_257.astype(np.float32)
    object_mask_257 = build_object_mask_257(dataset_root, terrain_data, tile_name)
    brush_mask_257 = load_binary_mask(resolve_brush_mask_path(dataset_root, tile_name), INPUT_SIZE)

    payload: dict[str, np.ndarray] = {
        "chunk_heights_256x145": chunk_heights,
        "height_257": height_257,
        "height_129": downsample_native_height(height_257, 129),
        "height_65": downsample_native_height(height_257, 65),
        "height_33": downsample_native_height(height_257, 33),
        "height_17": downsample_native_height(height_257, 17),
        "hole_mask_16x16": hole_mask_16,
        "normal_rgb_256": normal_rgb_256,
        "height_hints_v7": height_hints_v7,
        "liquid_mask_257": liquid_mask_257,
        "liquid_height_257": liquid_height_257.astype(np.float32),
        "object_mask_257": object_mask_257,
        "brush_mask_257": brush_mask_257,
    }
    if wdl_17 is not None:
        payload["wdl_17"] = wdl_17
        payload["wdl_delta_17"] = payload["height_17"] - wdl_17
    if minimap_rgb_256 is not None:
        payload["minimap_rgb_256"] = minimap_rgb_256

    liquid_coverage = float(liquid_mask_257.mean())
    object_coverage = float(object_mask_257.mean())
    brush_coverage = float(brush_mask_257.mean())
    hole_coverage = float(hole_mask_16.mean())
    minimap_variance = float(np.var(minimap_rgb_256.astype(np.float32) / 255.0)) if minimap_rgb_256 is not None else 0.0
    minimap_gradient = avg_gradient_magnitude(minimap_rgb_256) if minimap_rgb_256 is not None else 0.0
    detail_energy = compute_detail_energy(height_257, payload["height_65"])

    metadata = {
        "tile_name": tile_name,
        "source_json": str(json_path),
        "is_interleaved": is_interleaved,
        "height_min": float(np.min(height_257)),
        "height_max": float(np.max(height_257)),
        "has_wdl_17": wdl_17 is not None,
        "has_minimap_rgb_256": minimap_rgb_256 is not None,
        "has_normal_rgb_256": has_normal_rgb_256,
        "liquid_coverage": liquid_coverage,
        "object_coverage": object_coverage,
        "brush_coverage": brush_coverage,
        "hole_coverage": hole_coverage,
        "minimap_variance": minimap_variance,
        "minimap_gradient": minimap_gradient,
        "detail_energy": detail_energy,
        "minimap_source": minimap_source,
        "array_names": sorted(payload.keys()),
    }
    return payload, metadata


def write_shard(output_dir: Path, dataset_key: str, tile_name: str, payload: dict[str, np.ndarray], overwrite: bool) -> Path:
    shard_dir = output_dir / "shards" / dataset_key
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_path = shard_dir / f"{tile_name}.npz"
    if shard_path.exists() and not overwrite:
        return shard_path

    np.savez_compressed(shard_path, **payload)
    return shard_path


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    curated_manifest_path = Path(args.curated_manifest).resolve() if args.curated_manifest else None
    curated_entries = load_curated_manifest_entries(
        curated_manifest_path,
        allow_harvested_dataset_compat=args.allow_harvested_dataset_compat,
    ) if curated_manifest_path else None
    dataset_roots = [] if curated_entries is not None else resolve_dataset_roots(args)

    manifest_entries: list[dict[str, object]] = []
    processed = 0
    skipped = 0

    if curated_entries is not None:
        grouped_entries: dict[Path, list[Path]] = {}
        for dataset_root, json_path in curated_entries:
            grouped_entries.setdefault(dataset_root, []).append(json_path)
        entry_groups = list(grouped_entries.items())
    else:
        entry_groups = [(dataset_root, iter_tile_json_paths(dataset_root)) for dataset_root in dataset_roots]

    for dataset_root, json_paths in entry_groups:
        dataset_key = dataset_root_key(dataset_root)
        processed_for_root = 0
        for json_path in json_paths:
            if args.limit is not None and processed >= args.limit:
                break
            if args.limit_per_root is not None and processed_for_root >= args.limit_per_root:
                break

            built = build_shard_payload(dataset_root, json_path, args.default_interleaved)
            if built is None:
                skipped += 1
                continue

            payload, metadata = built
            tile_name = str(metadata["tile_name"])
            shard_path = write_shard(output_dir, dataset_key, tile_name, payload, overwrite=args.overwrite)

            manifest_entries.append(
                {
                    "dataset_root": str(dataset_root),
                    "dataset_key": dataset_key,
                    "tile_name": tile_name,
                    "shard_path": str(shard_path),
                    **metadata,
                }
            )
            processed += 1
            processed_for_root += 1

        if args.limit is not None and processed >= args.limit:
            break

    manifest = {
        "schema_version": "v9-native-tensor-cache.v2",
        "created_at_utc": utc_now_iso(),
        "output_dir": str(output_dir),
        "source_mode": "curated-manifest" if curated_manifest_path is not None else "dataset-roots",
        "source_curated_manifest": str(curated_manifest_path) if curated_manifest_path is not None else None,
        "dataset_roots": [str(root) for root in dataset_roots],
        "processed": processed,
        "skipped": skipped,
        "supported_native_sizes": list(SUPPORTED_NATIVE_SIZES),
        "entries": manifest_entries,
    }

    manifest_path = output_dir / MANIFEST_FILE
    with manifest_path.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2)

    print(f"Processed {processed} tile(s); skipped {skipped}; manifest: {manifest_path}")


if __name__ == "__main__":
    main()