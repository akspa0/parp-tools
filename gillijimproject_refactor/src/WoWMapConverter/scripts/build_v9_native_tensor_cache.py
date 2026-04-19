from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image


TILE_HEIGHTMAP_SIZE = 257
HALF_STEPS_PER_CHUNK = 16
DEFAULT_OUTPUT_DIR = Path("output/ml-training/v9_native_tensor_cache")
MANIFEST_FILE = "v9_tensor_cache_manifest.json"
SUPPORTED_NATIVE_SIZES = (257, 129, 65, 33, 17)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build V9 native tensor shards from harvested terrain dataset roots."
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
    if args.dataset_roots:
        roots = [Path(value) for value in args.dataset_roots]
    else:
        roots = discover_dataset_roots(args.search_root)

    if not roots:
        raise SystemExit("No dataset roots were found. Pass explicit dataset roots or use a valid --search-root.")

    return roots


def dataset_root_key(dataset_root: Path) -> str:
    parts = [part for part in dataset_root.parts if part and part != dataset_root.anchor]
    if len(parts) >= 2:
        return f"{parts[-2]}__{parts[-1]}"
    if parts:
        return parts[-1]
    return dataset_root.name or "dataset"


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


def load_minimap(dataset_root: Path, tile_json: dict) -> np.ndarray | None:
    minimap_path = resolve_dataset_path(dataset_root, tile_json.get("image"))
    if minimap_path is None:
        return None

    with Image.open(minimap_path) as image:
        rgb = image.convert("RGB")
        if rgb.size != (256, 256):
            rgb = rgb.resize((256, 256), Image.Resampling.BILINEAR)
        return np.asarray(rgb, dtype=np.uint8)


def build_shard_payload(dataset_root: Path, json_path: Path, default_interleaved: bool) -> tuple[dict[str, np.ndarray], dict[str, object]] | None:
    with json_path.open("r", encoding="utf-8") as handle:
        tile_json = json.load(handle)

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

    payload: dict[str, np.ndarray] = {
        "chunk_heights_256x145": chunk_heights,
        "height_257": height_257,
        "height_129": downsample_native_height(height_257, 129),
        "height_65": downsample_native_height(height_257, 65),
        "height_33": downsample_native_height(height_257, 33),
        "height_17": downsample_native_height(height_257, 17),
        "hole_mask_16x16": hole_mask_16,
    }
    if wdl_17 is not None:
        payload["wdl_17"] = wdl_17
        payload["wdl_delta_17"] = payload["height_17"] - wdl_17
    if minimap_rgb_256 is not None:
        payload["minimap_rgb_256"] = minimap_rgb_256

    metadata = {
        "tile_name": tile_name,
        "source_json": str(json_path),
        "is_interleaved": is_interleaved,
        "height_min": float(np.min(height_257)),
        "height_max": float(np.max(height_257)),
        "has_wdl_17": wdl_17 is not None,
        "has_minimap_rgb_256": minimap_rgb_256 is not None,
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
    dataset_roots = resolve_dataset_roots(args)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_entries: list[dict[str, object]] = []
    processed = 0
    skipped = 0

    for dataset_root in dataset_roots:
        dataset_key = dataset_root_key(dataset_root)
        processed_for_root = 0
        for json_path in iter_tile_json_paths(dataset_root):
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
        "schema_version": "v9-native-tensor-cache.v1",
        "created_at_utc": utc_now_iso(),
        "output_dir": str(output_dir),
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