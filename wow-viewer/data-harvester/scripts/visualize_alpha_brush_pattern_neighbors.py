"""Render exact alpha-scar patterns with nearest non-exact neighbors."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path

import numpy as np
import zarr
from PIL import Image, ImageDraw, ImageFont

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_DEDUPE_DIR = _PROJECT_ROOT / "output" / "analysis" / "alpha-brush-library" / "two-build-full" / "dedupe"

_LAYER_COLORS = {
    0: (120, 120, 120),
    1: (92, 177, 255),
    2: (124, 224, 123),
    3: (255, 184, 76),
}


class ZarrCache:
    def __init__(self, dataset_dir: Path) -> None:
        self.dataset_dir = dataset_dir
        self._stores: dict[str, zarr.storage.LocalStore] = {}
        self._roots: dict[str, zarr.Group] = {}
        self._tile_cache: dict[tuple[str, int], np.ndarray] = {}

    def tile(self, build: str, tile_id: int) -> np.ndarray:
        key = (build, int(tile_id))
        if key in self._tile_cache:
            return self._tile_cache[key]
        if build not in self._roots:
            zarr_path = self.dataset_dir / f"{build}.zarr"
            store = zarr.storage.LocalStore(str(zarr_path), read_only=True)
            self._stores[build] = store
            self._roots[build] = zarr.open_group(store=store, mode="r")
        tile = np.asarray(self._roots[build]["alpha_256"][int(tile_id)], dtype=np.float32)
        if len(self._tile_cache) > 16:
            self._tile_cache.clear()
        self._tile_cache[key] = tile
        return tile

    def close(self) -> None:
        for store in self._stores.values():
            store.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render exact scar patterns and nearest non-exact neighbors.")
    parser.add_argument("--dedupe-dir", type=Path, default=_DEFAULT_DEDUPE_DIR)
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-patterns", type=int, default=200)
    parser.add_argument("--neighbors", type=int, default=7)
    parser.add_argument("--patterns-per-page", type=int, default=40)
    parser.add_argument("--cell-size", type=int, default=160)
    parser.add_argument("--label-width", type=int, default=280)
    parser.add_argument("--padding", type=int, default=10)
    parser.add_argument("--split-by-layer", action="store_true", help="Write separate neighbor sheets by dominant pattern layer.")
    return parser.parse_args()


def _read_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def _render_example(cache: ZarrCache, example: dict, cell_size: int, subtitle: str) -> Image.Image:
    build = str(example.get("build", ""))
    map_name = str(example.get("map_name", "?"))
    tile_id = int(example.get("tile_id", -1))
    tile_x = int(example.get("tile_x", -1))
    tile_y = int(example.get("tile_y", -1))
    layer_idx = int(example.get("layer_idx", 0))
    threshold = float(example.get("threshold", 0.05))
    x, y, w, h = [int(value) for value in example.get("bbox_xywh", [0, 0, 1, 1])]

    tile = cache.tile(build, tile_id)
    crop = np.clip(tile[:, :, layer_idx], 0.0, 1.0)[max(0, y) : min(256, y + h), max(0, x) : min(256, x + w)]
    if crop.size == 0:
        crop = np.zeros((1, 1), dtype=np.float32)
    crop = np.where(crop > threshold, crop, 0.0)
    pad = max(2, int(cell_size * 0.07))
    padded = np.pad(crop, ((pad, pad), (pad, pad)), mode="constant", constant_values=0.0)
    max_draw = max(8, int(cell_size) - 52)
    scale = min(max_draw / max(1, padded.shape[1]), max_draw / max(1, padded.shape[0]))
    dst = (max(1, int(round(padded.shape[1] * scale))), max(1, int(round(padded.shape[0] * scale))))
    gray = Image.fromarray((padded * 255.0).astype(np.uint8), mode="L").resize(dst, Image.Resampling.BILINEAR)
    img = Image.new("RGB", (int(cell_size), int(cell_size)), color=(18, 18, 20))
    img.paste(Image.merge("RGB", (gray, gray, gray)), ((int(cell_size) - dst[0]) // 2, 6))
    draw = ImageDraw.Draw(img)
    color = _LAYER_COLORS.get(layer_idx, (255, 255, 255))
    draw.rectangle((0, 0, int(cell_size) - 1, int(cell_size) - 1), outline=color, width=3)
    label = [
        subtitle,
        build,
        f"{map_name[:11]} {tile_x},{tile_y}",
        f"box {x},{y} {w}x{h}",
        f"L{layer_idx} a{int(example.get('area', 0))}",
    ]
    draw.rectangle((2, int(cell_size) - 66, int(cell_size) - 3, int(cell_size) - 3), fill=(0, 0, 0))
    for idx, text in enumerate(label):
        draw.text((5, int(cell_size) - 64 + idx * 12), text, fill=color if idx in {0, 4} else (235, 235, 235))
    return img


def _write_page(
    path: Path,
    patterns: list[dict],
    pattern_by_id: dict[str, dict],
    neighbors_by_id: dict[str, list[dict]],
    cache: ZarrCache,
    args: argparse.Namespace,
) -> None:
    row_h = int(args.cell_size) + int(args.padding)
    cols = 1 + int(args.neighbors)
    legend_h = 86
    width = int(args.label_width) + cols * (int(args.cell_size) + int(args.padding))
    height = legend_h + max(1, len(patterns)) * row_h + int(args.padding)
    img = Image.new("RGB", (width, height), color=(10, 10, 12))
    draw = ImageDraw.Draw(img)
    font = ImageFont.load_default()
    draw.rectangle((0, 0, width - 1, legend_h - 8), fill=(20, 20, 24), outline=(58, 58, 64))
    draw.text((10, 8), "Exact scar patterns + nearest non-exact neighbors", fill=(245, 245, 245), font=font)
    draw.text((10, 26), "First cell is the exact canonical pattern. Following cells are nearest embedding neighbors, ranked by cosine similarity.", fill=(190, 190, 195), font=font)
    draw.text((10, 44), "These are micro-block scars; similar neighbors often represent hand fixups/blends of the same base brush idea.", fill=(190, 190, 195), font=font)

    for row_idx, pattern in enumerate(patterns):
        y = legend_h + int(args.padding) + row_idx * row_h
        pattern_id = str(pattern["pattern_id"])
        label = f"{pattern_id}\n{pattern['member_count']} exact\nC{pattern.get('dominant_cluster_id')} {pattern.get('dominant_map')} L{pattern.get('dominant_layer')}"
        draw.text((8, y + 8), label, fill=(230, 230, 230), font=font)
        examples = pattern.get("examples", [])
        if examples:
            cell = _render_example(cache, examples[0], int(args.cell_size), "exact")
            img.paste(cell, (int(args.label_width), y))
        for col_idx, neighbor in enumerate(neighbors_by_id.get(pattern_id, [])[: int(args.neighbors)], start=1):
            neighbor_pattern = pattern_by_id.get(str(neighbor.get("neighbor_pattern_id")))
            if not neighbor_pattern or not neighbor_pattern.get("examples"):
                continue
            subtitle = f"r{neighbor['rank']} {float(neighbor['cosine_similarity']):.3f}"
            cell = _render_example(cache, neighbor_pattern["examples"][0], int(args.cell_size), subtitle)
            x = int(args.label_width) + col_idx * (int(args.cell_size) + int(args.padding))
            img.paste(cell, (x, y))
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path)


def main() -> None:
    args = _parse_args()
    dedupe_dir = Path(args.dedupe_dir)
    output_dir = Path(args.output_dir) if args.output_dir is not None else dedupe_dir / "neighbor_montages"
    output_dir.mkdir(parents=True, exist_ok=True)
    patterns = list(_read_jsonl(dedupe_dir / "exact_patterns.jsonl"))[: int(args.max_patterns)]
    pattern_by_id = {str(row["pattern_id"]): row for row in _read_jsonl(dedupe_dir / "exact_patterns.jsonl")}
    selected = {str(row["pattern_id"]) for row in patterns}
    neighbors_by_id: dict[str, list[dict]] = {pattern_id: [] for pattern_id in selected}
    for row in _read_jsonl(dedupe_dir / "pattern_neighbors.jsonl"):
        pattern_id = str(row.get("pattern_id"))
        if pattern_id in neighbors_by_id and len(neighbors_by_id[pattern_id]) < int(args.neighbors):
            neighbors_by_id[pattern_id].append(row)

    cache = ZarrCache(Path(args.dataset_dir))
    try:
        if args.split_by_layer:
            all_pages: list[Path] = []
            for layer_idx in range(4):
                layer_patterns = [pattern for pattern in patterns if int(pattern.get("dominant_layer", -1)) == layer_idx]
                layer_dir = output_dir / f"layer_{layer_idx}"
                pages = _write_neighbor_sheet_set(layer_patterns, pattern_by_id, neighbors_by_id, cache, layer_dir, args)
                _write_html(layer_dir, pages)
                all_pages.extend(pages)
                print(f"Layer {layer_idx}: rendered {len(layer_patterns)} patterns to {layer_dir}", flush=True)
            pages = all_pages
        else:
            pages = _write_neighbor_sheet_set(patterns, pattern_by_id, neighbors_by_id, cache, output_dir, args)
    finally:
        cache.close()

    if not args.split_by_layer:
        _write_html(output_dir, pages)
    print(f"Index: {output_dir / 'index.html'}", flush=True)


def _write_neighbor_sheet_set(
    patterns: list[dict],
    pattern_by_id: dict[str, dict],
    neighbors_by_id: dict[str, list[dict]],
    cache: ZarrCache,
    output_dir: Path,
    args: argparse.Namespace,
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    pages: list[Path] = []
    per_page = max(1, int(args.patterns_per_page))
    for page_idx, start in enumerate(range(0, len(patterns), per_page)):
        page = output_dir / f"scar_neighbor_sheet_{page_idx:03d}.png"
        _write_page(page, patterns[start : start + per_page], pattern_by_id, neighbors_by_id, cache, args)
        pages.append(page)
        print(f"Wrote {page}", flush=True)
    return pages


def _write_html(output_dir: Path, pages: list[Path]) -> None:
    parts = ["<html><body><h1>Alpha Scar Neighbor Sheets</h1>"]
    parts.append("<p>Each row starts with one exact binary scar pattern, followed by nearest non-exact neighbors ranked by embedding similarity.</p>")
    for page in pages:
        rel = html.escape(page.name)
        parts.append(f'<h2>{rel}</h2><img src="{rel}" style="max-width:100%;height:auto">')
    parts.append("</body></html>")
    (output_dir / "index.html").write_text("\n".join(parts), encoding="utf-8")


if __name__ == "__main__":
    main()
