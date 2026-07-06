"""Mine repeated local heightmap motifs from Zarr height_257 tiles."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import zarr
import zarr.storage
from PIL import Image, ImageDraw


_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_DATASET_ROOT = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_OUTPUT_ROOT = _PROJECT_ROOT / "output" / "analysis" / "heightmap-patterns"


@dataclass
class Example:
    map: str
    tile_id: int
    tile_x: int
    tile_y: int
    patch_x: int
    patch_y: int
    patch_size: int
    cell_span: int
    chunk_x: int
    chunk_y: int
    patch_std: float


@dataclass
class PatternGroup:
    pattern_id: str
    patch_size: int
    cell_span: int
    count: int = 0
    sum_std: float = 0.0
    tiles: set[str] = field(default_factory=set)
    examples: list[Example] = field(default_factory=list)

    def add(
        self,
        row: dict[str, Any],
        x: int,
        y: int,
        patch_size: int,
        cell_span: int,
        chunk_cells: int,
        patch_std: float,
        max_examples: int,
    ) -> None:
        self.count += 1
        self.sum_std += float(patch_std)
        tile_key = f"{row.get('map')}:{row.get('tile_x')}:{row.get('tile_y')}"
        self.tiles.add(tile_key)
        if len(self.examples) < max_examples:
            self.examples.append(
                Example(
                    map=str(row.get("map")),
                    tile_id=int(row.get("tile_id")),
                    tile_x=int(row.get("tile_x")),
                    tile_y=int(row.get("tile_y")),
                    patch_x=int(x),
                    patch_y=int(y),
                    patch_size=int(patch_size),
                    cell_span=int(cell_span),
                    chunk_x=int(x // chunk_cells),
                    chunk_y=int(y // chunk_cells),
                    patch_std=float(patch_std),
                )
            )

    @property
    def distinct_tiles(self) -> int:
        return len(self.tiles)

    @property
    def mean_std(self) -> float:
        return self.sum_std / max(1, self.count)

    @property
    def score(self) -> float:
        return float(self.count * math.log1p(self.distinct_tiles) * math.log1p(self.mean_std))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find repeated locally normalized heightmap patch motifs."
    )
    parser.add_argument("--dataset-root", type=Path, default=_DEFAULT_DATASET_ROOT)
    parser.add_argument("--build", required=True, help="Build store name, for example 0_5_3_3368.")
    parser.add_argument("--maps", nargs="+", default=None, help="Optional map filters.")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-tiles", type=int, default=0, help="Zero means all filtered tiles.")
    parser.add_argument(
        "--cell-spans",
        nargs="+",
        type=int,
        default=[32, 64],
        help="Terrain-cell spans to match. Patch vertices are cell_span + 1.",
    )
    parser.add_argument(
        "--patch-sizes",
        nargs="+",
        type=int,
        default=None,
        help="Legacy vertex patch sizes. Converted to cell spans by subtracting one.",
    )
    parser.add_argument(
        "--min-cell-span",
        type=int,
        default=32,
        help="Reject candidate windows smaller than this many terrain cells.",
    )
    parser.add_argument("--cell-stride", type=int, default=16)
    parser.add_argument("--stride", type=int, default=None, help="Legacy alias for --cell-stride.")
    parser.add_argument("--chunk-cells", type=int, default=16)
    parser.add_argument("--chunk-aligned", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--hash-grid", type=int, default=4)
    parser.add_argument("--quant-levels", type=int, default=4)
    parser.add_argument("--min-std", type=float, default=0.5)
    parser.add_argument(
        "--max-saturated-ratio",
        type=float,
        default=0.55,
        help="Skip patches whose normalized pixels are mostly near black/white.",
    )
    parser.add_argument("--top-patterns", type=int, default=48)
    parser.add_argument("--examples-per-pattern", type=int, default=6)
    parser.add_argument("--max-patches", type=int, default=0, help="Zero means no patch cap.")
    return parser.parse_args()


def load_index(store_path: Path, maps: list[str] | None, max_tiles: int) -> list[dict[str, Any]]:
    table = pq.read_table(str(store_path / "index.parquet"))
    filters = {value.lower() for value in maps} if maps else None
    rows: list[dict[str, Any]] = []
    for i in range(table.num_rows):
        row = {col: table.column(col)[i].as_py() for col in table.column_names}
        if filters and str(row.get("map", "")).lower() not in filters:
            continue
        rows.append(row)
    rows.sort(key=lambda row: (str(row.get("map", "")), int(row.get("tile_y", 0)), int(row.get("tile_x", 0))))
    if max_tiles > 0:
        rows = rows[:max_tiles]
    return rows


def open_zarr_group(store_path: Path) -> Any:
    store = zarr.storage.LocalStore(str(store_path), read_only=True)
    return zarr.open_group(store=store, mode="r")


def normalize_patch_to_u8(patch: np.ndarray) -> np.ndarray:
    values = np.asarray(patch, dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return np.zeros(values.shape, dtype=np.uint8)
    lo = float(np.percentile(finite, 5.0))
    hi = float(np.percentile(finite, 95.0))
    if not math.isfinite(lo) or not math.isfinite(hi) or hi <= lo:
        lo = float(finite.min())
        hi = float(finite.max())
    if hi <= lo:
        return np.zeros(values.shape, dtype=np.uint8)
    scaled = (values - lo) / (hi - lo)
    return np.clip(scaled * 255.0, 0.0, 255.0).astype(np.uint8)


def patch_signature(patch: np.ndarray, hash_grid: int, quant_levels: int) -> tuple[str, float, float]:
    patch_std = float(np.nanstd(patch))
    patch_u8 = normalize_patch_to_u8(patch)
    saturated = float(np.mean((patch_u8 <= 5) | (patch_u8 >= 250)))
    image = Image.fromarray(patch_u8, mode="L").resize(
        (hash_grid, hash_grid), Image.Resampling.BILINEAR
    )
    low = np.asarray(image, dtype=np.uint8)
    levels = max(2, min(64, int(quant_levels)))
    quant = np.minimum((low.astype(np.uint16) * levels) // 256, levels - 1).astype(np.uint8)
    digest = hashlib.blake2b(quant.tobytes(), digest_size=8).hexdigest()
    return digest, patch_std, saturated


def mine_patterns(
    root: Any,
    rows: list[dict[str, Any]],
    cell_spans: list[int],
    cell_stride: int,
    chunk_cells: int,
    chunk_aligned: bool,
    hash_grid: int,
    quant_levels: int,
    min_std: float,
    max_saturated_ratio: float,
    max_examples: int,
    max_patches: int,
) -> tuple[dict[str, PatternGroup], dict[str, int]]:
    groups: dict[str, PatternGroup] = {}
    stats = defaultdict(int)
    height_array = root["height_257"]
    patch_cap_hit = False

    for row in rows:
        tile = height_array[int(row["tile_id"])].astype(np.float32)
        stats["tiles_read"] += 1
        tile_cell_width = tile.shape[1] - 1
        tile_cell_height = tile.shape[0] - 1
        for cell_span in cell_spans:
            patch_size = cell_span + 1
            if cell_span <= 0 or patch_size > min(tile.shape):
                continue
            step = chunk_cells if chunk_aligned else cell_stride
            y_limit = tile_cell_height - cell_span
            x_limit = tile_cell_width - cell_span
            for y in range(0, y_limit + 1, step):
                for x in range(0, x_limit + 1, step):
                    if max_patches > 0 and stats["patches_seen"] >= max_patches:
                        patch_cap_hit = True
                        break
                    stats["patches_seen"] += 1
                    patch = tile[y : y + patch_size, x : x + patch_size]
                    digest, patch_std, saturated = patch_signature(patch, hash_grid, quant_levels)
                    if patch_std < min_std:
                        stats["patches_skipped_low_std"] += 1
                        continue
                    if saturated > max_saturated_ratio:
                        stats["patches_skipped_saturated"] += 1
                        continue
                    key = f"cells{cell_span}_g{hash_grid}_q{quant_levels}_{digest}"
                    if key not in groups:
                        groups[key] = PatternGroup(
                            pattern_id=key,
                            patch_size=patch_size,
                            cell_span=cell_span,
                        )
                    groups[key].add(
                        row,
                        x,
                        y,
                        patch_size,
                        cell_span,
                        chunk_cells,
                        patch_std,
                        max_examples,
                    )
                    stats["patches_kept"] += 1
                if patch_cap_hit:
                    break
            if patch_cap_hit:
                break
        if patch_cap_hit:
            break

    stats["pattern_count"] = len(groups)
    return groups, dict(stats)


def group_to_json(group: PatternGroup) -> dict[str, Any]:
    return {
        "pattern_id": group.pattern_id,
        "patch_size": group.patch_size,
        "cell_span": group.cell_span,
        "count": group.count,
        "distinct_tiles": group.distinct_tiles,
        "mean_std": group.mean_std,
        "score": group.score,
        "examples": [example.__dict__ for example in group.examples],
    }


def write_atlas(
    root: Any,
    groups: list[PatternGroup],
    out_path: Path,
    examples_per_pattern: int,
) -> None:
    if not groups:
        Image.new("RGB", (640, 120), "white").save(out_path)
        return

    thumb = 96
    label_w = 360
    pad = 10
    row_h = thumb + pad * 2
    cols = max(1, examples_per_pattern)
    width = label_w + cols * (thumb + pad) + pad
    height = len(groups) * row_h
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    height_array = root["height_257"]
    tile_cache: dict[int, np.ndarray] = {}

    for row_idx, group in enumerate(groups):
        y0 = row_idx * row_h
        label = (
            f"{row_idx + 1}. {group.pattern_id}\n"
            f"cells={group.cell_span} count={group.count} tiles={group.distinct_tiles} "
            f"mean_std={group.mean_std:.3f} score={group.score:.1f}"
        )
        draw.text((pad, y0 + pad), label, fill="black")
        for ex_idx, example in enumerate(group.examples[:examples_per_pattern]):
            if example.tile_id not in tile_cache:
                tile_cache[example.tile_id] = height_array[example.tile_id].astype(np.float32)
            tile = tile_cache[example.tile_id]
            patch = tile[
                example.patch_y : example.patch_y + example.patch_size,
                example.patch_x : example.patch_x + example.patch_size,
            ]
            patch_u8 = normalize_patch_to_u8(patch)
            image = Image.fromarray(patch_u8, mode="L").resize(
                (thumb, thumb), Image.Resampling.BILINEAR
            ).convert("RGB")
            x0 = label_w + ex_idx * (thumb + pad)
            sheet.paste(image, (x0, y0 + pad))
            draw.text(
                (x0, y0 + pad + thumb - 24),
                f"{example.map} {example.tile_x},{example.tile_y}\n"
                f"chunk {example.chunk_x},{example.chunk_y}",
                fill=(255, 255, 0),
            )

    sheet.save(out_path)


def main() -> None:
    args = parse_args()
    store_path = args.dataset_root / f"{args.build}.zarr"
    if not store_path.exists():
        raise FileNotFoundError(f"Missing Zarr store: {store_path}")
    out_dir = args.output_dir or (_DEFAULT_OUTPUT_ROOT / args.build)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_index(store_path, args.maps, int(args.max_tiles))
    if not rows:
        raise RuntimeError("No index rows matched the requested filters.")
    root = open_zarr_group(store_path)
    min_cell_span = max(1, int(args.min_cell_span))
    if args.patch_sizes:
        requested_cell_spans = [int(value) - 1 for value in args.patch_sizes]
    else:
        requested_cell_spans = [int(value) for value in args.cell_spans]
    cell_spans = sorted({value for value in requested_cell_spans if value >= min_cell_span})
    if not cell_spans:
        raise RuntimeError(
            f"No usable cell spans remain after min-cell-span={min_cell_span}. "
            "Use --cell-spans with larger terrain-cell spans."
        )
    cell_stride = int(args.stride) if args.stride is not None else int(args.cell_stride)
    groups, stats = mine_patterns(
        root=root,
        rows=rows,
        cell_spans=cell_spans,
        cell_stride=max(1, cell_stride),
        chunk_cells=max(1, int(args.chunk_cells)),
        chunk_aligned=bool(args.chunk_aligned),
        hash_grid=max(2, int(args.hash_grid)),
        quant_levels=max(2, int(args.quant_levels)),
        min_std=float(args.min_std),
        max_saturated_ratio=float(args.max_saturated_ratio),
        max_examples=max(1, int(args.examples_per_pattern)),
        max_patches=max(0, int(args.max_patches)),
    )
    ranked = sorted(groups.values(), key=lambda group: group.score, reverse=True)
    top = ranked[: max(1, int(args.top_patterns))]
    atlas_path = out_dir / "pattern_atlas.png"
    write_atlas(root, top, atlas_path, max(1, int(args.examples_per_pattern)))

    summary = {
        "build": args.build,
        "dataset_root": str(args.dataset_root),
        "store_path": str(store_path),
        "maps": args.maps,
        "row_count": len(rows),
        "cell_spans": cell_spans,
        "patch_sizes": [value + 1 for value in cell_spans],
        "min_cell_span": min_cell_span,
        "cell_stride": cell_stride,
        "chunk_cells": int(args.chunk_cells),
        "chunk_aligned": bool(args.chunk_aligned),
        "hash_grid": int(args.hash_grid),
        "quant_levels": int(args.quant_levels),
        "min_std": float(args.min_std),
        "max_saturated_ratio": float(args.max_saturated_ratio),
        "stats": stats,
        "atlas": str(atlas_path),
        "patterns": [group_to_json(group) for group in top],
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"Rows scanned: {len(rows):,}")
    print(f"Patches kept: {stats.get('patches_kept', 0):,}")
    print(f"Patterns: {stats.get('pattern_count', 0):,}")
    print(f"Summary: {summary_path}")
    print(f"Atlas: {atlas_path}")


if __name__ == "__main__":
    main()
