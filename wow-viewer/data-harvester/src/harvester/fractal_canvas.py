"""Full-map canvas assembly helpers for spec 076.

ADT tiles are storage pages, not the authoring canvas. These helpers map V18
tile-local signals into a bounded map-canvas coordinate system while preserving
enough provenance to trace every canvas pixel back to its source tile.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr
from PIL import Image, ImageDraw

ALPHA_TILE_SIZE = 256
HEIGHT_TILE_STRIDE = 256
HEIGHT_TILE_SIZE = 257
MCLY_TILE_SIZE = 16


@dataclass(frozen=True, slots=True)
class CanvasTileRecord:
    build: str
    map_name: str
    tile_id: int
    tile_x: int
    tile_y: int
    has_alpha_256: bool = False
    has_height_257: bool = False
    has_normal_xyz: bool = False
    has_mcly_texture_ids: bool = False
    has_mcly_layer_mask: bool = False


@dataclass(frozen=True, slots=True)
class CanvasLayout:
    build: str
    map_name: str
    min_tile_x: int
    min_tile_y: int
    max_tile_x: int
    max_tile_y: int
    tile_count_x: int
    tile_count_y: int

    @property
    def alpha_shape(self) -> tuple[int, int]:
        return (self.tile_count_y * ALPHA_TILE_SIZE, self.tile_count_x * ALPHA_TILE_SIZE)

    @property
    def height_shape(self) -> tuple[int, int]:
        return (
            self.tile_count_y * HEIGHT_TILE_STRIDE + 1,
            self.tile_count_x * HEIGHT_TILE_STRIDE + 1,
        )

    @property
    def mcly_shape(self) -> tuple[int, int]:
        return (self.tile_count_y * MCLY_TILE_SIZE, self.tile_count_x * MCLY_TILE_SIZE)


def load_tile_records(
    zarr_path: str | Path,
    *,
    build: str,
    map_name: str | None = None,
    require_alpha: bool = True,
    tile_limit: int | None = None,
) -> list[CanvasTileRecord]:
    """Load canvas-capable tile rows from a V18 build index."""
    index_path = Path(zarr_path) / "index.parquet"
    if not index_path.exists():
        raise FileNotFoundError(f"Missing V18 index: {index_path}")
    table = pq.read_table(str(index_path))
    rows: list[CanvasTileRecord] = []
    for row_idx in range(table.num_rows):
        row_map = _cell(table, row_idx, "map", "unknown")
        if map_name is not None and str(row_map) != str(map_name):
            continue
        tile_id = int(_cell(table, row_idx, "tile_id", row_idx))
        tile_x = int(_cell(table, row_idx, "tile_x", -1) or -1)
        tile_y = int(_cell(table, row_idx, "tile_y", -1) or -1)
        if tile_id < 0 or tile_x < 0 or tile_y < 0:
            continue
        has_alpha = bool(_cell(table, row_idx, "has_alpha_256", False))
        if require_alpha and not has_alpha:
            continue
        rows.append(
            CanvasTileRecord(
                build=str(build),
                map_name=str(row_map),
                tile_id=tile_id,
                tile_x=tile_x,
                tile_y=tile_y,
                has_alpha_256=has_alpha,
                has_height_257=bool(_cell(table, row_idx, "has_height_257", False)),
                has_normal_xyz=bool(_cell(table, row_idx, "has_normal_xyz", False)),
                has_mcly_texture_ids=bool(_cell(table, row_idx, "has_mcly_texture_ids", False)),
                has_mcly_layer_mask=bool(_cell(table, row_idx, "has_mcly_layer_mask", False)),
            )
        )
    rows.sort(key=lambda record: (record.map_name, record.tile_y, record.tile_x, record.tile_id))
    if tile_limit is not None and int(tile_limit) > 0:
        rows = compact_tile_limit(rows, max(0, int(tile_limit)))
    return rows


def compact_tile_limit(records: list[CanvasTileRecord], tile_limit: int) -> list[CanvasTileRecord]:
    """Select a deterministic compact subset for bounded smoke/proof canvases."""
    if tile_limit <= 0:
        return []
    if len(records) <= int(tile_limit):
        return list(records)
    by_row: dict[int, list[CanvasTileRecord]] = {}
    for record in records:
        by_row.setdefault(int(record.tile_y), []).append(record)

    best: tuple[int, int, int, int, list[CanvasTileRecord]] | None = None
    for tile_y, row in by_row.items():
        row_sorted = sorted(row, key=lambda record: (record.tile_x, record.tile_id))
        if len(row_sorted) < int(tile_limit):
            continue
        for start in range(0, len(row_sorted) - int(tile_limit) + 1):
            subset = row_sorted[start : start + int(tile_limit)]
            span_x = int(subset[-1].tile_x - subset[0].tile_x)
            missing_slots = span_x + 1 - int(tile_limit)
            score = (missing_slots, span_x, int(tile_y), int(subset[0].tile_x), subset)
            if best is None or score[:4] < best[:4]:
                best = score
    if best is not None:
        return list(best[4])
    return list(records[: int(tile_limit)])


def build_canvas_layout(records: list[CanvasTileRecord]) -> CanvasLayout:
    if not records:
        raise ValueError("Cannot build canvas layout without tile records")
    builds = sorted({record.build for record in records})
    maps = sorted({record.map_name for record in records})
    if len(builds) != 1 or len(maps) != 1:
        raise ValueError(f"Canvas records must share one build/map, got builds={builds} maps={maps}")
    min_x = min(record.tile_x for record in records)
    max_x = max(record.tile_x for record in records)
    min_y = min(record.tile_y for record in records)
    max_y = max(record.tile_y for record in records)
    return CanvasLayout(
        build=builds[0],
        map_name=maps[0],
        min_tile_x=int(min_x),
        min_tile_y=int(min_y),
        max_tile_x=int(max_x),
        max_tile_y=int(max_y),
        tile_count_x=int(max_x - min_x + 1),
        tile_count_y=int(max_y - min_y + 1),
    )


def alpha_origin(record: CanvasTileRecord, layout: CanvasLayout) -> tuple[int, int]:
    return (
        (int(record.tile_x) - int(layout.min_tile_x)) * ALPHA_TILE_SIZE,
        (int(record.tile_y) - int(layout.min_tile_y)) * ALPHA_TILE_SIZE,
    )


def height_origin(record: CanvasTileRecord, layout: CanvasLayout) -> tuple[int, int]:
    return (
        (int(record.tile_x) - int(layout.min_tile_x)) * HEIGHT_TILE_STRIDE,
        (int(record.tile_y) - int(layout.min_tile_y)) * HEIGHT_TILE_STRIDE,
    )


def mcly_origin(record: CanvasTileRecord, layout: CanvasLayout) -> tuple[int, int]:
    return (
        (int(record.tile_x) - int(layout.min_tile_x)) * MCLY_TILE_SIZE,
        (int(record.tile_y) - int(layout.min_tile_y)) * MCLY_TILE_SIZE,
    )


def alpha_pixel_to_canvas(record: CanvasTileRecord, layout: CanvasLayout, x: int, y: int) -> tuple[int, int]:
    ox, oy = alpha_origin(record, layout)
    return ox + int(x), oy + int(y)


def height_vertex_to_canvas(record: CanvasTileRecord, layout: CanvasLayout, x: int, y: int) -> tuple[int, int]:
    ox, oy = height_origin(record, layout)
    return ox + int(x), oy + int(y)


def mcly_cell_to_canvas(record: CanvasTileRecord, layout: CanvasLayout, x: int, y: int) -> tuple[int, int]:
    ox, oy = mcly_origin(record, layout)
    return ox + int(x), oy + int(y)


def assemble_full_map_canvas(
    root: zarr.Group,
    records: list[CanvasTileRecord],
    *,
    layers: tuple[int, ...] = (0, 1, 2, 3),
) -> tuple[CanvasLayout, dict[str, np.ndarray], list[dict[str, Any]]]:
    """Assemble a bounded dense map canvas from V18 Zarr arrays."""
    layout = build_canvas_layout(records)
    layer_indices = tuple(int(layer) for layer in layers if 0 <= int(layer) < 4)
    if not layer_indices:
        raise ValueError("At least one alpha layer index in [0, 3] is required")

    alpha_h, alpha_w = layout.alpha_shape
    height_h, height_w = layout.height_shape
    mcly_h, mcly_w = layout.mcly_shape
    arrays: dict[str, np.ndarray] = {
        "alpha_256": np.zeros((alpha_h, alpha_w, len(layer_indices)), dtype=np.float32),
        "tile_id_256": np.full((alpha_h, alpha_w), -1, dtype=np.int32),
        "height_257": np.zeros((height_h, height_w), dtype=np.float32),
        "normal_xyz": np.zeros((height_h, height_w, 3), dtype=np.float32),
        "tile_id_257": np.full((height_h, height_w), -1, dtype=np.int32),
        "mcly_texture_ids": np.full((mcly_h, mcly_w, 4), -1, dtype=np.int32),
        "mcly_layer_mask": np.zeros((mcly_h, mcly_w, 4), dtype=np.float32),
        "tile_id_16": np.full((mcly_h, mcly_w), -1, dtype=np.int32),
    }
    index_rows: list[dict[str, Any]] = []

    for record in sorted(records, key=lambda item: (item.tile_y, item.tile_x, item.tile_id)):
        ax, ay = alpha_origin(record, layout)
        hx, hy = height_origin(record, layout)
        mx, my = mcly_origin(record, layout)
        tile_id = int(record.tile_id)

        if record.has_alpha_256 and "alpha_256" in root:
            alpha = np.clip(root["alpha_256"][tile_id].astype(np.float32), 0.0, 1.0)
            arrays["alpha_256"][ay : ay + ALPHA_TILE_SIZE, ax : ax + ALPHA_TILE_SIZE, :] = alpha[:, :, layer_indices]
            arrays["tile_id_256"][ay : ay + ALPHA_TILE_SIZE, ax : ax + ALPHA_TILE_SIZE] = tile_id

        if record.has_height_257 and "height_257" in root:
            arrays["height_257"][hy : hy + HEIGHT_TILE_SIZE, hx : hx + HEIGHT_TILE_SIZE] = root["height_257"][tile_id].astype(np.float32)
        if record.has_normal_xyz and "normal_xyz" in root:
            arrays["normal_xyz"][hy : hy + HEIGHT_TILE_SIZE, hx : hx + HEIGHT_TILE_SIZE, :] = root["normal_xyz"][tile_id].astype(np.float32)
        if (record.has_height_257 or record.has_normal_xyz) and ("height_257" in root or "normal_xyz" in root):
            arrays["tile_id_257"][hy : hy + HEIGHT_TILE_SIZE, hx : hx + HEIGHT_TILE_SIZE] = tile_id

        if record.has_mcly_texture_ids and "mcly_texture_ids" in root:
            arrays["mcly_texture_ids"][my : my + MCLY_TILE_SIZE, mx : mx + MCLY_TILE_SIZE, :] = root["mcly_texture_ids"][tile_id].astype(np.int32)
        if record.has_mcly_layer_mask and "mcly_layer_mask" in root:
            arrays["mcly_layer_mask"][my : my + MCLY_TILE_SIZE, mx : mx + MCLY_TILE_SIZE, :] = root["mcly_layer_mask"][tile_id].astype(np.float32)
        if (record.has_mcly_texture_ids or record.has_mcly_layer_mask) and ("mcly_texture_ids" in root or "mcly_layer_mask" in root):
            arrays["tile_id_16"][my : my + MCLY_TILE_SIZE, mx : mx + MCLY_TILE_SIZE] = tile_id

        index_rows.append(
            {
                **asdict(record),
                "alpha_canvas_x": int(ax),
                "alpha_canvas_y": int(ay),
                "alpha_canvas_w": ALPHA_TILE_SIZE,
                "alpha_canvas_h": ALPHA_TILE_SIZE,
                "height_canvas_x": int(hx),
                "height_canvas_y": int(hy),
                "height_canvas_w": HEIGHT_TILE_SIZE,
                "height_canvas_h": HEIGHT_TILE_SIZE,
                "mcly_canvas_x": int(mx),
                "mcly_canvas_y": int(my),
                "mcly_canvas_w": MCLY_TILE_SIZE,
                "mcly_canvas_h": MCLY_TILE_SIZE,
            }
        )

    arrays["alpha_layer_indices"] = np.asarray(layer_indices, dtype=np.int32)
    return layout, arrays, index_rows


def write_canvas_outputs(
    output_dir: str | Path,
    layout: CanvasLayout,
    arrays: dict[str, np.ndarray],
    index_rows: list[dict[str, Any]],
) -> None:
    """Write canvas arrays, provenance index, and summary files."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(out / "canvas.zarr"), mode="w")
    for name, array in arrays.items():
        group.create_array(name, data=array)
    group.attrs.update({"layout": _json_ready(asdict(layout))})

    pq.write_table(pa.Table.from_pylist([_json_ready(row) for row in index_rows]), out / "canvas_index.parquet")
    summary = {
        "layout": _json_ready(asdict(layout)),
        "tile_count": len(index_rows),
        "arrays": {name: {"shape": list(array.shape), "dtype": str(array.dtype)} for name, array in arrays.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")


def create_chunked_canvas_group(
    output_dir: str | Path,
    layout: CanvasLayout,
    *,
    layers: tuple[int, ...] = (0, 1, 2, 3),
) -> zarr.Group:
    """Create empty tile-chunked canvas arrays for memory-bounded full-map writes."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    group = zarr.open_group(str(out / "canvas.zarr"), mode="w")
    alpha_h, alpha_w = layout.alpha_shape
    height_h, height_w = layout.height_shape
    mcly_h, mcly_w = layout.mcly_shape
    n_layers = len(layers)
    group.create_array("alpha_256", shape=(alpha_h, alpha_w, n_layers), chunks=(ALPHA_TILE_SIZE, ALPHA_TILE_SIZE, 1), dtype=np.float32, fill_value=0.0)
    group.create_array("tile_id_256", shape=(alpha_h, alpha_w), chunks=(ALPHA_TILE_SIZE, ALPHA_TILE_SIZE), dtype=np.int32, fill_value=-1)
    group.create_array("height_257", shape=(height_h, height_w), chunks=(HEIGHT_TILE_SIZE, HEIGHT_TILE_SIZE), dtype=np.float32, fill_value=0.0)
    group.create_array("normal_xyz", shape=(height_h, height_w, 3), chunks=(HEIGHT_TILE_SIZE, HEIGHT_TILE_SIZE, 3), dtype=np.float32, fill_value=0.0)
    group.create_array("tile_id_257", shape=(height_h, height_w), chunks=(HEIGHT_TILE_SIZE, HEIGHT_TILE_SIZE), dtype=np.int32, fill_value=-1)
    group.create_array("mcly_texture_ids", shape=(mcly_h, mcly_w, 4), chunks=(MCLY_TILE_SIZE, MCLY_TILE_SIZE, 4), dtype=np.int32, fill_value=-1)
    group.create_array("mcly_layer_mask", shape=(mcly_h, mcly_w, 4), chunks=(MCLY_TILE_SIZE, MCLY_TILE_SIZE, 4), dtype=np.float32, fill_value=0.0)
    group.create_array("tile_id_16", shape=(mcly_h, mcly_w), chunks=(MCLY_TILE_SIZE, MCLY_TILE_SIZE), dtype=np.int32, fill_value=-1)
    group.create_array("alpha_layer_indices", data=np.asarray(layers, dtype=np.int32))
    group.attrs.update({"layout": _json_ready(asdict(layout))})
    return group


def write_tile_to_canvas(
    group: zarr.Group,
    record: CanvasTileRecord,
    layout: CanvasLayout,
    source_root: zarr.Group,
    *,
    layer_indices: tuple[int, ...] = (0, 1, 2, 3),
) -> None:
    """Write one tile's signals into a chunked canvas group."""
    ax, ay = alpha_origin(record, layout)
    hx, hy = height_origin(record, layout)
    mx, my = mcly_origin(record, layout)
    tile_id = int(record.tile_id)

    if record.has_alpha_256 and "alpha_256" in source_root:
        alpha = np.clip(source_root["alpha_256"][tile_id].astype(np.float32), 0.0, 1.0)
        group["alpha_256"][ay : ay + ALPHA_TILE_SIZE, ax : ax + ALPHA_TILE_SIZE, :] = alpha[:, :, layer_indices]
        group["tile_id_256"][ay : ay + ALPHA_TILE_SIZE, ax : ax + ALPHA_TILE_SIZE] = tile_id
    if record.has_height_257 and "height_257" in source_root:
        group["height_257"][hy : hy + HEIGHT_TILE_SIZE, hx : hx + HEIGHT_TILE_SIZE] = source_root["height_257"][tile_id].astype(np.float32)
    if record.has_normal_xyz and "normal_xyz" in source_root:
        group["normal_xyz"][hy : hy + HEIGHT_TILE_SIZE, hx : hx + HEIGHT_TILE_SIZE, :] = source_root["normal_xyz"][tile_id].astype(np.float32)
    if record.has_height_257 or record.has_normal_xyz:
        if "height_257" in source_root or "normal_xyz" in source_root:
            group["tile_id_257"][hy : hy + HEIGHT_TILE_SIZE, hx : hx + HEIGHT_TILE_SIZE] = tile_id
    if record.has_mcly_texture_ids and "mcly_texture_ids" in source_root:
        group["mcly_texture_ids"][my : my + MCLY_TILE_SIZE, mx : mx + MCLY_TILE_SIZE, :] = source_root["mcly_texture_ids"][tile_id].astype(np.int32)
    if record.has_mcly_layer_mask and "mcly_layer_mask" in source_root:
        group["mcly_layer_mask"][my : my + MCLY_TILE_SIZE, mx : mx + MCLY_TILE_SIZE, :] = source_root["mcly_layer_mask"][tile_id].astype(np.float32)
    if record.has_mcly_texture_ids or record.has_mcly_layer_mask:
        if "mcly_texture_ids" in source_root or "mcly_layer_mask" in source_root:
            group["tile_id_16"][my : my + MCLY_TILE_SIZE, mx : mx + MCLY_TILE_SIZE] = tile_id


def write_debug_overlay(output_dir: str | Path, layout: CanvasLayout, alpha_canvas: np.ndarray, *, layer_slot: int = 0) -> Path:
    """Write a small seam overlay for visual proof of tile/page boundaries."""
    out = Path(output_dir) / "overlays"
    out.mkdir(parents=True, exist_ok=True)
    alpha = np.clip(alpha_canvas[:, :, int(layer_slot)], 0.0, 1.0)
    img = Image.fromarray((alpha * 255.0).astype(np.uint8), mode="L").convert("RGB")
    draw = ImageDraw.Draw(img)
    for tx in range(1, layout.tile_count_x):
        x = tx * ALPHA_TILE_SIZE
        draw.line((x, 0, x, img.height), fill=(255, 0, 0), width=1)
    for ty in range(1, layout.tile_count_y):
        y = ty * ALPHA_TILE_SIZE
        draw.line((0, y, img.width, y), fill=(255, 0, 0), width=1)
    max_preview_side = 2048
    if max(img.size) > max_preview_side:
        img.thumbnail((max_preview_side, max_preview_side), Image.Resampling.NEAREST)
    path = out / f"alpha_layer_slot_{int(layer_slot)}_tile_seams.png"
    img.save(path)
    return path


def _cell(table: pa.Table, row_idx: int, column: str, default: Any) -> Any:
    if column not in table.column_names:
        return default
    return table.column(column)[row_idx].as_py()


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
