"""Render human-review contact sheets directly from a canonical v50 datastore.

The authored client minimap may contain baked objects while the synthesized minimaps are
terrain-only.  This review surface therefore presents the signals side by side and records object
placement counts; it does not claim that pixel equality is a validity requirement.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from PIL import Image, ImageDraw, ImageFont


CELL_SIZE = 256
HEADER_HEIGHT = 86
ROW_TITLE_HEIGHT = 28
CELL_LABEL_HEIGHT = 32
COLUMN_TITLES = (
    "Authored client 256",
    "Synthetic terrain 256",
    "Synthetic detail 1024 overview",
    "Synthetic detail native center crop",
    "Relative height target",
    "Terrain normals",
)


class StoreVisualReviewError(ValueError):
    """Raised when a store cannot produce an honest visual review."""


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _rgb_image(array: np.ndarray, *, size: int = CELL_SIZE) -> Image.Image:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.LANCZOS)
    return image


def _relative_height_rgb(height: np.ndarray) -> np.ndarray:
    values = np.asarray(height, dtype=np.float32)
    if not np.isfinite(values).all():
        raise StoreVisualReviewError("height_257 contains non-finite values")
    lo = float(values.min())
    hi = float(values.max())
    scale = max(hi - lo, 1.0)
    t = np.clip((values - lo) / scale, 0.0, 1.0)
    # Dark blue -> green -> tan -> white. The 1-unit floor matches the relative-height target
    # contract, so nearly-flat tiles remain visually nearly-flat instead of amplifying noise.
    stops = np.asarray(
        [
            [18, 34, 70],
            [42, 112, 80],
            [156, 145, 86],
            [238, 236, 220],
        ],
        dtype=np.float32,
    )
    position = t * (len(stops) - 1)
    lower = np.floor(position).astype(np.int32)
    upper = np.minimum(lower + 1, len(stops) - 1)
    blend = (position - lower)[..., None]
    return np.clip(stops[lower] * (1.0 - blend) + stops[upper] * blend, 0, 255).astype(np.uint8)


def _normal_rgb(normals: np.ndarray) -> np.ndarray:
    values = np.asarray(normals, dtype=np.float32)
    if not np.isfinite(values).all():
        raise StoreVisualReviewError("normal_xyz contains non-finite values")
    return np.clip((values + 1.0) * 127.5, 0, 255).astype(np.uint8)


def _populated(array, row: int) -> bool:
    return bool(np.asarray(array[row]).any())


def _select_rows(group, index_rows: list[dict], sample_count: int) -> list[tuple[int, str]]:
    candidates: list[tuple[int, float, int]] = []
    for row_id in range(len(index_rows)):
        if not _populated(group["minimap_rgb_authored"], row_id):
            continue
        if not _populated(group["minimap_rgb"], row_id):
            continue
        height = np.asarray(group["height_257"][row_id], dtype=np.float32)
        height_range = float(height.max() - height.min())
        placements = int(group["mddf_count"][row_id]) + int(group["modf_count"][row_id])
        candidates.append((row_id, height_range, placements))

    if not candidates:
        raise StoreVisualReviewError("no rows carry both authored and synthesized minimaps")

    by_range = sorted(candidates, key=lambda item: (item[1], item[0]))
    wanted = max(1, min(sample_count, len(by_range)))
    chosen: list[tuple[int, str]] = []
    used: set[int] = set()

    # Reserve one slot for the most object-rich authored tile. This makes the expected authored
    # versus terrain-only mismatch visible instead of accidentally sampling only empty terrain.
    quantile_slots = max(1, wanted - 1)
    for slot in range(quantile_slots):
        fraction = 0.5 if quantile_slots == 1 else slot / (quantile_slots - 1)
        position = round(fraction * (len(by_range) - 1))
        row_id = by_range[position][0]
        if row_id not in used:
            chosen.append((row_id, f"height-range q={fraction:.2f}"))
            used.add(row_id)

    object_rich = max(candidates, key=lambda item: (item[2], item[1], -item[0]))
    if object_rich[0] not in used and len(chosen) < wanted:
        chosen.append((object_rich[0], "highest placement count"))
        used.add(object_rich[0])

    for row_id, _, _ in reversed(by_range):
        if len(chosen) >= wanted:
            break
        if row_id not in used:
            chosen.append((row_id, "height-range fill"))
            used.add(row_id)
    return chosen


def render_store_review(store: Path, output: Path, *, sample_count: int = 6) -> dict:
    import zarr

    group = zarr.open_group(str(store), mode="r")
    required = {
        "minimap_rgb_authored",
        "minimap_rgb",
        "minimap_rgb_1024",
        "height_257",
        "normal_xyz",
        "mddf_count",
        "modf_count",
    }
    missing = sorted(required - set(group.array_keys()))
    if missing:
        raise StoreVisualReviewError(f"store is missing visual-review signals: {missing}")

    index_path = store / "index.parquet"
    if not index_path.exists():
        raise StoreVisualReviewError(f"store has no index.parquet: {store}")
    index_rows = pq.read_table(index_path).to_pylist()
    if len(index_rows) != int(group["height_257"].shape[0]):
        raise StoreVisualReviewError("index row count does not match height_257")

    selected = _select_rows(group, index_rows, sample_count)
    width = len(COLUMN_TITLES) * CELL_SIZE
    block_height = ROW_TITLE_HEIGHT + CELL_SIZE + CELL_LABEL_HEIGHT
    sheet = Image.new("RGB", (width, HEADER_HEIGHT + len(selected) * block_height), (20, 22, 26))
    draw = ImageDraw.Draw(sheet)
    font = _font()
    map_name = str(index_rows[selected[0][0]].get("map", store.stem))
    draw.text((10, 10), f"v50 datastore visual review: {map_name}", fill=(245, 245, 245), font=font)
    draw.text((10, 30), str(store.resolve()), fill=(178, 184, 194), font=font)
    draw.text(
        (10, 50),
        "Authored may contain client-baked objects; synthesized columns are terrain-only by design. Pixel equality is not required.",
        fill=(255, 210, 110),
        font=font,
    )

    summary_rows = []
    for sample_index, (row_id, selection_reason) in enumerate(selected):
        metadata = index_rows[row_id]
        authored = np.asarray(group["minimap_rgb_authored"][row_id], dtype=np.uint8)
        synthetic = np.asarray(group["minimap_rgb"][row_id], dtype=np.uint8)
        detail = np.asarray(group["minimap_rgb_1024"][row_id], dtype=np.uint8)
        if not detail.any():
            raise StoreVisualReviewError(f"selected row {row_id} has no populated minimap_rgb_1024")
        height = np.asarray(group["height_257"][row_id], dtype=np.float32)
        normals = np.asarray(group["normal_xyz"][row_id], dtype=np.float32)
        mddf_count = int(group["mddf_count"][row_id])
        modf_count = int(group["modf_count"][row_id])
        height_range = float(height.max() - height.min())

        detail_overview = _rgb_image(detail)
        detail_draw = ImageDraw.Draw(detail_overview)
        detail_draw.rectangle((96, 96, 159, 159), outline=(255, 70, 70), width=2)
        center = detail.shape[0] // 2
        half = CELL_SIZE // 2
        detail_crop = _rgb_image(detail[center - half : center + half, center - half : center + half])
        cells = (
            _rgb_image(authored),
            _rgb_image(synthetic),
            detail_overview,
            detail_crop,
            _rgb_image(_relative_height_rgb(height)),
            _rgb_image(_normal_rgb(normals)),
        )

        block_y = HEADER_HEIGHT + sample_index * block_height
        title = (
            f"row={row_id} tile={metadata.get('tile_x')},{metadata.get('tile_y')}  "
            f"height_range={height_range:.3f}  objects: MDDF={mddf_count} MODF={modf_count}  [{selection_reason}]"
        )
        draw.rectangle((0, block_y, width, block_y + ROW_TITLE_HEIGHT), fill=(34, 38, 45))
        draw.text((8, block_y + 8), title, fill=(230, 232, 236), font=font)
        image_y = block_y + ROW_TITLE_HEIGHT
        for column, (cell, label) in enumerate(zip(cells, COLUMN_TITLES)):
            x = column * CELL_SIZE
            sheet.paste(cell, (x, image_y))
            draw.rectangle(
                (x, image_y + CELL_SIZE, x + CELL_SIZE, image_y + CELL_SIZE + CELL_LABEL_HEIGHT),
                fill=(28, 31, 37),
            )
            draw.text((x + 6, image_y + CELL_SIZE + 10), label, fill=(220, 222, 226), font=font)

        summary_rows.append(
            {
                "row": row_id,
                "map": str(metadata.get("map", "")),
                "tile_x": int(metadata.get("tile_x", -1)),
                "tile_y": int(metadata.get("tile_y", -1)),
                "height_range": height_range,
                "mddf_count": mddf_count,
                "modf_count": modf_count,
                "selection_reason": selection_reason,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {
        "schema": "v50-store-visual-review-v1",
        "store": str(store.resolve()),
        "output": str(output.resolve()),
        "map": map_name,
        "authored_object_policy": "may_contain_client_baked_objects",
        "synthetic_object_policy": "terrain_only_no_objects",
        "pixel_equality_required": False,
        "rows": summary_rows,
    }


# ---------------------------------------------------------------------------
# Full-signal audit: every array the store actually carries, for the same rows
# render_store_review already selected -- discovered dynamically via
# group.array_keys() rather than a hardcoded list, so it degrades gracefully
# (skip + label) instead of crashing when a store's signal set differs from
# what any one map/build happened to produce.
# ---------------------------------------------------------------------------

# Signals already covered by the primary 6-column sheet; never duplicate them here.
_PRIMARY_SHEET_SIGNALS = frozenset(
    {"minimap_rgb_authored", "minimap_rgb", "minimap_rgb_1024", "height_257", "normal_xyz"}
)
# Not meaningfully visualizable as a 2D per-tile thumbnail: scalar/count columns, and the MDDF/MODF
# placement arrays, which are globally-flat variable-length lists across the WHOLE store (one entry
# per placement instance, not per row) sliced per-tile via *_placement_offset -- never a (H, W[, C])
# grid a generic thumbnail renderer could show. Their counts are already in each row's title text
# (from mddf_count/modf_count); a proper per-tile placement view needs a dedicated scatter-plot
# renderer, not a generic array thumbnail.
_NON_VISUAL_SIGNALS = frozenset(
    {
        "mddf_count", "modf_count", "source_group_id",
        "mddf_model_ids", "mddf_placement_data", "mddf_placement_offset", "mddf_unique_ids",
        "modf_model_ids", "modf_placement_data", "modf_placement_offset", "modf_unique_ids",
    }
)

_CATEGORICAL_PALETTE = np.asarray(
    [
        [31, 119, 180], [255, 127, 14], [44, 160, 44], [214, 39, 40],
        [148, 103, 189], [140, 86, 75], [227, 119, 194], [127, 127, 127],
        [188, 189, 34], [23, 190, 207],
    ],
    dtype=np.uint8,
)


def _normalize_grayscale_rgb(values: np.ndarray) -> np.ndarray:
    finite = np.isfinite(values)
    if not finite.any():
        return np.zeros((*values.shape, 3), dtype=np.uint8)
    lo = float(values[finite].min())
    hi = float(values[finite].max())
    scale = max(hi - lo, 1e-6)
    t = np.clip((np.nan_to_num(values, nan=lo) - lo) / scale, 0.0, 1.0)
    gray = (t * 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=-1)


def _categorical_rgb(values: np.ndarray) -> np.ndarray:
    ids = np.nan_to_num(values, nan=-1).astype(np.int64)
    palette_index = np.where(ids < 0, len(_CATEGORICAL_PALETTE) - 1, ids % len(_CATEGORICAL_PALETTE))
    return _CATEGORICAL_PALETTE[palette_index]


def _layered_grid_rgb(layers: np.ndarray, *, cell: int = 128) -> np.ndarray:
    """(H, W, K) -> a ceil(sqrt(K)) grid of per-layer grayscale panels, K <= 8."""
    height, width, count = layers.shape
    count = min(count, 8)
    cols = int(np.ceil(np.sqrt(count)))
    rows = int(np.ceil(count / cols))
    canvas = np.zeros((rows * cell, cols * cell, 3), dtype=np.uint8)
    for layer in range(count):
        panel = _normalize_grayscale_rgb(layers[:, :, layer].astype(np.float32))
        panel_image = Image.fromarray(panel, mode="RGB").resize((cell, cell), Image.Resampling.NEAREST)
        row, col = divmod(layer, cols)
        canvas[row * cell : (row + 1) * cell, col * cell : (col + 1) * cell] = np.asarray(panel_image)
    return canvas


def _auto_thumbnail(arr: np.ndarray, *, size: int = CELL_SIZE) -> Image.Image | None:
    """Best-effort RGB thumbnail for one signal array, or None if the shape isn't renderable."""
    values = np.asarray(arr)
    if values.ndim == 3 and values.shape[-1] == 3:
        rgb_arr = (
            values.astype(np.uint8)
            if values.dtype == np.uint8
            else _normalize_grayscale_rgb(values.mean(axis=-1))
        )
    elif values.ndim == 3 and values.shape[-1] in (4, 8):
        # Per-layer signals (alpha weights, texture IDs, layer masks): a small grid of panels
        # reads far more honestly than collapsing 4 independent layers into 3 RGB channels.
        rgb_arr = _layered_grid_rgb(values)
    elif values.ndim == 2 and values.dtype == np.bool_:
        rgb_arr = np.repeat((values.astype(np.uint8) * 255)[..., None], 3, axis=-1)
    elif values.ndim == 2 and np.issubdtype(values.dtype, np.integer) and int(values.max(initial=0)) < 64:
        rgb_arr = _categorical_rgb(values.astype(np.float32))
    elif values.ndim == 2:
        rgb_arr = _normalize_grayscale_rgb(values.astype(np.float32))
    else:
        return None

    image = Image.fromarray(rgb_arr, mode="RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return image


def render_full_signal_review(store: Path, output: Path, *, sample_count: int = 6) -> dict:
    """Every array the store carries for the same rows the primary sheet selected -- an honest
    cross-signal alignment check, not a promotion gate. Unrenderable/missing signals are labeled,
    never silently dropped or fabricated."""
    import zarr

    group = zarr.open_group(str(store), mode="r")
    required = {"minimap_rgb_authored", "minimap_rgb", "mddf_count", "modf_count"}
    missing = sorted(required - set(group.array_keys()))
    if missing:
        raise StoreVisualReviewError(f"store is missing signals required to select rows: {missing}")

    index_path = store / "index.parquet"
    if not index_path.exists():
        raise StoreVisualReviewError(f"store has no index.parquet: {store}")
    index_rows = pq.read_table(index_path).to_pylist()
    selected = _select_rows(group, index_rows, sample_count)

    all_keys = sorted(set(group.array_keys()))
    signal_keys = [k for k in all_keys if k not in _PRIMARY_SHEET_SIGNALS and k not in _NON_VISUAL_SIGNALS]
    if not signal_keys:
        raise StoreVisualReviewError("store has no signals beyond the primary sheet's five")

    thumb_size = 160
    label_h = 20
    cell_h = thumb_size + label_h
    cols = max(1, min(6, len(signal_keys)))
    rows_per_tile = int(np.ceil(len(signal_keys) / cols))
    block_height = ROW_TITLE_HEIGHT + rows_per_tile * cell_h
    width = cols * thumb_size
    sheet = Image.new("RGB", (width, HEADER_HEIGHT + len(selected) * block_height), (20, 22, 26))
    draw = ImageDraw.Draw(sheet)
    font = _font()
    map_name = str(index_rows[selected[0][0]].get("map", store.stem))
    draw.text((10, 10), f"v50 datastore full-signal audit: {map_name}", fill=(245, 245, 245), font=font)
    draw.text((10, 30), str(store.resolve()), fill=(178, 184, 194), font=font)
    draw.text(
        (10, 50),
        f"{len(signal_keys)} signals beyond the primary sheet, discovered from the store itself -- "
        "unrenderable or empty arrays are labeled, never fabricated.",
        fill=(255, 210, 110),
        font=font,
    )

    summary_rows = []
    for sample_index, (row_id, selection_reason) in enumerate(selected):
        metadata = index_rows[row_id]
        block_y = HEADER_HEIGHT + sample_index * block_height
        title = f"row={row_id} tile={metadata.get('tile_x')},{metadata.get('tile_y')}  [{selection_reason}]"
        draw.rectangle((0, block_y, width, block_y + ROW_TITLE_HEIGHT), fill=(34, 38, 45))
        draw.text((8, block_y + 8), title, fill=(230, 232, 236), font=font)
        image_y = block_y + ROW_TITLE_HEIGHT

        panel_status = {}
        for panel_index, key in enumerate(signal_keys):
            panel_row, panel_col = divmod(panel_index, cols)
            x = panel_col * thumb_size
            y = image_y + panel_row * cell_h
            try:
                raw = np.asarray(group[key][row_id])
                is_empty = raw.size == 0 or not raw.any()
                thumbnail = None if is_empty else _auto_thumbnail(raw, size=thumb_size)
                status = "empty" if is_empty else ("unrenderable" if thumbnail is None else "ok")
            except Exception as exc:  # noqa: BLE001 -- one bad signal must not sink the whole sheet
                thumbnail = None
                status = f"error:{type(exc).__name__}"
            panel_status[key] = status

            if thumbnail is not None:
                sheet.paste(thumbnail, (x, y))
            else:
                draw.rectangle((x, y, x + thumb_size, y + thumb_size), fill=(40, 40, 44))
                draw.text((x + 6, y + thumb_size // 2 - 6), status, fill=(180, 100, 100), font=font)
            draw.rectangle((x, y + thumb_size, x + thumb_size, y + thumb_size + label_h), fill=(28, 31, 37))
            draw.text((x + 4, y + thumb_size + 4), key, fill=(220, 222, 226), font=font)

        summary_rows.append(
            {
                "row": row_id,
                "tile_x": int(metadata.get("tile_x", -1)),
                "tile_y": int(metadata.get("tile_y", -1)),
                "selection_reason": selection_reason,
                "signal_status": panel_status,
            }
        )

    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return {
        "schema": "v50-store-full-signal-audit-v1",
        "store": str(store.resolve()),
        "output": str(output.resolve()),
        "map": map_name,
        "signals_audited": signal_keys,
        "rows": summary_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Render same-tile v50 datastore visual-review sheets")
    parser.add_argument("--store", action="append", required=True, type=Path, dest="stores")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--samples-per-store", type=int, default=6)
    parser.add_argument(
        "--full-signals",
        action="store_true",
        help="also render every other signal the store carries (beyond the primary 6-column sheet) "
        "for the same sampled rows, as a cross-signal alignment audit",
    )
    args = parser.parse_args()
    if args.samples_per_store < 1:
        raise SystemExit("--samples-per-store must be >= 1")

    reports = []
    for store in args.stores:
        output = args.output_dir / f"{store.stem}-visual-review.png"
        reports.append(render_store_review(store, output, sample_count=args.samples_per_store))
        print(f"visual review: {store} -> {output}")

        if args.full_signals:
            full_output = args.output_dir / f"{store.stem}-full-signal-audit.png"
            try:
                full_report = render_full_signal_review(store, full_output, sample_count=args.samples_per_store)
                reports.append(full_report)
                print(f"full-signal audit: {store} -> {full_output} ({len(full_report['signals_audited'])} signals)")
            except StoreVisualReviewError as exc:
                print(f"full-signal audit skipped for {store}: {exc}")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "visual-review.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
