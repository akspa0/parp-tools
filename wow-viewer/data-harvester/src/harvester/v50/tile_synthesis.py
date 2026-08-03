"""Render the weak-signal and white-plate tiles the training corpus has no visual record of.

These tiles are excluded from training for good reason — a tile with 0.4m of relief in a 500m world
teaches a height model nothing. But "excluded from training" was silently doubling as "never looked
at", and a tile whose relief is 1/1000th of the world scale is not the same thing as a tile whose
relief is exactly zero. This module makes that difference visible and keeps the record.

Every target tile gets a four-panel sheet:

- **height, per-tile autostretched** — the weak-signal amplifier as an image. A tile is normalized
  against its OWN min/max, so 0.4m of relief fills the same dynamic range 400m would. Whatever
  structure is in there shows up here or it is not there at all.
- **hillshade** of that autostretched field, through the established
  ``render_hillshade_torch`` forward model, because gradient structure reads where absolute height
  does not.
- **MCNR normals as RGB** — an independent signal. Normals can carry relief the heightmap lost.
- **minimap** where one exists.

The header carries the TRUE height range at full float precision, so "exactly 0.0" and "8e-06" are
never conflated — that distinction is the entire question for a white plate.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

SYNTHESIS_SCHEMA = "v50-weak-tile-synthesis-v1"

PANEL = 257
PANEL_LABEL_HEIGHT = 22
HEADER_HEIGHT = 58
PANEL_TITLES = ("Height (autostretched)", "Hillshade", "MCNR normals (XY amplified)", "Minimap")

MOSAIC_TILE = 96
MOSAIC_MARGIN = 64

# Signals that make a tile renderable at all; absent ones become labelled placeholders.
TARGET_CLASSIFICATIONS = frozenset(
    {"weak_signal", "weak_signal_with_minimap", "white_plate", "white_plate_with_minimap"}
)


def autostretch(field: np.ndarray) -> tuple[np.ndarray, float, float, bool]:
    """Normalize a field against its own extremes -> ``(image01, lo, hi, is_true_zero)``.

    float64 throughout: a float32 range of 1e-7 is real information about what the tile still
    carries, and computing the span in float32 can round it away.
    """
    values = np.asarray(field, dtype=np.float64)
    lo = float(values.min())
    hi = float(values.max())
    span = hi - lo
    if span <= 0.0:
        return np.full(values.shape, 0.5, dtype=np.float32), lo, hi, True
    return ((values - lo) / span).astype(np.float32), lo, hi, False


def hillshade(height: np.ndarray) -> np.ndarray:
    """Lambert hillshade in [0, 1] via the established Spec 126 forward model."""
    import torch

    from harvester.v50.terrain_lighting_torch import render_hillshade_torch

    tensor = torch.from_numpy(np.asarray(height, dtype=np.float32))[None, None]
    with torch.no_grad():
        # A low sun rakes across shallow relief; a zenith sun flattens exactly the tiles this tool
        # exists to inspect.
        shaded = render_hillshade_torch(tensor, azimuth_deg=315.0, elevation_deg=30.0)
    return shaded[0, 0].numpy()


def _gray(values01: np.ndarray) -> np.ndarray:
    gray = np.rint(np.clip(np.asarray(values01, dtype=np.float32), 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def normals_rgb(normal_xyz: np.ndarray, mcnr_mask: np.ndarray, *, amplify: bool = True) -> np.ndarray:
    """MCNR normals -> RGB, black where no MCNR vertex exists.

    The raw tangent-space encoding is useless on exactly the tiles this tool exists for: a normal
    tilted by half a degree is visually identical to straight up, so a tile measuring 24% tilted
    vertices still renders as flat blue. With ``amplify``, the XY deviation is autostretched against
    its own maximum — the same treatment the height panel gets — so weak tilt becomes visible while
    a genuinely flat tile stays flat (its maximum deviation is zero, so nothing is invented).
    """
    normals = np.asarray(normal_xyz, dtype=np.float32)
    mask = np.asarray(mcnr_mask).astype(bool)
    if amplify:
        xy = normals[..., :2].copy()
        # Scale to a HIGH PERCENTILE, not the maximum. These distributions are extremely
        # long-tailed — a real tile measures |xy| max 0.299 against a p95 of 0.0079 — so dividing
        # by the max leaves 95% of vertices within ±3/255 of neutral and the panel still reads
        # flat. The signed square root then lifts the small-tilt bulk into the visible range.
        scale = float(np.percentile(np.abs(xy[mask]), 99)) if mask.any() else 0.0
        if scale > 0.0:
            xy = np.clip(xy / scale, -1.0, 1.0)
            xy = np.sign(xy) * np.sqrt(np.abs(xy))
        normals = np.concatenate([xy, normals[..., 2:]], axis=-1)
    image = np.rint(np.clip((normals + 1.0) * 0.5, 0.0, 1.0) * 255.0).astype(np.uint8)
    image[~mask] = 0
    return image


def _placeholder(text: str, size: int = PANEL) -> np.ndarray:
    image = Image.new("RGB", (size, size), (32, 34, 40))
    draw = ImageDraw.Draw(image)
    draw.text((8, size // 2 - 6), text, fill=(150, 155, 165), font=ImageFont.load_default())
    return np.asarray(image, dtype=np.uint8)


def _fit(array: np.ndarray, size: int = PANEL) -> Image.Image:
    image = Image.fromarray(np.asarray(array, dtype=np.uint8), mode="RGB")
    if image.size != (size, size):
        image = image.resize((size, size), Image.Resampling.NEAREST)
    return image


def render_tile_sheet(panels: dict[str, np.ndarray], header: list[str], output: Path) -> None:
    """Write one tile's four-panel record."""
    width = len(PANEL_TITLES) * PANEL
    height = HEADER_HEIGHT + PANEL + PANEL_LABEL_HEIGHT
    sheet = Image.new("RGB", (width, height), (18, 20, 24))
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default()
    for line_index, line in enumerate(header[:3]):
        draw.text((8, 6 + line_index * 16), line, fill=(232, 234, 238), font=font)

    for column, title in enumerate(PANEL_TITLES):
        x = column * PANEL
        sheet.paste(_fit(panels[title]), (x, HEADER_HEIGHT))
        draw.rectangle(
            (x, HEADER_HEIGHT + PANEL, x + PANEL, HEADER_HEIGHT + PANEL + PANEL_LABEL_HEIGHT),
            fill=(28, 31, 37),
        )
        draw.text((x + 6, HEADER_HEIGHT + PANEL + 6), title, fill=(210, 214, 220), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)


def synthesize_tile(group: Any, record: dict[str, Any], output_dir: Path) -> dict[str, Any]:
    """Render one target tile and return its measurement record."""
    row = int(record["row_id"])
    height = np.asarray(group["height_257"][row], dtype=np.float32)
    stretched, lo, hi, true_zero = autostretch(height)

    panels = {
        PANEL_TITLES[0]: _gray(stretched),
        PANEL_TITLES[1]: _gray(hillshade(stretched)) if not true_zero else _placeholder("no relief"),
    }
    tilted = 0.0
    if "normal_xyz" in group and "mcnr_mask_257" in group:
        normals = np.asarray(group["normal_xyz"][row], dtype=np.float32)
        mask = np.asarray(group["mcnr_mask_257"][row]).astype(bool)
        panels[PANEL_TITLES[2]] = normals_rgb(normals, mask)
        if mask.any():
            tilted = float(np.mean(np.abs(normals[mask][:, 2]) < 0.999))
    else:
        panels[PANEL_TITLES[2]] = _placeholder("no MCNR")
    minimap = np.asarray(group["minimap_rgb"][row], dtype=np.uint8) if "minimap_rgb" in group else None
    panels[PANEL_TITLES[3]] = (
        minimap if minimap is not None and minimap.any() else _placeholder("no minimap")
    )

    span = hi - lo
    header = [
        f"{record['tile_key']}   [{record['classification']}]   row {row}",
        f"height {lo:.6f} .. {hi:.6f}   range {span:.9g}"
        + ("   TRUE ZERO (bit-exact flat)" if true_zero else ""),
        f"weak chunks {record.get('weak_chunk_count', 0)}/256   "
        f"MCNR tilted {tilted * 100:.2f}%   "
        f"neighbour range {record.get('neighbour_height_min')}..{record.get('neighbour_height_max')}   "
        f"suggested x{record.get('suggested_amplification_factor')}",
    ]
    tile_png = output_dir / "tiles" / f"{record['tile_key']}.png"
    render_tile_sheet(panels, header, tile_png)

    return {
        "tile_key": record["tile_key"],
        "map": record["map"],
        "tile_x": record["tile_x"],
        "tile_y": record["tile_y"],
        "classification": record["classification"],
        "height_min": lo,
        "height_max": hi,
        "height_range": span,
        "bit_exact_flat": true_zero,
        "mcnr_tilted_fraction": tilted,
        "has_minimap": bool(minimap is not None and minimap.any()),
        "weak_chunk_count": record.get("weak_chunk_count"),
        "suggested_amplification_factor": record.get("suggested_amplification_factor"),
        "image": str(tile_png.resolve()),
        # The hillshade panel, kept in memory for the mosaic rather than re-read from disk.
        "_mosaic": _gray(hillshade(stretched)) if not true_zero else _placeholder("", MOSAIC_TILE),
    }


def render_mosaic(records: list[dict[str, Any]], map_name: str, output: Path) -> None:
    """Stitch a map's synthesized tiles at their true (tile_x, tile_y) grid positions."""
    if not records:
        return
    xs = [r["tile_x"] for r in records]
    ys = [r["tile_y"] for r in records]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    width = (x1 - x0 + 1) * MOSAIC_TILE + MOSAIC_MARGIN
    height = (y1 - y0 + 1) * MOSAIC_TILE + MOSAIC_MARGIN
    canvas = Image.new("RGB", (width, height), (12, 13, 16))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    for record in records:
        px = MOSAIC_MARGIN + (record["tile_x"] - x0) * MOSAIC_TILE
        py = MOSAIC_MARGIN + (record["tile_y"] - y0) * MOSAIC_TILE
        canvas.paste(_fit(record["_mosaic"], MOSAIC_TILE), (px, py))
        # White plates get a distinct border so a flat cell is never mistaken for an empty cell.
        border = (200, 80, 80) if record["classification"].startswith("white_plate") else (90, 170, 220)
        draw.rectangle((px, py, px + MOSAIC_TILE - 1, py + MOSAIC_TILE - 1), outline=border)
        draw.text((px + 3, py + 3), f"{record['tile_x']},{record['tile_y']}",
                  fill=(235, 235, 235), font=font)

    draw.text((8, 8), f"{map_name}: {len(records)} synthesized tiles "
                      f"(x {x0}..{x1}, y {y0}..{y1})", fill=(240, 240, 240), font=font)
    draw.text((8, 24), "blue = weak signal (autostretched relief)   red = white plate",
              fill=(180, 185, 195), font=font)
    draw.text((8, 40), "each cell is per-tile autostretched: brightness is RELATIVE to that tile only",
              fill=(180, 185, 195), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    import argparse

    import zarr

    parser = argparse.ArgumentParser(
        description="Render the weak-signal and white-plate tiles as a historical record"
    )
    parser.add_argument("--inventory", required=True, type=Path,
                        help="a v50-tile-inventory-v1 directory (from v50_tile_inventory.py)")
    parser.add_argument("--store", required=True, type=Path, action="append", dest="stores",
                        metavar="STORE", help="a per-map v50 store; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--classification", action="append", dest="classifications", default=None,
                        help=f"override which classes to render (default: {sorted(TARGET_CLASSIFICATIONS)})")
    args = parser.parse_args()

    wanted = frozenset(args.classifications) if args.classifications else TARGET_CLASSIFICATIONS
    tiles = json.loads((args.inventory / "tiles.json").read_text(encoding="utf-8"))["tiles"]
    by_map: dict[str, list[dict[str, Any]]] = {}
    for record in tiles:
        if record["classification"] in wanted:
            by_map.setdefault(record["map"], []).append(record)
    if not by_map:
        raise SystemExit(f"inventory has no tiles in {sorted(wanted)}")

    stores = {}
    for store in args.stores:
        group = zarr.open_group(str(store), mode="r")
        import pyarrow.parquet as pq

        index = pq.read_table(store / "index.parquet").to_pylist()
        stores[str(index[0].get("map", store.stem))] = group

    all_records: list[dict[str, Any]] = []
    for map_name, records in sorted(by_map.items()):
        group = stores.get(map_name)
        if group is None:
            print(f"WARNING: no --store supplied for {map_name}; skipping {len(records)} tiles", flush=True)
            continue
        rendered = [synthesize_tile(group, record, args.output) for record in records]
        render_mosaic(rendered, map_name, args.output / f"mosaic-{map_name}.png")
        flat = sum(1 for r in rendered if r["bit_exact_flat"])
        print(f"{map_name:12s} {len(rendered):>4} tiles rendered  "
              f"({flat} bit-exact flat, {len(rendered) - flat} carry SOME relief)", flush=True)
        all_records.extend(rendered)

    for record in all_records:
        record.pop("_mosaic", None)
    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "manifest.json").write_text(
        json.dumps({"schema": SYNTHESIS_SCHEMA, "tiles": all_records}, indent=2), encoding="utf-8"
    )
    carrying = [r for r in all_records if not r["bit_exact_flat"]]
    print(f"\n[DONE] {len(all_records)} tiles -> {args.output}")
    print(f"       {len(carrying)} carry non-zero relief; "
          f"{len(all_records) - len(carrying)} are bit-exact flat")
    return 0
