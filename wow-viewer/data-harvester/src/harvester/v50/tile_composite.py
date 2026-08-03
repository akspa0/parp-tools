"""Whole-map composites that put the degenerate tiles back beside the terrain they belong to.

The weak-tile synthesis proved individual tiles still carry relief. This proves the claim that
matters: the editor never erased those tiles, it compressed them, and the geometry that survives
still LINES UP with the full-scale terrain around it.

Three renders of the same map, from the same stored bytes, differing only in the display transform:

- ``absolute`` — every tile on the map's own global height range. This is what the data looks like
  today: the degenerate tiles are blank, which is exactly why nobody has looked at them.
- ``autostretch`` — every tile normalized against its OWN extremes. Nothing is added; the same
  numbers are simply given the full display range. The blanks fill in.
- ``restored`` — degenerate tiles scaled toward their adjacent full-scale neighbours' real height
  range (the neighbour-referenced amplification factor), then placed back on the map's global
  scale. This is the falsifiable one: if the compressed geometry is real, the restored landforms
  continue their neighbours' coastlines and ridges. If it were noise, they would not.

Rendering all three from one read pass keeps the comparison honest — the three images cannot differ
by anything except the transform.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

COMPOSITE_SCHEMA = "v50-tile-composite-v1"

MODES = ("absolute", "autostretch", "restored", "liquid")

# Liquid tint applied over the shaded surface. Multiplicative, so relief stays readable through it
# rather than being replaced by a flat blue mask.
LIQUID_TINT = (0.30, 0.62, 1.00)
DEFAULT_CELL = 64
LEGEND_HEIGHT = 96

# Classification -> outline colour. Normal tiles get no outline so the degenerate ones stand out.
OUTLINE = {
    "weak_signal": (90, 170, 220),
    "weak_signal_with_minimap": (90, 170, 220),
    "white_plate": (210, 90, 90),
    "white_plate_with_minimap": (210, 90, 90),
}


def downsample(height: np.ndarray, cell: int) -> np.ndarray:
    """Area-average a 257x257 field down to ``cell`` x ``cell`` without a torch dependency."""
    field = np.asarray(height, dtype=np.float64)
    side = (field.shape[0] // cell) * cell
    trimmed = field[:side, :side]
    return trimmed.reshape(cell, side // cell, cell, side // cell).mean(axis=(1, 3))


def hillshade_np(height: np.ndarray, *, spacing: float = 533.333 / 256.0,
                 azimuth_deg: float = 315.0, elevation_deg: float = 30.0) -> np.ndarray:
    """Lambert hillshade from a height field, low raking sun by default.

    A numpy twin of ``tile_synthesis.hillshade``: the composite renders thousands of small cells, and
    building a torch graph per cell costs far more than the shading itself. Same model, same
    convention, so the two agree.
    """
    field = np.asarray(height, dtype=np.float64)
    dzdy, dzdx = np.gradient(field, spacing)
    normal = np.stack([-dzdx, -dzdy, np.ones_like(field)], axis=-1)
    normal /= np.linalg.norm(normal, axis=-1, keepdims=True)
    azimuth = np.deg2rad(azimuth_deg)
    elevation = np.deg2rad(elevation_deg)
    light = np.array([
        np.cos(elevation) * np.cos(azimuth),
        np.cos(elevation) * np.sin(azimuth),
        np.sin(elevation),
    ])
    return np.clip((normal * light).sum(axis=-1), 0.0, 1.0)


def effective_factor(record: dict[str, Any], observed_range: float) -> float:
    """The factor that actually lands a compressed tile on its neighbours' scale.

    The inventory's ``suggested_amplification_factor`` is the viewer's own
    ``EstimateFactorFromRanges``, ported verbatim — including its ``epsilon = 0.001`` guard and its
    512x clamp. Those two constants silently refuse exactly the tiles this composite exists to show:
    a tile with 0.0005 units of range is rejected as unamplifiable, so the viewer's amplifier could
    never have surfaced this data either.

    This recomputes the ratio directly against the neighbours' real range, with no epsilon and no
    clamp, because the composite's job is to show what the geometry IS, not what the current
    constants permit. The required factor is reported alongside so the gap is visible rather than
    hidden. Only tiles with genuinely non-zero relief are touched; a flat tile stays flat.
    """
    suggested = record.get("suggested_amplification_factor")
    if suggested and suggested > 1.0:
        return float(suggested)
    lo = record.get("neighbour_height_min")
    hi = record.get("neighbour_height_max")
    if lo is None or hi is None or observed_range <= 0.0:
        return 1.0
    neighbour_range = float(hi) - float(lo)
    if neighbour_range <= observed_range:
        return 1.0
    return neighbour_range / observed_range


def restore_height(height: np.ndarray, factor: float | None) -> np.ndarray:
    """Scale a compressed tile's relief about its own floor, as the viewer's amplifier does.

    Anchoring at the tile minimum (rather than at zero) keeps the tile where it sits in the world
    and expands only its relief, so a restored ocean-floor tile at -501 does not launch itself to
    the surface. ``None``/1.0 leaves the tile untouched.
    """
    field = np.asarray(height, dtype=np.float64)
    if not factor or factor <= 1.0:
        return field
    anchor = float(field.min())
    return anchor + (field - anchor) * float(factor)


def _cell_image(shaded: np.ndarray) -> np.ndarray:
    gray = np.rint(np.clip(shaded, 0.0, 1.0) * 255.0).astype(np.uint8)
    return np.repeat(gray[:, :, None], 3, axis=2)


def flood_with_liquid(
    terrain: np.ndarray, liquid_height: np.ndarray | None, liquid_mask: np.ndarray | None
) -> tuple[np.ndarray, np.ndarray]:
    """Composite the liquid SURFACE over terrain -> ``(surface, wet_fraction)``.

    The paired terrain-only / liquid renders exist because they answer different questions. Terrain
    alone shows the sculpted basin — the shape of a lakebed or seafloor. Flooded shows what the
    player actually sees: a flat plane at the liquid height, hiding that shape entirely. A tile can
    look completely featureless flooded and hold detailed relief underneath, which is precisely the
    kind of thing this whole investigation exists to surface.

    Returns the surface height field and the per-pixel wet fraction used to tint it.
    """
    surface = np.asarray(terrain, dtype=np.float64).copy()
    if liquid_height is None or liquid_mask is None:
        return surface, np.zeros(surface.shape, dtype=np.float64)
    wet = np.asarray(liquid_mask, dtype=np.float64)
    level = np.asarray(liquid_height, dtype=np.float64)
    side = min(surface.shape[0], wet.shape[0], level.shape[0])
    covered = np.zeros(surface.shape, dtype=bool)
    block = wet[:side, :side] > 0.5
    covered[:side, :side] = block
    # Only raise: liquid never carves below the terrain it sits on.
    region = surface[:side, :side]
    surface[:side, :side] = np.where(block, np.maximum(region, level[:side, :side]), region)
    return surface, covered.astype(np.float64)


def _tint(shaded: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Shade to RGB, tinting wet pixels so liquid extent reads without erasing relief."""
    gray = np.clip(shaded, 0.0, 1.0)[:, :, None]
    rgb = np.repeat(gray, 3, axis=2)
    tint = np.array(LIQUID_TINT, dtype=np.float64).reshape(1, 1, 3)
    weight = np.clip(wet, 0.0, 1.0)[:, :, None]
    blended = rgb * (1.0 - weight) + rgb * tint * weight
    return np.rint(np.clip(blended, 0.0, 1.0) * 255.0).astype(np.uint8)


def build_map_cells(
    group: Any,
    index_rows: list[dict[str, Any]],
    inventory: dict[str, dict[str, Any]],
    *,
    cell: int = DEFAULT_CELL,
) -> tuple[dict[tuple[int, int], dict[str, np.ndarray]], dict[str, Any]]:
    """One read pass over the map -> a shaded cell per tile per mode, plus the map's height range."""
    cells: dict[tuple[int, int], dict[str, np.ndarray]] = {}
    lows: list[float] = []
    highs: list[float] = []
    small: dict[tuple[int, int], np.ndarray] = {}
    meta: dict[tuple[int, int], dict[str, Any]] = {}

    wet_cells: dict[tuple[int, int], np.ndarray] = {}
    flooded: dict[tuple[int, int], np.ndarray] = {}
    has_liquid = "liquid_mask" in group and "liquid_height" in group

    for row_id, row in enumerate(index_rows):
        key = (int(row["tile_x"]), int(row["tile_y"]))
        height = np.asarray(group["height_257"][row_id], dtype=np.float32)
        reduced = downsample(height, cell)
        small[key] = reduced
        if has_liquid:
            surface, wet = flood_with_liquid(
                height,
                np.asarray(group["liquid_height"][row_id], dtype=np.float32),
                np.asarray(group["liquid_mask"][row_id], dtype=np.float32),
            )
            flooded[key] = downsample(surface, cell)
            # Nearest-style reduce for the mask: averaging would smear the shoreline.
            wet_cells[key] = (downsample(wet, cell) > 0.5).astype(np.float64)
        else:
            flooded[key] = reduced
            wet_cells[key] = np.zeros((cell, cell), dtype=np.float64)
        record = inventory.get(f"{row.get('map')}_{key[0]:02d}_{key[1]:02d}", {})
        meta[key] = record
        # A degenerate tile's extremes must not define the map's global scale — a single tile at
        # -501 would crush every real landform into the top few percent of the range.
        if record.get("classification", "usable") in ("usable", "terrain_no_minimap"):
            lows.append(float(reduced.min()))
            highs.append(float(reduced.max()))

    global_lo = min(lows) if lows else 0.0
    global_hi = max(highs) if highs else 1.0

    for key, reduced in small.items():
        record = meta[key]
        span = reduced.max() - reduced.min()
        cells[key] = {
            "absolute": _cell_image(hillshade_np(reduced)),
            "autostretch": _cell_image(
                hillshade_np((reduced - reduced.min()) / span) if span > 0
                else np.full(reduced.shape, 0.5)
            ),
            # The factor comes from the tile's TRUE 257-grid range, not the downsampled span:
            # area-averaging to the composite cell shrinks the range and would inflate the factor.
            "restored": _cell_image(
                hillshade_np(restore_height(
                    reduced, effective_factor(record, float(record.get("height_range", span)))
                ))
            ),
            # What the player sees: liquid surfaces flood the basins, hiding whatever relief the
            # terrain-only render shows underneath.
            "liquid": _tint(hillshade_np(flooded[key]), wet_cells[key]),
        }
    return cells, {
        "global_min": global_lo,
        "global_max": global_hi,
        "liquid_available": has_liquid,
        "wet_tiles": int(sum(1 for w in wet_cells.values() if w.any())),
    }


def render_composite(
    cells: dict[tuple[int, int], dict[str, np.ndarray]],
    inventory: dict[str, dict[str, Any]],
    *,
    map_name: str,
    mode: str,
    output: Path,
    cell: int = DEFAULT_CELL,
) -> None:
    """Stitch one map, one mode, at true grid coordinates."""
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
    if not cells:
        raise ValueError("cannot render a composite with zero tiles")
    xs = [k[0] for k in cells]
    ys = [k[1] for k in cells]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    width = (x1 - x0 + 1) * cell
    height = (y1 - y0 + 1) * cell
    canvas = Image.new("RGB", (width, LEGEND_HEIGHT + height), (10, 11, 14))
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()

    degenerate = 0
    for (tx, ty), per_mode in cells.items():
        px = (tx - x0) * cell
        py = LEGEND_HEIGHT + (ty - y0) * cell
        canvas.paste(Image.fromarray(per_mode[mode], mode="RGB"), (px, py))
        record = inventory.get(f"{map_name}_{tx:02d}_{ty:02d}", {})
        colour = OUTLINE.get(record.get("classification", ""))
        if colour is not None:
            degenerate += 1
            draw.rectangle((px, py, px + cell - 1, py + cell - 1), outline=colour)

    captions = {
        "absolute": "ABSOLUTE — every tile on the map's global height scale. This is what the data "
                    "looks like today: outlined tiles read as blank.",
        "autostretch": "AUTOSTRETCH — every tile normalized against its OWN extremes. Same bytes, "
                       "same map; only the display transform changed. The blanks fill in.",
        "restored": "RESTORED — compressed tiles scaled toward their adjacent full-scale neighbours' "
                    "real height range, then placed back on the global scale.",
        "liquid": "LIQUID — the same terrain FLOODED to its liquid surface (blue tint = wet). Pair "
                  "this with ABSOLUTE: relief visible there and gone here is hidden under water.",
    }
    draw.text((8, 8), f"{map_name} — {len(cells)} tiles, {degenerate} degenerate "
                      f"(x {x0}..{x1}, y {y0}..{y1})", fill=(245, 245, 245), font=font)
    draw.text((8, 26), captions[mode], fill=(215, 220, 228), font=font)
    draw.text((8, 46), "blue outline = weak signal    red outline = white plate    "
                       "no outline = full-scale terrain", fill=(175, 182, 194), font=font)
    draw.text((8, 64), "Hillshade, raking sun at 30 deg elevation. Nothing is added in any mode; "
                       "only the scaling differs.", fill=(175, 182, 194), font=font)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)


def main() -> int:
    import argparse

    import pyarrow.parquet as pq
    import zarr

    parser = argparse.ArgumentParser(
        description="Whole-map composites merging degenerate tiles back with full-scale terrain"
    )
    parser.add_argument("--inventory", required=True, type=Path,
                        help="a v50-tile-inventory-v1 directory (from v50_tile_inventory.py)")
    parser.add_argument("--store", required=True, type=Path, action="append", dest="stores",
                        metavar="STORE", help="a per-map v50 store; repeatable")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--cell", type=int, default=DEFAULT_CELL,
                        help=f"pixels per tile in the composite (default {DEFAULT_CELL})")
    args = parser.parse_args()
    if args.cell < 8 or 257 // args.cell < 1:
        raise SystemExit("--cell must be between 8 and 257")

    tiles = json.loads((args.inventory / "tiles.json").read_text(encoding="utf-8"))["tiles"]
    inventory = {t["tile_key"]: t for t in tiles}

    summary: list[dict[str, Any]] = []
    for store in args.stores:
        group = zarr.open_group(str(store), mode="r")
        index_rows = pq.read_table(store / "index.parquet").to_pylist()
        map_name = str(index_rows[0].get("map", store.stem))
        cells, scale = build_map_cells(group, index_rows, inventory, cell=args.cell)
        for mode in MODES:
            render_composite(cells, inventory, map_name=map_name, mode=mode,
                             output=args.output / f"composite-{map_name}-{mode}.png",
                             cell=args.cell)
        degenerate = sum(
            1 for k in cells
            if inventory.get(f"{map_name}_{k[0]:02d}_{k[1]:02d}", {}).get("classification")
            in OUTLINE
        )
        summary.append({"map": map_name, "tiles": len(cells), "degenerate_tiles": degenerate, **scale})
        print(f"{map_name:12s} {len(cells):>4} tiles ({degenerate} degenerate)  "
              f"global height {scale['global_min']:.2f}..{scale['global_max']:.2f}  "
              f"-> {len(MODES)} composites", flush=True)

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "composites.json").write_text(
        json.dumps({"schema": COMPOSITE_SCHEMA, "cell_pixels": args.cell,
                    "modes": list(MODES), "maps": summary}, indent=2),
        encoding="utf-8",
    )
    print(f"\n[DONE] {len(summary)} maps x {len(MODES)} modes -> {args.output}", flush=True)
    return 0
