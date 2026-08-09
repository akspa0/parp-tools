"""Render operator-facing visual coverage atlases for the v60 control corpus.

The script consumes the C#-generated NPZ rows. It does not synthesize a second terrain or lighting
signal. The outputs are deliberately review-oriented: one family atlas, one variant atlas, and a
JSON coverage report that shows which complexity buckets and signal panels are present.
"""

# The repository's CLI convention bootstraps ``src`` before importing the package.
# Ruff's import sorter cannot safely move that import above the path setup.
# ruff: noqa: I001

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v60.control_corpus import (  # noqa: E402
    CONTROL_FAMILY_BUCKETS,
    CROSS_TILE_FAMILIES,
    EXPECTED_CONTROL_FAMILIES,
    load_control_manifest,
    validate_control_corpus,
)


PANEL = 192
HEADER = 58
ROW_LABEL = 32
GAP = 4
BUCKET_COLORS = {
    "easy": (72, 150, 92),
    "medium": (63, 130, 190),
    "hard": (214, 150, 45),
    "pathological": (190, 68, 68),
}
HEIGHT_STOPS = np.asarray(
    [[18, 37, 82], [36, 104, 102], [116, 145, 84], [220, 186, 112], [248, 245, 225]],
    dtype=np.float32,
)
EDGE_STOPS = np.asarray(
    [[8, 12, 28], [43, 35, 106], [154, 50, 86], [242, 133, 53], [255, 238, 170]],
    dtype=np.float32,
)


def _font() -> ImageFont.ImageFont:
    return ImageFont.load_default()


def _finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = np.asarray(values)[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError("control signal contains no finite values")
    return float(finite.min()), float(finite.max())


def _palette(values: np.ndarray, stops: np.ndarray, lo: float, hi: float) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    normalized = np.clip((np.nan_to_num(data, nan=lo) - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    position = normalized * (len(stops) - 1)
    lower = np.floor(position).astype(np.int32)
    upper = np.minimum(lower + 1, len(stops) - 1)
    blend = (position - lower)[..., None]
    return np.clip(
        stops[lower] * (1.0 - blend) + stops[upper] * blend,
        0,
        255,
    ).astype(np.uint8)


def _height_rgb(height: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return _palette(height, HEIGHT_STOPS, lo, hi)


def _shadow_rgb(shadow: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(shadow, dtype=np.float32), 0.0, 1.0)
    gray = np.rint(values * 255.0).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _normal_rgb(normals: np.ndarray) -> np.ndarray:
    values = np.asarray(normals, dtype=np.float32)
    if values.ndim != 3 or values.shape[-1] != 3:
        raise ValueError(f"expected normal_xyz (H,W,3), got {values.shape}")
    return np.clip((values * 0.5 + 0.5) * 255.0, 0, 255).astype(np.uint8)


def _height_edge(height: np.ndarray) -> np.ndarray:
    values = np.asarray(height, dtype=np.float32)
    gy, gx = np.gradient(values)
    return np.sqrt((gx * gx) + (gy * gy)).astype(np.float32)


def _resize_rgb(values: np.ndarray) -> Image.Image:
    image = Image.fromarray(np.asarray(values, dtype=np.uint8), mode="RGB")
    return image.resize((PANEL, PANEL), Image.Resampling.NEAREST)


def _resize_rgb_to(values: np.ndarray, width: int, height: int) -> Image.Image:
    image = Image.fromarray(np.asarray(values, dtype=np.uint8), mode="RGB")
    return image.resize((int(width), int(height)), Image.Resampling.NEAREST)


def _load_row(root: Path, row: dict[str, Any]) -> dict[str, Any]:
    path = root / str(row["npz"])
    with np.load(path, allow_pickle=False) as payload:
        required = ("height_257", "terrain_shadow_256", "mcnr_normal_xyz")
        missing = [name for name in required if name not in payload]
        if missing:
            raise ValueError(f"{path.name}: missing visual signals {missing}")
        height = np.asarray(payload["height_257"], dtype=np.float32)
        shadow = np.asarray(payload["terrain_shadow_256"], dtype=np.float32)
        normals = np.asarray(payload["mcnr_normal_xyz"], dtype=np.float32)
    if height.shape != (257, 257) or shadow.shape != (256, 256) or normals.shape != (257, 257, 3):
        raise ValueError(f"{path.name}: unexpected visual shapes {height.shape}, {shadow.shape}, {normals.shape}")
    if not np.isfinite(height).all() or not np.isfinite(shadow).all() or not np.isfinite(normals).all():
        raise ValueError(f"{path.name}: non-finite visual signal")
    return {"row": row, "height": height, "shadow": shadow, "normals": normals}


def _draw_panel(sheet: Image.Image, image: Image.Image, x: int, y: int, label: str) -> None:
    draw = ImageDraw.Draw(sheet)
    sheet.paste(image, (x, y))
    draw.rectangle((x, y + PANEL, x + PANEL, y + PANEL + ROW_LABEL), fill=(29, 32, 39))
    draw.text((x + 5, y + PANEL + 9), label, fill=(230, 232, 236), font=_font())


def _family_order(families: list[str]) -> list[str]:
    rank = {family: index for index, family in enumerate(EXPECTED_CONTROL_FAMILIES)}
    return sorted(families, key=lambda family: (rank.get(family, len(rank)), family))


def _stitch_cross_tiles(items: list[dict[str, Any]], key: str) -> np.ndarray:
    canvas = np.zeros((512, 512) if key != "normals" else (512, 512, 3), dtype=np.float32)
    positions: set[tuple[int, int]] = set()
    for item in items:
        row = item["row"]
        tile_x = int(row.get("pattern_tile_x", -1))
        tile_y = int(row.get("pattern_tile_y", -1))
        if tile_x not in (0, 1) or tile_y not in (0, 1):
            continue
        positions.add((tile_x, tile_y))
        values = np.asarray(item[key])[:256, :256]
        canvas[tile_y * 256 : (tile_y + 1) * 256, tile_x * 256 : (tile_x + 1) * 256] = values
    if positions != {(0, 0), (0, 1), (1, 0), (1, 1)}:
        missing = sorted({(0, 0), (0, 1), (1, 0), (1, 1)} - positions)
        raise ValueError(f"cross-tile visual stitch is missing positions {missing}")
    return canvas


def render_visual_review(corpus: Path, output_dir: Path, variants_per_family: int) -> dict[str, Any]:
    report = validate_control_corpus(corpus)
    if not report["valid"]:
        raise ValueError("control corpus is invalid; run the validator first: " + "; ".join(report["failures"][:5]))

    manifest = load_control_manifest(corpus)
    rows_by_family: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in manifest["rows"]:
        rows_by_family[str(row["control_family"])].append(row)
    for rows in rows_by_family.values():
        rows.sort(key=lambda row: int(row.get("variant", 0)))

    loaded: dict[str, list[dict[str, Any]]] = {}
    all_heights: list[np.ndarray] = []
    for family in _family_order(list(rows_by_family)):
        loaded[family] = []
        for row in rows_by_family[family]:
            item = _load_row(corpus, row)
            loaded[family].append(item)
            all_heights.append(item["height"])
    height_lo = min(float(values.min()) for values in all_heights)
    height_hi = max(float(values.max()) for values in all_heights)

    output_dir.mkdir(parents=True, exist_ok=True)
    families = _family_order(list(loaded))
    family_width = 4 * PANEL + 3 * GAP
    family_sheet = Image.new("RGB", (family_width, HEADER + len(families) * (PANEL + ROW_LABEL)), (18, 21, 27))
    draw = ImageDraw.Draw(family_sheet)
    draw.text((8, 8), "v60 control family atlas: height / shadow / normals / height edges", fill=(245, 245, 245), font=_font())
    draw.text((8, 28), f"rows={len(manifest['rows'])} families={len(families)} height_range=[{height_lo:.1f}, {height_hi:.1f}]", fill=(180, 186, 194), font=_font())

    family_summary: list[dict[str, Any]] = []
    for family_index, family in enumerate(families):
        representative = loaded[family][0]
        height = representative["height"]
        shadow = representative["shadow"]
        normals = representative["normals"]
        edge = _height_edge(height)
        y = HEADER + family_index * (PANEL + ROW_LABEL)
        bucket = str(representative["row"].get("complexity_bucket", CONTROL_FAMILY_BUCKETS.get(family, "unknown")))
        draw.rectangle((0, y, family_width, y + PANEL + ROW_LABEL), outline=BUCKET_COLORS.get(bucket, (130, 130, 130)), width=2)
        draw.text((5, y + 7), f"{family} [{bucket}]", fill=(245, 245, 245), font=_font())
        panels = (
            _height_rgb(height, height_lo, height_hi),
            _shadow_rgb(shadow),
            _normal_rgb(normals),
            _palette(edge, EDGE_STOPS, float(edge.min()), float(edge.max())),
        )
        for panel_index, (panel, label) in enumerate(
            zip(panels, ("height", "shadow", "normals", "edges"), strict=True)
        ):
            _draw_panel(family_sheet, _resize_rgb(panel), panel_index * (PANEL + GAP), y, label)

        family_summary.append(
            {
                "family": family,
                "complexity_bucket": bucket,
                "variant_count": len(loaded[family]),
                "height_min": float(min(item["height"].min() for item in loaded[family])),
                "height_max": float(max(item["height"].max() for item in loaded[family])),
                "height_gradient_mean": float(np.mean([_height_edge(item["height"]).mean() for item in loaded[family]])),
                "shadow_std_mean": float(np.mean([item["shadow"].std() for item in loaded[family]])),
            }
        )
    family_path = output_dir / "control-family-atlas.png"
    family_sheet.save(family_path)

    max_variants = max((min(variants_per_family, len(loaded[family])) for family in families), default=0)
    variant_width = max_variants * PANEL + max(0, max_variants - 1) * GAP
    variant_sheet = Image.new("RGB", (variant_width, HEADER + len(families) * (PANEL + ROW_LABEL)), (18, 21, 27))
    draw = ImageDraw.Draw(variant_sheet)
    draw.text((8, 8), "v60 control variant atlas: height fields", fill=(245, 245, 245), font=_font())
    draw.text((8, 28), "Each row is one complexity family; columns are deterministic variants.", fill=(180, 186, 194), font=_font())
    for family_index, family in enumerate(families):
        y = HEADER + family_index * (PANEL + ROW_LABEL)
        bucket = str(loaded[family][0]["row"].get("complexity_bucket", CONTROL_FAMILY_BUCKETS.get(family, "unknown")))
        for variant_index, item in enumerate(loaded[family][:variants_per_family]):
            x = variant_index * (PANEL + GAP)
            tile_label = ""
            if int(item["row"].get("pattern_tile_span", 1)) > 1:
                tile_label = f" tile={int(item['row'].get('pattern_tile_x', -1))},{int(item['row'].get('pattern_tile_y', -1))}"
            _draw_panel(
                variant_sheet,
                _resize_rgb(_height_rgb(item["height"], height_lo, height_hi)),
                x,
                y,
                f"{family} v{int(item['row'].get('variant', variant_index)):02d}{tile_label}",
            )
        draw.rectangle((0, y, variant_width, y + PANEL + ROW_LABEL), outline=BUCKET_COLORS.get(bucket, (130, 130, 130)), width=2)
    variant_path = output_dir / "control-variant-atlas.png"
    variant_sheet.save(variant_path)

    cross_tile_summary: list[dict[str, Any]] = []
    cross_families = [family for family in families if family in CROSS_TILE_FAMILIES]
    cross_path: Path | None = None
    if cross_families:
        cross_panel = (2 * PANEL) + GAP
        cross_width = (2 * cross_panel) + GAP
        cross_sheet = Image.new(
            "RGB",
            (cross_width, HEADER + len(cross_families) * (cross_panel + ROW_LABEL)),
            (18, 21, 27),
        )
        cross_draw = ImageDraw.Draw(cross_sheet)
        cross_draw.text((8, 8), "v60 cross-tile atlas: global 2x2 pattern continuity", fill=(245, 245, 245), font=_font())
        cross_draw.text((8, 28), "Each large panel stitches four tile payloads; seams are intentional review boundaries.", fill=(180, 186, 194), font=_font())

        for family_index, family in enumerate(cross_families):
            items = loaded[family]
            y = HEADER + family_index * (cross_panel + ROW_LABEL)
            bucket = str(items[0]["row"].get("complexity_bucket", CONTROL_FAMILY_BUCKETS[family]))
            stitched_height = _stitch_cross_tiles(items, "height")
            stitched_shadow = _stitch_cross_tiles(items, "shadow")
            height_image = _resize_rgb_to(_height_rgb(stitched_height, height_lo, height_hi), cross_panel, cross_panel)
            shadow_image = _resize_rgb_to(_shadow_rgb(stitched_shadow), cross_panel, cross_panel)
            cross_sheet.paste(height_image, (0, y))
            cross_sheet.paste(shadow_image, (cross_panel + GAP, y))
            cross_draw.rectangle((0, y, cross_panel - 1, y + cross_panel - 1), outline=BUCKET_COLORS[bucket], width=2)
            cross_draw.rectangle((cross_panel + GAP, y, (2 * cross_panel) + GAP - 1, y + cross_panel - 1), outline=BUCKET_COLORS[bucket], width=2)
            cross_draw.rectangle((0, y + cross_panel, cross_panel, y + cross_panel + ROW_LABEL), fill=(29, 32, 39))
            cross_draw.rectangle((cross_panel + GAP, y + cross_panel, (2 * cross_panel) + GAP, y + cross_panel + ROW_LABEL), fill=(29, 32, 39))
            cross_draw.text((6, y + cross_panel + 9), f"{family} [{bucket}] height", fill=(230, 232, 236), font=_font())
            cross_draw.text((cross_panel + GAP + 6, y + cross_panel + 9), f"{family} [{bucket}] shadow", fill=(230, 232, 236), font=_font())
            cross_draw.line((cross_panel, y, cross_panel, y + cross_panel), fill=(255, 255, 255), width=2)
            cross_draw.line((0, y + cross_panel, cross_panel, y + cross_panel), fill=(255, 255, 255), width=2)
            cross_draw.line((cross_panel + GAP + cross_panel, y, cross_panel + GAP + cross_panel, y + cross_panel), fill=(255, 255, 255), width=2)
            cross_draw.line((cross_panel + GAP, y + cross_panel, (2 * cross_panel) + GAP, y + cross_panel), fill=(255, 255, 255), width=2)
            cross_tile_summary.append(
                {
                    "family": family,
                    "pattern_id": str(items[0]["row"].get("pattern_id", "")),
                    "tile_positions": sorted(
                        [
                            [int(item["row"].get("pattern_tile_x", -1)), int(item["row"].get("pattern_tile_y", -1))]
                            for item in items
                        ]
                    ),
                    "stitched": True,
                }
            )
        cross_path = output_dir / "control-cross-tile-atlas.png"
        cross_sheet.save(cross_path)

    family_names = set(families)
    bucket_family_counts: dict[str, int] = defaultdict(int)
    for summary in family_summary:
        bucket_family_counts[str(summary["complexity_bucket"])] += 1
    cross_tile_complete = all(
        {tuple(position) for position in summary["tile_positions"]} == {(0, 0), (0, 1), (1, 0), (1, 1)}
        for summary in cross_tile_summary
    ) if cross_tile_summary else not cross_families
    coverage = {
        "schema": "v60-control-visual-review-v1",
        "corpus": str(corpus.resolve()),
        "outputs": [
            str(family_path.resolve()),
            str(variant_path.resolve()),
            *([] if cross_path is None else [str(cross_path.resolve())]),
        ],
        "signals_rendered": ["height_257", "terrain_shadow_256", "mcnr_normal_xyz", "height_edges"],
        "row_count": len(manifest["rows"]),
        "family_count": len(families),
        "expected_family_count": len(EXPECTED_CONTROL_FAMILIES),
        "missing_expected_families": [family for family in EXPECTED_CONTROL_FAMILIES if family not in family_names],
        "unexpected_families": [family for family in families if family not in CONTROL_FAMILY_BUCKETS],
        "coverage_complete": family_names == set(EXPECTED_CONTROL_FAMILIES) and cross_tile_complete,
        "cross_tile_complete": cross_tile_complete,
        "cross_tile_coverage": cross_tile_summary,
        "complexity_bucket_family_counts": dict(sorted(bucket_family_counts.items())),
        "alignment_policy": report.get("alignment_policy"),
        "alignment_modes": report.get("alignment_modes", {}),
        "field_offset_ranges": report.get("field_offset_ranges", {}),
        "families": family_summary,
    }
    coverage_path = output_dir / "control-visual-review.json"
    coverage_path.write_text(json.dumps(coverage, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {family_path}")
    print(f"Wrote {variant_path}")
    print(f"Wrote {coverage_path}")
    return coverage


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--variants-per-family", type=int, default=4)
    args = parser.parse_args(argv)
    if args.variants_per_family < 1:
        parser.error("--variants-per-family must be >= 1")
    render_visual_review(args.corpus, args.output_dir, args.variants_per_family)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
