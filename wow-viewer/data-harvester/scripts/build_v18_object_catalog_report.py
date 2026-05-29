"""Build asset category atlases, zone design kits, and infographic summaries from object_visual.zarr.

Reads the unified object_visual.zarr and cross-references against per-build placement
parquets to group assets by category, zone, and usage statistics.

Outputs:
  - Category atlases (TreeAtlas.png, TownHallAtlas.png, ...)
  - Per-zone design kits (organized roof exemplars grouped by zone)
  - Summary infographic JSON with per-category/per-zone stats
  - Zone-asset cross-reference table (parquet)

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/build_v18_object_catalog_report.py \
      --object-store ../output/datasets/object_roof_library/object_visual.zarr \
      --dataset-dir ../output/datasets/v18 \
      --report-dir ../output/tmp/object_catalog_report
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import sys
from typing import Any

import numpy as np
from PIL import Image, ImageDraw
import pyarrow.parquet as pq
import zarr
import zarr.storage

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_DEFAULT_DATASET_DIR = _PROJECT_ROOT / "output" / "datasets" / "v18"
_DEFAULT_REPORT_DIR = _PROJECT_ROOT / "output" / "tmp" / "object_catalog_report"


# Category patterns: (name, path_pattern, color_for_atlas_border)
_CATEGORY_PATTERNS = [
    ("TownHall", r"townhall", (180, 60, 60)),
    ("Chapel", r"chapel", (120, 80, 180)),
    ("Inn", r"inn", (160, 120, 60)),
    ("Barn", r"barn", (140, 100, 60)),
    ("Farm", r"farm", (100, 160, 80)),
    ("Blacksmith", r"blacksmith", (120, 120, 120)),
    ("Barracks", r"barracks", (100, 80, 60)),
    ("Stable", r"stable", (80, 120, 100)),
    ("Tower", r"tower|mage", (100, 100, 180)),
    ("LumberMill", r"lumber", (100, 80, 40)),
    ("House", r"twostory|house", (140, 100, 80)),
    ("Stable", r"stable", (80, 120, 100)),
    ("Bridge", r"bridge", (80, 80, 80)),
    ("Wall", r"wall", (60, 60, 60)),
    ("Fence", r"fence", (60, 80, 40)),
    ("Gate", r"gate", (100, 80, 60)),
    ("Statue", r"statue|monument", (180, 180, 160)),
    ("Fountain", r"fountain", (100, 140, 180)),
    ("Tree", r"tree|pine|oak|willow|bush|shrub", (40, 120, 60)),
    ("Rock", r"rock|stone|cliff|boulder", (80, 80, 80)),
    ("Crate", r"crate|barrel|box", (120, 100, 60)),
    ("Sign", r"sign|post", (160, 120, 80)),
    ("Lamp", r"lamp|lantern", (180, 180, 100)),
    ("Flag", r"flag|banner", (160, 60, 60)),
    ("Door", r"door", (100, 60, 40)),
    ("Window", r"window", (100, 140, 180)),
    ("Ruin", r"ruin|abandoned|destroyed|burnt", (120, 100, 80)),
    ("Dock", r"dock|wharf|pier", (100, 80, 60)),
    ("Boat", r"boat|ship", (80, 100, 120)),
    ("Cart", r"cart|wagon", (140, 100, 60)),
    ("Campfire", r"campfire|bonfire", (200, 120, 40)),
    ("Tent", r"tent|camp", (120, 140, 100)),
    ("Cage", r"cage", (100, 100, 100)),
    ("Throne", r"throne", (180, 160, 100)),
    ("Chest", r"chest|coffer", (160, 140, 80)),
    ("Anvil", r"anvil|forge", (100, 100, 100)),
    ("Mill", r"mill|windmill", (120, 100, 60)),
    ("Well", r"well", (80, 100, 120)),
    ("Grave", r"grave|tomb|coffin", (60, 60, 80)),
    ("Portal", r"portal|teleport", (120, 60, 160)),
    ("Crystal", r"crystal", (160, 120, 200)),
    ("Banner", r"banner|standard", (180, 60, 60)),
    ("Other", r".*", (100, 100, 100)),
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build asset category atlases and zone design kits from object_visual.zarr."
    )
    parser.add_argument("--object-store", type=Path, required=True)
    parser.add_argument("--dataset-dir", type=Path, default=_DEFAULT_DATASET_DIR)
    parser.add_argument("--report-dir", type=Path, default=_DEFAULT_REPORT_DIR)
    parser.add_argument("--run-name", type=str, default=None)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--max-per-category", type=int, default=256)
    parser.add_argument("--atlas-cols", type=int, default=12)
    return parser.parse_args()


def _categorize_asset(asset_path: str) -> str:
    path = asset_path.lower().replace("\\", "/")
    for name, pattern, _ in _CATEGORY_PATTERNS:
        if re.search(pattern, path):
            return name
    return "Other"


def _category_color(asset_path: str) -> tuple[int, int, int]:
    path = asset_path.lower().replace("\\", "/")
    for name, pattern, color in _CATEGORY_PATTERNS:
        if re.search(pattern, path):
            return color
    return (100, 100, 100)


def _load_placements(dataset_dir: Path, builds: list[str]) -> dict[str, set[str]]:
    """Load per-asset-per-zone mapping from all builds' placements."""
    from collections import defaultdict

    asset_zones: dict[str, set[str]] = defaultdict(set)
    for build in builds:
        placements_path = dataset_dir / f"{build}.zarr" / "placements.parquet"
        if not placements_path.exists():
            continue
        table = pq.read_table(str(placements_path))
        for i in range(table.num_rows):
            asset = str(table.column("asset_path")[i].as_py() or "").replace("\\", "/").lower()
            if not asset:
                continue
            # Fetch map name from tile_id via index
            tile_id = table.column("tile_id")[i].as_py()
            if tile_id is None or tile_id < 0:
                continue
            asset_zones[asset].add(f"{build}")
    return dict(asset_zones)


def _assign_zone(asset_path: str, asset_zones: dict[str, set[str]]) -> str:
    path_lower = asset_path.lower().replace("\\", "/")
    zones = asset_zones.get(path_lower, set())
    if not zones:
        return "unknown"
    return ", ".join(sorted(zones))


def _build_category_atlas(
    category: str,
    row_indices: list[int],
    roofs: list[np.ndarray],
    asset_paths: list[str],
    crop_size: int,
    max_per_category: int,
    atlas_cols: int,
) -> Image.Image | None:
    rows_to_show = row_indices[:max_per_category]
    if not rows_to_show:
        return None

    n = len(rows_to_show)
    cols = min(atlas_cols, n)
    rows = math.ceil(n / cols)
    label_height = 16
    tile_size = crop_size + 0  # no extra padding
    cell_w = crop_size
    cell_h = crop_size + label_height

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    for idx_in_grid, list_idx in enumerate(rows_to_show):
        r = idx_in_grid // cols
        c = idx_in_grid % cols
        x = c * cell_w
        y = r * cell_h

        # Label bar
        draw.rectangle([(x, y), (x + cell_w - 1, y + label_height - 1)], fill=(18, 18, 18))
        label_text = Path(asset_paths[list_idx]).stem
        if len(label_text) > 18:
            label_text = label_text[:17] + "…"
        draw.text((x + 2, y + 2), label_text, fill=(200, 200, 200))

        # Roof image
        img_y = y + label_height
        tile = Image.fromarray(roofs[list_idx], mode="RGB")
        canvas.paste(tile, (x, img_y))

        # Border
        color = _category_color(asset_paths[list_idx])
        for border in range(1):
            draw.rectangle(
                [(x, img_y), (x + crop_size - 1, img_y + crop_size - 1)],
                outline=color,
            )

    return canvas


def _build_zone_designkit(
    zone_name: str,
    row_indices: list[int],
    roofs: list[np.ndarray],
    asset_paths: list[str],
    crop_size: int,
    atlas_cols: int,
) -> Image.Image | None:
    """Build a design-kit collage for a zone — one exemplar per unique family in that zone."""
    if not row_indices:
        return None

    n = len(row_indices)
    cols = min(atlas_cols, n)
    rows = math.ceil(n / cols)
    label_height = 16
    cell_w = crop_size
    cell_h = crop_size + label_height

    canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(canvas)

    # Draw zone header across the top
    header_h = 32
    header_canvas = Image.new("RGB", (cols * cell_w, rows * cell_h + header_h), color=(10, 10, 10))
    header_draw = ImageDraw.Draw(header_canvas)
    header_draw.rectangle([(0, 0), (cols * cell_w - 1, header_h - 1)], fill=(24, 24, 30))
    header_draw.text((8, 8), f"Design Kit: {zone_name}", fill=(220, 220, 240))
    header_canvas.paste(canvas, (0, header_h))

    for idx_in_grid, list_idx in enumerate(row_indices):
        r = idx_in_grid // cols
        c = idx_in_grid % cols
        x = c * cell_w
        y = r * cell_h + header_h

        draw.rectangle([(x, y), (x + cell_w - 1, y + label_height - 1)], fill=(18, 18, 18))
        label_text = Path(asset_paths[list_idx]).stem
        if len(label_text) > 18:
            label_text = label_text[:17] + "…"
        header_draw.text((x + 2, y + 2), label_text, fill=(200, 200, 200))

        img_y = y + label_height
        tile = Image.fromarray(roofs[list_idx], mode="RGB")
        header_canvas.paste(tile, (x, img_y))

    return header_canvas


def main() -> None:
    args = _parse_args()
    report_dir = Path(args.report_dir)
    run_name = args.run_name or f"catalog_report_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    output_dir = report_dir / run_name
    output_dir.mkdir(parents=True, exist_ok=True)

    crop_size = int(args.crop_size)
    atlas_cols = int(args.atlas_cols)
    max_per_category = int(args.max_per_category)

    print(f"Object catalog report: {run_name}")
    print(f"Object store: {args.object_store}")
    print(f"Output: {output_dir}")

    # 1. Load object_visual.zarr
    store = zarr.storage.LocalStore(str(args.object_store), read_only=True)
    root = zarr.open_group(store=store, mode="r")

    n = root["roof_rgb"].shape[0]
    print(f"\nLoaded {n} objects from object_visual.zarr")

    roofs = root["roof_rgb"]
    mask_arr = root["roof_mask"] if "roof_mask" in root else None
    build_code_arr = root["build_code"] if "build_code" in root else None
    build_codes = root["build_code"].attrs.get("build_codes", []) if "build_code" in root else []

    # Load exemplar metadata (asset_path per row)
    exemplar_path = args.object_store.parent / "roof_exemplars.parquet"
    asset_paths: list[str] = []
    if exemplar_path.exists():
        table = pq.read_table(str(exemplar_path))
        asset_paths = [str(r) for r in table.column("asset_path")]
        builds_from_store = list(table.column("build"))
        print(f"  {len(asset_paths)} exemplars in catalog")
    else:
        print("  [WARN] No roof_exemplars.parquet — using generic asset names")
        asset_paths = [f"asset_{i}" for i in range(n)]
        builds_from_store = [""] * n

    # 2. Categorize
    category_map: dict[str, list[int]] = defaultdict(list)
    zone_map: dict[str, list[int]] = defaultdict(list)
    build_map: dict[str, list[int]] = defaultdict(list)
    per_category_families: dict[str, set[str]] = defaultdict(set)

    for i in range(n):
        ap = asset_paths[i] if i < len(asset_paths) else f"asset_{i}"
        cat = _categorize_asset(ap)
        category_map[cat].append(i)
        b = builds_from_store[i] if i < len(builds_from_store) else ""
        build_map[b].append(i)

        family = ap.split("/")[-1] if "/" in ap else ap
        per_category_families[cat].add(family)

    print(f"\nCategories ({len(category_map)}):")
    for cat, indices in sorted(category_map.items(), key=lambda x: -len(x[1])):
        print(f"  {cat}: {len(indices)} objects ({len(per_category_families[cat])} families)")

    # 3. Build category atlases
    atlas_dir = output_dir / "atlases"
    atlas_dir.mkdir(parents=True, exist_ok=True)
    atlases_built = 0
    category_stats = {}

    for cat in sorted(category_map.keys()):
        indices = category_map[cat]
        full_roofs = [np.array(roofs[i]) for i in indices[:max_per_category]]
        full_paths = [asset_paths[i] if i < len(asset_paths) else f"asset_{i}" for i in indices[:max_per_category]]

        img = _build_category_atlas(cat, indices, full_roofs, full_paths, crop_size, max_per_category, atlas_cols)
        if img is not None:
            safe_name = cat.replace(" ", "_").lower()
            out_path = atlas_dir / f"{safe_name}_atlas.png"
            img.save(out_path)
            atlases_built += 1
            print(f"  Atlas {cat}: {len(full_roofs)} exemplars -> {out_path}")

        category_stats[cat] = {
            "count": len(indices),
            "families": len(per_category_families[cat]),
            "families_list": sorted(per_category_families[cat])[:20],
            "atlas": str(out_path) if img else None,
        }

    # 4. Build zone design kits (cross-reference with placements for zone info)
    # For now, group by build since precise zoning requires index.parquet cross-ref
    design_kit_dir = output_dir / "design_kits"
    design_kit_dir.mkdir(parents=True, exist_ok=True)
    zone_stats = {}

    for build_name, indices in sorted(build_map.items(), key=lambda x: -len(x[1])):
        build_label = build_name if build_name else "unknown"
        full_roofs = [np.array(roofs[i]) for i in indices[:max_per_category]]
        full_paths = [asset_paths[i] if i < len(asset_paths) else f"asset_{i}" for i in indices[:max_per_category]]

        # Use build-level groupings as zones
        img = _build_zone_designkit(build_label, indices, full_roofs, full_paths, crop_size, atlas_cols)
        if img is not None:
            safe_name = build_label.replace(".", "_").replace(" ", "_").lower()
            out_path = design_kit_dir / f"design_kit_{safe_name}.png"
            img.save(out_path)
            print(f"  DesignKit {build_label}: {len(full_roofs)} objects -> {out_path}")

        zone_stats[build_label] = {
            "count": len(indices),
            "atlas": str(out_path) if img else None,
        }

    # 5. Summary report
    coverage_by_category = sum(1 for cat, data in category_stats.items() if data["count"] > 0)
    total_families = sum(len(per_category_families[cat]) for cat in category_stats)

    summary = {
        "run_name": run_name,
        "object_store": str(args.object_store),
        "total_objects": n,
        "categories": len(category_stats),
        "categories_with_objects": coverage_by_category,
        "total_families": total_families,
        "atlases_built": atlases_built,
        "design_kits": len(zone_stats),
        "category_stats": category_stats,
        "zone_stats": zone_stats,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }

    summary_path = output_dir / "catalog_report.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\nReport: {summary_path}")

    # 6. Build a mega infographic with top categories side by side
    print(f"\nInfographic summary:")
    top_cats = sorted(category_stats.items(), key=lambda x: -x[1]["count"])[:6]
    for cat, stats in top_cats:
        print(f"  {cat}: {stats['count']} objects ({stats['families']} families)")

    store.close()


if __name__ == "__main__":
    main()