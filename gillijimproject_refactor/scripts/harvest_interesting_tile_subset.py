#!/usr/bin/env python3
"""Build a focused subset manifest from known interesting Azeroth/EmeraldDream tiles.

By default this script derives interesting tiles from prior validation preview files:
  output/ml-training/**/previews/val_epoch_*.json

It then scans current dataset manifests under datasets/*/*/ml_dataset_manifest.json,
retains rows whose tile_name is in the interesting set, and writes:
  1) a compact subset manifest
  2) a missing-tile plan for targeted exports
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


MAPS_OF_INTEREST = {"Azeroth", "EmeraldDream"}


@dataclass(frozen=True)
class InterestingTile:
    map_name: str
    tile_name: str
    frequency: int


def normalize_map_and_tile(label: str) -> tuple[str, str] | None:
    # Preview labels are usually "Map:Map_x_y".
    if ":" in label:
        map_name, tile_name = label.split(":", 1)
        map_name = map_name.strip()
        tile_name = tile_name.strip()
    else:
        tile_name = label.strip()
        parts = tile_name.split("_")
        if len(parts) < 3:
            return None
        map_name = parts[0]

    if map_name not in MAPS_OF_INTEREST:
        return None

    if not tile_name.startswith(f"{map_name}_"):
        return None

    parts = tile_name.split("_")
    if len(parts) < 3:
        return None

    try:
        int(parts[-2])
        int(parts[-1])
    except ValueError:
        return None

    return map_name, tile_name


def load_interesting_tiles_from_previews(previews_root: Path) -> list[InterestingTile]:
    counts: Counter[tuple[str, str]] = Counter()

    files = sorted(previews_root.glob("**/previews/val_epoch_*.json"))
    for file_path in files:
        try:
            payload = json.loads(file_path.read_text(encoding="utf-8"))
        except Exception:
            continue

        labels = payload.get("labels")
        if not isinstance(labels, list):
            continue

        for raw in labels:
            if not isinstance(raw, str):
                continue
            normalized = normalize_map_and_tile(raw)
            if normalized is None:
                continue
            counts[normalized] += 1

    rows = [
        InterestingTile(map_name=map_name, tile_name=tile_name, frequency=freq)
        for (map_name, tile_name), freq in counts.items()
    ]
    rows.sort(key=lambda row: (-row.frequency, row.map_name, row.tile_name))
    return rows


def discover_manifests(datasets_root: Path) -> list[Path]:
    return sorted(path for path in datasets_root.glob("*/*/ml_dataset_manifest.json") if path.is_file())


def find_matching_tiles(
    datasets_root: Path,
    interesting_tile_names: set[str],
) -> tuple[list[dict[str, Any]], set[str], list[str]]:
    selected_rows: list[dict[str, Any]] = []
    seen_tile_names: set[str] = set()
    failures: list[str] = []

    for manifest_path in discover_manifests(datasets_root):
        try:
            payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception as exc:
            failures.append(f"{manifest_path}: {exc}")
            continue

        tiles = payload.get("tiles")
        if not isinstance(tiles, list):
            continue

        client_label = manifest_path.parent.parent.name
        map_name_from_path = manifest_path.parent.name

        for tile in tiles:
            if not isinstance(tile, dict):
                continue
            tile_name = str(tile.get("tile_name") or "").strip()
            if tile_name not in interesting_tile_names:
                continue

            map_name = str(tile.get("map_name") or map_name_from_path).strip()
            if map_name not in MAPS_OF_INTEREST:
                continue

            seen_tile_names.add(tile_name)
            selected_rows.append(
                {
                    "client_label": client_label,
                    "map_name": map_name,
                    "tile_name": tile_name,
                    "dataset_map_root": manifest_path.parent.relative_to(datasets_root).as_posix(),
                    "tile_json_path": tile.get("tile_json_path"),
                    "source_minimap_path": tile.get("source_minimap_path"),
                    "source_minimap_exists": bool(tile.get("source_minimap_exists")),
                    "heightmap_local_exists": bool(tile.get("heightmap_local_exists")),
                    "heightmap_global_exists": bool(tile.get("heightmap_global_exists")),
                    "object_count": int(tile.get("object_count") or 0),
                    "existing_alpha_mask_count": int(tile.get("existing_alpha_mask_count") or 0),
                    "existing_shadow_map_count": int(tile.get("existing_shadow_map_count") or 0),
                    "completeness_class": tile.get("completeness_class"),
                }
            )

    selected_rows.sort(key=lambda row: (row["map_name"], row["tile_name"], row["client_label"]))
    return selected_rows, seen_tile_names, failures


def to_xy(tile_name: str) -> str:
    parts = tile_name.split("_")
    return f"{parts[-2]}_{parts[-1]}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Harvest focused Azeroth/EmeraldDream tile subset from current datasets.")
    parser.add_argument("--datasets-root", default="i:/parp/parp-tools/datasets", help="Root containing datasets/<client>/<map> outputs.")
    parser.add_argument("--previews-root", default="i:/parp/parp-tools/output/ml-training", help="Root for historical val preview files.")
    parser.add_argument("--output", default="i:/parp/parp-tools/output/build-validation/ml-audit/interesting_tile_subset_manifest.json", help="Subset manifest output path.")
    parser.add_argument("--missing-plan", default="i:/parp/parp-tools/output/build-validation/ml-audit/interesting_tile_subset_missing_plan.json", help="Missing-tile export plan output path.")
    parser.add_argument("--min-frequency", type=int, default=1, help="Only keep interesting tiles observed at least this many times in previews.")
    args = parser.parse_args()

    datasets_root = Path(args.datasets_root).resolve()
    previews_root = Path(args.previews_root).resolve()
    output_path = Path(args.output).resolve()
    missing_plan_path = Path(args.missing_plan).resolve()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    missing_plan_path.parent.mkdir(parents=True, exist_ok=True)

    interesting_rows = load_interesting_tiles_from_previews(previews_root)
    interesting_rows = [row for row in interesting_rows if row.frequency >= max(args.min_frequency, 1)]

    interesting_tile_names = {row.tile_name for row in interesting_rows}
    selected_rows, seen_tile_names, failures = find_matching_tiles(datasets_root, interesting_tile_names)

    missing_tiles = [
        {
            "map_name": row.map_name,
            "tile_name": row.tile_name,
            "tile_xy": to_xy(row.tile_name),
            "frequency": row.frequency,
        }
        for row in interesting_rows
        if row.tile_name not in seen_tile_names
    ]
    missing_tiles.sort(key=lambda row: (row["map_name"], -row["frequency"], row["tile_name"]))

    missing_plan = {
        "schema_version": "interesting-tile-missing-plan.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets_root": datasets_root.as_posix(),
        "notes": "Use map_name + tile_xy with ml-export --map <map_name> --tile <tile_xy> for targeted fills.",
        "missing_tiles": missing_tiles,
    }

    manifest = {
        "schema_version": "interesting-tile-subset.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "datasets_root": datasets_root.as_posix(),
        "previews_root": previews_root.as_posix(),
        "maps_of_interest": sorted(MAPS_OF_INTEREST),
        "min_frequency": max(args.min_frequency, 1),
        "interesting_tile_count": len(interesting_rows),
        "harvested_tile_rows": len(selected_rows),
        "missing_tile_count": len(missing_tiles),
        "interesting_tiles": [
            {
                "map_name": row.map_name,
                "tile_name": row.tile_name,
                "tile_xy": to_xy(row.tile_name),
                "frequency": row.frequency,
            }
            for row in interesting_rows
        ],
        "selected_tiles": selected_rows,
        "failures": failures,
    }

    output_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    missing_plan_path.write_text(json.dumps(missing_plan, indent=2), encoding="utf-8")

    print(f"Interesting tiles: {len(interesting_rows)}")
    print(f"Selected rows: {len(selected_rows)}")
    print(f"Missing tiles: {len(missing_tiles)}")
    print(f"Subset manifest: {output_path.as_posix()}")
    print(f"Missing plan: {missing_plan_path.as_posix()}")
    if failures:
        print(f"Warnings: {len(failures)} manifest parse failures")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
