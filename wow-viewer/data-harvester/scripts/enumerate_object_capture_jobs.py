"""Enumerate per-object capture jobs from V18 placement tables (spec 077 T011).

The output is a JSONL ledger that the C# capture lane consumes one row at a
time. Each row pins a single (build, map, asset_path) tuple plus a
representative first placement so the capture tool knows what to load and
where to position the capture camera.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq

_SCRIPT_DIR = Path(__file__).resolve().parent
_SRC_DIR = _SCRIPT_DIR.parent / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from harvester.object_library import (  # noqa: E402
    detect_asset_type,
    is_clutter_asset,
    library_id_from_asset_path,
    normalize_asset_path,
)


def _read_table_rows(path: Path) -> list[dict[str, Any]]:
    table = pq.read_table(str(path))
    columns = table.column_names
    return [
        {col: table.column(col)[idx].as_py() for col in columns}
        for idx in range(table.num_rows)
    ]


def _build_tile_to_map(index_rows: list[dict[str, Any]]) -> dict[int, str]:
    """Map tile_id → map name from the per-tile index."""
    out: dict[int, str] = {}
    for row in index_rows:
        tile_id = row.get("tile_id")
        map_name = row.get("map")
        if tile_id is None or map_name is None:
            continue
        out[int(tile_id)] = str(map_name)
    return out


def _enumerate_jobs_for_build(
    build: str,
    placements: list[dict[str, Any]],
    tile_to_map: dict[int, str],
    *,
    include_mddf: bool,
    include_modf: bool,
    skip_clutter: bool,
) -> list[dict[str, Any]]:
    """Collapse placements into one job per (instance_type, normalized path)."""
    by_asset: dict[tuple[str, str], dict[str, Any]] = {}

    for row in placements:
        instance_type = str(row.get("instance_type", "")).lower()
        if instance_type == "mddf" and not include_mddf:
            continue
        if instance_type == "modf" and not include_modf:
            continue
        asset_path_raw = str(row.get("asset_path", "") or "")
        if not asset_path_raw:
            continue
        normalized = normalize_asset_path(asset_path_raw)
        if not normalized:
            continue
        if skip_clutter and is_clutter_asset(normalized):
            continue

        tile_id = row.get("tile_id")
        map_name = tile_to_map.get(int(tile_id)) if tile_id is not None else None
        key = (instance_type, normalized)
        bucket = by_asset.get(key)
        if bucket is None:
            bucket = {
                "build": build,
                "instance_type": instance_type,
                "asset_path": asset_path_raw,
                "normalized_asset_path": normalized,
                "library_id": library_id_from_asset_path(normalized),
                "asset_type": detect_asset_type(normalized),
                "observation_count": 0,
                "source_builds": {build},
                "source_maps": {map_name} if map_name else set(),
                "first_tile_id": tile_id,
                "first_unique_id": row.get("uniqueId"),
                "first_pos_x": row.get("posX"),
                "first_pos_y": row.get("posY"),
                "first_pos_z": row.get("posZ"),
                "first_rot_x": row.get("rotX"),
                "first_rot_y": row.get("rotY"),
                "first_rot_z": row.get("rotZ"),
                "first_scale": row.get("scale"),
            }
            by_asset[key] = bucket
        else:
            bucket["observation_count"] += 1
            if map_name:
                bucket["source_maps"].add(str(map_name))

    jobs: list[dict[str, Any]] = []
    for bucket in by_asset.values():
        bucket["source_builds"] = sorted(bucket["source_builds"])
        bucket["source_maps"] = sorted(bucket["source_maps"])
        jobs.append(bucket)
    jobs.sort(key=lambda j: (j["instance_type"], j["normalized_asset_path"]))
    return jobs


def _resolve_builds(dataset_dir: Path, args: argparse.Namespace) -> list[str]:
    if args.builds:
        return [str(item) for item in args.builds]
    if args.build:
        return [str(args.build)]
    return sorted(path.stem.replace(".zarr", "") for path in dataset_dir.glob("*.zarr"))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Enumerate per-object capture jobs from V18 placement tables."
    )
    parser.add_argument("--dataset-dir", type=Path, default=None,
                        help="V18 dataset directory (default: wow-viewer/output/datasets/v18).")
    parser.add_argument("--build", type=str, default=None)
    parser.add_argument("--builds", nargs="+", default=None)
    parser.add_argument("--output", type=Path, required=True,
                        help="Output JSONL file path.")
    parser.add_argument("--include-mddf", action="store_true", default=False)
    parser.add_argument("--include-modf", action="store_true", default=True)
    parser.add_argument("--no-modf", action="store_false", dest="include_modf")
    parser.add_argument("--skip-clutter", action="store_true", default=True,
                        help="Filter out clutter/tree/plant assets before emitting jobs.")
    parser.add_argument("--keep-clutter", action="store_false", dest="skip_clutter")
    return parser.parse_args()


def _default_dataset_dir() -> Path:
    return Path(__file__).resolve().parent.parent.parent / "output" / "datasets" / "v18"


def main() -> int:
    args = _parse_args()
    dataset_dir = args.dataset_dir or _default_dataset_dir()
    if not dataset_dir.exists():
        print(f"Dataset directory not found: {dataset_dir}", file=sys.stderr)
        return 2

    args.output.parent.mkdir(parents=True, exist_ok=True)
    builds = _resolve_builds(dataset_dir, args)
    if not builds:
        print(f"No builds found under {dataset_dir}", file=sys.stderr)
        return 2

    total_jobs = 0
    with open(args.output, "w", encoding="utf-8") as handle:
        for build in builds:
            placements_path = dataset_dir / f"{build}.zarr" / "placements.parquet"
            if not placements_path.exists():
                print(f"Skipping build {build}: no placements.parquet", file=sys.stderr)
                continue
            index_path = dataset_dir / f"{build}.zarr" / "index.parquet"
            index_rows = _read_table_rows(index_path) if index_path.exists() else []
            tile_to_map = _build_tile_to_map(index_rows)
            placements = _read_table_rows(placements_path)
            jobs = _enumerate_jobs_for_build(
                build,
                placements,
                tile_to_map,
                include_mddf=args.include_mddf,
                include_modf=args.include_modf,
                skip_clutter=args.skip_clutter,
            )
            for job in jobs:
                handle.write(json.dumps(job, sort_keys=True) + "\n")
                total_jobs += 1
    print(f"Wrote {total_jobs} capture job rows to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
