"""Patch hole_mask_16 in existing V16 Zarr stores using correct 16-bit bitmasks from game client WDT files.

Usage:
  1. Run extract-holes for each build:
     dotnet run --project wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj -- extract-holes --client-root "path" --build 3_3_5_12340 --map Azeroth --output holes_335.json

  2. Run this script:
     python scripts/patch_holes_zarr.py --holes-json holes_335.json --build 3_3_5_12340
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr
import zarr.storage

_SYS_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SYS_SRC) not in sys.path:
    sys.path.insert(0, str(_SYS_SRC))

DATASET_ROOT = Path("I:/parp/parp-tools/wow-viewer/output/datasets/v16")
HARVESTER_DLL = Path("I:/parp/parp-tools/wow-viewer/tools/harvest/WowViewer.Tool.Harvest/WowViewer.Tool.Harvest.csproj")

CLIENT_ROOTS = {
    "0_5_3_3368": Path("I:/parp/parp-tools/output/tmp/wowarchive-clients/0_5_3_3368/World of Warcraft"),
    "0_5_5_3494": Path("I:/parp/parp-tools/output/tmp/wowarchive-clients/0_5_5_3494/World of Warcraft"),
    "0_7_0_3694": Path("I:/parp/parp-tools/output/tmp/wowarchive-clients/0_7_0_3694/World of Warcraft"),
    "3_0_1_8303": Path("I:/parp/parp-tools/output/tmp/wowarchive-clients/3_0_1_8303/World of Warcraft"),
    "3_3_5_12340": Path("I:/parp/parp-tools/output/tmp/wowarchive-clients/3_3_5_12340/World of Warcraft"),
    "4_0_0_11927": Path("I:/parp/parp-tools/output/tmp/wowarchive-clients/4_0_0_11927/World of Warcraft"),
}

MAP_NAMES = {
    "0_5_3_3368": "development",
    "0_5_5_3494": "development",
    "0_7_0_3694": "development",
    "3_0_1_8303": "development",
    "3_3_5_12340": "Azeroth",
    "4_0_0_11927": "Azeroth",
}


def extract_holes_from_client(build: str, output_json: Path) -> bool:
    """Run extract-holes command to get hole masks from game client."""
    client_root = CLIENT_ROOTS.get(build)
    if client_root is None or not client_root.exists():
        print(f"  Client root not found for {build}")
        return False

    map_name = MAP_NAMES.get(build, "Azeroth")
    cmd = [
        "dotnet", "run", "--project", str(HARVESTER_DLL), "--",
        "extract-holes",
        "--client-root", str(client_root),
        "--build", build,
        "--map", map_name,
        "--output", str(output_json),
    ]
    print(f"  Running: {' '.join(cmd[-8:])}")
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
    if result.returncode != 0:
        print(f"  Error: {result.stderr[:500]}")
        return False
    print(f"  {result.stdout.strip()}")
    return output_json.exists()


def patch_zarr_from_json(build: str, holes_json: Path, dry_run: bool = False) -> int:
    """Patch Zarr store from extract-holes JSON output."""
    zarr_path = DATASET_ROOT / f"{build}.zarr"
    if not zarr_path.exists():
        print(f"  Zarr not found: {zarr_path}")
        return 0

    with holes_json.open() as f:
        hole_data = json.load(f)

    print(f"  Loaded {len(hole_data)} tiles from JSON")

    index_path = zarr_path / "index.parquet"
    if not index_path.exists():
        print(f"  No index.parquet in {zarr_path}")
        return 0

    table = pq.read_table(str(index_path))

    store = zarr.storage.LocalStore(str(zarr_path), read_only=False)
    root = zarr.open_group(store=store, mode="r+")

    if "holes_16" in root:
        holes_arr = root["holes_16"]
    else:
        holes_arr = root.create_array(
            "holes_16",
            shape=(len(root["minimap_rgb"]), 16, 16),
            chunks=(64, 16, 16),
            dtype=np.uint16,
        )

    patched = 0
    for i in range(table.num_rows):
        tile_x = int(table.column("tile_x")[i].as_py())
        tile_y = int(table.column("tile_y")[i].as_py())
        tile_id = int(table.column("tile_id")[i].as_py())

        key = f"{tile_x}_{tile_y}"
        if key in hole_data:
            new_val = np.array(hole_data[key], dtype=np.uint16)
            old_val = holes_arr[tile_id]
            if not np.array_equal(old_val, new_val):
                if not dry_run:
                    holes_arr[tile_id] = new_val
                patched += 1

    print(f"  Patched {patched} tiles")
    return patched


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Patch hole masks in V16 Zarr stores")
    p.add_argument("--builds", nargs="*", default=None, help="Builds to patch")
    p.add_argument("--holes-json", type=Path, default=None, help="Pre-extracted holes JSON (skip extraction)")
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args()

    builds = args.builds or list(CLIENT_ROOTS.keys())
    total_patched = 0

    for build in builds:
        print(f"\nPatching {build}...")

        if args.holes_json and args.holes_json.exists():
            holes_json = args.holes_json
        else:
            holes_json = DATASET_ROOT / f"_holes_{build}.json"
            if not extract_holes_from_client(build, holes_json):
                continue

        patched = patch_zarr_from_json(build, holes_json, dry_run=args.dry_run)
        total_patched += patched

    print(f"\nTotal patched: {total_patched}")


if __name__ == "__main__":
    main()
