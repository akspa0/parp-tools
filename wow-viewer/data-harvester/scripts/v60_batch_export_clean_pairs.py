#!/usr/bin/env python3
"""Batch-Export Clean Minimap / Elevation Pairs Across Zarr v3 Maps (Spec 274).

Usage:
    # Dry-run inspection
    uv run python scripts/v60_batch_export_clean_pairs.py --dry-run

    # Targeted dry-run on Azeroth and Kalimdor
    uv run python scripts/v60_batch_export_clean_pairs.py --maps Azeroth,Kalimdor --dry-run

    # Execute export to output/datasets/v60_clean_pairs.zarr
    uv run python scripts/v60_batch_export_clean_pairs.py --confirm-build
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from harvester.v60.batch_export_clean_pairs import (
    export_clean_pairs_dataset,
    find_zarr_map_stores,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("batch_export_clean_pairs")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("output/datasets/v60"),
        help="Root directory containing source Zarr v3 map stores (default: output/datasets/v60)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("output/datasets/v60_clean_pairs.zarr"),
        help="Destination path for clean paired Zarr dataset (default: output/datasets/v60_clean_pairs.zarr)",
    )
    parser.add_argument(
        "--maps",
        type=str,
        default=None,
        help="Optional comma-separated map filter (e.g. 'Azeroth,Kalimdor')",
    )
    parser.add_argument(
        "--val-map",
        type=str,
        default=None,
        help="Map name to hold out for validation",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of candidate tiles to export",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Scan and report candidate counts without writing to disk",
    )
    parser.add_argument(
        "--confirm-build",
        action="store_true",
        help="Required confirmation flag to perform real disk writes",
    )

    args = parser.parse_args()

    # Discover stores with intelligent candidate fallback
    candidate_roots = [
        args.source_root,
        Path("../output/datasets/v60"),
        Path("output/datasets/v60"),
        Path("../../output/datasets/v60"),
    ]
    stores = []
    effective_root = args.source_root
    for root in candidate_roots:
        if root.exists():
            found = find_zarr_map_stores(root)
            if found:
                stores = found
                effective_root = root
                break

    if not stores:
        logger.error(f"No Zarr map stores found under {args.source_root.resolve()}")
        return 1

    logger.info(f"Discovered {len(stores)} Zarr map stores under {effective_root}")

    target_maps = [m.strip() for m in args.maps.split(",")] if args.maps else None
    dry_run = args.dry_run or not args.confirm_build

    if not args.dry_run and not args.confirm_build:
        logger.warning("Neither --dry-run nor --confirm-build was specified; defaulting to DRY RUN.")

    result = export_clean_pairs_dataset(
        source_stores=stores,
        output_path=args.output,
        target_maps=target_maps,
        validation_map=args.val_map,
        limit=args.limit,
        dry_run=dry_run,
    )

    print(json.dumps(result, indent=2, sort_keys=True))
    if dry_run:
        print("\n[DRY RUN COMPLETE] To execute full export, pass --confirm-build")
    else:
        logger.info(f"Clean paired dataset successfully written to {args.output.resolve()}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
