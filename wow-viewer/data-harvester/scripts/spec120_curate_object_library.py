#!/usr/bin/env python3
"""Spec 120 — Object Library Curation & Blank Asset Pruning Pipeline.

Filters embeddings.parquet to build a clean, verified retrieval index (curated_embeddings.parquet):
1. Prunes blank / low-coverage captures (< 5% foreground).
2. Prunes non-world assets (UI menus, character items, particle effects, skyboxes).
3. Prunes mislabeled assets identified in quality_report.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

# Add src directory to path if running directly
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Curate Object Library Embeddings for Minimap Retrieval.")
    parser.add_argument(
        "--embeddings-in",
        type=Path,
        default=Path("../output/object-library/runs/classifier_v1/embeddings.parquet"),
        help="Path to raw Object Library embeddings.parquet.",
    )
    parser.add_argument(
        "--assets-parquet",
        type=Path,
        default=Path("../output/object-library/objlib_0_5_3_3368.zarr/assets.parquet"),
        help="Path to assets.parquet containing original asset paths.",
    )
    parser.add_argument(
        "--quality-report",
        type=Path,
        default=Path("../output/object-library/runs/classifier_v1/quality_report.json"),
        help="Path to quality_report.json.",
    )
    parser.add_argument(
        "--output-curated",
        type=Path,
        default=Path("../output/spec120/curated_embeddings.parquet"),
        help="Path to save clean curated embeddings parquet.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print("=== Spec 120 Object Library Curation & Blank Asset Pruning Pipeline ===")
    print(f"Raw Embeddings: {args.embeddings_in.resolve()}")
    print(f"Quality Report: {args.quality_report.resolve()}")

    if not args.embeddings_in.exists() or not args.quality_report.exists():
        print("[ERROR] Input files missing.")
        sys.exit(1)

    table = pq.read_table(args.embeddings_in)
    raw_records = table.to_pylist()
    print(f"Loaded {len(raw_records):,} raw asset embeddings.")

    # Load quality report audit lists
    with open(args.quality_report, "r", encoding="utf-8") as f:
        q_report = json.load(f)

    bad_library_ids = set()
    for item in q_report.get("low_coverage", []):
        bad_library_ids.add(item["library_id"])
    for item in q_report.get("mislabels", []):
        bad_library_ids.add(item["library_id"])

    print(f"Blacklisted {len(bad_library_ids):,} blank/low-coverage/mislabel library IDs.")

    # Load asset path mapping
    asset_path_map = {}
    if args.assets_parquet.exists():
        a_tbl = pq.read_table(args.assets_parquet)
        l_ids = a_tbl["library_id"].to_pylist()
        orig_paths = a_tbl["original_asset_path"].to_pylist()
        norm_paths = a_tbl["normalized_asset_path"].to_pylist()
        for lid, op, np in zip(l_ids, orig_paths, norm_paths):
            asset_path_map[lid] = op if op else np

    curated_records = []
    pruned_counts = {"blank": 0, "non_world": 0}

    for r in raw_records:
        lid = r["library_id"]
        path = asset_path_map.get(lid, "").lower().replace("\\", "/")

        # 1. Prune quality report blacklist (blanks & mislabels)
        if lid in bad_library_ids:
            pruned_counts["blank"] += 1
            continue

        # 2. Prune non-world assets (UI menus, character items, ammo, particle effects)
        if path.startswith("interface/") or path.startswith("item/") or path.startswith("environments/") or path.startswith("character/"):
            pruned_counts["non_world"] += 1
            continue

        # Must be world asset
        r["asset_path"] = asset_path_map.get(lid, f"World/assets/{lid}.mdx")
        curated_records.append(r)

    print("\n--- Curation Audit Summary ---")
    print(f"  Raw Assets:         {len(raw_records):>6,}")
    print(f"  Pruned Blank/Empty: {pruned_counts['blank']:>6,}")
    print(f"  Pruned Non-World:   {pruned_counts['non_world']:>6,}")
    print(f"  Curated World Assets: {len(curated_records):>6,}")

    args.output_curated.parent.mkdir(parents=True, exist_ok=True)
    curated_table = pa.Table.from_pylist(curated_records)
    pq.write_table(curated_table, args.output_curated)
    print(f"\n[CURATION SUCCESS] Clean index written to {args.output_curated.resolve()}")


if __name__ == "__main__":
    main()
