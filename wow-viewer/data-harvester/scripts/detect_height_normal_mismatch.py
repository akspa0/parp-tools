"""Detect tiles where normal data encodes terrain variation but height data is flat.

Scans V18/V16 Zarr stores and writes a mismatch report as parquet.
Uses full-array reads for height, then selective reads for normals.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import zarr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from harvester.mismatch_detector import (
    MismatchTile,
    MismatchReport,
)


def load_curation_rows(manifest_path: str | Path) -> list[dict]:
    path = Path(manifest_path)
    if path.is_dir():
        kept = path / "kept_tiles.parquet"
        tiles = path / "tiles.parquet"
        path = kept if kept.exists() else tiles
    if not path.exists():
        raise FileNotFoundError(f"Curation manifest not found: {path}")

    import pyarrow.parquet as pq
    table = pq.read_table(str(path))
    return table.to_pylist()


def main() -> None:
    parser = argparse.ArgumentParser(description="Detect height-normal mismatch tiles in Zarr stores")
    parser.add_argument("--dataset-dir", type=str, required=True,
                        help="Root directory containing <build>.zarr stores")
    parser.add_argument("--curation-manifest", type=str, required=True,
                        help="Path to curation manifest directory or kept_tiles.parquet")
    parser.add_argument("--builds", type=str, nargs="*", default=None,
                        help="Builds to scan (default: auto-detect from manifest)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output parquet path (default: <dataset-dir>/curation/v18_mismatch_report.parquet)")
    parser.add_argument("--normal-relief-threshold", type=float, default=0.02,
                        help="Minimum normal_relief_mean to flag (default: 0.02)")
    parser.add_argument("--height-range-threshold", type=float, default=3.0,
                        help="Maximum height_range to flag as flat (default: 3.0)")
    parser.add_argument("--normal-cov-threshold", type=float, default=0.10,
                        help="Minimum normal_mask coverage (default: 0.10)")
    parser.add_argument("--summary-json", type=str, default=None,
                        help="Optional path for summary JSON")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    curation_rows = load_curation_rows(args.curation_manifest)
    print(f"Loaded {len(curation_rows)} rows from curation manifest")

    builds_to_scan = set(args.builds) if args.builds else set()
    if not builds_to_scan:
        for row in curation_rows:
            builds_to_scan.add(row.get("build", ""))

    if not builds_to_scan:
        print("ERROR: no builds in manifest and none provided via --builds", file=sys.stderr)
        sys.exit(1)

    mismatch_tiles: list[MismatchTile] = []
    total_kept = 0
    total_skipped = 0
    total_mismatch = 0

    for build in sorted(builds_to_scan):
        store_path = dataset_dir / f"{build}.zarr"
        if not store_path.exists():
            print(f"WARNING: store not found: {store_path}", file=sys.stderr)
            continue

        store = zarr.storage.LocalStore(str(store_path), read_only=True)
        root = zarr.open_group(store, mode="r")

        build_rows = [r for r in curation_rows if r.get("build") == build]
        if not build_rows:
            continue

        has_normals = "normal_xyz" in root and "normal_mask" in root
        print(f"  {build}: {len(build_rows)} kept tiles, has_normals={has_normals}")

        # Read all heights at once
        print(f"    reading height_257...")
        all_heights = root["height_257"][:].astype(np.float32, copy=False)
        hr_all = all_heights.max(axis=(1, 2)) - all_heights.min(axis=(1, 2))

        build_mismatch = 0

        for row in build_rows:
            tid = int(row["tile_id"])
            total_kept += 1

            if not has_normals:
                total_skipped += 1
                continue

            hr = float(hr_all[tid])
            if hr >= args.height_range_threshold:
                continue

            # Only read normals for height-filtered tiles
            normals = root["normal_xyz"][tid].astype(np.float32, copy=False)
            nmask = root["normal_mask"][tid].astype(np.float32, copy=False)
            nc = float(nmask.mean())
            if nc < args.normal_cov_threshold:
                total_skipped += 1
                continue

            nx = normals[:, :, 0]
            ny = normals[:, :, 1]
            relief = np.sqrt(np.maximum(0.0, nx * nx + ny * ny)) * nmask
            rm = float(relief.mean())

            if rm < args.normal_relief_threshold:
                continue

            if hr < 0.001:
                severity = "high" if rm > 0.10 else "medium" if rm > 0.03 else "low"
            else:
                ratio = rm / max(hr, 1e-6)
                severity = "high" if ratio > 0.10 else "medium" if ratio > 0.03 else "low"

            build_mismatch += 1
            total_mismatch += 1
            mismatch_tiles.append(MismatchTile(
                build=build,
                tile_id=tid,
                tile_x=int(row.get("tile_x", 0)),
                tile_y=int(row.get("tile_y", 0)),
                map_name=str(row.get("map", "")),
                height_range=hr,
                height_std=float(all_heights[tid].std()),
                normal_relief_mean=rm,
                normal_cov=nc,
                normal_edge_frac=0.0,
                minimap_gray_std=0.0,
                mismatch_severity=severity,
                mismatch_reason="height_flat_vs_normal_varied",
                object_cov=0.0,
                has_normals=True,
            ))

        print(f"  {build}: {build_mismatch} mismatch tiles out of {len(build_rows)} kept")

    report = MismatchReport(tiles=mismatch_tiles)

    output_path = args.output
    if not output_path:
        output_path = str(dataset_dir / "curation" / "v18_mismatch_report.parquet")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    report.to_parquet(output_path)

    print(f"\nWrote {len(mismatch_tiles)} mismatch tiles to {output_path}")
    print(f"  Total kept tiles audited: {total_kept}")
    print(f"  Skipped (insufficient normal data): {total_skipped}")
    print(f"  Mismatched (height flat vs normal varied): {total_mismatch}")

    if args.summary_json:
        severity_counts: dict[str, int] = {}
        for t in mismatch_tiles:
            severity_counts[t.mismatch_severity] = severity_counts.get(t.mismatch_severity, 0) + 1
        summary = {
            "total_audited": total_kept,
            "total_skipped": total_skipped,
            "total_mismatched": total_mismatch,
            "severity_counts": severity_counts,
        }
        Path(args.summary_json).write_text(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
