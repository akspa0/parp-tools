"""Build the lean V25 Zarr datastore (Spec 102).

Reads the V18 substrate store plus (optionally) the V22 store for the tileset
vocabulary and object placements, and the V24 store for pre-computed cleaned
minimaps.  Only V25-relevant signals are carried over — see
``harvester/v25/dataset.py`` for the schema and the documented omissions.

Example (curated two-era corpus, 0.5.3 + 3.3.5, shared path-keyed tileset
vocabulary; store lists are index-paired with --v18-store, '-' = none):

    uv run python scripts/build_v25_dataset.py \
        --v18-store ../output/datasets/v18/0_5_3_3368.zarr ../output/datasets/v18/3_3_5_12340.zarr \
        --v22-store ../output/datasets/v22/0_5_3_3368.zarr ../output/datasets/v22/3_3_5_12340.zarr \
        --v24-store - ../output/datasets/v24/3_3_5_12340_openworld_curated.zarr \
        --curation-manifest ../output/datasets/v18/curation/v18_focus_terrain_all_v1/kept_tiles.parquet \
        --output ../output/datasets/v25/0_5_3+3_3_5_v25_curated_v1.zarr

Attach pre-parsed PM4 segment records (from the C# segment export JSON) after
the build with ``--attach-pm4-segments`` — Python never parses raw .pm4 files.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

DATA_HARVESTER_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(DATA_HARVESTER_ROOT / "src"))

from harvester.v25.dataset import (  # noqa: E402
    attach_holes_bits,
    attach_pm4_segments,
    build_v25_dataset,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="V25 lean Zarr dataset builder (Spec 102)")
    parser.add_argument("--v18-store", required=True, nargs="+", type=Path,
                        help="V18 substrate Zarr store(s), one per build")
    parser.add_argument("--v22-store", nargs="*", default=None,
                        help="V22 store(s) for tileset vocab + placements, index-paired with "
                             "--v18-store ('-' for a build without one)")
    parser.add_argument("--v24-store", nargs="*", default=None,
                        help="V24 store(s) with pre-computed cleaned_minimap_256, index-paired "
                             "with --v18-store ('-' for a build without one)")
    parser.add_argument("--output", required=True, type=Path, help="output V25 Zarr store path")
    parser.add_argument("--maps", nargs="*", default=None, help="restrict to these map names")
    parser.add_argument("--curation-manifest", type=Path, default=None,
                        help="V18 curation kept_tiles.parquet; keeps only keep==True tiles")
    parser.add_argument("--difficulty-bucket", default=None,
                        help="optional curation difficulty bucket filter (e.g. hard)")
    parser.add_argument("--limit", type=int, default=None, help="cap the number of tiles")
    parser.add_argument("--vocab-size", type=int, default=256,
                        help="tileset vocabulary size incl. the OOV bucket (default 256)")
    parser.add_argument("--batch-rows", type=int, default=64, help="source read batch size")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output store")
    parser.add_argument("--attach-pm4-segments", type=Path, default=None,
                        help="C# PM4 segment export JSON to attach as pm4_segments.parquet")
    parser.add_argument("--height-repair-root", type=Path, default=None,
                        help="mismatch-repair store root (per-build height_corrected_257 "
                             "replaces raw heights)")
    parser.add_argument("--mismatch-report", type=Path, default=None,
                        help="mismatch audit parquet; joins severity/reason per tile into the index")
    parser.add_argument("--attach-holes", nargs="+", type=Path, default=None,
                        help="extract-holes JSON export(s) to attach as holes_bits_16 "
                             "(from WowViewer.Tool.Harvest extract-holes)")
    args = parser.parse_args()

    try:
        out = build_v25_dataset(
            v18_store=args.v18_store,
            output=args.output,
            v22_store=args.v22_store,
            v24_store=args.v24_store,
            maps=args.maps,
            curation_manifest=args.curation_manifest,
            difficulty_bucket=args.difficulty_bucket,
            limit=args.limit,
            vocab_size=args.vocab_size,
            batch_rows=args.batch_rows,
            overwrite=args.overwrite,
            height_repair_root=args.height_repair_root,
            mismatch_report=args.mismatch_report,
        )
        if args.attach_pm4_segments is not None:
            n = attach_pm4_segments(out, args.attach_pm4_segments)
            print(f"[v25-build] attached {n} pre-parsed PM4 segment records", flush=True)
        if args.attach_holes is not None:
            stats = attach_holes_bits(out, args.attach_holes)
            print(
                f"[v25-build] holes_bits_16 attached: {stats['matched']}/{stats['rows']} rows "
                f"matched, {stats['holed']} with holes",
                flush=True,
            )
        print(f"[v25-build] store ready: {out}", flush=True)
        return 0
    except Exception as ex:  # surface a clean CLI error, not a stack dump
        print(f"Error: {ex}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    sys.exit(main())
