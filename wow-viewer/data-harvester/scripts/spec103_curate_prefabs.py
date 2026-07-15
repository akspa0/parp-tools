"""Build the Spec 103 map-canvas prefab evidence ledger and reduced manifest.

This is CPU-only dataset curation.  It reads existing V18 stores and Spec 076
analysis outputs; it does not harvest clients, train a model, or mutate inputs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parents[1] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.spec103.prefab_curation import (  # noqa: E402
    PrefabCurationConfig,
    run_prefab_curation,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spec 103: curate V8 tiles by full-map terrain-art prefab coverage"
    )
    parser.add_argument(
        "--store",
        action="append",
        type=Path,
        required=True,
        help="V18 Zarr store; repeat for multiple builds",
    )
    parser.add_argument(
        "--analysis-root",
        type=Path,
        required=True,
        help="Spec 076 full-map analysis root containing canvas.zarr and region outputs",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--regions",
        action="append",
        type=Path,
        default=[],
        help="explicit region Parquet; repeat to avoid auto-discovery",
    )
    parser.add_argument(
        "--prefab-members",
        action="append",
        type=Path,
        default=[],
        help="trusted prefab placement membership Parquet; repeat as needed",
    )
    parser.add_argument(
        "--clean-manifest",
        action="append",
        type=Path,
        default=[],
        help="existing clean-tile manifest; repeat per build and apply its hard gates first",
    )
    parser.add_argument(
        "--val-map",
        action="append",
        default=[],
        help="complete holdout map; repeat as needed (family-connected tiles follow it)",
    )
    parser.add_argument("--thumbnail-size", type=int, default=16)
    parser.add_argument("--alpha-threshold", type=float, default=0.05)
    parser.add_argument("--family-hamming-radius", type=int, default=4)
    parser.add_argument("--neighbor-radii", default="256,1024,4096")
    parser.add_argument("--global-tileset-rarity", type=float, default=0.01)
    parser.add_argument("--local-tileset-rarity", type=float, default=0.02)
    parser.add_argument("--min-family-tileset-support", type=int, default=2)
    parser.add_argument(
        "--max-selected-tiles",
        type=int,
        default=0,
        help="hard diversity budget; 0 means cover every discovered evidence token",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_arg_parser().parse_args(argv)
    radii = tuple(float(value.strip()) for value in args.neighbor_radii.split(",") if value.strip())
    if not radii or any(value <= 0 for value in radii):
        raise SystemExit("--neighbor-radii must contain positive comma-separated values")
    if not args.val_map:
        raise SystemExit("At least one --val-map is required for an auditable complete-map holdout")
    config = PrefabCurationConfig(
        thumbnail_size=args.thumbnail_size,
        alpha_threshold=args.alpha_threshold,
        family_hamming_radius=args.family_hamming_radius,
        neighbor_radii=tuple(sorted(radii)),
        global_tileset_rarity=args.global_tileset_rarity,
        local_tileset_rarity=args.local_tileset_rarity,
        min_family_tileset_support=args.min_family_tileset_support,
        max_selected_tiles=args.max_selected_tiles or None,
    )
    summary = run_prefab_curation(
        store_paths=args.store,
        analysis_root=args.analysis_root,
        output_dir=args.output,
        region_paths=args.regions,
        member_paths=args.prefab_members,
        clean_manifest=args.clean_manifest,
        val_maps=set(args.val_map),
        config=config,
    )
    print(f"[prefab-curation] families={summary['prefab_family_count']}")
    print(
        f"[prefab-curation] tiles={summary['selected_tile_count']}/"
        f"{summary['eligible_tile_count']} selected"
    )
    print(f"[prefab-curation] ledger={summary['ledger_row_count']} rows")
    print(f"[prefab-curation] output={args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
