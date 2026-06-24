"""Build a cross-map brush-family catalog from Spec 076 near-duplicate clusters."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from harvester.fractal_family_catalog import (
    CanvasCache,
    build_families,
    discover_canvas_dirs,
    filter_families,
    group_members_by_cluster,
    load_near_clusters,
    render_family_contact_sheet,
    write_family_outputs,
)

_DEFAULT_ANALYSIS_ROOT = Path(__file__).resolve().parents[2] / "output" / "analysis" / "full-map-fractal-brush-library" / "full_map_Azeroth_0_5_3_3368_rectangles"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a brush-family catalog from near-duplicate cluster output.")
    parser.add_argument("--analysis-root", type=Path, default=_DEFAULT_ANALYSIS_ROOT)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-members", type=int, default=2)
    parser.add_argument("--min-builds", type=int, default=1)
    parser.add_argument("--min-maps", type=int, default=1)
    parser.add_argument("--max-families", type=int, default=None)
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--families-per-page", type=int, default=20)
    parser.add_argument("--no-contact-sheet", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analysis_root = Path(args.analysis_root)
    output_dir = Path(args.output_dir) if args.output_dir else analysis_root / "family_catalog"
    output_dir.mkdir(parents=True, exist_ok=True)

    patterns, members = load_near_clusters(analysis_root)
    selected = filter_families(
        patterns,
        min_members=int(args.min_members),
        min_builds=int(args.min_builds),
        min_maps=int(args.min_maps),
        max_families=args.max_families,
    )
    members_by_cluster = group_members_by_cluster(members)

    target_index = discover_canvas_dirs(analysis_root)
    cache = CanvasCache(target_index)
    families, tensor = build_families(selected, members_by_cluster, cache, crop_size=int(args.crop_size))
    cache.close()

    summary = write_family_outputs(output_dir, families, tensor)

    pages: list[Path] = []
    if not bool(args.no_contact_sheet):
        pages = render_family_contact_sheet(
            families,
            tensor,
            output_dir / "contact_sheets" / "family_catalog_page_001.png",
            families_per_page=int(args.families_per_page),
        )

    summary["contact_sheet_pages"] = [str(path) for path in pages]
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")

    print("Brush-family catalog complete", flush=True)
    print(f"  output_dir: {output_dir}", flush=True)
    print(f"  families: {len(families)}", flush=True)
    print(f"  pages: {len(pages)}", flush=True)


if __name__ == "__main__":
    main()
