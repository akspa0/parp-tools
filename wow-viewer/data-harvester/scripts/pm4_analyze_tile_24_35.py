"""Cross-reference PM4 segments against WMO placements for tile 24_35.

This script compares the PM4 collision data against real WMO assets to
understand how the PM4 format relates to the original game assets.
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harvester.pm4_asset_matching import import_asset_corpus, import_segment_export


def main():
    dev = Path(r"I:\parp\parp-tools\wow-viewer\test_data\development\World\Maps\development")
    tmp = Path(r"I:\parp\parp-tools\output\tmp")

    # Load PM4 segments for tile 24_35
    segments = import_segment_export(tmp / "pm4_24_35_segments.json")
    print(f"=== PM4 Segments for tile 24_35 ({len(segments)} total) ===\n")

    # Group by Ck24 low-16 ObjectID
    from collections import defaultdict
    by_objid = defaultdict(list)
    for seg in segments:
        # Extract the low16 object ID from segment_id or use placeholder
        by_objid[seg.segment_id.split("-")[-1]].append(seg)

    for seg in segments:
        print(f"  Segment: {seg.segment_id}")
        if seg.bounds:
            b = seg.bounds
            span_x = b.max[0] - b.min[0]
            span_y = b.max[1] - b.min[1]
            span_z = b.max[2] - b.min[2]
            cx = (b.min[0] + b.max[0]) / 2
            cy = (b.min[1] + b.max[1]) / 2
            cz = (b.min[2] + b.max[2]) / 2
            print(f"    center: ({cx:.2f}, {cy:.2f}, {cz:.2f})")
            print(f"    span:   ({span_x:.2f}, {span_y:.2f}, {span_z:.2f})")
            print(f"    bounds: ({b.min[0]:.2f},{b.min[1]:.2f})-({b.max[0]:.2f},{b.max[1]:.2f})")
        if seg.footprint_hull:
            print(f"    hull points: {len(seg.footprint_hull)}")
        print(f"    topology: {seg.topology_stats.surface_count} surfaces, {seg.topology_stats.total_index_count} indices")
        print(f"    anchors: {seg.anchor_signals.linked_position_ref_count} pos refs, {seg.anchor_signals.normal_heading_count} heading refs")
        print()

    # Load the WMO asset corpus for tile 24_35
    assets = import_asset_corpus(tmp / "pm4_24_35_placements_corpus.json")
    print(f"\n=== WMO Assets placed on tile 24_35 ({len(assets)} total) ===\n")
    for asset in assets:
        name = asset.asset_path.split("\\")[-1]
        print(f"  {name}")
        if asset.bounds:
            b = asset.bounds
            span_x = b.max[0] - b.min[0]
            span_y = b.max[1] - b.min[1]
            span_z = b.max[2] - b.min[2]
            print(f"    model-space bounds: ({b.min[0]:.2f},{b.min[1]:.2f})-({b.max[0]:.2f},{b.max[1]:.2f})")
            print(f"    model-space span:   ({span_x:.2f}, {span_y:.2f}, {span_z:.2f})")
            print(f"    footprint area: {asset.footprint_area:.2f}")
        print()

    # Analysis: compare PM4 segment total extent vs WMO model-space bounds
    print("=== COMPARISON ===\n")
    # Get the combined extents of all PM4 segments on this tile
    all_bounds = [s.bounds for s in segments if s.bounds]
    if all_bounds:
        min_x = min(b.min[0] for b in all_bounds)
        min_y = min(b.min[1] for b in all_bounds)
        max_x = max(b.max[0] for b in all_bounds)
        max_y = max(b.max[1] for b in all_bounds)
        total_span_x = max_x - min_x
        total_span_y = max_y - min_y
        print(f"PM4 tile extent: ({min_x:.2f},{min_y:.2f})-({max_x:.2f},{max_y:.2f})")
        print(f"PM4 tile span: ({total_span_x:.2f}, {total_span_y:.2f})")

        # Compare against the large wall stairs (asset #4)
        wall = assets[3]  # WALLPIECESTAIRS01
        if wall.bounds:
            ws = wall.bounds
            w_span_x = ws.max[0] - ws.min[0]
            w_span_y = ws.max[1] - ws.min[1]
            print(f"\nWallStairs01 model span: ({w_span_x:.2f}, {w_span_y:.2f})")
            print(f"  The PM4 segment spans should be SIMILAR to model spans after")
            print(f"  applying the MODF placement transform (translation + rotation + scale).")
            print(f"  Any DIFFERENCE reveals the simplification algorithm.")

    # Compare individual assets against segments that have similar spans
    print("\n=== SPAN COMPARISON: PM4 segments vs WMO assets ===")
    for seg in segments:
        if not seg.bounds:
            continue
        sb = seg.bounds
        s_span = (sb.max[0]-sb.min[0], sb.max[1]-sb.min[1], sb.max[2]-sb.min[2])
        for asset in assets:
            if not asset.bounds:
                continue
            ab = asset.bounds
            a_span = (ab.max[0]-ab.min[0], ab.max[1]-ab.min[1], ab.max[2]-ab.min[2])
            # Check if spans are similar (within 20%)
            ratio_x = min(s_span[0], a_span[0]) / max(s_span[0], a_span[0]) if max(s_span[0], a_span[0]) > 0 else 0
            ratio_y = min(s_span[1], a_span[1]) / max(s_span[1], a_span[1]) if max(s_span[1], a_span[1]) > 0 else 0
            ratio_z = min(s_span[2], a_span[2]) / max(s_span[2], a_span[2]) if max(s_span[2], a_span[2]) > 0 else 0
            if ratio_x > 0.5 and ratio_y > 0.5:
                aname = asset.asset_path.split("\\")[-1]
                print(f"  Potential match: {seg.segment_id} <-> {aname}")
                print(f"    PM4 span: ({s_span[0]:.2f}, {s_span[1]:.2f}, {s_span[2]:.2f})")
                print(f"    Asset span: ({a_span[0]:.2f}, {a_span[1]:.2f}, {a_span[2]:.2f})")
                print(f"    Ratios: X={ratio_x:.3f} Y={ratio_y:.3f} Z={ratio_z:.3f}")


if __name__ == "__main__":
    main()
