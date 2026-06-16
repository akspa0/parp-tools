"""CLI: Validate Python PM4 scorer against C# match report ground truth.

Compares Python scorer output against C# match report. Matches segments by ID.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from harvester.pm4_asset_matching import (
    import_asset_corpus,
    import_match_report,
    import_segment_export,
    score_segment,
)


def _status_str(s) -> str:
    return s.value if hasattr(s, "value") else str(s)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Python PM4 scorer against C# ground truth")
    parser.add_argument("--segments", "-s", required=True)
    parser.add_argument("--corpus", "-c", required=True)
    parser.add_argument("--expected", "-e", required=True)
    args = parser.parse_args()

    for pname in ["segments", "corpus", "expected"]:
        p = Path(getattr(args, pname))
        if not p.exists():
            print(f"Error: '{p}' not found.", file=sys.stderr)
            sys.exit(1)

    segments = import_segment_export(args.segments)
    assets = import_asset_corpus(args.corpus)
    expected_results, _ = import_match_report(args.expected)

    expected_by_id = {r.segment.segment_id: r for r in expected_results}

    print(f"Loaded {len(segments)} segments, {len(assets)} assets, {len(expected_results)} expected results")

    matched = 0
    failed = 0
    tested = 0

    for seg in segments:
        exp = expected_by_id.get(seg.segment_id)
        if exp is None:
            continue
        tested += 1

        ck24_type = {None: 0, "wmo": 0x42, "m2": 0x40}.get(exp.expected_asset_kind, 0)
        result = score_segment(
            seg, assets, max_candidates=5,
            ck24_type=ck24_type,
            segment_tiles=seg.tile_coordinates,
        )

        status_ok = result.status == exp.status
        count_ok = len(result.candidates) == len(exp.candidates)

        if status_ok and count_ok:
            matched += 1
            print(f"  [PASS] {seg.segment_id}: {result.status.value} ({len(result.candidates)} candidates)")
        else:
            failed += 1
            ec = exp.candidates[0] if exp.candidates else None
            rc = result.candidates[0] if result.candidates else None
            print(f"  [FAIL] {seg.segment_id}")
            if not status_ok:
                print(f"    Status: C#={_status_str(exp.status)} Py={_status_str(result.status)}")
            if not count_ok:
                print(f"    Candidates: C#={len(exp.candidates)} Py={len(result.candidates)}")
            if ec and rc:
                print(f"    Top score: C#={ec.overall_score:.4f} Py={rc.overall_score:.4f}")

    print(f"\nResults: {matched} passed, {failed} failed out of {tested} tested")
    print(f"Skipped {len(segments) - tested - (len(expected_results) - len(expected_by_id))} segments not in match report")

    if failed > 0:
        print("\nWARNING: Python and C# scorers diverge.")
        print("Check: footprint area computation, typed bounds handling, same-tile bonus")
        sys.exit(1)
    print("All segments match C# ground truth!")


if __name__ == "__main__":
    main()
