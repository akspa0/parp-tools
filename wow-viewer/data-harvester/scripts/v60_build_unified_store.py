#!/usr/bin/env python3
"""Build the unified v60 Zarr store from existing v50.1 stores (Spec 134 US1).

Consolidates all per-build, per-map v50.1 Zarr stores into a single v60 Zarr store
with a unified index across all builds and maps. New signals (terrain_shadow_256,
signal_class, surviving_height_levels) are included where available; missing signals
are recorded as unavailable-with-reason.

Usage:
    cd wow-viewer/data-harvester
    uv run python scripts/v60_build_unified_store.py \\
        --source-root ../output/datasets/v50/v50.1 \\
        --output ../output/datasets/v60/v60.1/unified.zarr
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.v50.v60_store import build_v60_store  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Build unified v60 Zarr store from existing v50.1 stores"
    )
    parser.add_argument("--source-root", required=True, type=Path, action="append",
                        dest="source_roots", metavar="ROOT",
                        help="A directory containing v50.1 Zarr stores; repeatable")
    parser.add_argument("--output", required=True, type=Path,
                        help="Output v60 Zarr store path")
    parser.add_argument("--release", default="v60.1",
                        help="Release identifier (default: v60.1)")
    args = parser.parse_args()

    if not args.source_roots:
        raise SystemExit("at least one --source-root is required")

    result = build_v60_store(
        args.source_roots,
        args.output,
        release=args.release,
    )

    print(f"\n[DONE] v60 unified store: {result.store_path}")
    print(f"       {result.row_count} rows, {result.signal_count} signals")
    print(f"       {len(result.source_stores)} source stores consolidated")
    if result.unavailable_signals:
        print(f"       {len(result.unavailable_signals)} signals unavailable:")
        for u in result.unavailable_signals[:5]:
            print(f"         {u.name}: {u.reason}")
        if len(result.unavailable_signals) > 5:
            print(f"         ... and {len(result.unavailable_signals) - 5} more")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())