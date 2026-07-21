#!/usr/bin/env python3
"""Spec 116 US2 CLI: shape->coverage coupling measurement (dry-run by default).

Prints per-(tile, layer) explained variance, the bimodality coefficient, the 1-vs-2 component
mixture BIC, the high-coupling tile share, and the coverage_derivable/coverage_independent
decision. Writes the durable ``v116-analysis-report-v1`` artifact only with ``--write``. No model
is trained.

Run from ``wow-viewer/data-harvester/``::

    uv run python scripts/spec116_shape_coverage_coupling.py \
        --store <v50 curriculum store> [--output <report.json>] [--write]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from harvester.spec116.shape_coverage_coupling import measure_shape_coverage_coupling


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 116 US2: shape->coverage coupling measurement (dry-run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="v50 curriculum Zarr store")
    ap.add_argument("--build-id", default="", help="build id recorded for provenance")
    ap.add_argument("--output", type=Path, default=None, help="output report JSON path (required with --write)")
    ap.add_argument("--write", action="store_true", help="write the report artifact (default: print only)")
    args = ap.parse_args(argv)

    if args.write and args.output is None:
        ap.error("--write requires --output")

    report = measure_shape_coverage_coupling(store=args.store, build_id=args.build_id)
    scc = report["shape_coverage_coupling"]

    print("=== Spec 116 US2: shape->coverage coupling ===")
    print(f"store: {report['identity']['store']['path']}")
    print(f"fittable (tile,layer) pairs: {report['fittable_tile_layer_count']}  "
          f"skipped_zero_coverage: {report['skipped_zero_coverage']}")
    print()
    evs = [e["explained_variance"] for e in scc["per_tile_layer_explained_variance"]]
    if evs:
        ev_arr = evs  # noqa: F841
        print(f"explained variance: min={min(evs):.3f} median={sorted(evs)[len(evs)//2]:.3f} max={max(evs):.3f}")
    print(f"high_coupling_tile_share (ev>={report['high_coupling_variance_threshold']}): "
          f"{scc['high_coupling_tile_share']:.3f}")
    print(f"bimodality_coefficient: {scc['bimodality_coefficient']:.3f} "
          f"(threshold {report['bimodality_coefficient_threshold']:.3f})")
    bic = scc["mixture_bic"]
    print(f"mixture_bic: one={bic['one_component']:.2f} two={bic['two_component']:.2f} "
          f"two_preferred={bic['two_preferred']}")
    print(f"decision: {report['decision']['value']}")
    print(f"note: {scc['linear_underpowered_note']}")

    if not args.write:
        print("\nDRY RUN ONLY -- pass --write to emit the report artifact.")
        return 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nwrote report: {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
