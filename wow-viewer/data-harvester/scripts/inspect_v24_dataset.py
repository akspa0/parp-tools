"""Inspect a V24 store: coverage summary and per-tile dumps (Spec 094)."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from harvester.v24 import store  # noqa: E402


def _summary(args: argparse.Namespace) -> int:
    group = store.open_v24_store(args.store)
    index = store.read_index(args.store)
    stats = store.coverage_stats(group)

    audit_empty = np.asarray(group["wdl_prior_audit_empty"][:])
    real_available = np.asarray(group["wdl_prior_real_available"][:])
    disagree = np.asarray(group["wdl_prior_disagree_ratio"][:])

    report = {
        "store": str(args.store),
        "tiles": len(index["tile_id"]),
        "maps": sorted(set(index["map"])),
        "audit_empty_tiles": int(audit_empty.sum()),
        "real_wdl_available_tiles": int(real_available.sum()),
        "tiles_with_disagreement": int((disagree > 0).sum()),
        "max_disagree_ratio": float(disagree.max()) if disagree.size else 0.0,
        **stats,
        "attrs": {k: v for k, v in group.attrs.items() if not k.startswith("coverage_")},
    }
    print(json.dumps(report, indent=2, default=str))
    return 0


def _tile(args: argparse.Namespace) -> int:
    group = store.open_v24_store(args.store)
    index = store.read_index(args.store)
    matches = [
        i
        for i in range(len(index["tile_id"]))
        if index["map"][i].lower() == args.map.lower()
        and index["tile_x"][i] == args.tile_x
        and index["tile_y"][i] == args.tile_y
    ]
    if not matches:
        print(f"No tile {args.map} ({args.tile_x}, {args.tile_y}) in the store.", file=sys.stderr)
        return 1

    row = matches[0]
    outer = np.asarray(group["wdl_prior_outer"][row])
    inner = np.asarray(group["wdl_prior_inner"][row])
    source_outer = np.asarray(group["wdl_prior_source_outer"][row])
    print(
        json.dumps(
            {
                "row": row,
                "tile_id": index["tile_id"][row],
                "v18_row": index["v18_row"][row],
                "outer_min": float(outer.min()),
                "outer_max": float(outer.max()),
                "inner_min": float(inner.min()),
                "inner_max": float(inner.max()),
                "source_counts": {
                    str(v): int((source_outer == v).sum()) for v in (0, 1, 2)
                },
                "disagree_ratio": float(group["wdl_prior_disagree_ratio"][row]),
                "audit_empty": bool(group["wdl_prior_audit_empty"][row]),
                "real_available": bool(group["wdl_prior_real_available"][row]),
            },
            indent=2,
        )
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    summary = sub.add_parser("summary")
    summary.add_argument("--store", required=True)

    tile = sub.add_parser("tile")
    tile.add_argument("--store", required=True)
    tile.add_argument("--map", required=True)
    tile.add_argument("--tile-x", type=int, required=True)
    tile.add_argument("--tile-y", type=int, required=True)

    args = parser.parse_args()
    return _summary(args) if args.command == "summary" else _tile(args)


if __name__ == "__main__":
    raise SystemExit(main())
