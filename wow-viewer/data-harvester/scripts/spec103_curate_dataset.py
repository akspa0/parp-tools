"""Spec 103 — curate + bucket the training corpus into a clean, coherent tile set.

The governing law (spec Principle #5) makes this mandatory, not optional: a supervised target
must contain only information the input image supports. Height under an object is occluded in
the minimap, so an object tile is an impossible target and must be DROPPED, not learned. Blank
minimap art carries no input signal. A flat-height tile whose normals show relief is a harvest
failure (mismatched signals). This pass removes all three and writes an auditable manifest the
trainer consumes — clean data in, clean model out.

Drop reasons (each tile gets exactly one, most-severe first):
  missing_signal      required array absent (height / minimap / normals)
  blank_minimap       per-tile RGB std < --min-rgb-std (dead-space art)
  object_contaminated object_precise_mask coverage > --max-object-coverage
  height_normal_mismatch  height flat (range < --flat-height-range) but normals show relief
                          (relief mean > --normal-relief, coverage > --normal-cov)
  kept                everything else — coherent terrain

Buckets (stratification tags on kept tiles, for representative holdouts): map name, and a
height-regime tertile (flat / rolling / steep) from per-tile height std.

Run from wow-viewer/data-harvester/ (CPU, read-only; a few minutes for a full scan):

    uv run python scripts/spec103_curate_dataset.py \
        --store ../output/datasets/v18/3_3_5_12340.zarr \
        --output ../output/spec103/curation_v18_v1
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

REQUIRED = ("minimap_rgb", "height_257", "normal_xyz")
DROP_PRECEDENCE = ("missing_signal", "blank_minimap", "object_contaminated", "height_normal_mismatch")


def _height_regime(h_std: float, edges: tuple[float, float]) -> str:
    if h_std < edges[0]:
        return "flat"
    if h_std < edges[1]:
        return "rolling"
    return "steep"


def main() -> int:
    ap = argparse.ArgumentParser(description="Spec 103 dataset curation + bucketing")
    ap.add_argument("--store", required=True, type=Path)
    ap.add_argument("--output", required=True, type=Path, help="directory for the curation manifest + summary")
    ap.add_argument("--min-rgb-std", type=float, default=1.0, help="drop blank-minimap tiles below this RGB std")
    ap.add_argument("--max-object-coverage", type=float, default=0.02,
                    help="drop tiles whose object_precise_mask covers more than this fraction "
                         "(0.0 = zero-object purist set; 0.02 tolerates a few stray pixels)")
    ap.add_argument("--flat-height-range", type=float, default=3.0,
                    help="a tile is 'flat' below this world-unit height range (mismatch check only)")
    ap.add_argument("--normal-relief", type=float, default=0.02,
                    help="normals 'show relief' above this mean sqrt(nx^2+ny^2)")
    ap.add_argument("--normal-cov", type=float, default=0.10, help="min normal_mask coverage for the mismatch check")
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    group = zarr.open_group(str(args.store), mode="r")
    index = pq.read_table(args.store / "index.parquet").to_pylist()
    n = len(index)
    present = {name: name in group for name in REQUIRED}
    has_object = "object_precise_mask" in group
    has_normal_mask = "normal_mask" in group

    cov = np.zeros(n, np.float32)
    rgb_std = np.zeros(n, np.float32)
    h_range = np.zeros(n, np.float32)
    h_std = np.zeros(n, np.float32)
    relief = np.zeros(n, np.float32)
    normal_cov = np.zeros(n, np.float32)

    rgb = group["minimap_rgb"]
    height = group["height_257"]
    for a in range(0, n, args.batch):
        b = min(n, a + args.batch)
        r = np.asarray(rgb[a:b]).reshape(b - a, -1).astype(np.float32)
        rgb_std[a:b] = r.std(axis=1)
        hh = np.asarray(height[a:b]).reshape(b - a, -1)
        h_range[a:b] = hh.max(axis=1) - hh.min(axis=1)
        h_std[a:b] = hh.std(axis=1)
        if has_object:
            m = np.asarray(group["object_precise_mask"][a:b]) > 0.5
            cov[a:b] = m.reshape(b - a, -1).mean(axis=1)
        if present["normal_xyz"]:
            nrm = np.asarray(group["normal_xyz"][a:b]).astype(np.float32)
            if np.abs(nrm).max(initial=0.0) > 1.5:
                nrm = nrm / 127.0
            nx, ny = nrm[..., 0], nrm[..., 1]
            rel = np.sqrt(np.clip(nx * nx + ny * ny, 0.0, None))
            if has_normal_mask:
                nmask = np.asarray(group["normal_mask"][a:b]).astype(np.float32)
                normal_cov[a:b] = nmask.reshape(b - a, -1).mean(axis=1)
                rel = rel * nmask
            else:
                normal_cov[a:b] = 1.0
            relief[a:b] = rel.reshape(b - a, -1).mean(axis=1)

    # height-regime tertile edges from the kept-eligible population (finite, non-blank)
    eligible = rgb_std >= args.min_rgb_std
    hstd_pop = h_std[eligible] if eligible.any() else h_std
    edges = (float(np.percentile(hstd_pop, 33)), float(np.percentile(hstd_pop, 66)))

    rows = []
    reason_counter: Counter = Counter()
    for i, meta in enumerate(index):
        missing_required = [name for name in REQUIRED if not present[name]] + \
            [name for name in REQUIRED if present[name] and not bool(meta.get(f"has_{name}", True))]
        if missing_required:
            reason = "missing_signal"
        elif rgb_std[i] < args.min_rgb_std:
            reason = "blank_minimap"
        elif has_object and cov[i] > args.max_object_coverage:
            reason = "object_contaminated"
        elif (h_range[i] < args.flat_height_range and relief[i] > args.normal_relief
              and normal_cov[i] >= args.normal_cov):
            reason = "height_normal_mismatch"
        else:
            reason = "kept"
        reason_counter[reason] += 1
        rows.append({
            "tile_id": int(meta["tile_id"]), "build": str(meta["build"]), "map": str(meta["map"]),
            "tile_x": int(meta["tile_x"]), "tile_y": int(meta["tile_y"]),
            "keep": reason == "kept", "reason": reason,
            "object_coverage": round(float(cov[i]), 5), "rgb_std": round(float(rgb_std[i]), 3),
            "height_range": round(float(h_range[i]), 3), "height_std": round(float(h_std[i]), 3),
            "normal_relief": round(float(relief[i]), 5),
            "height_regime": _height_regime(float(h_std[i]), edges),
            "bucket": f"{meta['map']}|{_height_regime(float(h_std[i]), edges)}",
        })

    args.output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(rows), args.output / "curation_manifest.parquet")

    kept = [r for r in rows if r["keep"]]
    by_map = Counter(r["map"] for r in kept)
    by_regime = Counter(r["height_regime"] for r in kept)
    summary = {
        "schema": "spec103-curation-v1",
        "created_utc": datetime.now(UTC).isoformat(),
        "store": str(args.store.resolve()),
        "index_sha256": sha256((args.store / "index.parquet").read_bytes()).hexdigest(),
        "thresholds": {
            "min_rgb_std": args.min_rgb_std, "max_object_coverage": args.max_object_coverage,
            "flat_height_range": args.flat_height_range, "normal_relief": args.normal_relief,
            "normal_cov": args.normal_cov, "height_regime_edges": edges,
        },
        "total_tiles": n, "kept": len(kept),
        "drop_reasons": {k: reason_counter[k] for k in DROP_PRECEDENCE},
        "kept_by_regime": dict(by_regime.most_common()),
        "kept_by_map": dict(by_map.most_common()),
        "object_coverage_alternatives": {
            "zero_objects": int((cov == 0).sum()),
            "leq_0.005": int((cov <= 0.005).sum()),
            "leq_0.02": int((cov <= 0.02).sum()),
            "leq_0.05": int((cov <= 0.05).sum()),
        },
    }
    (args.output / "curation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[curate] {args.store}")
    print(f"[curate] total={n} kept={len(kept)} ({100 * len(kept) / max(n, 1):.1f}%)")
    for reason in DROP_PRECEDENCE:
        print(f"  drop {reason:22s} {reason_counter[reason]}")
    print(f"  kept by regime: {dict(by_regime.most_common())}")
    print(f"  object-coverage alternatives: {summary['object_coverage_alternatives']}")
    print(f"[curate] manifest -> {args.output / 'curation_manifest.parquet'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
