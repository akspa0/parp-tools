"""Spec 118 US1: read-only audit of the harvested strict visible-object signals.

Emits a ``v118-object-mask-audit-v1`` document (data-model.md Mask Audit Record) answering the
spec's US1 acceptance questions against a real rebuilt store:

- marked-fraction distribution (p05/p50/p95) per map and corpus-wide -- visibly-covered fraction,
  NOT the 80-90% the full-footprint mask produced (FR-004 / SC-001);
- instance-id/mask consistency: instance ids appear only where the mask is positive (FR-002);
- class-per-instance consistency: each instance's pixels carry a single source class, up to a
  documented mixed-pixel tolerance (front-most overlap seams can legitimately switch class at
  shared edges);
- visible-vs-footprint reduction factor where ``object_mask_257`` is also present (SC-001's
  >=3x target on underground-heavy tiles);
- exclusion counts: rows lacking the arrays are counted, never silently skipped (the store
  builder's excluded-and-counted behavior surfaces here as missing rows).

The audit never writes to the store. Dry-run prints the document; ``--write`` persists it.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "v118-object-mask-audit-v1"
MASK_ARRAY = "object_precise_mask"
SOURCE_ARRAY = "object_geometry_visible_source_257"  # optional: absent for the binary v18 masks
INSTANCE_ARRAY = "object_instance_mask"
FOOTPRINT_ARRAY = "object_mask"

#: Per-instance pixels whose class disagrees with the instance's modal class beyond this share
#: count as a consistency violation. Front-most overlap seams can legitimately flip class at a
#: shared edge, so the tolerance is small but nonzero.
CLASS_MIX_TOLERANCE = 0.05


class ObjectMaskAuditError(ValueError):
    """Raised when the store cannot support the audit (missing arrays, empty index)."""


def _percentiles(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"p05": 0.0, "p50": 0.0, "p95": 0.0}
    return {
        "p05": float(np.percentile(values, 5)),
        "p50": float(np.percentile(values, 50)),
        "p95": float(np.percentile(values, 95)),
    }


def audit_object_masks(store: Path, *, map_filter: str | None = None) -> dict[str, Any]:
    """Audit one v50 store and return the ``v118-object-mask-audit-v1`` document."""
    import zarr  # local import: keeps module importable without zarr for contract tests

    store = Path(store)
    group = zarr.open_group(str(store), mode="r")
    if MASK_ARRAY not in group:
        raise ObjectMaskAuditError(
            f"store lacks {MASK_ARRAY!r} -- expected until a US1 rebuild lands; "
            "the audit cannot run against a store harvested before the Spec 118 catalog amendment"
        )
    if INSTANCE_ARRAY not in group:
        raise ObjectMaskAuditError(
            f"store has {MASK_ARRAY!r} but lacks {INSTANCE_ARRAY!r}: the mask and its instance ids "
            "are painted together; a partial set indicates a mixed-version store rebuild"
        )

    index_path = store / "index.parquet"
    if not index_path.exists():
        raise ObjectMaskAuditError(f"store has no index.parquet: {store}")
    import pyarrow.parquet as pq

    index = pq.read_table(index_path).to_pylist()
    # Keep the ORIGINAL store row indices: arrays are read positionally, so a map filter must
    # filter (position, row) pairs, not just the rows.
    indexed_rows = list(enumerate(index))
    if map_filter is not None:
        indexed_rows = [(i, row) for i, row in indexed_rows if row.get("map") == map_filter]

    masks = group[MASK_ARRAY]
    sources = group[SOURCE_ARRAY] if SOURCE_ARRAY in group else None
    instances = group[INSTANCE_ARRAY]
    footprint = group[FOOTPRINT_ARRAY] if FOOTPRINT_ARRAY in group else None

    fractions: list[float] = []
    footprint_fractions: list[float] = []
    instance_counts: list[int] = []
    instance_pixel_counts: list[int] = []
    class_violations = 0
    mask_instance_mismatches = 0
    touched_tiles = 0
    per_map: dict[str, list[float]] = {}

    class_consistency_evaluated = sources is not None
    for row_idx, row in indexed_rows:
        mask = np.asarray(masks[row_idx], dtype=np.float32)
        source = np.asarray(sources[row_idx], dtype=np.uint8) if sources is not None else None
        instance = np.asarray(instances[row_idx], dtype=np.int32)
        fraction = float((mask > 0).mean())
        fractions.append(fraction)
        per_map.setdefault(str(row.get("map", "unknown")), []).append(fraction)
        if mask.size and (mask > 0).any():
            touched_tiles += 1

        # FR-002 consistency: instance ids only where the mask is positive.
        mismatches = int(((instance > 0) != (mask > 0)).sum())
        mask_instance_mismatches += mismatches

        ids = np.unique(instance[instance > 0])
        instance_counts.append(int(ids.size))
        for iid in ids:
            region = instance == iid
            pixel_count = int(region.sum())
            instance_pixel_counts.append(pixel_count)
            if source is not None:
                classes = source[region]
                modal = int(np.bincount(classes, minlength=3).argmax())
                mixed = float((classes != modal).mean()) if pixel_count else 0.0
                if mixed > CLASS_MIX_TOLERANCE:
                    class_violations += 1

        if footprint is not None:
            footprint_fractions.append(float((np.asarray(footprint[row_idx]) > 0).mean()))

    reduction = None
    if footprint_fractions:
        # Median per-tile ratio of footprint-marked to visible-marked fraction on tiles where the
        # footprint marked anything. Values >> 1 confirm the over-masking reduction (SC-001).
        ratios = [
            fb / vis
            for fb, vis in zip(footprint_fractions, fractions, strict=True)
            if fb > 0 and vis > 0
        ]
        reduction = {
            "footprint_fraction": _percentiles(np.asarray(footprint_fractions, dtype=np.float64)),
            "median_footprint_to_visible_ratio": float(np.median(ratios)) if ratios else None,
            "tiles_compared": len(ratios),
        }

    document: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "store": str(store),
        "map_filter": map_filter,
        "tile_count": len(indexed_rows),
        "object_touched_tile_count": touched_tiles,
        "marked_fraction": _percentiles(np.asarray(fractions, dtype=np.float64)),
        "per_map_marked_fraction": {
            name: _percentiles(np.asarray(values, dtype=np.float64)) for name, values in sorted(per_map.items())
        },
        "instance_count_per_tile": _percentiles(np.asarray(instance_counts, dtype=np.float64)),
        "instance_visible_pixel_count": _percentiles(np.asarray(instance_pixel_counts, dtype=np.float64)),
        "class_consistency": {
            "evaluated": class_consistency_evaluated,
            "tolerance": CLASS_MIX_TOLERANCE,
            "violation_count": class_violations if class_consistency_evaluated else None,
            "note": None if class_consistency_evaluated
            else "binary object mask carries no per-pixel class source; check skipped",
        },
        "mask_instance_mismatch_pixel_count": mask_instance_mismatches,
        "visible_vs_footprint": reduction,
    }
    return document


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Audit harvested strict visible-object masks (read-only).")
    ap.add_argument("--store", required=True, type=Path, help="v50 Zarr store to audit")
    ap.add_argument("--map", dest="map_filter", default=None, help="restrict to one map (e.g. Kalimdor)")
    ap.add_argument("--output", type=Path, default=None, help="where --write persists the audit JSON")
    ap.add_argument("--write", action="store_true", help="persist the audit document (default: print only)")
    args = ap.parse_args(argv)

    try:
        document = audit_object_masks(args.store, map_filter=args.map_filter)
    except ObjectMaskAuditError as exc:
        print(f"REFUSING: {exc}", file=sys.stderr)
        return 2

    text = json.dumps(document, indent=2, sort_keys=True)
    if args.write:
        if args.output is None:
            print("REFUSING: --write requires --output", file=sys.stderr)
            return 2
        if args.output.exists():
            print(f"REFUSING: output already exists: {args.output}", file=sys.stderr)
            return 2
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
        print(f"wrote {args.output}")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
