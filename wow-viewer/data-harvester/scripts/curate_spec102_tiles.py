"""Build the only split manifest accepted by Spec 102 trainers."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from harvester.spec102.curation import (
    DEFAULT_M0_MAX_LIQUID_COVERAGE,
    STRICT_TARGET_FRAGMENT_TRACE_FIELDS,
    STRICT_TARGET_LIQUID_COUNTER_FIELDS,
    classify_tile,
)
from harvester.spec102.m0 import STRICT_OBJECT_TARGET_KEY

ERA_HOLDOUT = "0_5_3_3368"
MAP_HOLDOUT = ("3_3_5_12340", "Northrend")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def split_for(build: str, map_name: str) -> str:
    if build == ERA_HOLDOUT:
        return "test_era"
    if (build, map_name) == MAP_HOLDOUT:
        return "validation_map"
    return "train"


def main() -> int:
    parser = argparse.ArgumentParser(description="Curate aligned, terrain-visible Spec 102 tiles")
    parser.add_argument("--v25-store", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-liquid-coverage",
        type=float,
        default=DEFAULT_M0_MAX_LIQUID_COVERAGE,
        help="fixed initial-M0 dry-only ceiling; must remain 0.0 until a valid-loss mask exists",
    )
    args = parser.parse_args()
    if args.max_liquid_coverage != DEFAULT_M0_MAX_LIQUID_COVERAGE:
        raise ValueError(
            "initial M0 is dry-only: --max-liquid-coverage must be 0.0 until a per-pixel valid-loss mask exists"
        )

    group = zarr.open_group(str(args.v25_store), mode="r")
    required = [
        "minimap_rgb", STRICT_OBJECT_TARGET_KEY, "liquid_mask_256", "mcnk_flags_16",
        "normal_xyz_257", "height_257",
    ]
    missing = [name for name in required if name not in group]
    if missing:
        raise RuntimeError(f"Spec 102 curation refuses incomplete store; missing {missing}")
    index_path = args.v25_store / "index.parquet"
    index = pq.read_table(index_path).to_pylist()
    required_provenance = (
        "strict_target_materialized",
        "strict_target_version",
        "strict_target_status",
        "strict_target_arrays_present",
        "strict_target_geometry_unresolved_placement_count",
        "strict_target_fallback_required_placement_count",
        "strict_target_terrain_unknown_pixel_count",
        "strict_target_liquid_evidence_status",
        *STRICT_TARGET_LIQUID_COUNTER_FIELDS,
        *STRICT_TARGET_FRAGMENT_TRACE_FIELDS,
    )
    if index:
        missing_provenance = [name for name in required_provenance if name not in index[0]]
        if missing_provenance:
            raise RuntimeError(
                "Spec 102 curation requires strict transformed-geometry target provenance; "
                f"missing {missing_provenance}"
            )
    has_paired_wdl = "wdl_outer_17" in group and "wdl_inner_16" in group
    rows: list[dict] = []
    reasons = Counter()
    stage_counts = Counter()
    for expected_row, source in enumerate(index):
        row = int(source["row"])
        if row != expected_row:
            raise RuntimeError(f"index/array row mismatch at {expected_row}: index says {row}")
        result = classify_tile(
            minimap_rgb=np.asarray(group["minimap_rgb"][row]),
            strict_mask_257=np.asarray(group[STRICT_OBJECT_TARGET_KEY][row]),
            strict_target_version=str(source.get("strict_target_version", "missing")),
            strict_target_materialized=bool(source.get("strict_target_materialized", False)),
            strict_target_status=str(source.get("strict_target_status", "missing")),
            strict_target_arrays_present=bool(source.get("strict_target_arrays_present", False)),
            strict_target_geometry_unresolved_placement_count=int(
                source.get("strict_target_geometry_unresolved_placement_count", 0) or 0
            ),
            strict_target_fallback_required_placement_count=int(
                source.get("strict_target_fallback_required_placement_count", 0) or 0
            ),
            strict_target_terrain_unknown_pixel_count=int(
                source.get("strict_target_terrain_unknown_pixel_count", 0) or 0
            ),
            strict_target_liquid_evidence_status=str(
                source.get("strict_target_liquid_evidence_status", "missing")
            ),
            **{
                field: int(source.get(field, 0) or 0)
                for field in STRICT_TARGET_LIQUID_COUNTER_FIELDS
            },
            **{field: source.get(field) for field in STRICT_TARGET_FRAGMENT_TRACE_FIELDS},
            liquid_mask_256=np.asarray(group["liquid_mask_256"][row]),
            liquid_signal_present=bool(source.get("has_liquid_mask", False)),
            mcnk_flags_16=np.asarray(group["mcnk_flags_16"][row]),
            normal_xyz_257=np.asarray(group["normal_xyz_257"][row]),
            height_257=np.asarray(group["height_257"][row]),
            height_repaired=bool(source.get("height_repaired", False)),
            mismatch_reason=source.get("mismatch_reason"),
            has_paired_wdl=has_paired_wdl,
            max_liquid_coverage=args.max_liquid_coverage,
        )
        for reason in result.rejection_reasons:
            reasons[reason] += 1
        for stage in ("m0", "w1", "h2"):
            if getattr(result, f"eligible_{stage}"):
                stage_counts[f"{split_for(str(source['build']), str(source['map']))}:{stage}"] += 1
        rows.append({
            "row": row, "tile_id": int(source["tile_id"]), "build": str(source["build"]),
            "map": str(source["map"]), "tile_x": int(source["tile_x"]), "tile_y": int(source["tile_y"]),
            "split": split_for(str(source["build"]), str(source["map"])),
            "liquid_coverage": result.liquid_coverage,
            "liquid_flag_chunk_coverage": result.liquid_flag_chunk_coverage,
            "visible_terrain_coverage": result.visible_terrain_coverage,
            "minimap_dominant_color_fraction": result.minimap_dominant_color_fraction,
            "minimap_blue_fraction": result.minimap_blue_fraction,
            "liquid_signal_present": result.liquid_signal_present,
            "liquid_source": source.get("liquid_source"),
            "strict_target_version": str(source.get("strict_target_version", "missing")),
            "strict_target_materialized": bool(source.get("strict_target_materialized", False)),
            "strict_target_status": str(source.get("strict_target_status", "missing")),
            "strict_target_arrays_present": bool(source.get("strict_target_arrays_present", False)),
            "strict_target_geometry_unresolved_placement_count": int(
                source.get("strict_target_geometry_unresolved_placement_count", 0) or 0
            ),
            "strict_target_fallback_required_placement_count": int(
                source.get("strict_target_fallback_required_placement_count", 0) or 0
            ),
            "strict_target_terrain_unknown_pixel_count": int(
                source.get("strict_target_terrain_unknown_pixel_count", 0) or 0
            ),
            "strict_target_liquid_evidence_status": str(
                source.get("strict_target_liquid_evidence_status", "missing")
            ),
            **{
                field: int(source.get(field, 0) or 0)
                for field in STRICT_TARGET_LIQUID_COUNTER_FIELDS
            },
            **{field: source.get(field) for field in STRICT_TARGET_FRAGMENT_TRACE_FIELDS},
            "eligible_m0": result.eligible_m0, "eligible_w1": result.eligible_w1,
            "eligible_h2": result.eligible_h2, "rejection_reasons": list(result.rejection_reasons),
        })
    report = {
        "schema": "spec102-curated-split-v5",
        "source_store": str(args.v25_store.resolve()),
        "source_index_sha256": sha256_file(index_path),
        "max_liquid_coverage": args.max_liquid_coverage,
        "m0_liquid_policy": "dry_only_no_per_pixel_valid_loss_mask",
        "paired_wdl_contract": "wdl_outer_17 + wdl_inner_16; derived wdl_height_33 is prohibited",
        "counts": dict(stage_counts), "rejection_counts": dict(reasons), "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({key: report[key] for key in ("schema", "counts", "rejection_counts")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
