"""Curate every row of the raw 3.3.5 M0 corpus without silently dropping maps."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

from harvester.spec102.curation import (
    DEFAULT_M0_MAX_LIQUID_COVERAGE,
    STRICT_LIQUID_EVIDENCE_DRY,
    STRICT_TARGET_FRAGMENT_TRACE_FIELDS,
    STRICT_TARGET_LIQUID_COUNTER_FIELDS,
    classify_tile,
)
from harvester.spec102.m0 import STRICT_OBJECT_TARGET_KEY
from harvester.spec102.numeric_store import (
    RAW_V18_IDENTITY_SELECTION_MODE,
    STRICT_LIQUID_COUNTER_FIELDS,
    STRICT_LIQUID_EVIDENCE_STATUS_FIELD,
    STRICT_OBJECT_TARGET_VERSION_FIELD,
)
from harvester.spec102.strict_target_contract import REQUIRED_STRICT_OBJECT_TARGET_VERSION
from harvester.spec102.strict_target_contract import (
    STRICT_FRAGMENT_COUNT_FIELD,
    STRICT_FRAGMENT_SHA256_FIELD,
    STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD,
    STRICT_FRAGMENT_TRACE_SCHEMA_FIELD,
    STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY,
    STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD,
    STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD,
    STRICT_FRAGMENT_TRACE_VALIDATED_FIELD,
    sha256_tree,
)

SCHEMA = "spec102-m0-full-3_3_5-curation-v4"
BUILD = "3_3_5_12340"
IDENTITY_FIELDS = ("build", "map", "tile_id", "tile_x", "tile_y")
RAW_STRICT_LIQUID_PROVENANCE_FIELDS = (
    STRICT_LIQUID_EVIDENCE_STATUS_FIELD,
    *STRICT_LIQUID_COUNTER_FIELDS,
)
RAW_STRICT_FRAGMENT_PROVENANCE_FIELDS = (
    STRICT_FRAGMENT_TRACE_SCHEMA_FIELD,
    STRICT_FRAGMENT_COUNT_FIELD,
    STRICT_FRAGMENT_SHA256_FIELD,
    "object_geometry_target_assets_json",
    "object_geometry_target_unresolved_placements_json",
    STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD,
    STRICT_FRAGMENT_TRACE_VALIDATED_FIELD,
    STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD,
    STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD,
)
STRICT_FRAGMENT_PROVENANCE_PAIRS = (
    ("strict_target_fragment_trace_schema", STRICT_FRAGMENT_TRACE_SCHEMA_FIELD),
    ("strict_target_fragment_count", STRICT_FRAGMENT_COUNT_FIELD),
    ("strict_target_fragment_sha256", STRICT_FRAGMENT_SHA256_FIELD),
    ("strict_target_assets_json", "object_geometry_target_assets_json"),
    ("strict_target_unresolved_placements_json", "object_geometry_target_unresolved_placements_json"),
    ("strict_target_fragment_trace_arrays_present", STRICT_FRAGMENT_TRACE_ARRAYS_PRESENT_FIELD),
    ("strict_target_fragment_trace_validated", STRICT_FRAGMENT_TRACE_VALIDATED_FIELD),
    ("strict_target_fragment_trace_sidecar_start", STRICT_FRAGMENT_TRACE_SIDECAR_START_FIELD),
    ("strict_target_fragment_trace_sidecar_end", STRICT_FRAGMENT_TRACE_SIDECAR_END_FIELD),
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _require_exact_identity(
    numeric: dict,
    raw: dict,
    *,
    numeric_row: int,
) -> int:
    try:
        source_row = int(numeric["source_v18_row"])
        source_tile = int(numeric["source_v18_tile_id"])
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f"numeric row {numeric_row} lacks explicit raw-V18 identity") from error
    if source_row != numeric_row:
        raise RuntimeError(f"numeric row {numeric_row} is not an ordinal raw-V18 identity copy")
    if source_tile != int(raw["tile_id"]):
        raise RuntimeError(f"numeric row {numeric_row} source_v18_tile_id does not match raw V18")
    for field in IDENTITY_FIELDS:
        if str(numeric.get(field)) != str(raw.get(field)):
            raise RuntimeError(f"numeric row {numeric_row} mismatches raw V18 {field}")
    return source_row


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Curate all raw 3.3.5 Spec 102 rows with explicit M0 eligibility reasons"
    )
    parser.add_argument("--store", required=True, type=Path)
    parser.add_argument("--raw-v18-store", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--max-liquid-coverage",
        type=float,
        default=DEFAULT_M0_MAX_LIQUID_COVERAGE,
        help="fixed initial-M0 dry-only ceiling; must remain 0.0 until a valid-loss mask exists",
    )
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite curation manifest: {args.output}")
    if args.max_liquid_coverage != DEFAULT_M0_MAX_LIQUID_COVERAGE:
        raise ValueError(
            "initial M0 is dry-only: --max-liquid-coverage must be 0.0 until a per-pixel valid-loss mask exists"
        )

    contract = json.loads((args.store / "contract.json").read_text(encoding="utf-8"))
    if contract.get("selection_mode") != RAW_V18_IDENTITY_SELECTION_MODE:
        raise RuntimeError("full 3.3.5 curation requires the explicit raw_v18_identity numeric store")
    if Path(contract.get("source_selection_store", "")).resolve() != args.raw_v18_store.resolve():
        raise RuntimeError("numeric store is not bound to the supplied raw 3.3.5 V18 source")
    target_provenance = contract.get("object_target_provenance")
    if not isinstance(target_provenance, dict):
        raise RuntimeError("numeric store does not declare strict object-target provenance")
    if target_provenance.get("target_array") != STRICT_OBJECT_TARGET_KEY:
        raise RuntimeError("numeric store target is not the strict terrain-visible geometry array")
    if target_provenance.get("legacy_precise_mask_policy") != "prohibited":
        raise RuntimeError("numeric store does not explicitly prohibit the legacy object_precise_mask")
    if target_provenance.get("terrain_occlusion_clipped") is not True:
        raise RuntimeError("numeric store does not prove raw-MCVT terrain-Z clipping")
    fragment_trace_contract = target_provenance.get("fragment_trace")
    if not isinstance(fragment_trace_contract, dict):
        raise RuntimeError("numeric store does not declare the strict v3 fragment-trace audit sidecar")
    if fragment_trace_contract.get("schema") != REQUIRED_STRICT_OBJECT_TARGET_VERSION:
        raise RuntimeError("numeric store fragment-trace schema is not the exact C# v3 contract")
    if fragment_trace_contract.get("sidecar") != STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY:
        raise RuntimeError("numeric store fragment-trace sidecar declaration is wrong")
    if fragment_trace_contract.get("model_input_policy") != "audit_sidecar_only_not_a_numeric_model_input":
        raise RuntimeError("numeric store must keep fragment trace out of model inputs")
    group = zarr.open_group(str(args.store), mode="r")
    raw_group = zarr.open_group(str(args.raw_v18_store), mode="r")
    required_numeric = (
        "minimap_rgb", STRICT_OBJECT_TARGET_KEY,
        "object_geometry_visible_top_elevation_257",
        "object_geometry_visible_terrain_elevation_257",
        "object_geometry_visible_source_257",
        "liquid_mask_256", "mcnk_flags_16", "normal_xyz_257", "height_257",
    )
    missing = [name for name in required_numeric if name not in group]
    if missing:
        raise RuntimeError(f"numeric store lacks M0 curation signals: {missing}")
    required_raw = (
        "object_geometry_visible_mask",
        "object_geometry_visible_top_elevation",
        "object_geometry_visible_terrain_elevation",
        "object_geometry_visible_source",
    )
    missing_raw = [name for name in required_raw if name not in raw_group]
    if missing_raw:
        raise RuntimeError(f"raw V18 source lacks strict object-target arrays: {missing_raw}")
    numeric_index = pq.read_table(args.store / "index.parquet").to_pylist()
    raw_index = pq.read_table(args.raw_v18_store / "index.parquet").to_pylist()
    if len(numeric_index) != len(raw_index):
        raise RuntimeError("full raw-V18 numeric store must preserve every source row")
    if {str(row.get("build")) for row in numeric_index} != {BUILD}:
        raise RuntimeError("full M0 corpus must contain only 3_3_5_12340")
    source_trace = (
        contract.get("source_v18_contracts", {})
        .get(BUILD, {})
        .get("strict_fragment_trace_sidecar", {})
    )
    raw_trace_hash = sha256_tree(args.raw_v18_store / STRICT_FRAGMENT_TRACE_SIDECAR_DIRECTORY)
    if source_trace.get("sha256") != raw_trace_hash:
        raise RuntimeError("numeric store is not bound to the supplied raw V18 fragment-trace sidecar")

    rows: list[dict] = []
    reasons = Counter()
    map_counts: dict[str, Counter] = defaultdict(Counter)
    for row_number, numeric in enumerate(numeric_index):
        if int(numeric.get("row", -1)) != row_number:
            raise RuntimeError(f"numeric index row numbering is not contiguous at {row_number}")
        raw = raw_index[row_number]
        source_row = _require_exact_identity(numeric, raw, numeric_row=row_number)
        strict_fields = (
            STRICT_OBJECT_TARGET_VERSION_FIELD,
            "object_geometry_target_status",
            "object_geometry_target_materialized",
            "object_geometry_target_arrays_present",
            "object_geometry_target_geometry_unresolved_placement_count",
            "object_geometry_target_fallback_required_placement_count",
            "object_geometry_target_terrain_unknown_pixel_count",
            *RAW_STRICT_LIQUID_PROVENANCE_FIELDS,
            *RAW_STRICT_FRAGMENT_PROVENANCE_FIELDS,
        )
        if any(field not in raw for field in strict_fields):
            raise RuntimeError(f"raw V18 row {row_number} lacks strict object-target provenance")
        if bool(numeric.get("strict_target_materialized", False)) != bool(
            raw.get("object_geometry_target_materialized", False)
        ):
            raise RuntimeError(f"numeric row {row_number} strict target materialization differs from raw V18")
        if str(numeric.get("strict_target_version", "")) != str(
            raw.get(STRICT_OBJECT_TARGET_VERSION_FIELD, "")
        ):
            raise RuntimeError(f"numeric row {row_number} strict target version differs from raw V18")
        if str(numeric.get("strict_target_status", "")) != str(raw.get("object_geometry_target_status", "")):
            raise RuntimeError(f"numeric row {row_number} strict target status differs from raw V18")
        if str(numeric.get("strict_target_liquid_evidence_status", "")) != str(
            raw.get(STRICT_LIQUID_EVIDENCE_STATUS_FIELD, "")
        ):
            raise RuntimeError(f"numeric row {row_number} liquid-evidence status differs from raw V18")
        for raw_field in STRICT_LIQUID_COUNTER_FIELDS:
            numeric_field = f"strict_target_{raw_field.removeprefix('object_geometry_target_')}"
            if numeric.get(numeric_field) != raw.get(raw_field):
                raise RuntimeError(
                    f"numeric row {row_number} liquid-evidence {raw_field} differs from raw V18"
                )
        for numeric_field, raw_field in STRICT_FRAGMENT_PROVENANCE_PAIRS:
            if numeric.get(numeric_field) != raw.get(raw_field):
                raise RuntimeError(
                    f"numeric row {row_number} fragment-trace {raw_field} differs from raw V18"
                )
        if numeric.get("strict_target_fragment_trace_source_sidecar_sha256") != raw_trace_hash:
            raise RuntimeError(f"numeric row {row_number} is not bound to raw V18 fragment-trace sidecar")
        result = classify_tile(
            minimap_rgb=np.asarray(group["minimap_rgb"][row_number]),
            strict_mask_257=np.asarray(group[STRICT_OBJECT_TARGET_KEY][row_number]),
            strict_target_version=str(numeric.get("strict_target_version", "missing")),
            strict_target_materialized=bool(numeric.get("strict_target_materialized", False)),
            strict_target_status=str(numeric.get("strict_target_status", "missing")),
            strict_target_arrays_present=bool(numeric.get("strict_target_arrays_present", False)),
            strict_target_geometry_unresolved_placement_count=int(
                numeric.get("strict_target_geometry_unresolved_placement_count", 0) or 0
            ),
            strict_target_fallback_required_placement_count=int(
                numeric.get("strict_target_fallback_required_placement_count", 0) or 0
            ),
            strict_target_terrain_unknown_pixel_count=int(
                numeric.get("strict_target_terrain_unknown_pixel_count", 0) or 0
            ),
            strict_target_liquid_evidence_status=str(
                numeric.get("strict_target_liquid_evidence_status", "missing")
            ),
            **{
                field: int(numeric.get(field, 0) or 0)
                for field in STRICT_TARGET_LIQUID_COUNTER_FIELDS
            },
            **{field: numeric.get(field) for field in STRICT_TARGET_FRAGMENT_TRACE_FIELDS},
            liquid_mask_256=np.asarray(group["liquid_mask_256"][row_number]),
            liquid_signal_present=bool(numeric.get("has_liquid_mask", False)),
            mcnk_flags_16=np.asarray(group["mcnk_flags_16"][row_number]),
            normal_xyz_257=np.asarray(group["normal_xyz_257"][row_number]),
            height_257=np.asarray(group["height_257"][row_number]),
            height_repaired=bool(numeric.get("height_repaired", False)),
            mismatch_reason=numeric.get("mismatch_reason"),
            has_paired_wdl=False,
            max_liquid_coverage=args.max_liquid_coverage,
        )
        for reason in result.rejection_reasons:
            reasons[reason] += 1
        map_name = str(numeric["map"])
        map_counts[map_name]["rows"] += 1
        map_counts[map_name]["eligible_m0"] += int(result.eligible_m0)
        map_counts[map_name]["rejected_m0"] += int(not result.eligible_m0)
        rows.append({
            "row": row_number,
            "source_v18_row": source_row,
            "source_v18_tile_id": int(raw["tile_id"]),
            **{field: numeric[field] for field in IDENTITY_FIELDS},
            "strict_target_materialized": bool(numeric.get("strict_target_materialized", False)),
            "strict_target_version": str(numeric.get("strict_target_version", "missing")),
            "strict_target_status": str(numeric.get("strict_target_status", "missing")),
            "strict_target_arrays_present": bool(numeric.get("strict_target_arrays_present", False)),
            "strict_target_geometry_unresolved_placement_count": int(
                numeric.get("strict_target_geometry_unresolved_placement_count", 0) or 0
            ),
            "strict_target_fallback_required_placement_count": int(
                numeric.get("strict_target_fallback_required_placement_count", 0) or 0
            ),
            "strict_target_terrain_unknown_pixel_count": int(
                numeric.get("strict_target_terrain_unknown_pixel_count", 0) or 0
            ),
            "strict_target_liquid_evidence_status": str(
                numeric.get("strict_target_liquid_evidence_status", "missing")
            ),
            **{
                field: int(numeric.get(field, 0) or 0)
                for field in STRICT_TARGET_LIQUID_COUNTER_FIELDS
            },
            **{field: numeric.get(field) for field in STRICT_TARGET_FRAGMENT_TRACE_FIELDS},
            "liquid_coverage": result.liquid_coverage,
            "liquid_flag_chunk_coverage": result.liquid_flag_chunk_coverage,
            "visible_terrain_coverage": result.visible_terrain_coverage,
            "minimap_dominant_color_fraction": result.minimap_dominant_color_fraction,
            "minimap_blue_fraction": result.minimap_blue_fraction,
            "liquid_signal_present": result.liquid_signal_present,
            "liquid_source": numeric.get("liquid_source"),
            "eligible_m0": result.eligible_m0,
            "rejection_reasons": list(result.rejection_reasons),
        })

    report = {
        "schema": SCHEMA,
        "build": BUILD,
        "store": str(args.store.resolve()),
        "store_contract_sha256": sha256_file(args.store / "contract.json"),
        "store_index_sha256": sha256_file(args.store / "index.parquet"),
        "raw_v18_store": str(args.raw_v18_store.resolve()),
        "raw_v18_index_sha256": sha256_file(args.raw_v18_store / "index.parquet"),
        "raw_v18_fragment_trace_sidecar_sha256": raw_trace_hash,
        "max_liquid_coverage": args.max_liquid_coverage,
        "m0_liquid_policy": "dry_only_no_per_pixel_valid_loss_mask",
        "target_policy": (
            "only CompleteEmpty/CompleteVisible strict transformed-geometry targets with raw-MCVT-Z "
            "visibility and liquid-evidence proof are eligible; target version must be "
            f"{REQUIRED_STRICT_OBJECT_TARGET_VERSION} with a lossless audit-only fragment trace; "
            "legacy object_precise_mask is prohibited"
        ),
        "counts": {
            "rows": len(rows),
            "maps": len(map_counts),
            "eligible_m0": sum(int(row["eligible_m0"]) for row in rows),
            "rejected_m0": sum(int(not row["eligible_m0"]) for row in rows),
            "strict_target_materialized": sum(int(row["strict_target_materialized"]) for row in rows),
            "unmaterialized_strict_target": sum(int(not row["strict_target_materialized"]) for row in rows),
            "strict_target_liquid_evidence_status": dict(
                Counter(str(row["strict_target_liquid_evidence_status"]) for row in rows)
            ),
            "strict_target_non_dry": sum(
                int(row["strict_target_liquid_evidence_status"] != STRICT_LIQUID_EVIDENCE_DRY)
                for row in rows
            ),
        },
        "rejection_counts": dict(reasons),
        "map_counts": {name: dict(count) for name, count in sorted(map_counts.items())},
        "rows": rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(args.output.resolve()),
        "schema": report["schema"],
        "counts": report["counts"],
        "rejection_counts": report["rejection_counts"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
