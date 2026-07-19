"""Same-tile (authored LR, terrain-only detail HR) pair set (Spec 113 US2, T011).

A pair exists only for a tile that carries BOTH a populated authored client minimap and a
successful detail 1024 render — a tile missing either is excluded and counted, never zero-filled
into a pair (FR-004). A conventional pixel-SR pairing may use the US1 gate's corrective transform.
The explicitly selected terrain-only cross-domain mode instead requires a persisted visual-review
report, keeps the same-row identity orientation, and records that authored objects are intentionally
absent from the target. The split is deterministic per tile
(``source_group_id``) within each map, so no tile's pair crosses train/eval (FR-005). Kalimdor and
Azeroth only.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from harvester.v50.minimap_alignment import apply_dihedral, apply_translation

PAIRSET_SCHEMA = "v50-sr-pairset-v1"
ALLOWED_MAPS = frozenset({"Kalimdor", "Azeroth"})
SCALE = 4  # 256 -> 1024; recorded in attrs, not hardcoded downstream (FR-011)


class PairSetBuildError(ValueError):
    """Raised when the pair set cannot be built as declared."""


def assign_pair_split(entries: list[dict], val_fraction: float) -> None:
    """Deterministic within-map holdout of TILES (each tile = one pair = one group)."""
    by_map: dict[str, list[dict]] = defaultdict(list)
    for entry in entries:
        by_map[entry["map"]].append(entry)
    for rows in by_map.values():
        rows.sort(key=lambda row: hashlib.sha256(row["source_group_id"].encode()).hexdigest())
        val_count = max(1, round(len(rows) * val_fraction))
        for position, row in enumerate(rows):
            row["split"] = "val" if position < val_count else "train"


def build_sr_pairset(
    *,
    stores: list[Path],
    output: Path,
    alignment_report: Path,
    val_fraction: float,
    terrain_only_cross_domain: bool = False,
    visual_review_report: Path | None = None,
) -> dict:
    import pyarrow as pa
    import pyarrow.parquet as pq
    import zarr

    if not 0.0 < val_fraction < 1.0:
        raise PairSetBuildError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if output.exists():
        raise PairSetBuildError(f"refusing to overwrite existing output: {output}")

    alignment = json.loads(alignment_report.read_text(encoding="utf-8"))
    gate = str(alignment.get("gate", ""))
    pixel_gate_passed = gate in ("pass_identity", "pass_with_transform")
    if not pixel_gate_passed and not terrain_only_cross_domain:
        raise PairSetBuildError(
            f"US1 alignment gate is {gate!r}; pairs may only be built after pass_identity/pass_with_transform"
        )
    if terrain_only_cross_domain:
        if visual_review_report is None:
            raise PairSetBuildError(
                "terrain-only cross-domain pairing requires --visual-review evidence"
            )
        corrective = "identity"
        corrective_offset_lr = [0, 0]
        pairing_mode = "terrain_only_cross_domain_same_tile"
    else:
        corrective = alignment.get("corrective_transform") or "identity"
        corrective_offset_lr = alignment.get("corrective_offset_lr") or [0, 0]
        pairing_mode = "pixel_registered_sr"
    if len(corrective_offset_lr) != 2:
        raise PairSetBuildError(
            f"alignment corrective_offset_lr must be [dy, dx], got {corrective_offset_lr!r}"
        )
    aligned_stores = {
        str(Path(path).resolve())
        for path in alignment.get("stores", [alignment.get("store")])
        if path
    }
    requested_stores = {str(path.resolve()) for path in stores}
    if aligned_stores and not requested_stores.issubset(aligned_stores):
        missing = sorted(requested_stores - aligned_stores)
        raise PairSetBuildError(
            f"alignment report does not cover requested stores: {missing}"
        )
    visual_review_path = None
    if terrain_only_cross_domain:
        visual_review_path = Path(visual_review_report).resolve()
        reviews = json.loads(Path(visual_review_report).read_text(encoding="utf-8"))
        if not isinstance(reviews, list) or not reviews:
            raise PairSetBuildError("visual review must be a non-empty report list")
        reviewed_stores: set[str] = set()
        for review in reviews:
            if review.get("schema") != "v50-store-visual-review-v1":
                raise PairSetBuildError("visual review has an unsupported schema")
            if review.get("pixel_equality_required") is not False:
                raise PairSetBuildError("visual review must explicitly declare pixel_equality_required=false")
            if review.get("authored_object_policy") != "may_contain_client_baked_objects":
                raise PairSetBuildError("visual review does not declare the authored object policy")
            if review.get("synthetic_object_policy") != "terrain_only_no_objects":
                raise PairSetBuildError("visual review does not declare the terrain-only synthetic policy")
            if not review.get("rows"):
                raise PairSetBuildError("visual review contains no reviewed rows")
            reviewed_stores.add(str(Path(review["store"]).resolve()))
        if not requested_stores.issubset(reviewed_stores):
            missing = sorted(requested_stores - reviewed_stores)
            raise PairSetBuildError(f"visual review does not cover requested stores: {missing}")

    entries: list[dict] = []
    excluded_missing_authored = 0
    excluded_missing_detail = 0
    groups = []
    for store_path in stores:
        group = zarr.open_group(str(store_path), mode="r")
        if group.attrs.get("minimap_rgb_1024_render_mode") != "detail":
            raise PairSetBuildError(
                f"store {store_path} is not provenance-marked with minimap_rgb_1024 render_mode=detail"
            )
        index = pq.read_table(store_path / "index.parquet").to_pylist()
        maps_here = {str(r.get("map", "")) for r in index}
        out_of_scope = sorted(maps_here - ALLOWED_MAPS)
        if out_of_scope:
            raise PairSetBuildError(f"store {store_path} contains out-of-scope maps {out_of_scope}")
        groups.append(group)
        for row_id, row in enumerate(index):
            has_authored = "minimap_rgb_authored" in group and bool(np.asarray(group["minimap_rgb_authored"][row_id]).any())
            has_detail = "minimap_rgb_1024" in group and bool(np.asarray(group["minimap_rgb_1024"][row_id]).any())
            if not has_authored:
                excluded_missing_authored += 1
            if not has_detail:
                excluded_missing_detail += 1
            if not has_authored or not has_detail:
                continue
            entries.append(
                {
                    "build": str(row["build"]),
                    "map": str(row["map"]),
                    "tile_x": int(row["tile_x"]),
                    "tile_y": int(row["tile_y"]),
                    "source_tile_id": row_id,
                    "source_store_index": len(groups) - 1,
                    "source_store": str(store_path.resolve()),
                    "source_group_id": (
                        f"real:{row['build']}:{row['map']}:{row['tile_x']}:{row['tile_y']}"
                    ),
                }
            )

    if not entries:
        raise PairSetBuildError("no tile carries both an authored minimap and a detail render")
    assign_pair_split(entries, val_fraction)

    lr_shape = np.asarray(groups[entries[0]["source_store_index"]]["minimap_rgb_authored"][entries[0]["source_tile_id"]]).shape
    hr_shape = np.asarray(groups[entries[0]["source_store_index"]]["minimap_rgb_1024"][entries[0]["source_tile_id"]]).shape
    if len(lr_shape) != 3 or len(hr_shape) != 3 or lr_shape[2] != 3 or hr_shape[2] != 3:
        raise PairSetBuildError(f"expected RGB LR/HR arrays, got lr={lr_shape} hr={hr_shape}")
    scale_y = hr_shape[0] // lr_shape[0]
    scale_x = hr_shape[1] // lr_shape[1]
    if (
        hr_shape[0] != lr_shape[0] * scale_y
        or hr_shape[1] != lr_shape[1] * scale_x
        or scale_y != scale_x
        or scale_y != SCALE
    ):
        raise PairSetBuildError(f"expected exact x{SCALE} LR/HR shapes, got lr={lr_shape} hr={hr_shape}")

    out = zarr.open_group(str(output), mode="w")
    out.attrs.update(
        {
            "schema": PAIRSET_SCHEMA,
            "scale": SCALE,
            "maps": sorted({e["map"] for e in entries}),
            "corrective_transform": corrective,
            "corrective_offset_lr": [int(corrective_offset_lr[0]), int(corrective_offset_lr[1])],
            "pairing_mode": pairing_mode,
            "authored_object_policy": "may_contain_client_baked_objects",
            "synthetic_object_policy": "terrain_only_no_objects",
            "alignment_report": str(alignment_report.resolve()),
            "visual_review_report": str(visual_review_path) if visual_review_path else "",
            "source_stores": [str(p.resolve()) for p in stores],
        }
    )
    out.create_array("lr", shape=(len(entries), *lr_shape), dtype=np.uint8, chunks=(1, *lr_shape))
    out.create_array("hr", shape=(len(entries), *hr_shape), dtype=np.uint8, chunks=(1, *hr_shape))

    index_rows = []
    for target, entry in enumerate(entries):
        source = groups[entry["source_store_index"]]
        tile_id = entry["source_tile_id"]
        out["lr"][target] = np.asarray(source["minimap_rgb_authored"][tile_id], dtype=np.uint8)
        hr = np.asarray(source["minimap_rgb_1024"][tile_id], dtype=np.uint8)
        if corrective != "identity":
            hr = np.ascontiguousarray(apply_dihedral(hr, corrective))
        if corrective_offset_lr != [0, 0]:
            hr = np.ascontiguousarray(
                apply_translation(
                    hr,
                    (
                        int(corrective_offset_lr[0]) * SCALE,
                        int(corrective_offset_lr[1]) * SCALE,
                    ),
                )
            )
        out["hr"][target] = hr
        index_rows.append(
            {k: v for k, v in entry.items() if k != "source_store_index"} | {"pair_id": target}
        )
    pq.write_table(pa.Table.from_pylist(index_rows), output / "index.parquet")

    maps_present = sorted({e["map"] for e in entries})
    summary = {
        "schema": PAIRSET_SCHEMA,
        "scale": SCALE,
        "maps": maps_present,
        "corrective_transform": corrective,
        "corrective_offset_lr": [int(corrective_offset_lr[0]), int(corrective_offset_lr[1])],
        "pairing_mode": pairing_mode,
        "authored_object_policy": "may_contain_client_baked_objects",
        "synthetic_object_policy": "terrain_only_no_objects",
        "visual_review_report": str(visual_review_path) if visual_review_path else "",
        "total_pairs": len(entries),
        "splits": {
            "train": sum(e["split"] == "train" for e in entries),
            "val": sum(e["split"] == "val" for e in entries),
        },
        "per_map": {name: sum(e["map"] == name for e in entries) for name in maps_present},
        "excluded": {
            "missing_authored": excluded_missing_authored,
            "missing_detail_render": excluded_missing_detail,
        },
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the Spec 113 SR pair set (authored LR, detail HR)")
    ap.add_argument("--store", action="append", required=True, type=Path, dest="stores")
    ap.add_argument("--alignment", required=True, type=Path, help="US1 raw pixel-alignment report JSON")
    ap.add_argument(
        "--terrain-only-cross-domain",
        action="store_true",
        help="pair same-tile authored LR with terrain-only HR; objects are intentionally removed",
    )
    ap.add_argument(
        "--visual-review",
        type=Path,
        help="v50-store-visual-review-v1 report required by --terrain-only-cross-domain",
    )
    ap.add_argument("--val-fraction", type=float, default=0.15)
    ap.add_argument("--output", required=True, type=Path)
    args = ap.parse_args()
    try:
        summary = build_sr_pairset(
            stores=args.stores, output=args.output,
            alignment_report=args.alignment, val_fraction=args.val_fraction,
            terrain_only_cross_domain=args.terrain_only_cross_domain,
            visual_review_report=args.visual_review,
        )
    except PairSetBuildError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"pairset: {summary['total_pairs']} pairs (train={summary['splits']['train']} val={summary['splits']['val']}) "
        f"transform={summary['corrective_transform']} excluded={summary['excluded']} -> {args.output}"
    )
    return 0
