"""Build a trainer-consumable v50 curriculum store from Spec 109 complete per-map stores.

Both canonical v50 trainers (``wdl_prior_train``/``terrain_refiner_train``) gate their input on
``require_store_release``, which demands ``schema == "v50-mixed-curriculum-v1"`` plus a ``split``
column in ``index.parquet`` -- the Spec 109 clean-room builder emits ``v50-complete-store-v1``
per-map stores, so neither trainer can consume them directly. This module is the bridge: it selects
the ``keep`` rows named by each map's reviewed strict curation manifest (the object-free profile --
correct for height-supervision training, where an object occludes the ground it sits on), assigns a
whole-map holdout split, and writes one training store with full source lineage.

Selection is manifest-driven on purpose: this builder never re-derives its own quality policy. The
strict curation manifest is the reviewed artifact that already dropped blank-minimap,
object-contaminated, and height/normal-mismatched tiles (Spec 109 Phase 9); rows it kept are copied
bit-for-bit, rows it dropped never enter the store. Because the split is assigned per whole map and
real 0.5.3 tiles have no time/color variants, every ``source_group_id`` lands entirely in one
partition, satisfying ``validate_source_group_split`` by construction.

The schema name ``v50-mixed-curriculum-v1`` is historical (Spec 108 mixed real/synthetic); an
all-real curriculum is still that schema -- it is what the release contract requires of any
trainer-facing store, and ``source_kind`` in the index records honestly that every row here is
``real_053``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from harvester.v50.contracts import MIXED_STORE_SCHEMA, MODEL_FAMILY, validate_release

# Superset of what both trainers read (wdl_prior_train: minimap_rgb/height_257;
# terrain_refiner_train additionally object_precise_mask); matches the proven Spec 108
# builder's field list so downstream inference/visualization tooling sees the same layout.
CURRICULUM_FIELDS = (
    "minimap_rgb",
    "height_257",
    "normal_xyz",
    "liquid_mask",
    "liquid_height",
    "object_precise_mask",
    "alpha_256",
)


class CurriculumBuildError(ValueError):
    """Raised when the requested curriculum cannot be built as declared."""


def _load_keep_rows(manifest_path: Path) -> list[dict]:
    path = manifest_path / "curation_manifest.parquet" if manifest_path.is_dir() else manifest_path
    rows = pq.read_table(path).to_pylist()
    kept = [row for row in rows if bool(row.get("keep", False))]
    if not kept:
        raise CurriculumBuildError(f"curation manifest kept zero rows: {path}")
    return kept


def _stratified_split(selected: list[dict], val_fraction: float) -> None:
    """Assign a deterministic within-map holdout: each map contributes ``val_fraction`` of its
    kept rows (at least one) to val, ordered by a stable hash of ``source_group_id``.

    This is the established WDL-prior evaluation regime (Spec 108's group split): val tiles come
    from the *same* maps as train. A whole-map holdout is NOT interchangeable with it -- the WDL
    target is absolute elevation on a global scale, and real 0.5.3 maps sit at very different
    absolute altitudes (measured on the v50.1 corpus: PVPZone02 mean +381 vs Azeroth -150), so a
    fully held-out map mostly measures an altitude offset the model has never seen and val loss
    *worsens* as training progresses. Use ``val_map`` only for deliberate cross-map
    generalization experiments, knowing that is what it measures.
    """
    by_map: dict[str, list[dict]] = {}
    for row in selected:
        by_map.setdefault(row["map"], []).append(row)
    for rows in by_map.values():
        rows.sort(key=lambda row: hashlib.sha256(row["source_group_id"].encode()).hexdigest())
        val_count = max(1, round(len(rows) * val_fraction))
        for position, row in enumerate(rows):
            row["split"] = "val" if position < val_count else "train"


def build_training_curriculum(
    *,
    stores: list[Path],
    curation_manifests: list[Path],
    output: Path,
    release: str,
    val_map: str | None = None,
    val_fraction: float | None = None,
) -> dict:
    """Write the curriculum store and return its summary dict. Exactly one of ``val_map`` (whole
    held-out map; cross-map generalization regime) or ``val_fraction`` (within-map stratified
    holdout; the standard regime) selects the split."""
    import zarr

    release = validate_release(release)
    if (val_map is None) == (val_fraction is None):
        raise CurriculumBuildError("pass exactly one of val_map or val_fraction")
    if val_fraction is not None and not 0.0 < val_fraction < 1.0:
        raise CurriculumBuildError(f"val_fraction must be in (0, 1), got {val_fraction}")
    if len(stores) != len(curation_manifests):
        raise CurriculumBuildError(
            f"got {len(stores)} stores but {len(curation_manifests)} curation manifests; "
            "pass one --curation-manifest per --store, in the same order"
        )
    if output.exists():
        raise CurriculumBuildError(f"refusing to overwrite existing output: {output}")

    selected: list[dict] = []
    source_groups = []
    for store_path, manifest_path in zip(stores, curation_manifests):
        group = zarr.open_group(str(store_path), mode="r")
        if str(group.attrs.get("release", "")) != release:
            raise CurriculumBuildError(
                f"source store release {group.attrs.get('release')!r} != requested {release!r}: {store_path}"
            )
        source_groups.append(group)
        for row in _load_keep_rows(manifest_path):
            selected.append(
                {
                    "build": str(row["build"]),
                    "map": str(row["map"]),
                    "tile_x": int(row["tile_x"]),
                    "tile_y": int(row["tile_y"]),
                    "source_tile_id": int(row["tile_id"]),
                    "source_store_index": len(source_groups) - 1,
                    "source_store": str(store_path.resolve()),
                    "source_curation_manifest": str(manifest_path.resolve()),
                    "source_kind": "real_053",
                    "source_group_id": f"real:{row['build']}:{row['map']}:{int(row['tile_id'])}",
                    "height_regime": str(row.get("height_regime", "")),
                }
            )

    maps_present = sorted({row["map"] for row in selected})
    if val_map is not None:
        if val_map not in maps_present:
            raise CurriculumBuildError(f"--val-map {val_map!r} matched no kept rows; maps present: {maps_present}")
        for row in selected:
            row["split"] = "val" if row["map"] == val_map else "train"
        split_mode = f"whole_map_holdout:{val_map}"
    else:
        _stratified_split(selected, float(val_fraction))
        split_mode = f"within_map_stratified:{val_fraction}"
    train_count = sum(row["split"] == "train" for row in selected)
    val_count = len(selected) - train_count
    if train_count == 0:
        raise CurriculumBuildError("every kept row landed in val; holdout map cannot be the whole corpus")

    reference = {}
    for field in CURRICULUM_FIELDS:
        for group in source_groups:
            if field in group:
                reference[field] = group[field]
                break
    for required in ("minimap_rgb", "height_257"):
        if required not in reference:
            raise CurriculumBuildError(f"no source store carries required trainer signal {required!r}")

    out = zarr.open_group(str(output), mode="w")
    out.attrs.update(
        {
            "schema": MIXED_STORE_SCHEMA,
            "model_family": MODEL_FAMILY,
            "release": release,
            "curriculum_kind": "all_real_strict_curated",
            "split_mode": split_mode,
            "source_stores": [str(path.resolve()) for path in stores],
            "source_curation_manifests": [str(path.resolve()) for path in curation_manifests],
        }
    )
    for field, array in reference.items():
        out.create_array(field, shape=(len(selected), *array.shape[1:]), dtype=array.dtype, chunks=(1, *array.shape[1:]))

    index_rows = []
    for target_row, row in enumerate(selected):
        source = source_groups[row["source_store_index"]]
        for field, target in reference.items():
            if field in source:
                out[field][target_row] = source[field][row["source_tile_id"]]
            else:
                out[field][target_row] = np.zeros(target.shape[1:], dtype=target.dtype)
        index_rows.append(
            {key: value for key, value in row.items() if key != "source_store_index"}
            | {"tile_id": target_row, "model_family": MODEL_FAMILY, "release": release}
        )
    pq.write_table(pa.Table.from_pylist(index_rows), output / "index.parquet")

    summary = {
        "schema": MIXED_STORE_SCHEMA,
        "model_family": MODEL_FAMILY,
        "release": release,
        "curriculum_kind": "all_real_strict_curated",
        "total_rows": len(selected),
        "splits": {"train": train_count, "val": val_count},
        "split_mode": split_mode,
        "val_rows_per_map": {name: sum(row["map"] == name and row["split"] == "val" for row in selected) for name in maps_present},
        "maps": maps_present,
        "rows_per_map": {name: sum(row["map"] == name for row in selected) for name in maps_present},
        "sources": [
            {"store": str(store.resolve()), "curation_manifest": str(manifest.resolve())}
            for store, manifest in zip(stores, curation_manifests)
        ],
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    ap = argparse.ArgumentParser(description="Build the trainer-facing v50 curriculum store from complete per-map stores")
    ap.add_argument("--store", action="append", required=True, type=Path, dest="stores",
                    help="a Spec 109 complete per-map store (repeatable)")
    ap.add_argument("--curation-manifest", action="append", required=True, type=Path, dest="curation_manifests",
                    help="the strict curation manifest (dir or parquet) for the store at the same position (repeatable)")
    ap.add_argument("--output", required=True, type=Path)
    split_group = ap.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--val-map", default=None,
                             help="whole map held out as validation -- cross-map generalization regime ONLY; "
                                  "the WDL target is absolute elevation, so an unseen map's altitude offset "
                                  "dominates this metric (see _stratified_split docstring)")
    split_group.add_argument("--val-fraction", type=float, default=None,
                             help="standard regime: deterministic within-map stratified holdout (e.g. 0.15)")
    ap.add_argument("--release", default="v50.1", type=validate_release)
    args = ap.parse_args()
    try:
        summary = build_training_curriculum(
            stores=args.stores,
            curation_manifests=args.curation_manifests,
            output=args.output,
            val_map=args.val_map,
            val_fraction=args.val_fraction,
            release=args.release,
        )
    except CurriculumBuildError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"[{summary['release']}] curriculum rows={summary['total_rows']} "
        f"train={summary['splits']['train']} val={summary['splits']['val']} ({summary['split_mode']}) "
        f"-> {args.output}"
    )
    return 0
