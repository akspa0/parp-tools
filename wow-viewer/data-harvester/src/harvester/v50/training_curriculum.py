"""Build a trainer-consumable v50 curriculum store from Spec 109 complete per-map stores.

Both canonical v50 trainers gate their input on ``require_store_release``, which demands
``schema == "v50-mixed-curriculum-v1"`` plus a ``split`` column in ``index.parquet`` -- the Spec 109
clean-room builder emits ``v50-complete-store-v1`` per-map stores, so neither trainer can consume
them directly. This module is the bridge: it selects the ``keep`` rows named by each map's reviewed
strict curation manifest (the object-free profile -- correct for height-supervision training, where
an object occludes the ground it sits on) and writes one trainer-facing store with full lineage.

Dual minimap sources (Spec 112, user-directed 2026-07-18): the per-map store carries both the
synthesized compositor minimap (``minimap_rgb``) and the authored client minimap
(``minimap_rgb_authored``). A model that decompiles real minimaps must see the authored image, but
synthetic imagery is valuable augmentation. So each kept tile emits UP TO TWO rows -- one per
available minimap source -- paired with the SAME height target and all the same auxiliary terrain
signals. The curriculum's ``minimap_rgb`` column is the per-row model input (synthetic or authored);
an ``minimap_source`` index column records which. Both rows of a tile share one ``source_group_id``,
and the split is assigned per group, so a tile's two rows can never straddle the train/val boundary
(leak safety, checked again by ``validate_source_group_split`` in the trainer).

Selection is manifest-driven on purpose: this builder never re-derives its own quality policy. The
strict curation manifest already dropped blank-minimap, object-contaminated, and
height/normal-mismatched tiles; rows it kept are copied bit-for-bit, rows it dropped never enter the
store. The schema name ``v50-mixed-curriculum-v1`` is historical (Spec 108 mixed real/synthetic); an
all-real dual-source curriculum is still that schema -- it is what the release contract requires of
any trainer-facing store.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from harvester.v50.contracts import MIXED_STORE_SCHEMA, MODEL_FAMILY, validate_release

# The curriculum's model-input minimap column. For each row it holds EITHER the synthesized or the
# authored image, per that row's ``minimap_source``; it is written from the source field named in
# ``_MINIMAP_SOURCE_FIELDS``.
MINIMAP_INPUT_FIELD = "minimap_rgb"
_MINIMAP_SOURCE_FIELDS = {"synthetic": "minimap_rgb", "authored": "minimap_rgb_authored"}

# Never copied as their own curriculum column: the two minimap source arrays (folded into per-row
# ``minimap_rgb``) and the synthetic 1024px upscaler target (a separate lane, synthetic-only, with
# no authored counterpart -- undefined for an authored row).
_EXCLUDED_FIELDS = frozenset({"minimap_rgb", "minimap_rgb_authored", "minimap_rgb_1024"})


class CurriculumBuildError(ValueError):
    """Raised when the requested curriculum cannot be built as declared."""


def _load_keep_rows(manifest_path: Path) -> list[dict]:
    path = manifest_path / "curation_manifest.parquet" if manifest_path.is_dir() else manifest_path
    rows = pq.read_table(path).to_pylist()
    kept = [row for row in rows if bool(row.get("keep", False))]
    if not kept:
        raise CurriculumBuildError(f"curation manifest kept zero rows: {path}")
    return kept


def _store_row_count(group) -> int:
    declared = int(group.attrs.get("row_count", 0))
    return declared if declared > 0 else int(group["height_257"].shape[0])


def _per_tile_fields(group) -> list[str]:
    """Per-tile terrain signals to copy identically for both rows of a tile: row-aligned arrays of
    rank >= 2, excluding the minimap family (handled polymorphically). Flat placement arrays are
    1-D or not row-aligned, so they are naturally excluded rather than copied without their data."""
    row_count = _store_row_count(group)
    fields = []
    for name in group.array_keys():
        if name in _EXCLUDED_FIELDS:
            continue
        array = group[name]
        if array.ndim >= 2 and array.shape[0] == row_count:
            fields.append(name)
    return fields


def _minimap_usable(group, field: str, tile_id: int, min_rgb_std: float) -> bool:
    """A minimap source is usable for a training row when it is present, non-empty, and not a
    near-uniform blank (RGB std >= ``min_rgb_std``, matching the curation blank-minimap check).

    This is applied PER SOURCE (Spec 112, user-directed 2026-07-18): a tile whose synthesized
    minimap failed/blanked but whose authored client minimap is valid still contributes an authored
    training row. Synthetic-centric curation would have discarded ~275 real authored tiles on the
    0.5.3.3368 corpus purely because our renderer couldn't decode their textures."""
    if field not in group:
        return False
    array = np.asarray(group[field][tile_id], dtype=np.float32)
    if not array.any():
        return False
    return float(array.std()) >= min_rgb_std


def _assign_group_split(selected: list[dict], *, val_map: str | None, val_fraction: float | None) -> str:
    """Assign each row a ``split``. Splits are decided per ``source_group_id`` (a whole tile, i.e.
    both its minimap-source rows together), so a tile's rows never cross the train/val boundary.

    ``val_map``: hold out one whole map (cross-map generalization regime). ``val_fraction``: within
    each map, hold out that fraction of TILES (not rows), the standard regime -- absolute altitude
    is not a fair cross-map target, see the Spec 112 within-map ruling."""
    if val_map is not None:
        for row in selected:
            row["split"] = "val" if row["map"] == val_map else "train"
        return f"whole_map_holdout:{val_map}"

    by_map_group: dict[str, dict[str, list[dict]]] = defaultdict(lambda: defaultdict(list))
    for row in selected:
        by_map_group[row["map"]][row["source_group_id"]].append(row)
    for groups in by_map_group.values():
        ordered = sorted(groups, key=lambda gid: hashlib.sha256(gid.encode()).hexdigest())
        val_count = max(1, round(len(ordered) * float(val_fraction)))
        val_groups = set(ordered[:val_count])
        for gid, rows in groups.items():
            split = "val" if gid in val_groups else "train"
            for row in rows:
                row["split"] = split
    return f"within_map_stratified:{val_fraction}"


def _surviving_height_levels(group, tile_id: int) -> int:
    """Distinct height values for a tile — the Spec 134 curation gate. A tile whose surviving shape
    is compressed to ≤ ``max_height_levels`` distinct heights teaches the model a broken relationship
    (measured: 4 Azeroth tiles hold exactly 2 levels across a 516-unit range); a compressed-RANGE
    tile with full level count is real terrain worth keeping."""
    if "height_257" not in group:
        return 0
    height = np.asarray(group["height_257"][tile_id], dtype=np.float32)
    return int(np.unique(height).size)


def _height_levels_acceptable(group, tile_id: int,
                              min_levels: int | None, max_levels: int | None) -> bool:
    """True when a tile's surviving height levels are within the allowed range (or no range set)."""
    if min_levels is None and max_levels is None:
        return True
    levels = _surviving_height_levels(group, tile_id)
    if min_levels is not None and levels < min_levels:
        return False
    if max_levels is not None and levels > max_levels:
        return False
    return True


def build_training_curriculum(
    *,
    stores: list[Path],
    curation_manifests: list[Path],
    output: Path,
    release: str,
    val_map: str | None = None,
    val_fraction: float | None = None,
    min_rgb_std: float = 1.0,
    min_height_levels: int | None = None,
    max_height_levels: int | None = None,
) -> dict:
    """Write the curriculum store and return its summary dict. Exactly one of ``val_map`` (whole
    held-out map; cross-map generalization regime) or ``val_fraction`` (within-map stratified
    holdout; the standard regime) selects the split. ``min_rgb_std`` is the per-source blank-minimap
    threshold (matches the curation default); pass a terrain-quality-only curation manifest so the
    per-source blank check here is the sole minimap gate.

    Spec 134 (US2) curation gating: ``min_height_levels``/``max_height_levels`` filter kept tiles
    by their distinct height count (``surviving_height_levels``). The default recommendation is to
    exclude tiles with ≤64 levels (``max_height_levels=64``) — those teach a texture edge as a
    vertical wall — while admitting compressed-rich tiles (high level count, low range). When no
    gate is passed the legacy behavior is unchanged (no level filtering)."""
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

    selected: list[dict] = []  # one entry PER ROW (a tile can yield up to two)
    source_groups = []
    for store_path, manifest_path in zip(stores, curation_manifests):
        group = zarr.open_group(str(store_path), mode="r")
        if str(group.attrs.get("release", "")) != release:
            raise CurriculumBuildError(
                f"source store release {group.attrs.get('release')!r} != requested {release!r}: {store_path}"
            )
        source_index = len(source_groups)
        source_groups.append(group)
        kept_rows = _load_keep_rows(manifest_path)
        if min_height_levels is not None or max_height_levels is not None:
            before = len(kept_rows)
            kept_rows = [
                row for row in kept_rows
                if _height_levels_acceptable(group, int(row["tile_id"]),
                                             min_height_levels, max_height_levels)
            ]
            dropped = before - len(kept_rows)
            if dropped:
                print(f"  height-level gating: dropped {dropped}/{before} kept tiles "
                      f"({min_height_levels}<=levels<={max_height_levels})", flush=True)
        for row in kept_rows:
            tile_id = int(row["tile_id"])
            base = {
                "build": str(row["build"]),
                "map": str(row["map"]),
                "tile_x": int(row["tile_x"]),
                "tile_y": int(row["tile_y"]),
                "source_tile_id": tile_id,
                "source_store_index": source_index,
                "source_store": str(store_path.resolve()),
                "source_curation_manifest": str(manifest_path.resolve()),
                "source_kind": "real_053",
                "source_group_id": f"real:{row['build']}:{row['map']}:{tile_id}",
                "height_regime": str(row.get("height_regime", "")),
            }
            for source_name, source_field in _MINIMAP_SOURCE_FIELDS.items():
                if _minimap_usable(group, source_field, tile_id, min_rgb_std):
                    selected.append({**base, "minimap_source": source_name, "minimap_field": source_field})

    if not selected:
        raise CurriculumBuildError("no kept tile carries any minimap source; nothing to train on")

    maps_present = sorted({row["map"] for row in selected})
    if val_map is not None and val_map not in maps_present:
        raise CurriculumBuildError(f"--val-map {val_map!r} matched no kept rows; maps present: {maps_present}")
    split_mode = _assign_group_split(selected, val_map=val_map, val_fraction=val_fraction)
    train_count = sum(row["split"] == "train" for row in selected)
    val_count = len(selected) - train_count
    if train_count == 0:
        raise CurriculumBuildError("every kept row landed in val; holdout map cannot be the whole corpus")

    # Copied per-tile fields: union across source stores, minus the minimap family.
    copied_field_names: list[str] = []
    seen: set[str] = set()
    for group in source_groups:
        for name in _per_tile_fields(group):
            if name not in seen:
                seen.add(name)
                copied_field_names.append(name)
    if "height_257" not in copied_field_names:
        raise CurriculumBuildError("no source store carries the required height_257 signal")

    reference = {}
    for name in copied_field_names:
        for group in source_groups:
            if name in group:
                reference[name] = group[name]
                break
    minimap_reference = None
    for group in source_groups:
        if "minimap_rgb" in group:
            minimap_reference = group["minimap_rgb"]
            break
    if minimap_reference is None:
        raise CurriculumBuildError("no source store carries minimap_rgb to shape the input column")

    out = zarr.open_group(str(output), mode="w")
    out.attrs.update(
        {
            "schema": MIXED_STORE_SCHEMA,
            "model_family": MODEL_FAMILY,
            "release": release,
            "curriculum_kind": "all_real_dual_minimap",
            "split_mode": split_mode,
            "min_rgb_std": min_rgb_std,
            "minimap_sources": sorted({row["minimap_source"] for row in selected}),
            "source_stores": [str(path.resolve()) for path in stores],
            "source_curation_manifests": [str(path.resolve()) for path in curation_manifests],
        }
    )
    out.create_array(
        MINIMAP_INPUT_FIELD,
        shape=(len(selected), *minimap_reference.shape[1:]),
        dtype=minimap_reference.dtype,
        chunks=(1, *minimap_reference.shape[1:]),
    )
    for name in copied_field_names:
        array = reference[name]
        out.create_array(name, shape=(len(selected), *array.shape[1:]), dtype=array.dtype, chunks=(1, *array.shape[1:]))

    index_rows = []
    for target_row, row in enumerate(selected):
        source = source_groups[row["source_store_index"]]
        tile_id = row["source_tile_id"]
        out[MINIMAP_INPUT_FIELD][target_row] = source[row["minimap_field"]][tile_id]
        for name in copied_field_names:
            if name in source:
                out[name][target_row] = source[name][tile_id]
            else:
                out[name][target_row] = np.zeros(reference[name].shape[1:], dtype=reference[name].dtype)
        index_rows.append(
            {key: value for key, value in row.items() if key not in ("source_store_index", "minimap_field")}
            | {"tile_id": target_row, "model_family": MODEL_FAMILY, "release": release}
        )
    pq.write_table(pa.Table.from_pylist(index_rows), output / "index.parquet")

    def _count(pred) -> int:
        return sum(1 for row in selected if pred(row))

    summary = {
        "schema": MIXED_STORE_SCHEMA,
        "model_family": MODEL_FAMILY,
        "release": release,
        "curriculum_kind": "all_real_dual_minimap",
        "total_rows": len(selected),
        "splits": {"train": train_count, "val": val_count},
        "split_mode": split_mode,
        "minimap_source_counts": {
            name: _count(lambda r, n=name: r["minimap_source"] == n) for name in sorted({r["minimap_source"] for r in selected})
        },
        "val_rows_per_map": {name: _count(lambda r, n=name: r["map"] == n and r["split"] == "val") for name in maps_present},
        "maps": maps_present,
        "rows_per_map": {name: _count(lambda r, n=name: r["map"] == n) for name in maps_present},
        "copied_fields": copied_field_names,
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
                             help="whole map held out as validation -- cross-map generalization regime ONLY")
    split_group.add_argument("--val-fraction", type=float, default=None,
                             help="standard regime: deterministic within-map stratified holdout of TILES (e.g. 0.15)")
    ap.add_argument("--min-rgb-std", type=float, default=1.0,
                    help="per-source blank-minimap threshold (default 1.0, matches curation); pass a "
                         "terrain-quality-only curation manifest so this is the sole minimap gate")
    ap.add_argument("--min-height-levels", type=int, default=None,
                    help="Spec 134 curation gate: exclude tiles with fewer than this many distinct "
                         "height values. Default: no min (legacy behavior).")
    ap.add_argument("--max-height-levels", type=int, default=None,
                    help="Spec 134 curation gate: exclude tiles with more than this many distinct "
                         "height values. Recommended: 64 (excludes ≤64-level tiles that teach broken "
                         "relationships). Default: no max (legacy behavior).")
    ap.add_argument("--release", default="v50.1", type=validate_release)
    args = ap.parse_args()
    try:
        summary = build_training_curriculum(
            stores=args.stores,
            curation_manifests=args.curation_manifests,
            output=args.output,
            val_map=args.val_map,
            val_fraction=args.val_fraction,
            min_rgb_std=args.min_rgb_std,
            release=args.release,
            min_height_levels=args.min_height_levels,
            max_height_levels=args.max_height_levels,
        )
    except CurriculumBuildError as exc:
        raise SystemExit(str(exc)) from exc
    print(
        f"[{summary['release']}] curriculum rows={summary['total_rows']} "
        f"({summary['minimap_source_counts']}) "
        f"train={summary['splits']['train']} val={summary['splits']['val']} ({summary['split_mode']}) "
        f"-> {args.output}"
    )
    return 0
