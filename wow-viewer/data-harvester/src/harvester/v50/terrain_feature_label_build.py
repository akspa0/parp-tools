"""Spec 115: build the derived terrain-feature label store for a v50 curriculum.

Reads the curriculum's real texture-family ground truth, resolves each pixel's dominant texture
layer to a canonical family (see ``terrain_feature_labels``), and persists a row-aligned label store
beside the curriculum. Row order is preserved exactly so the curriculum's frozen train/val split
applies to the labels unchanged.

Dry run by default: prints the full coverage/exclusion report and writes nothing without ``--write``.
Rows whose tile has no texture-name dump entry, or no usable MTEX table, are excluded wholesale and
counted -- never emitted as an all-``unknown`` row (spec FR-004).
"""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.v50.model_stage_contract import sha256_file
from harvester.v50.terrain_feature_labels import (
    CLASS_COUNT,
    DOMINANT_ALPHA_THRESHOLD,
    FAMILY_NAMES,
    TAXONOMY_REVISION,
    TILE_PIXELS,
    TerrainFeatureLabelError,
    derive_row_labels,
    load_texture_name_dump,
    rule_set_sha256,
)

LABEL_STORE_SCHEMA = "v115-terrain-feature-labels-v1"


def build_label_store(
    *,
    store: Path,
    dumps: list[Path],
    output: Path,
    write: bool,
) -> dict:
    """Derive labels for every curriculum row; return the coverage report. Writes only if ``write``."""
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(store), mode="r")
    for required in ("mcly_texture_ids", "minimap_rgb"):
        if required not in group:
            raise TerrainFeatureLabelError(f"curriculum store is missing {required!r}: {store}")

    index_path = store / "index.parquet"
    if not index_path.exists():
        raise TerrainFeatureLabelError(f"curriculum store has no index.parquet: {store}")
    index_rows = pq.read_table(index_path).to_pylist()

    row_count = int(group["mcly_texture_ids"].shape[0])
    if row_count != len(index_rows):
        raise TerrainFeatureLabelError(
            f"index rows ({len(index_rows)}) != mcly_texture_ids rows ({row_count})"
        )

    names_by_tile = load_texture_name_dump(dumps)
    has_alpha = "alpha_256" in group
    has_layer_mask = "mcly_layer_mask" in group

    labels_all = np.zeros((row_count, TILE_PIXELS, TILE_PIXELS), dtype=np.uint8)
    valid_all = np.zeros((row_count, TILE_PIXELS, TILE_PIXELS), dtype=bool)
    included = np.zeros(row_count, dtype=bool)

    family_pixels = dict.fromkeys(FAMILY_NAMES, 0)
    invalid_pixels = 0
    excluded: dict[str, int] = {"no_texture_name_dump_entry": 0, "empty_mtex_table": 0}
    rows_with_road = 0

    for row in range(row_count):
        meta = index_rows[row]
        key = (str(meta.get("map")), int(meta.get("tile_x", -1)), int(meta.get("tile_y", -1)))
        texture_names = names_by_tile.get(key)
        if not texture_names:
            excluded["no_texture_name_dump_entry"] += 1
            continue

        labels, valid = derive_row_labels(
            texture_ids=np.asarray(group["mcly_texture_ids"][row]),
            texture_names=texture_names,
            alpha_256=np.asarray(group["alpha_256"][row]) if has_alpha else None,
            layer_mask=np.asarray(group["mcly_layer_mask"][row]) if has_layer_mask else None,
        )
        if not valid.any():
            excluded["empty_mtex_table"] += 1
            continue

        labels_all[row] = labels
        valid_all[row] = valid
        included[row] = True
        for family in range(CLASS_COUNT):
            family_pixels[FAMILY_NAMES[family]] += int(np.count_nonzero((labels == family) & valid))
        invalid_pixels += int(np.count_nonzero(~valid))
        if np.any((labels == 2) & valid):  # ROAD
            rows_with_road += 1

    rows_labelled = int(included.sum())
    if rows_labelled == 0:
        raise TerrainFeatureLabelError("no curriculum row produced usable terrain-feature labels")

    labelled_pixels = sum(family_pixels.values())
    report = {
        "schema": LABEL_STORE_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "curriculum": {"path": str(store.resolve()), "sha256": sha256_file(index_path)},
        "texture_name_dumps": [
            {"path": str(Path(d).resolve()), "sha256": sha256_file(Path(d))} for d in dumps
        ],
        "taxonomy_revision": TAXONOMY_REVISION,
        "rule_set_sha256": rule_set_sha256(),
        "dominant_alpha_threshold": DOMINANT_ALPHA_THRESHOLD,
        "row_count": row_count,
        "rows_labelled": rows_labelled,
        "rows_excluded": int(row_count - rows_labelled),
        "excluded_counts": excluded,
        "rows_with_any_road": rows_with_road,
        "family_pixels": family_pixels,
        "family_fraction": {
            name: (count / labelled_pixels if labelled_pixels else 0.0)
            for name, count in family_pixels.items()
        },
        "invalid_pixels": invalid_pixels,
        "output": str(output.resolve()),
        "written": bool(write),
    }

    # Reconciliation: every labelled pixel lands in exactly one family, and excluded rows account
    # for the difference. A mismatch means the derivation dropped or double-counted pixels.
    expected = rows_labelled * TILE_PIXELS * TILE_PIXELS
    if labelled_pixels + invalid_pixels != expected:
        raise TerrainFeatureLabelError(
            f"pixel reconciliation failed: {labelled_pixels} labelled + {invalid_pixels} invalid "
            f"!= {expected} expected"
        )
    if sum(excluded.values()) != report["rows_excluded"]:
        raise TerrainFeatureLabelError("row reconciliation failed: excluded reasons do not sum")

    if write:
        if output.exists() and any(output.iterdir()):
            raise TerrainFeatureLabelError(f"refusing to overwrite non-empty output: {output}")
        output.mkdir(parents=True, exist_ok=True)
        out_group = zarr.open_group(str(output), mode="w")
        out_group.create_array(
            "labels", shape=labels_all.shape, chunks=(1, TILE_PIXELS, TILE_PIXELS), dtype="uint8"
        )[:] = labels_all
        out_group.create_array(
            "valid", shape=valid_all.shape, chunks=(1, TILE_PIXELS, TILE_PIXELS), dtype="bool"
        )[:] = valid_all
        out_group.create_array("included", shape=included.shape, chunks=(row_count,), dtype="bool")[
            :
        ] = included
        out_group.attrs.update(
            {
                "schema": LABEL_STORE_SCHEMA,
                "taxonomy_revision": TAXONOMY_REVISION,
                "rule_set_sha256": rule_set_sha256(),
                "dominant_alpha_threshold": DOMINANT_ALPHA_THRESHOLD,
                "family_names": list(FAMILY_NAMES),
                "curriculum": report["curriculum"],
                "texture_name_dumps": report["texture_name_dumps"],
                "rows_labelled": rows_labelled,
                "rows_excluded": report["rows_excluded"],
            }
        )
        (output / "label_build_report.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )

    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 115: derive terrain-feature labels for a v50 curriculum (dry run by default)"
    )
    ap.add_argument("--store", required=True, type=Path, help="curriculum Zarr store")
    ap.add_argument(
        "--texture-names", required=True, type=Path, action="append", dest="dumps",
        help="dump-texture-names JSON (repeat once per map)",
    )
    ap.add_argument("--output", required=True, type=Path, help="derived label store path")
    ap.add_argument("--write", action="store_true", help="persist the label store; default prints only")
    args = ap.parse_args(argv)

    try:
        report = build_label_store(
            store=args.store, dumps=args.dumps, output=args.output, write=args.write
        )
    except TerrainFeatureLabelError as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(report, indent=2), flush=True)
    if not args.write:
        print("DRY RUN ONLY: add --write to persist the label store.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
