"""Spec 121 B-reframe: within-map WDL completion split (v121-within-map-split-v1).

The Spec 116 8-neighbour isolation split proved that RGB→WDL does NOT transfer across regions
(research.md R-1). The honest reframe: train on WDL-covered tiles of a map, predict missing WDL
tiles of the SAME map — v7's actual deployment constraint. This split assigns tiles per map:
a random fraction of each map's tiles become held-out; the rest train. Adjacent tiles are
allowed in both splits (deployment reality for completion). A configurable buffer ring can
exclude tiles adjacent to held-out tiles for a more conservative evaluation.

Schema ``v121-within-map-split-v1`` — intentionally NOT ``v50-held-out-split-v1`` (which
hard-requires ``adjacency_rule="8-neighbour"``). The ``apply_within_map_split`` function mirrors
``apply_held_out_split``'s (map, tile_x, tile_y) lookup semantics exactly so the trainer
dispatch is a one-line import swap.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from harvester.v50.model_stage_contract import sha256_file

WITHIN_MAP_SPLIT_SCHEMA = "v121-within-map-split-v1"
SPLIT_TRAIN = "train"
SPLIT_HELD_OUT = "held_out"
SPLIT_EXCLUDED = "excluded"


class WithinMapSplitError(ValueError):
    """Raised when a within-map split violates its contract."""


def build_within_map_split(
    index_rows: list[dict],
    *,
    held_out_fraction: float = 0.15,
    buffer_rings: int = 0,
    seed: int = 121,
) -> dict:
    """Build a per-map random split. Returns the same shape as ``build_held_out_split``.

    For each map present in the index, shuffle its tiles (by (tile_x, tile_y) deduped),
    assign ``held_out_fraction`` to held-out, optionally buffer adjacent tiles as excluded,
    then assign every row of each tile the same label (so authored+synthetic views of the
    same tile never leak across splits — Spec 114 acceptance 2).
    """
    if not 0.0 < held_out_fraction < 1.0:
        raise WithinMapSplitError(f"held_out_fraction must be in (0, 1), got {held_out_fraction}")
    if buffer_rings < 0:
        raise WithinMapSplitError(f"buffer_rings must be >= 0, got {buffer_rings}")

    # Group unique (map, tile_x, tile_y) per map.
    per_map: dict[str, set[tuple[int, int]]] = {}
    for row in index_rows:
        m = str(row.get("map", "?"))
        per_map.setdefault(m, set()).add((int(row.get("tile_x", -1)), int(row.get("tile_y", -1))))

    rng = np.random.default_rng(seed)
    assignments: list[dict] = []
    total_train = total_held = total_excluded = 0
    per_map_counts: dict[str, dict[str, int]] = {}
    overlap_count = 0

    for map_name, coords in sorted(per_map.items()):
        coord_list = sorted(coords)
        rng.shuffle(coord_list)
        n_held = max(1, round(len(coord_list) * held_out_fraction))
        held_set = set(coord_list[:n_held])
        # Buffer: tiles adjacent to any held-out tile.
        if buffer_rings > 0:
            buf_set: set[tuple[int, int]] = set()
            for hx, hy in held_set:
                for dx in range(-buffer_rings, buffer_rings + 1):
                    for dy in range(-buffer_rings, buffer_rings + 1):
                        if dx == 0 and dy == 0:
                            continue
                        buf_set.add((hx + dx, hy + dy))
            buf_set -= held_set
        else:
            buf_set = set()
        train_set = set(coord_list) - held_set - buf_set

        # Verify no tile appears in both held and train.
        if held_set & train_set:
            overlap_count += len(held_set & train_set)

        for coord in coord_list:
            if coord in held_set:
                split = SPLIT_HELD_OUT
            elif coord in train_set:
                split = SPLIT_TRAIN
            else:
                split = SPLIT_EXCLUDED
            assignments.append({
                "tile_row": -1,  # not row-specific; resolved at apply time
                "map": map_name,
                "tile_x": int(coord[0]),
                "tile_y": int(coord[1]),
                "split": split,
            })

        mc = {"train": len(train_set), "held_out": len(held_set), "excluded": len(buf_set)}
        per_map_counts[map_name] = mc
        total_train += mc["train"]
        total_held += mc["held_out"]
        total_excluded += mc["excluded"]

    if total_held == 0:
        raise WithinMapSplitError("no held-out tiles after split construction")
    if total_train == 0:
        raise WithinMapSplitError("no training tiles after split construction")

    return {
        "assignments": assignments,
        "split_counts": {"train": total_train, "held_out": total_held},
        "excluded_count": total_excluded,
        "verified_overlap_count": overlap_count,
        "buffer_rings": buffer_rings,
        "held_out_fraction": held_out_fraction,
        "seed": seed,
        "per_map_counts": per_map_counts,
    }


def write_within_map_split(
    *,
    store: Path,
    output: Path,
    split: dict,
    build_id: str = "",
) -> dict:
    """Persist ``split`` as ``split.parquet`` + ``split.json`` (``v121-within-map-split-v1``)."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    output.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pylist(split["assignments"]), output / "split.parquet")

    manifest = {
        "schema": WITHIN_MAP_SPLIT_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "build_id": build_id,
        "store": {"path": str(store.resolve()), "sha256": sha256_file(store / "index.parquet")},
        "split_rule": "within-map-random",
        "leakage_rule": "no (map,tile_x,tile_y) appears in both train and held_out; all views of a tile share one label",
        "buffer_rings": split["buffer_rings"],
        "held_out_fraction": split["held_out_fraction"],
        "seed": split["seed"],
        "split_counts": split["split_counts"],
        "per_map_counts": split["per_map_counts"],
        "excluded_count": split["excluded_count"],
        "verified_overlap_count": split["verified_overlap_count"],
        "absolute_comparison_to_region_isolated_runs_invalid": True,
    }
    validate_within_map_split(manifest)
    (output / "split.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def validate_within_map_split(doc: Any) -> None:
    """Validate a ``v121-within-map-split-v1`` manifest."""
    path = "$"
    if not isinstance(doc, dict):
        raise WithinMapSplitError(f"{path}: must be a dict, got {type(doc).__name__}")
    required = {
        "schema", "created_utc", "build_id", "store", "split_rule", "leakage_rule",
        "buffer_rings", "held_out_fraction", "seed", "split_counts", "per_map_counts",
        "verified_overlap_count", "absolute_comparison_to_region_isolated_runs_invalid",
    }
    missing = required - set(doc)
    if missing:
        raise WithinMapSplitError(f"{path}: missing required keys {sorted(missing)}")
    if doc["schema"] != WITHIN_MAP_SPLIT_SCHEMA:
        raise WithinMapSplitError(
            f"{path}.schema: expected {WITHIN_MAP_SPLIT_SCHEMA!r}, got {doc['schema']!r}"
        )
    if doc["verified_overlap_count"] != 0:
        raise WithinMapSplitError(
            f"{path}.verified_overlap_count: must be 0, got {doc['verified_overlap_count']}"
        )
    counts = doc["split_counts"]
    if not isinstance(counts, dict) or "train" not in counts or "held_out" not in counts:
        raise WithinMapSplitError(f"{path}.split_counts: must have 'train' and 'held_out'")
    if int(counts["train"]) < 0 or int(counts["held_out"]) < 0:
        raise WithinMapSplitError(f"{path}.split_counts: counts must be non-negative")


def load_within_map_split(split_dir: Path) -> tuple[dict, list[dict]]:
    """Read a persisted within-map split: (manifest, assignment rows)."""
    import pyarrow.parquet as pq

    manifest = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))
    validate_within_map_split(manifest)
    rows = pq.read_table(split_dir / "split.parquet").to_pylist()
    return manifest, rows


def apply_within_map_split(
    *,
    index_rows: list[dict],
    selected_rows: list[int],
    split_dir: Path,
) -> tuple[list[int], list[int], dict]:
    """Partition ``selected_rows`` by a within-map split. Mirrors ``apply_held_out_split``
    semantics exactly: (map, tile_x, tile_y) lookup; held_out→val; train→train; missing→excluded.
    """
    split_manifest, split_rows = load_within_map_split(split_dir)
    if split_manifest["verified_overlap_count"] != 0:
        raise WithinMapSplitError(
            f"within-map split has {split_manifest['verified_overlap_count']} overlaps; "
            "refusing to train on a split with tile identity violations"
        )
    split_map: dict[tuple[str, int, int], str] = {
        (str(row["map"]), int(row["tile_x"]), int(row["tile_y"])): str(row["split"])
        for row in split_rows
    }
    train_rows: list[int] = []
    val_rows: list[int] = []
    for i in selected_rows:
        meta = index_rows[i]
        key = (str(meta.get("map")), int(meta.get("tile_x", -1)), int(meta.get("tile_y", -1)))
        label = split_map.get(key, SPLIT_EXCLUDED)
        if label == SPLIT_TRAIN:
            train_rows.append(i)
        elif label == SPLIT_HELD_OUT:
            val_rows.append(i)
    return train_rows, val_rows, split_manifest


def detect_split_schema(split_dir: Path) -> str:
    """Read the schema field from a split dir's manifest without full validation."""
    manifest = json.loads((split_dir / "split.json").read_text(encoding="utf-8"))
    return str(manifest.get("schema", ""))


__all__ = [
    "WITHIN_MAP_SPLIT_SCHEMA",
    "WithinMapSplitError",
    "build_within_map_split",
    "write_within_map_split",
    "validate_within_map_split",
    "load_within_map_split",
    "apply_within_map_split",
    "detect_split_schema",
]
