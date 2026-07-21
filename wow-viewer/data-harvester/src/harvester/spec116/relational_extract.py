"""Spec 116 US1/US3: extract relational layer-entry rows from a v50 curriculum store.

A terrain tile is a serialized relational schema. This module turns the per-chunk/per-slot arrays
already in the v50 store into **rows** -- the central object of Spec 116:

    (tile_row, map, tile_x, tile_y, chunk_y, chunk_x, slot, local_texture_id, coverage, family)

The texture reference (``local_texture_id``) is a foreign key into that tile's own MTEX table. It is
resolved to a **surface family** by joining against a texture-name dump and classifying the leaf
name with the frozen Spec 115 taxonomy (``harvester.v50.terrain_feature_labels``, revision
``v115.1``). The local index needs no global registry, and the dump is verified to reproduce the
real client table exactly (see Spec 115).

Rows whose tile has no texture-name dump entry, or an empty MTEX table, are excluded wholesale and
counted -- never emitted as all-``unknown`` (mirrors the Spec 115 label builder). Absent slots
(``local_texture_id < 0``) are not rows at all and are skipped silently.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from harvester.v50.terrain_feature_labels import (
    FAMILY_NAMES,
    TAXONOMY_REVISION,
    classify_texture_name,
    load_texture_name_dump,
    rule_set_sha256,
)

CHUNKS_PER_AXIS = 16
MAX_LAYERS = 4


class RelationalExtractError(ValueError):
    """Raised when layer-entry rows cannot be extracted honestly."""


@dataclass(frozen=True)
class LayerEntry:
    """One ordered row of a chunk's layer table (spec Key Entity: Layer Entry)."""

    tile_row: int
    map: str
    tile_x: int
    tile_y: int
    chunk_y: int
    chunk_x: int
    slot: int
    local_texture_id: int
    coverage: float
    family: int

    def as_dict(self) -> dict:
        return {
            "tile_row": self.tile_row,
            "map": self.map,
            "tile_x": self.tile_x,
            "tile_y": self.tile_y,
            "chunk_y": self.chunk_y,
            "chunk_x": self.chunk_x,
            "slot": self.slot,
            "local_texture_id": self.local_texture_id,
            "coverage": float(self.coverage),
            "family": int(self.family),
            "family_name": FAMILY_NAMES[self.family],
        }


@dataclass(frozen=True)
class ExtractResult:
    rows: tuple[LayerEntry, ...]
    row_count: int
    excluded: dict[str, int] = field(default_factory=dict)
    taxonomy_revision: str = TAXONOMY_REVISION
    rule_set_sha256: str = field(default_factory=rule_set_sha256)

    def family_slot_counts(self) -> np.ndarray:
        """(CLASS_COUNT, MAX_LAYERS) int64 counts of rows per (family, slot)."""
        counts = np.zeros((len(FAMILY_NAMES), MAX_LAYERS), dtype=np.int64)
        for row in self.rows:
            counts[row.family, row.slot] += 1
        return counts


def extract_layer_entries(
    *,
    store: Path,
    dumps: Iterable[Path],
) -> ExtractResult:
    """Extract every layer-entry row from a v50 curriculum store.

    Reads ``mcly_texture_ids`` and ``mcly_layer_mask`` plus ``index.parquet``; joins each tile's
    local MTEX indices to surface families via the texture-name dump. Returns the rows and an
    exclusion report. Never emits all-``unknown`` rows.
    """
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(store), mode="r")
    for required in ("mcly_texture_ids", "mcly_layer_mask"):
        if required not in group:
            raise RelationalExtractError(f"store is missing {required!r}: {store}")
    if "mcly_texture_ids" not in group or "mcly_layer_mask" not in group:
        raise RelationalExtractError(f"store missing mcly arrays: {store}")

    index_path = store / "index.parquet"
    if not index_path.exists():
        raise RelationalExtractError(f"store has no index.parquet: {store}")
    index_rows = pq.read_table(index_path).to_pylist()

    ids_array = group["mcly_texture_ids"]
    mask_array = group["mcly_layer_mask"]
    row_count = int(ids_array.shape[0])
    if row_count != len(index_rows):
        raise RelationalExtractError(
            f"index rows ({len(index_rows)}) != mcly_texture_ids rows ({row_count})"
        )
    if ids_array.shape[1:] != (CHUNKS_PER_AXIS, CHUNKS_PER_AXIS, MAX_LAYERS):
        raise RelationalExtractError(
            f"mcly_texture_ids must be (N, {CHUNKS_PER_AXIS}, {CHUNKS_PER_AXIS}, {MAX_LAYERS}), "
            f"got {ids_array.shape}"
        )

    names_by_tile = load_texture_name_dump(dumps)

    rows: list[LayerEntry] = []
    excluded: dict[str, int] = {"no_texture_name_dump_entry": 0, "empty_mtex_table": 0}

    for row in range(row_count):
        meta = index_rows[row]
        key = (str(meta.get("map")), int(meta.get("tile_x", -1)), int(meta.get("tile_y", -1)))
        texture_names = names_by_tile.get(key)
        if not texture_names:
            excluded["no_texture_name_dump_entry"] += 1
            continue
        if len(texture_names) == 0:
            excluded["empty_mtex_table"] += 1
            continue

        tile_ids = np.asarray(ids_array[row], dtype=np.int32)
        tile_mask = np.asarray(mask_array[row], dtype=np.float32)
        map_name, tile_x, tile_y = key

        for cy in range(CHUNKS_PER_AXIS):
            for cx in range(CHUNKS_PER_AXIS):
                for slot in range(MAX_LAYERS):
                    local_id = int(tile_ids[cy, cx, slot])
                    if local_id < 0:
                        continue  # absent slot is not a row
                    coverage = float(tile_mask[cy, cx, slot])
                    if 0 <= local_id < len(texture_names):
                        family = classify_texture_name(texture_names[local_id])
                    else:
                        # Local index outside this tile's own table: no honest family.
                        family = 0  # UNKNOWN
                    rows.append(
                        LayerEntry(
                            tile_row=row, map=map_name, tile_x=tile_x, tile_y=tile_y,
                            chunk_y=cy, chunk_x=cx, slot=slot,
                            local_texture_id=local_id, coverage=coverage, family=family,
                        )
                    )

    if not rows:
        raise RelationalExtractError("no layer-entry rows could be extracted")

    return ExtractResult(
        rows=tuple(rows),
        row_count=row_count,
        excluded=excluded,
    )


__all__ = [
    "LayerEntry",
    "ExtractResult",
    "RelationalExtractError",
    "extract_layer_entries",
    "CHUNKS_PER_AXIS",
    "MAX_LAYERS",
]
