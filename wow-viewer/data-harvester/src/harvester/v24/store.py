"""V24 Zarr store layout (FR-007, amended per spec A5).

The store carries the merged WDL prior as paired outer/inner arrays plus an
``index.parquet`` describing each row (including ``v18_row``, the row index in
the source V18 store, so consumers can join back to the V18 substrate without
copying V18 arrays).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

PRIOR_ARRAYS: dict[str, tuple[tuple[int, ...], str]] = {
    "wdl_prior_outer": ((17, 17), "float32"),
    "wdl_prior_inner": ((16, 16), "float32"),
    "wdl_prior_source_outer": ((17, 17), "uint8"),
    "wdl_prior_source_inner": ((16, 16), "uint8"),
    "wdl_prior_confidence_outer": ((17, 17), "float32"),
    "wdl_prior_confidence_inner": ((16, 16), "float32"),
}

SCALAR_ARRAYS: dict[str, str] = {
    "wdl_prior_disagree_ratio": "float32",
    "wdl_prior_audit_empty": "bool",
    "wdl_prior_real_available": "bool",
}


def create_v24_store(path: Path | str, n_tiles: int, attrs: dict[str, Any]) -> zarr.Group:
    """Create (overwrite) a V24 store with the prior arrays sized for ``n_tiles``."""
    group = zarr.open_group(str(path), mode="w", zarr_format=2)
    for name, (tile_shape, dtype) in PRIOR_ARRAYS.items():
        group.create_array(
            name=name,
            shape=(n_tiles, *tile_shape),
            chunks=(min(256, max(1, n_tiles)), *tile_shape),
            dtype=dtype,
        )
    for name, dtype in SCALAR_ARRAYS.items():
        group.create_array(
            name=name,
            shape=(n_tiles,),
            chunks=(max(1, n_tiles),),
            dtype=dtype,
        )
    group.attrs.update(
        {
            "spec": "094-wdl-prior-v24",
            "wdl_grid_shape": {"outer": [17, 17], "inner": [16, 16]},
            "wdl_grid_shape_source": "csharp_wdl_reader",
            **attrs,
        }
    )
    return group


def open_v24_store(path: Path | str) -> zarr.Group:
    return zarr.open_group(str(path), mode="r")


def write_index(path: Path | str, columns: dict[str, list]) -> None:
    """Write the V24 index.parquet next to the Zarr arrays."""
    table = pa.table(columns)
    pq.write_table(table, str(Path(path) / "index.parquet"))


def read_index(path: Path | str) -> dict[str, list]:
    """Read a store index.parquet (works for V18, V22, and V24 stores)."""
    table = pq.read_table(str(Path(path) / "index.parquet"))
    return table.to_pydict()


def coverage_stats(group: zarr.Group) -> dict[str, float]:
    """Aggregate per-source cell ratios across the whole store (SC-001)."""
    counts = np.zeros(3, dtype=np.int64)
    for name in ("wdl_prior_source_outer", "wdl_prior_source_inner"):
        source = np.asarray(group[name][:])
        for value in (0, 1, 2):
            counts[value] += int((source == value).sum())

    total = int(counts.sum())
    cells_per_tile = 17 * 17 + 16 * 16
    audit_empty = np.asarray(group["wdl_prior_audit_empty"][:])
    non_empty_cells = int((~audit_empty).sum()) * cells_per_tile
    real_plus_synth = int(counts[0] + counts[1])
    return {
        "total_cells": total,
        "real_cell_ratio": counts[0] / total if total else 0.0,
        "synthetic_cell_ratio": counts[1] / total if total else 0.0,
        "learned_fill_cell_ratio": counts[2] / total if total else 0.0,
        # SC-001 gate: real+synthetic coverage over the cells of non-audit-empty
        # tiles ("all non-empty cells" in the spec's wording).
        "real_plus_synthetic_ratio_of_non_empty": (
            real_plus_synth / non_empty_cells if non_empty_cells else 1.0
        ),
    }
