"""Spec 119 shared object-library reading (read-only; FR-011).

One canonical place for opening the object-library zarr, aligning ``assets.parquet`` rows with
``capture_rgb``/``capture_mask`` array rows, and deriving per-row coverage/labels. Both trainers,
the inference batch path, and the quality lens consume this; no module duplicates the row
alignment (captured-only filtering keeps the ORIGINAL zarr row index so array reads stay
positionally correct).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from harvester.spec119.object_library_contract import (
    ObjectLibraryContractError,
    coarse_label_for_row,
    derive_fine_family_label,
    mask_coverage,
)

CAPTURE_RGB = "capture_rgb"
CAPTURE_MASK = "capture_mask"


def load_asset_rows(store: Path) -> list[dict[str, Any]]:
    """All ``assets.parquet`` rows, each annotated with its original zarr row index."""
    import pyarrow.parquet as pq

    parquet = Path(store) / "assets.parquet"
    if not parquet.is_file():
        raise ObjectLibraryContractError(f"{store}: missing assets.parquet (not an object-library store)")
    rows = pq.read_table(parquet).to_pylist()
    for index, row in enumerate(rows):
        row["_row_index"] = index
    return rows


def captured_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Only ``captured`` rows are trainable (data-model.md)."""
    return [row for row in rows if row.get("capture_status") == "captured"]


def open_library(store: Path):
    """Open the zarr read-only and fail closed when the capture arrays are absent."""
    import zarr

    try:
        group = zarr.open_group(str(store), mode="r")
    except FileNotFoundError as exc:
        raise ObjectLibraryContractError(f"{store}: store does not exist") from exc
    missing = [name for name in (CAPTURE_RGB, CAPTURE_MASK) if name not in group]
    if missing:
        raise ObjectLibraryContractError(
            f"{store}: missing capture arrays {missing}; expected the Spec 118 capture-objects "
            "object-library layout (capture_rgb + capture_mask)"
        )
    return group


def row_coverages(group, rows: list[dict[str, Any]]) -> list[float]:
    """Per-row mask coverage in the same order as ``rows``."""
    mask = group[CAPTURE_MASK]
    return [mask_coverage(np.asarray(mask[row["_row_index"]])) for row in rows]


def coarse_labels(
    rows: list[dict[str, Any]], coverages: list[float], blank_threshold: float
) -> list[str]:
    """Coarse labels (blank -> ``empty``, D-04) in the same order as ``rows``."""
    return [
        coarse_label_for_row(str(row["asset_type"]), coverage, blank_threshold)
        for row, coverage in zip(rows, coverages, strict=True)
    ]


def fine_labels(
    rows: list[dict[str, Any]], coverages: list[float], blank_threshold: float
) -> list[str]:
    """Heuristic fine-family labels (D-03); blank rows are still ``empty``."""
    labels = []
    for row, coverage in zip(rows, coverages, strict=True):
        coarse = coarse_label_for_row(str(row["asset_type"]), coverage, blank_threshold)
        labels.append(
            coarse if coarse == "empty" else derive_fine_family_label(str(row["normalized_asset_path"]))
        )
    return labels


def label_index_map(labels: list[str]) -> dict[str, int]:
    """Stable label->index map; ``empty`` is always index 0, the rest sorted."""
    rest = sorted({label for label in labels if label != "empty"})
    return {"empty": 0, **{label: i + 1 for i, label in enumerate(rest)}}


def read_image(group, row_index: int) -> np.ndarray:
    """One capture image as float32 HWC in [0, 1]."""
    return np.asarray(group[CAPTURE_RGB][row_index], dtype=np.float32) / 255.0


def require_new_output(path: Path) -> None:
    """Refuse to overwrite an existing run directory (immutable run outputs)."""
    if Path(path).exists():
        raise ObjectLibraryContractError(f"{path} already exists; refusing to overwrite a run output")


__all__ = [
    "CAPTURE_MASK",
    "CAPTURE_RGB",
    "captured_rows",
    "coarse_labels",
    "fine_labels",
    "label_index_map",
    "load_asset_rows",
    "open_library",
    "read_image",
    "require_new_output",
    "row_coverages",
]
