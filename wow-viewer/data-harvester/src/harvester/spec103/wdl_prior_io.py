"""Portable Spec 108 generated-WDL archive contract."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from harvester.spec103.wdl_prior_model import WDL_INNER_SIZE, WDL_OUTER_SIZE


def write_prediction_archive(path: Path, rows: np.ndarray, outer_17: np.ndarray, inner_16: np.ndarray, metadata: dict) -> None:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    outer = np.asarray(outer_17, dtype=np.float32)
    inner = np.asarray(inner_16, dtype=np.float32)
    if len(np.unique(rows)) != len(rows) or outer.shape != (len(rows), WDL_OUTER_SIZE, WDL_OUTER_SIZE) or inner.shape != (len(rows), WDL_INNER_SIZE, WDL_INNER_SIZE):
        raise ValueError("invalid generated WDL archive shapes or duplicate rows")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, rows=rows, outer_17=outer, inner_16=inner, metadata_json=json.dumps(metadata, sort_keys=True))


def read_prediction_archive(path: Path) -> tuple[dict[int, np.ndarray], dict]:
    with np.load(path, allow_pickle=False) as archive:
        rows = np.asarray(archive["rows"], dtype=np.int64).reshape(-1)
        outer = np.asarray(archive["outer_17"], dtype=np.float32)
        inner = np.asarray(archive["inner_16"], dtype=np.float32)
        metadata = json.loads(str(archive["metadata_json"].item()))
    if len(np.unique(rows)) != len(rows) or outer.shape != (len(rows), WDL_OUTER_SIZE, WDL_OUTER_SIZE) or inner.shape != (len(rows), WDL_INNER_SIZE, WDL_INNER_SIZE):
        raise ValueError(f"invalid generated WDL archive: {path}")
    if not np.all(np.isfinite(outer)) or not np.all(np.isfinite(inner)):
        raise ValueError(f"generated WDL archive contains non-finite values: {path}")
    return {int(row): outer[idx] for idx, row in enumerate(rows)}, metadata
