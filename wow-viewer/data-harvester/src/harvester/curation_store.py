"""Thin reader for the canonical C#-produced dataset curation manifest (Spec 122).

This module reads ``curation_manifest.parquet`` / ``curation_findings.parquet`` written by
``WowViewer.Tool.Harvest curate`` (see ``wow-viewer/src/core/WowViewer.Core.Curation``). It contains
no curation *logic* of its own -- bucketing and mismatch detection are computed exactly once, in
C#, and this module only reads the result. See ``specs/122-dataset-curation/`` for the full design.

Design commitment (User Story 2, FR-009): querying a non-clean bucket (e.g. ``coverage_bucket ==
"blank"``) is the exact same operation -- a column filter on the same table -- as querying the
clean bucket. There is no separate "recovery" path for bad/mismatched data; both loaders return the
complete table with zero default filtering, so every bucket stays equally accessible.
"""

from __future__ import annotations

from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def _resolve_run_dir(store_path: str | Path, curation_run_id: str | None) -> Path:
    curation_root = Path(store_path) / "curation"
    if curation_run_id is None:
        pointer_path = curation_root / "latest"
        if not pointer_path.exists():
            raise FileNotFoundError(
                f"No curation manifest found for store {store_path!r} -- run "
                "`WowViewer.Tool.Harvest curate --store <store> --client-root <root> --write` first."
            )
        curation_run_id = pointer_path.read_text(encoding="utf-8").strip()

    run_dir = curation_root / curation_run_id
    if not run_dir.exists():
        raise FileNotFoundError(f"Curation run {curation_run_id!r} not found under {curation_root}.")
    return run_dir


def load_curation_manifest(store_path: str | Path, curation_run_id: str | None = None) -> pa.Table:
    """Read the per-tile curation manifest (one row per tile) for a store.

    Resolves ``<store>/curation/latest`` by default; pass ``curation_run_id`` to pin a specific run
    instead of "most recent". Returns the full table -- no bucket is filtered out by default.
    """
    run_dir = _resolve_run_dir(store_path, curation_run_id)
    return pq.read_table(str(run_dir / "curation_manifest.parquet"))


def load_curation_findings(store_path: str | Path, curation_run_id: str | None = None) -> pa.Table:
    """Read the per-finding curation table (one row per (tile, finding)) for a store.

    Same resolution rule as :func:`load_curation_manifest`. A tile with zero findings simply has no
    rows here -- this is distinct from a tile whose check was not_evaluable, which does get a row
    (see data-model.md "Mismatch Finding" validation rules).
    """
    run_dir = _resolve_run_dir(store_path, curation_run_id)
    return pq.read_table(str(run_dir / "curation_findings.parquet"))


def resolve_curation_run_id(store_path: str | Path, curation_run_id: str | None = None) -> str:
    """The curation_run_id that :func:`load_curation_manifest`/:func:`load_curation_findings` would
    resolve to -- useful for a downstream consumer's own run record to cite exactly which curation
    pass it selected from (the "Selection Record" audit trail in data-model.md)."""
    return _resolve_run_dir(store_path, curation_run_id).name
