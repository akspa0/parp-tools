"""Spec 114 dual-view reconstruction curriculum selection and summary (T008).

Consumes the existing trainer-facing dual-source curriculum store (Spec 112's
``v50-mixed-curriculum-v1``: each terrain tile contributes up to two rows — authored and synthetic
minimap views of the same ``height_257`` target, sharing one ``source_group_id`` and split) and
produces the Spec 114 ``v50-reconstruction-curriculum-v1`` summary plus an explicit row selection.

Admission policy, fail-closed:

- Both views of one tile MUST share one split; any cross-split ``source_group_id`` refuses the
  whole build (the published contract requires ``group_leak_count == 0``).
- Synthetic rows are admitted as ``synthetic_noon_white`` ONLY when the source store records
  ``synthetic_lighting_contract=NoonWhiteGlobal``. Today's dual store predates the corrected
  compositor, so its synthetic rows are honestly excluded and counted under
  ``synthetic_stale_lighting`` — never silently trained on, never zero-filled.
- Rows missing required index fields are excluded and counted, never repaired with placeholders.
- Invalid enum values (``minimap_source``/``split``) are corruption and refuse the build.

The summary's ``synthetic_lighting_contract=NoonWhiteGlobal`` is the ADMISSION contract for the
rows it contains: every synthetic row in the selection was proven to carry that provenance. Rows
that could not prove it are excluded, so the const stays honest even for the current stale store.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from harvester.v50.model_stage_contract import (
    SYNTHETIC_LIGHTING_CONTRACT,
    TARGET_SIGNAL,
    ContractViolation,
    identity_for_path,
    validate_curriculum_summary,
)

SOURCE_CURRICULUM_SCHEMA = "v50-mixed-curriculum-v1"
REQUIRED_INDEX_FIELDS = frozenset({"map", "source_group_id", "minimap_source", "split"})
VALID_SOURCES = frozenset({"authored", "synthetic"})
VALID_SPLITS = frozenset({"train", "val", "validation", "test"})

SUMMARY_FILENAME = "summary.json"
SELECTION_FILENAME = "selection.parquet"


class ReconstructionCurriculumError(ValueError):
    """Raised when a reconstruction curriculum cannot be built as declared."""


@dataclass(frozen=True)
class CurriculumSelection:
    """Row-level result of the dual-view admission policy."""

    rows: list[dict[str, Any]]
    excluded_counts: dict[str, int]
    source_group_count: int
    split_counts: dict[str, int]
    input_origins: dict[str, int]
    synthetic_contract_proven: bool = field(compare=False)


def _normalize_split(value: str) -> str:
    return "validation" if value == "val" else value


def select_reconstruction_rows(
    index_rows: list[dict],
    *,
    synthetic_lighting_contract: str | None,
) -> CurriculumSelection:
    """Apply the dual-view admission policy to one store's index rows (pure, CPU-testable).

    ``synthetic_lighting_contract`` is the source store's recorded provenance attr (``None`` when
    the store predates the corrected compositor).
    """
    if not index_rows:
        raise ReconstructionCurriculumError("source curriculum index contains zero rows")

    kept: list[dict[str, Any]] = []
    excluded: dict[str, int] = {}
    synthetics_proven = synthetic_lighting_contract == SYNTHETIC_LIGHTING_CONTRACT

    def _exclude(reason: str) -> None:
        excluded[reason] = excluded.get(reason, 0) + 1

    for row_index, row in enumerate(index_rows):
        missing = sorted(field for field in REQUIRED_INDEX_FIELDS if field not in row)
        if missing:
            _exclude("missing_required_field")
            continue
        source = str(row["minimap_source"])
        if source not in VALID_SOURCES:
            raise ReconstructionCurriculumError(
                f"index row {row_index} has invalid minimap_source {source!r}; "
                f"expected one of {sorted(VALID_SOURCES)}"
            )
        split = str(row["split"])
        if split not in VALID_SPLITS:
            raise ReconstructionCurriculumError(
                f"index row {row_index} has invalid split {split!r}; "
                f"expected one of {sorted(VALID_SPLITS)}"
            )
        if source == "synthetic" and not synthetics_proven:
            _exclude("synthetic_stale_lighting")
            continue
        kept.append(
            {
                "row_index": row_index,
                "map": str(row["map"]),
                "source_group_id": str(row["source_group_id"]),
                "split": _normalize_split(split),
                "input_origin": "authored" if source == "authored" else "synthetic_noon_white",
            }
        )

    if not kept:
        raise ReconstructionCurriculumError(
            f"admission policy kept zero rows; excluded counts: {excluded}"
        )

    splits_by_group: dict[str, set[str]] = {}
    for row in kept:
        splits_by_group.setdefault(row["source_group_id"], set()).add(row["split"])
    leaked = sorted(group for group, splits in splits_by_group.items() if len(splits) != 1)
    if leaked:
        raise ReconstructionCurriculumError(
            f"source groups leak across splits; refusing to build (first groups: {leaked[:5]})"
        )

    split_counts = {"train": 0, "validation": 0, "test": 0}
    input_origins = {"authored": 0, "synthetic_noon_white": 0}
    for row in kept:
        split_counts[row["split"]] += 1
        input_origins[row["input_origin"]] += 1

    return CurriculumSelection(
        rows=kept,
        excluded_counts=excluded,
        source_group_count=len(splits_by_group),
        split_counts=split_counts,
        input_origins=input_origins,
        synthetic_contract_proven=synthetics_proven,
    )


def build_reconstruction_summary(
    *,
    curriculum_id: str,
    source_stores: list[dict[str, str]],
    selection: CurriculumSelection,
    created_utc: str | None = None,
) -> dict[str, Any]:
    """Assemble and self-validate the published ``v50-reconstruction-curriculum-v1`` document."""
    if not curriculum_id:
        raise ReconstructionCurriculumError("curriculum_id must be non-empty")
    summary: dict[str, Any] = {
        "schema": "v50-reconstruction-curriculum-v1",
        "curriculum_id": curriculum_id,
        "created_utc": created_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "source_stores": source_stores,
        "row_count": len(selection.rows),
        "source_group_count": selection.source_group_count,
        "split_counts": selection.split_counts,
        "input_origins": selection.input_origins,
        "target_signal": TARGET_SIGNAL,
        "synthetic_lighting_contract": SYNTHETIC_LIGHTING_CONTRACT,
        "group_leak_count": 0,
        "excluded_counts": selection.excluded_counts,
    }
    try:
        validate_curriculum_summary(summary)
    except ContractViolation as exc:
        raise ReconstructionCurriculumError(f"built summary violates its own contract: {exc}") from exc
    return summary


def load_source_curriculum(store: Path) -> tuple[dict, list[dict]]:
    """Read a dual-source curriculum store's attrs and index rows (the builder's only inputs)."""
    import pyarrow.parquet as pq
    import zarr

    if not store.is_dir():
        raise ReconstructionCurriculumError(f"source curriculum store not found: {store}")
    group = zarr.open_group(str(store), mode="r")
    attrs = dict(group.attrs)
    if attrs.get("schema") != SOURCE_CURRICULUM_SCHEMA:
        raise ReconstructionCurriculumError(
            f"source store schema must be {SOURCE_CURRICULUM_SCHEMA!r}, got {attrs.get('schema')!r}"
        )
    index_path = store / "index.parquet"
    if not index_path.is_file():
        raise ReconstructionCurriculumError(f"source store is missing index.parquet: {store}")
    return attrs, pq.read_table(index_path).to_pylist()


def write_reconstruction_curriculum(
    *,
    summary: dict[str, Any],
    selection: CurriculumSelection,
    output: Path,
) -> dict[str, Path]:
    """Persist the summary and row selection. Refuses to overwrite a non-empty directory."""
    import pyarrow as pa
    import pyarrow.parquet as pq

    if output.exists() and any(output.iterdir()):
        raise ReconstructionCurriculumError(
            f"refusing to overwrite non-empty curriculum output {output}; choose a new path"
        )
    output.mkdir(parents=True, exist_ok=True)
    summary_path = output / SUMMARY_FILENAME
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    table = pa.Table.from_pylist(selection.rows)
    selection_path = output / SELECTION_FILENAME
    pq.write_table(table, selection_path)
    return {"summary": summary_path, "selection": selection_path}


def run_builder(
    *,
    stores: list[Path],
    output: Path,
    curriculum_id: str,
    write: bool,
) -> dict[str, Any]:
    """Shared CLI/test path: select rows, build the summary, optionally persist it."""
    all_rows: list[dict] = []
    source_identities: list[dict[str, str]] = []
    synthetic_contract: str | None = None
    contract_seen = False

    for store in stores:
        attrs, rows = load_source_curriculum(store)
        contract = attrs.get("synthetic_lighting_contract")
        if not contract_seen:
            synthetic_contract = contract
            contract_seen = True
        elif contract != synthetic_contract:
            raise ReconstructionCurriculumError(
                "source stores disagree on synthetic_lighting_contract "
                f"({synthetic_contract!r} vs {contract!r}); build each era's curriculum separately"
            )
        source_identities.append(identity_for_path(store / "index.parquet", display_path=str(store)))
        all_rows.extend(rows)

    selection = select_reconstruction_rows(
        all_rows, synthetic_lighting_contract=synthetic_contract
    )
    summary = build_reconstruction_summary(
        curriculum_id=curriculum_id,
        source_stores=source_identities,
        selection=selection,
    )
    if write:
        write_reconstruction_curriculum(summary=summary, selection=selection, output=output)
    return summary


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description="Spec 114 dual-view reconstruction curriculum builder (dry run by default)"
    )
    ap.add_argument(
        "--store",
        required=True,
        type=Path,
        action="append",
        help="dual-source curriculum zarr (repeatable for multi-store curricula)",
    )
    ap.add_argument("--output", required=True, type=Path)
    ap.add_argument("--curriculum-id", default="reconstruction-0_5_3_3368-dual-v1")
    ap.add_argument(
        "--write",
        action="store_true",
        help="persist summary.json + selection.parquet; without it only print the summary",
    )
    args = ap.parse_args(argv)

    try:
        summary = run_builder(
            stores=args.store, output=args.output, curriculum_id=args.curriculum_id, write=args.write
        )
    except (ReconstructionCurriculumError, ContractViolation) as exc:
        raise SystemExit(str(exc)) from exc

    print(json.dumps(summary, indent=2), flush=True)
    if not args.write:
        print("DRY RUN ONLY: add --write to persist summary.json and selection.parquet.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
