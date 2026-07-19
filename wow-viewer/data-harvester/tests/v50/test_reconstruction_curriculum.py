"""Spec 114 T007: grouped-split, missing-signal, stale-lighting, and provenance gates."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq
import pytest
import zarr

from harvester.v50.model_stage_contract import validate_curriculum_summary
from harvester.v50.reconstruction_curriculum import (
    ReconstructionCurriculumError,
    build_reconstruction_summary,
    main,
    run_builder,
    select_reconstruction_rows,
    write_reconstruction_curriculum,
)


def _row(index: int, group: str, source: str, split: str, map_name: str = "Kalimdor") -> dict:
    return {
        "row_index_hint": index,
        "map": map_name,
        "source_group_id": group,
        "minimap_source": source,
        "split": split,
    }


def _dual_rows() -> list[dict]:
    return [
        _row(0, "tile-a", "authored", "train"),
        _row(1, "tile-a", "synthetic", "train"),
        _row(2, "tile-b", "authored", "val"),
        _row(3, "tile-b", "synthetic", "val"),
    ]


def test_grouped_split_leak_refuses_build() -> None:
    rows = [
        _row(0, "tile-a", "authored", "train"),
        _row(1, "tile-a", "synthetic", "val"),
    ]
    with pytest.raises(ReconstructionCurriculumError, match="leak across splits"):
        select_reconstruction_rows(rows, synthetic_lighting_contract="NoonWhiteGlobal")


def test_missing_required_field_is_excluded_never_zero_filled() -> None:
    rows = _dual_rows()
    del rows[0]["minimap_source"]  # missing signal metadata: exclude honestly
    selection = select_reconstruction_rows(rows, synthetic_lighting_contract="NoonWhiteGlobal")
    assert selection.excluded_counts == {"missing_required_field": 1}
    kept_indices = [row["row_index"] for row in selection.rows]
    assert 0 not in kept_indices
    # The excluded row leaves no placeholder behind: every kept row has a real origin.
    assert all(row["input_origin"] in {"authored", "synthetic_noon_white"} for row in selection.rows)


def test_stale_synthetic_lighting_is_excluded_and_counted() -> None:
    selection = select_reconstruction_rows(_dual_rows(), synthetic_lighting_contract=None)
    assert selection.excluded_counts == {"synthetic_stale_lighting": 2}
    assert selection.input_origins == {"authored": 2, "synthetic_noon_white": 0}
    assert selection.synthetic_contract_proven is False
    assert {row["source_group_id"] for row in selection.rows} == {"tile-a", "tile-b"}


def test_noon_white_synthetics_are_admitted_with_dual_origins() -> None:
    selection = select_reconstruction_rows(
        _dual_rows(), synthetic_lighting_contract="NoonWhiteGlobal"
    )
    assert selection.excluded_counts == {}
    assert selection.input_origins == {"authored": 2, "synthetic_noon_white": 2}
    assert selection.split_counts == {"train": 2, "validation": 2, "test": 0}
    assert selection.source_group_count == 2
    assert selection.synthetic_contract_proven is True


def test_invalid_source_or_split_fails_closed() -> None:
    with pytest.raises(ReconstructionCurriculumError, match="invalid minimap_source"):
        select_reconstruction_rows(
            [_row(0, "tile-a", "painted", "train")], synthetic_lighting_contract=None
        )
    with pytest.raises(ReconstructionCurriculumError, match="invalid split"):
        select_reconstruction_rows(
            [_row(0, "tile-a", "authored", "holdout")], synthetic_lighting_contract=None
        )


def test_empty_index_and_empty_selection_fail() -> None:
    with pytest.raises(ReconstructionCurriculumError, match="zero rows"):
        select_reconstruction_rows([], synthetic_lighting_contract=None)
    with pytest.raises(ReconstructionCurriculumError, match="kept zero rows"):
        select_reconstruction_rows(
            [_row(0, "tile-a", "synthetic", "train")], synthetic_lighting_contract=None
        )


def test_summary_validates_against_published_schema() -> None:
    selection = select_reconstruction_rows(_dual_rows(), synthetic_lighting_contract=None)
    summary = build_reconstruction_summary(
        curriculum_id="test-curriculum",
        source_stores=[{"path": "store.zarr", "sha256": "a" * 64}],
        selection=selection,
        created_utc="2026-07-19T00:00:00Z",
    )
    validate_curriculum_summary(summary)
    assert summary["row_count"] == 2
    assert summary["input_origins"]["synthetic_noon_white"] == 0
    assert summary["excluded_counts"] == {"synthetic_stale_lighting": 2}


def test_write_is_immutable_and_refuses_overwrite(tmp_path: Path) -> None:
    selection = select_reconstruction_rows(_dual_rows(), synthetic_lighting_contract=None)
    summary = build_reconstruction_summary(
        curriculum_id="test-curriculum",
        source_stores=[{"path": "store.zarr", "sha256": "a" * 64}],
        selection=selection,
        created_utc="2026-07-19T00:00:00Z",
    )
    output = tmp_path / "curriculum"
    paths = write_reconstruction_curriculum(summary=summary, selection=selection, output=output)
    persisted = json.loads(paths["summary"].read_text(encoding="utf-8"))
    validate_curriculum_summary(persisted)
    selected = pq.read_table(paths["selection"]).to_pylist()
    assert len(selected) == 2
    assert {row["input_origin"] for row in selected} == {"authored"}
    with pytest.raises(ReconstructionCurriculumError, match="refusing to overwrite"):
        write_reconstruction_curriculum(summary=summary, selection=selection, output=output)


def _write_source_store(path: Path, rows: list[dict], *, contract: str | None) -> None:
    import pyarrow as pa

    group = zarr.open_group(str(path), mode="w")
    group.attrs["schema"] = "v50-mixed-curriculum-v1"
    if contract is not None:
        group.attrs["synthetic_lighting_contract"] = contract
    pq.write_table(pa.Table.from_pylist(rows), path / "index.parquet")


def test_run_builder_dry_run_prints_summary_and_writes_nothing(tmp_path: Path, capsys) -> None:
    store = tmp_path / "dual.zarr"
    store.mkdir()
    _write_source_store(store, _dual_rows(), contract=None)
    output = tmp_path / "out"
    exit_code = main(
        [
            "--store", str(store),
            "--output", str(output),
            "--curriculum-id", "dry-run-test",
        ]
    )
    assert exit_code == 0
    assert not output.exists()
    printed = capsys.readouterr().out
    assert "DRY RUN ONLY" in printed
    summary = json.loads(printed[: printed.index("DRY RUN ONLY")].strip())
    validate_curriculum_summary(summary)
    assert summary["row_count"] == 2


def test_run_builder_write_persists_and_refuses_rerun(tmp_path: Path) -> None:
    store = tmp_path / "dual.zarr"
    store.mkdir()
    _write_source_store(store, _dual_rows(), contract="NoonWhiteGlobal")
    output = tmp_path / "out"
    summary = run_builder(
        stores=[store], output=output, curriculum_id="write-test", write=True
    )
    assert summary["input_origins"] == {"authored": 2, "synthetic_noon_white": 2}
    assert (output / "summary.json").is_file()
    assert (output / "selection.parquet").is_file()
    with pytest.raises(ReconstructionCurriculumError, match="refusing to overwrite"):
        run_builder(stores=[store], output=output, curriculum_id="write-test", write=True)


def test_run_builder_refuses_mixed_lighting_provenance(tmp_path: Path) -> None:
    stale = tmp_path / "stale.zarr"
    fresh = tmp_path / "fresh.zarr"
    stale.mkdir()
    fresh.mkdir()
    _write_source_store(stale, _dual_rows(), contract=None)
    _write_source_store(fresh, _dual_rows(), contract="NoonWhiteGlobal")
    with pytest.raises(ReconstructionCurriculumError, match="disagree on synthetic_lighting"):
        run_builder(
            stores=[stale, fresh],
            output=tmp_path / "out",
            curriculum_id="mixed-test",
            write=False,
        )


def test_run_builder_refuses_wrong_source_schema(tmp_path: Path) -> None:
    store = tmp_path / "wrong.zarr"
    store.mkdir()
    group = zarr.open_group(str(store), mode="w")
    group.attrs["schema"] = "v50-complete-store-v1"
    with pytest.raises(ReconstructionCurriculumError, match="source store schema"):
        run_builder(stores=[store], output=tmp_path / "out", curriculum_id="x", write=False)
