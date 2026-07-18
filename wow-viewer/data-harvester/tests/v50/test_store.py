"""Spec 109 T029: complete-store/finalization tests. A store can only finalize as COMPLETE when
every signal's declared content_identity matches what was actually written and every row has real
lineage -- never because the manifest says so."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from harvester.v50.contracts import DatasetSignal, DatasetStoreManifest, FinalizationState, MigrationPolicy, RowLineage
from harvester.v50.curriculum import CurriculumRowRef, build_curriculum
from harvester.v50.identity import hash_array
from harvester.v50.store import (
    StoreWriteError,
    finalize_store,
    finalize_store_report,
    read_v50_manifest,
    write_v50_store,
)

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64


def _height_array(row_count: int = 2) -> np.ndarray:
    return np.stack([np.full((4, 4), float(i), dtype=np.float32) for i in range(row_count)])


def _manifest(row_count: int = 2, height_hash: str | None = None) -> DatasetStoreManifest:
    height = _height_array(row_count)
    return DatasetStoreManifest(
        release="v50.1",
        store_id=_HASH_A,
        build_id="0.5.3.3368",
        producer_identity=_HASH_B,
        client_build_evidence_id=_HASH_C,
        index_identity=_HASH_A,
        row_count=row_count,
        signals=(
            DatasetSignal(
                name="height_4",
                dtype="float32",
                row_shape=(4, 4),
                required=True,
                authoritative_source="wowviewer.core.io.adt_reader",
                content_identity=height_hash or hash_array(height),
                coverage_count=row_count,
                migration_policy=MigrationPolicy.COPY_IF_VERIFIED,
            ),
        ),
        row_lineage_identity=_HASH_B,
        finalization_state=FinalizationState.COMPLETE,
    )


def _lineages(row_count: int = 2) -> list[RowLineage]:
    return [
        RowLineage(
            store_row=row,
            build_id="0.5.3.3368",
            map_name="Azeroth",
            tile_x=row,
            tile_y=row,
            source_group=f"azeroth:{row}",
            signal_actions={"height_4": "copied"},
        )
        for row in range(row_count)
    ]


def test_write_v50_store_writes_a_real_readable_store(tmp_path: Path):
    manifest = _manifest()
    store_path = tmp_path / "store.zarr"

    write_v50_store(store_path, manifest, {"height_4": _height_array()})

    attrs = read_v50_manifest(store_path)
    assert attrs["release"] == "v50.1"
    assert attrs["schema"] == "v50-complete-store-v1"


def test_write_v50_store_refuses_when_a_required_signal_array_is_missing(tmp_path: Path):
    manifest = _manifest()
    with pytest.raises(StoreWriteError, match="missing arrays"):
        write_v50_store(tmp_path / "store.zarr", manifest, {})


def test_write_v50_store_refuses_a_row_count_mismatch(tmp_path: Path):
    manifest = _manifest(row_count=2)
    wrong_count_array = _height_array(row_count=5)
    with pytest.raises(StoreWriteError, match="row count mismatches"):
        write_v50_store(tmp_path / "store.zarr", manifest, {"height_4": wrong_count_array})


def test_write_v50_store_leaves_no_staging_directory_behind_on_success(tmp_path: Path):
    manifest = _manifest()
    store_path = tmp_path / "store.zarr"

    write_v50_store(store_path, manifest, {"height_4": _height_array()})

    leftovers = [p for p in tmp_path.iterdir() if p != store_path]
    assert leftovers == []


def test_write_v50_store_preserves_a_prior_good_store_when_the_new_write_fails(tmp_path: Path):
    # Regression for Spec 109 Phase 8: a build that crashes/is killed partway through writing must
    # never destroy a prior store that already finished successfully -- a bad manifest here models
    # any mid-write failure (I/O error, process kill, bad array) that previously wiped the target
    # path via an unconditional `mode="w"` before anything new had actually landed.
    good_manifest = _manifest()
    store_path = tmp_path / "store.zarr"
    write_v50_store(store_path, good_manifest, {"height_4": _height_array()})
    good_attrs = read_v50_manifest(store_path)

    broken_manifest = _manifest(row_count=5)  # declares 5 rows but we hand it a 2-row array
    with pytest.raises(StoreWriteError, match="row count mismatches"):
        write_v50_store(store_path, broken_manifest, {"height_4": _height_array(row_count=2)})

    assert read_v50_manifest(store_path) == good_attrs


def test_write_v50_store_replaces_a_prior_store_on_a_second_successful_write(tmp_path: Path):
    manifest_a = _manifest(row_count=2)
    manifest_b = _manifest(row_count=3)
    store_path = tmp_path / "store.zarr"

    write_v50_store(store_path, manifest_a, {"height_4": _height_array(row_count=2)})
    write_v50_store(store_path, manifest_b, {"height_4": _height_array(row_count=3)})

    assert read_v50_manifest(store_path)["row_count"] == 3


def test_finalize_store_reaches_complete_when_everything_reconciles(tmp_path: Path):
    manifest = _manifest()
    store_path = tmp_path / "store.zarr"
    write_v50_store(store_path, manifest, {"height_4": _height_array()})

    finalized = finalize_store(store_path, manifest, _lineages())

    assert finalized.finalization_state is FinalizationState.COMPLETE


def test_finalize_store_stays_incomplete_when_manifest_declares_a_wrong_content_hash(tmp_path: Path):
    # The manifest *claims* a hash that does not match what was actually written -- finalization
    # must catch this, not trust the declared value.
    manifest = _manifest(height_hash="sha256:" + "9" * 64)
    store_path = tmp_path / "store.zarr"
    write_v50_store(store_path, manifest, {"height_4": _height_array()})

    finalized = finalize_store(store_path, manifest, _lineages())

    assert finalized.finalization_state is FinalizationState.INCOMPLETE


def test_finalize_store_stays_incomplete_when_row_lineage_count_disagrees(tmp_path: Path):
    manifest = _manifest(row_count=2)
    store_path = tmp_path / "store.zarr"
    write_v50_store(store_path, manifest, {"height_4": _height_array()})

    finalized = finalize_store(store_path, manifest, _lineages(row_count=1))

    assert finalized.finalization_state is FinalizationState.INCOMPLETE


def test_finalize_store_stays_incomplete_when_a_required_signal_has_no_real_lineage_action(tmp_path: Path):
    manifest = _manifest(row_count=1)
    store_path = tmp_path / "store.zarr"
    write_v50_store(store_path, manifest, {"height_4": _height_array(row_count=1)})

    unlineaged = [
        RowLineage(
            store_row=0,
            build_id="0.5.3.3368",
            map_name="Azeroth",
            tile_x=0,
            tile_y=0,
            source_group="azeroth:0",
            signal_actions={},  # height_4 never actually recorded
        )
    ]

    finalized = finalize_store(store_path, manifest, unlineaged)

    assert finalized.finalization_state is FinalizationState.INCOMPLETE


def test_finalize_store_report_names_the_specific_row_and_signal_that_stayed_incomplete(tmp_path: Path):
    # Regression for Spec 109 Phase 9: `finalize` used to report only the bare
    # `finalization_state=incomplete` with no way to tell *why* short of hand-rolling a diagnostic
    # script against the store -- this happened on a real build (a legitimate tile lacking a
    # required signal) and cost real debugging time. `finalize_store_report` must name the exact
    # signal and row so that reason is visible immediately.
    manifest = _manifest(row_count=2)
    store_path = tmp_path / "store.zarr"
    write_v50_store(store_path, manifest, {"height_4": _height_array(row_count=2)})

    lineages = _lineages(row_count=2)
    lineages[1] = RowLineage(
        store_row=1,
        build_id="0.5.3.3368",
        map_name="Azeroth",
        tile_x=1,
        tile_y=1,
        source_group="azeroth:1",
        signal_actions={},  # height_4 missing for this one row only
    )

    report = finalize_store_report(store_path, manifest, lineages)

    assert report.manifest.finalization_state is FinalizationState.INCOMPLETE
    assert len(report.mismatches) == 1
    assert "height_4" in report.mismatches[0]
    assert "1 row(s)" in report.mismatches[0]
    assert "[1]" in report.mismatches[0]


def test_finalize_store_report_has_no_mismatches_when_complete(tmp_path: Path):
    manifest = _manifest()
    store_path = tmp_path / "store.zarr"
    write_v50_store(store_path, manifest, {"height_4": _height_array()})

    report = finalize_store_report(store_path, manifest, _lineages())

    assert report.manifest.finalization_state is FinalizationState.COMPLETE
    assert report.mismatches == ()


class TestCurriculumManifest:
    def test_a_valid_curriculum_contains_no_array_payloads_only_references(self):
        rows = [
            CurriculumRowRef(store_id=_HASH_A, row_id=0, source_group="azeroth:0", split="train"),
            CurriculumRowRef(store_id=_HASH_A, row_id=1, source_group="azeroth:1", split="val"),
        ]

        manifest = build_curriculum(
            release="v50.1", rows=rows, selection_reason="height-regime stratified sample", policy_identity=_HASH_B
        )

        payload = manifest.to_dict()
        assert payload["store_ids"] == [_HASH_A]
        assert all(set(row.keys()) == {"store_id", "row_id", "source_group", "split"} for row in payload["rows"])

    def test_rejects_duplicate_row_selection(self):
        rows = [
            CurriculumRowRef(store_id=_HASH_A, row_id=0, source_group="azeroth:0", split="train"),
            CurriculumRowRef(store_id=_HASH_A, row_id=0, source_group="azeroth:0", split="val"),
        ]

        with pytest.raises(ValueError, match="duplicate"):
            build_curriculum(release="v50.1", rows=rows, selection_reason="x", policy_identity=_HASH_B)

    def test_rejects_a_source_group_split_across_train_and_val(self):
        # Same time/color-variant source group must stay in one partition (leak check reused from
        # the existing prefab_curation helper).
        rows = [
            CurriculumRowRef(store_id=_HASH_A, row_id=0, source_group="shared-group", split="train"),
            CurriculumRowRef(store_id=_HASH_A, row_id=1, source_group="shared-group", split="val"),
        ]

        with pytest.raises(ValueError, match="leakage"):
            build_curriculum(release="v50.1", rows=rows, selection_reason="x", policy_identity=_HASH_B)

    def test_manifest_id_is_deterministic_for_the_same_selection(self):
        rows = [CurriculumRowRef(store_id=_HASH_A, row_id=0, source_group="azeroth:0", split="train")]

        first = build_curriculum(release="v50.1", rows=rows, selection_reason="x", policy_identity=_HASH_B)
        second = build_curriculum(release="v50.1", rows=rows, selection_reason="x", policy_identity=_HASH_B)

        assert first.manifest_id == second.manifest_id

    def test_rejects_a_non_v50_release(self):
        rows = [CurriculumRowRef(store_id=_HASH_A, row_id=0, source_group="azeroth:0", split="train")]
        with pytest.raises(ValueError, match="v50.N"):
            build_curriculum(release="v8", rows=rows, selection_reason="x", policy_identity=_HASH_B)
