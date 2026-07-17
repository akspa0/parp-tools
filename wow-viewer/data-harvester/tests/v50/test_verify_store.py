"""Spec 109 (T017 coverage): the promotion gate. A store can only pass when its schema, row count,
required-signal truthfulness, content-integrity hashes, and partition leakage all check out --
never because of its name or declared attributes alone (FR-002/FR-005)."""

from __future__ import annotations

from harvester.v50.contracts import DatasetSignal, DatasetStoreManifest, FinalizationState, MigrationPolicy, RowLineage
from harvester.v50.verify_store import verify_store

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64
_HEIGHT_HASH = "sha256:" + "1" * 64


def _manifest(row_count: int = 2, required: bool = True) -> DatasetStoreManifest:
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
                name="height_257",
                dtype="float32",
                row_shape=(257, 257),
                required=required,
                authoritative_source="wowviewer.core.io.adt_reader",
                content_identity=_HEIGHT_HASH,
                coverage_count=row_count,
                migration_policy=MigrationPolicy.COPY_IF_VERIFIED,
            ),
        ),
        row_lineage_identity=_HASH_B,
        finalization_state=FinalizationState.COMPLETE,
    )


def _lineage(row: int, *, action: str = "copied", source_group: str | None = None, split_group: str | None = None) -> RowLineage:
    return RowLineage(
        store_row=row,
        build_id="0.5.3.3368",
        map_name="Azeroth",
        tile_x=row,
        tile_y=row,
        source_group=source_group or f"azeroth:{row}",
        signal_actions={"height_257": action},
        split_group=split_group,
    )


def test_a_fully_consistent_store_passes_every_check():
    manifest = _manifest(row_count=2)
    lineages = [_lineage(0), _lineage(1)]

    result = verify_store(manifest, lineages, observed_signal_hashes={"height_257": _HEIGHT_HASH})

    assert result.passed is True
    assert result.failure_reasons == ()


def test_row_count_mismatch_fails_even_if_everything_else_looks_fine():
    manifest = _manifest(row_count=5)  # manifest claims 5 rows
    lineages = [_lineage(0), _lineage(1)]  # only 2 actually observed

    result = verify_store(manifest, lineages, observed_signal_hashes={"height_257": _HEIGHT_HASH})

    assert result.passed is False
    assert any("row_count" in reason for reason in result.failure_reasons)


def test_content_hash_mismatch_catches_a_manifest_that_lies_about_its_own_content():
    manifest = _manifest(row_count=1)
    lineages = [_lineage(0)]

    result = verify_store(manifest, lineages, observed_signal_hashes={"height_257": "sha256:" + "9" * 64})

    assert result.passed is False
    assert any("content_integrity" in reason for reason in result.failure_reasons)


def test_a_required_signal_missing_its_action_in_one_row_fails_even_if_others_are_fine():
    manifest = _manifest(row_count=2)
    lineages = [
        _lineage(0),
        RowLineage(
            store_row=1,
            build_id="0.5.3.3368",
            map_name="Azeroth",
            tile_x=1,
            tile_y=1,
            source_group="azeroth:1",
            signal_actions={},  # height_257 missing entirely for this row
        ),
    ]

    result = verify_store(manifest, lineages, observed_signal_hashes={"height_257": _HEIGHT_HASH})

    assert result.passed is False
    assert any("required_signal_truthfulness" in reason for reason in result.failure_reasons)


def test_required_signal_marked_unavailable_per_row_but_not_store_wide_fails():
    manifest = _manifest(row_count=1)
    lineages = [_lineage(0, action="unavailable")]

    result = verify_store(manifest, lineages, observed_signal_hashes={"height_257": _HEIGHT_HASH})

    assert result.passed is False


def test_a_required_signal_declared_unavailable_store_wide_does_not_need_a_per_row_action():
    from harvester.v50.contracts import UnavailableSignal

    manifest = DatasetStoreManifest(
        release="v50.1",
        store_id=_HASH_A,
        build_id="0.5.3.3368",
        producer_identity=_HASH_B,
        client_build_evidence_id=_HASH_C,
        index_identity=_HASH_A,
        row_count=1,
        signals=(
            DatasetSignal(
                name="liquid_mask",
                dtype="float32",
                row_shape=(257, 257),
                required=True,
                authoritative_source="wowviewer.core.io.wl_reader",
                content_identity=_HEIGHT_HASH,
                coverage_count=0,
                migration_policy=MigrationPolicy.FRESH_ONLY,
            ),
        ),
        row_lineage_identity=_HASH_B,
        finalization_state=FinalizationState.COMPLETE,
        unavailable_signals=(UnavailableSignal(name="liquid_mask", reason="no WL fallback for this tile"),),
    )
    lineages = [
        RowLineage(
            store_row=0,
            build_id="0.5.3.3368",
            map_name="Azeroth",
            tile_x=0,
            tile_y=0,
            source_group="azeroth:0",
            signal_actions={},  # no action recorded, but that's fine: declared unavailable store-wide
        )
    ]

    result = verify_store(manifest, lineages, observed_signal_hashes={"liquid_mask": _HEIGHT_HASH})

    assert result.passed is True


def test_partition_leakage_across_train_and_val_fails_the_store():
    manifest = _manifest(row_count=2)
    lineages = [
        _lineage(0, source_group="shared-group", split_group="train"),
        _lineage(1, source_group="shared-group", split_group="val"),
    ]

    result = verify_store(manifest, lineages, observed_signal_hashes={"height_257": _HEIGHT_HASH})

    assert result.passed is False
    assert any("partition_leakage" in reason for reason in result.failure_reasons)


def test_no_declared_splits_skips_the_leakage_check_without_failing():
    manifest = _manifest(row_count=2)
    lineages = [_lineage(0), _lineage(1)]  # split_group=None on both

    result = verify_store(manifest, lineages, observed_signal_hashes={"height_257": _HEIGHT_HASH})

    assert result.passed is True
