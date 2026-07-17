"""Spec 109 T006: contract tests for ArtifactRecord, DatasetStoreManifest, DatasetSignal,
RowLineage, and the verification enums. These must fail before harvester.v50.contracts exists and
pass once it enforces the frozen schema (FR-001, FR-002, FR-005)."""

from __future__ import annotations

import pytest

from harvester.v50.contracts import (
    ArtifactRecord,
    DatasetSignal,
    DatasetStoreManifest,
    Disposition,
    FinalizationState,
    MigrationPolicy,
    ProofLevel,
    RowLineage,
    TrustState,
    UnavailableSignal,
    migration_policy_for_signal,
    require_liquid_source_provenance,
    require_metadata_release,
    validate_release,
)

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64


def _signal(name: str = "height_257", policy: MigrationPolicy = MigrationPolicy.COPY_IF_VERIFIED) -> DatasetSignal:
    return DatasetSignal(
        name=name,
        dtype="float32",
        row_shape=(257, 257),
        required=True,
        authoritative_source="wowviewer.core.io.adt_reader",
        content_identity=_HASH_B,
        coverage_count=42,
        migration_policy=policy,
    )


class TestArtifactRecord:
    def test_new_artifact_defaults_to_unverified_and_never_promotes_by_construction(self):
        # FR-001/FR-002: a name/label alone cannot establish trust. Every fresh record starts
        # unverified regardless of what path or kind it names.
        record = ArtifactRecord(
            artifact_id=_HASH_A,
            kind="dataset",
            resolved_path="/datasets/v50/v50.1/azeroth.zarr",
            observed_bytes=1024,
            content_identity=_HASH_B,
            owner="harvester.v50",
        )
        assert record.trust_state is TrustState.UNVERIFIED
        assert record.proof_level is ProofLevel.INVENTORY

    def test_rejects_malformed_identity_hashes(self):
        with pytest.raises(ValueError, match="sha256"):
            ArtifactRecord(
                artifact_id="not-a-hash",
                kind="dataset",
                resolved_path="/x",
                observed_bytes=0,
                content_identity=_HASH_B,
                owner="unknown",
            )

    def test_rejects_negative_observed_bytes(self):
        with pytest.raises(ValueError, match="observed_bytes"):
            ArtifactRecord(
                artifact_id=_HASH_A,
                kind="dataset",
                resolved_path="/x",
                observed_bytes=-1,
                content_identity=_HASH_B,
                owner="unknown",
            )


class TestDatasetStoreManifest:
    def test_valid_manifest_round_trips_to_the_frozen_schema_shape(self):
        manifest = DatasetStoreManifest(
            release="v50.1",
            store_id=_HASH_A,
            build_id="0.5.3.3368",
            producer_identity=_HASH_B,
            client_build_evidence_id=_HASH_C,
            index_identity=_HASH_A,
            row_count=100,
            signals=(_signal(),),
            row_lineage_identity=_HASH_B,
            finalization_state=FinalizationState.COMPLETE,
        )

        payload = manifest.to_dict()

        assert payload["model_family"] == "v50"
        assert payload["schema"] == "v50-complete-store-v1"
        assert payload["release"] == "v50.1"
        assert payload["finalization_state"] == "complete"
        assert payload["signals"][0]["migration_policy"] == "copy-if-verified"
        assert "unavailable_signals" not in payload  # schema: optional, omitted when empty

    def test_rejects_non_v50_release_string(self):
        with pytest.raises(ValueError, match="v50.N"):
            DatasetStoreManifest(
                release="v8",
                store_id=_HASH_A,
                build_id="0.5.3.3368",
                producer_identity=_HASH_B,
                client_build_evidence_id=_HASH_C,
                index_identity=_HASH_A,
                row_count=1,
                signals=(_signal(),),
                row_lineage_identity=_HASH_B,
                finalization_state=FinalizationState.COMPLETE,
            ).validate()

    def test_rejects_a_v50_labeled_manifest_with_zero_signals(self):
        # FR-002: a v50 label/attribute alone cannot establish trust -- a manifest claiming v50
        # with no declared signals is not a valid complete store no matter what it is named.
        with pytest.raises(ValueError, match="at least one signal"):
            DatasetStoreManifest(
                release="v50.1",
                store_id=_HASH_A,
                build_id="0.5.3.3368",
                producer_identity=_HASH_B,
                client_build_evidence_id=_HASH_C,
                index_identity=_HASH_A,
                row_count=0,
                signals=(),
                row_lineage_identity=_HASH_B,
                finalization_state=FinalizationState.COMPLETE,
            ).validate()

    def test_rejects_duplicate_signal_names(self):
        with pytest.raises(ValueError, match="duplicate signal names"):
            DatasetStoreManifest(
                release="v50.1",
                store_id=_HASH_A,
                build_id="0.5.3.3368",
                producer_identity=_HASH_B,
                client_build_evidence_id=_HASH_C,
                index_identity=_HASH_A,
                row_count=1,
                signals=(_signal("height_257"), _signal("height_257")),
                row_lineage_identity=_HASH_B,
                finalization_state=FinalizationState.COMPLETE,
            ).validate()

    def test_liquid_signals_are_fresh_only_and_reflected_in_a_real_manifest(self):
        liquid_signal = _signal("liquid_mask", MigrationPolicy(migration_policy_for_signal("liquid_mask")))
        assert liquid_signal.migration_policy is MigrationPolicy.FRESH_ONLY

        manifest = DatasetStoreManifest(
            release="v50.1",
            store_id=_HASH_A,
            build_id="0.5.3.3368",
            producer_identity=_HASH_B,
            client_build_evidence_id=_HASH_C,
            index_identity=_HASH_A,
            row_count=1,
            signals=(_signal(), liquid_signal),
            row_lineage_identity=_HASH_B,
            finalization_state=FinalizationState.INCOMPLETE,
            unavailable_signals=(UnavailableSignal(name="wl_liquid_mask", reason="no WL fallback in this tile"),),
        )
        payload = manifest.to_dict()
        assert payload["unavailable_signals"][0]["reason"] == "no WL fallback in this tile"


class TestRowLineage:
    def test_records_per_signal_action_and_hashes(self):
        lineage = RowLineage(
            store_row=0,
            build_id="0.5.3.3368",
            map_name="Azeroth",
            tile_x=32,
            tile_y=48,
            source_group="azeroth:32:48",
            signal_actions={"height_257": "copied", "liquid_mask": "freshly_extracted"},
            signal_source_hashes={"height_257": _HASH_A},
            signal_destination_hashes={"height_257": _HASH_A},
        )
        assert lineage.signal_actions["height_257"] == "copied"

    def test_rejects_an_invalid_signal_action(self):
        with pytest.raises(ValueError, match="invalid signal actions"):
            RowLineage(
                store_row=0,
                build_id="0.5.3.3368",
                map_name="Azeroth",
                tile_x=0,
                tile_y=0,
                source_group="azeroth:0:0",
                signal_actions={"height_257": "relabeled"},
            )

    def test_rejects_negative_store_row(self):
        with pytest.raises(ValueError, match="store_row"):
            RowLineage(
                store_row=-1,
                build_id="0.5.3.3368",
                map_name="Azeroth",
                tile_x=0,
                tile_y=0,
                source_group="azeroth:0:0",
                signal_actions={},
            )


class TestMigratedReleaseGates:
    """The v50_contract.py shim must still expose these identically (T012)."""

    def test_validate_release_and_disposition_enum_membership(self):
        assert validate_release("v50.1") == "v50.1"
        with pytest.raises(ValueError):
            validate_release("v8")
        assert Disposition.REMOVE_CANDIDATE.value == "remove-candidate"

    def test_require_metadata_release_rejects_cross_release(self):
        require_metadata_release({"model_family": "v50", "release": "v50.1"}, "v50.1", artifact="x")
        with pytest.raises(ValueError, match="not compatible"):
            require_metadata_release({"model_family": "v50", "release": "v50.1"}, "v50.2", artifact="x")

    def test_wl_liquid_provenance_gate_still_enforced(self):
        with pytest.raises(ValueError, match="re-extract"):
            require_liquid_source_provenance(["wl_liquid_mask"], artifact="historical shard")
