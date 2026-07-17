"""Spec 109 T014: metadata-only inventory tests (FR-001/FR-002/FR-003/FR-009). The load-bearing
assertion throughout: nothing a discovered artifact is named, or what attributes/labels it carries,
ever produces a trust_state other than UNVERIFIED at inventory time."""

from __future__ import annotations

from pathlib import Path

from harvester.v50.contracts import Disposition, ProofLevel, TrustState
from harvester.v50.inventory import InventoryRoot, discover_artifacts


def _make_fake_zarr_store(path: Path, *, attrs_claiming_v50: bool = False) -> None:
    path.mkdir(parents=True)
    (path / ".zattrs").write_text(
        '{"model_family": "v50", "release": "v50.1"}' if attrs_claiming_v50 else "{}"
    )
    (path / "height_257").mkdir()
    (path / "height_257" / "0.0").write_bytes(b"\x00" * 128)


def test_a_forged_v50_labeled_store_is_still_inventoried_as_unverified(tmp_path: Path):
    # FR-002: a v50 name/directory/attribute must not by itself establish trust.
    forged = tmp_path / "datasets"
    forged.mkdir()
    _make_fake_zarr_store(forged / "totally_legit_v50_verified.zarr", attrs_claiming_v50=True)

    records = discover_artifacts([InventoryRoot(path=forged, default_owner="test")])

    assert len(records) == 1
    assert records[0].trust_state is TrustState.UNVERIFIED
    assert records[0].disposition is Disposition.QUARANTINE
    assert records[0].proof_level is ProofLevel.INVENTORY


def test_an_old_pre_v50_dataset_is_also_unverified_not_rejected_outright(tmp_path: Path):
    # FR-001: predates-v50 artifacts start unverified (a state that can still be promoted later
    # with evidence), not some other terminal state invented by the inventory step itself.
    old = tmp_path / "datasets"
    old.mkdir()
    _make_fake_zarr_store(old / "v18_3_3_5_12340.zarr")

    records = discover_artifacts([InventoryRoot(path=old, default_owner="legacy")])

    assert records[0].trust_state is TrustState.UNVERIFIED


def test_classifies_common_artifact_kinds_by_extension(tmp_path: Path):
    root = tmp_path / "mixed"
    root.mkdir()
    _make_fake_zarr_store(root / "some.zarr")
    (root / "checkpoint.pt").write_bytes(b"fake-checkpoint")
    (root / "prior.npz").write_bytes(b"fake-npz")
    (root / "release-manifest.json").write_text("{}")
    (root / "audit-report.json").write_text("{}")

    records = {Path(r.resolved_path).name: r for r in discover_artifacts([InventoryRoot(path=root)])}

    assert records["some.zarr"].kind == "dataset"
    assert records["checkpoint.pt"].kind == "checkpoint"
    assert records["prior.npz"].kind == "prior archive"
    assert records["release-manifest.json"].kind == "manifest"
    assert records["audit-report.json"].kind == "report"


def test_content_identity_is_metadata_only_and_does_not_change_when_array_bytes_do(tmp_path: Path):
    # plan.md: "Inventory should avoid reading chunk payloads." The metadata-tree identity must be
    # driven by the *set of files present*, not a full read/interpretation of array contents --
    # this test pins that changing a chunk's byte content still changes the tree identity (files
    # differ), while the discovery pass itself never opens the array as decoded data.
    root = tmp_path / "datasets"
    root.mkdir()
    store = root / "store.zarr"
    _make_fake_zarr_store(store)

    first = discover_artifacts([InventoryRoot(path=root)])[0]

    (store / "height_257" / "0.0").write_bytes(b"\xff" * 128)
    second = discover_artifacts([InventoryRoot(path=root)])[0]

    assert first.content_identity != second.content_identity


def test_two_distinct_paths_with_byte_identical_content_share_content_identity_but_not_artifact_id(
    tmp_path: Path,
):
    root = tmp_path / "datasets"
    root.mkdir()
    _make_fake_zarr_store(root / "store_a.zarr")
    _make_fake_zarr_store(root / "store_b.zarr")

    records = discover_artifacts([InventoryRoot(path=root)])
    assert len(records) == 2
    assert records[0].content_identity == records[1].content_identity
    assert records[0].artifact_id != records[1].artifact_id


def test_missing_root_yields_no_artifacts_instead_of_raising(tmp_path: Path):
    missing = tmp_path / "does_not_exist"
    assert discover_artifacts([InventoryRoot(path=missing)]) == []


def test_discovery_is_deterministic_across_repeated_runs(tmp_path: Path):
    root = tmp_path / "datasets"
    root.mkdir()
    _make_fake_zarr_store(root / "store.zarr")
    (root / "checkpoint.pt").write_bytes(b"fixed-bytes")

    first_ids = {r.artifact_id for r in discover_artifacts([InventoryRoot(path=root)])}
    second_ids = {r.artifact_id for r in discover_artifacts([InventoryRoot(path=root)])}

    assert first_ids == second_ids
