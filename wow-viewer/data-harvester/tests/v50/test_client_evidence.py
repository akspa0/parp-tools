"""Spec 109 T006 (client-evidence half): configurable client-library/build fingerprinting without
hardcoded paths. Every path/filename this test exercises is fixture-local -- no real client root
or filename convention is assumed."""

from __future__ import annotations

from pathlib import Path

import pytest

from harvester.v50.client_evidence import RESULT_FAIL, RESULT_PASS, collect_client_build_evidence


def _make_fixture_client(root: Path, *, with_executable: bool = True) -> None:
    (root / "Data").mkdir(parents=True)
    (root / "Data" / "common.MPQ").write_bytes(b"fixture-archive-bytes")
    if with_executable:
        (root / "FixtureClient.exe").write_bytes(b"fixture-executable-bytes")


def test_collect_evidence_passes_when_required_paths_and_executable_are_present(tmp_path: Path):
    root = tmp_path / "client"
    _make_fixture_client(root)

    evidence = collect_client_build_evidence(
        root,
        client_library_id="fixture-library",
        build_id="0.5.3.3368",
        required_relative_paths=["Data"],
        reader_identity="wowviewer.core.io.alpha_wdt_reader",
        executable_candidates=["FixtureClient.exe", "Wow.exe"],
        archive_glob="Data/*.MPQ",
    )

    assert evidence.result == RESULT_PASS
    assert evidence.missing_paths == ()
    assert evidence.executable_relative_path == "FixtureClient.exe"
    assert evidence.executable_identity is not None
    assert evidence.root_argument == str(root)  # the actual run's root, never a hardcoded default


def test_collect_evidence_fails_closed_when_a_required_path_is_missing(tmp_path: Path):
    root = tmp_path / "incomplete-client"
    root.mkdir()
    # No "Data" directory at all.

    evidence = collect_client_build_evidence(
        root,
        client_library_id="fixture-library",
        build_id="0.5.3.3368",
        required_relative_paths=["Data"],
        reader_identity="wowviewer.core.io.alpha_wdt_reader",
    )

    assert evidence.result == RESULT_FAIL
    assert "Data" in evidence.missing_paths


def test_collect_evidence_fails_closed_when_no_executable_candidate_matches(tmp_path: Path):
    root = tmp_path / "client-no-exe"
    _make_fixture_client(root, with_executable=False)

    evidence = collect_client_build_evidence(
        root,
        client_library_id="fixture-library",
        build_id="0.5.3.3368",
        required_relative_paths=["Data"],
        reader_identity="wowviewer.core.io.alpha_wdt_reader",
        executable_candidates=["Wow.exe", "WoWClient.exe"],
    )

    assert evidence.result == RESULT_FAIL
    assert evidence.executable_identity is None


def test_collect_evidence_raises_only_when_the_root_itself_does_not_exist(tmp_path: Path):
    missing_root = tmp_path / "does-not-exist"
    with pytest.raises(FileNotFoundError):
        collect_client_build_evidence(
            missing_root,
            client_library_id="fixture-library",
            build_id="0.5.3.3368",
            required_relative_paths=["Data"],
            reader_identity="wowviewer.core.io.alpha_wdt_reader",
        )


def test_two_identical_fixture_clients_produce_the_same_archive_catalog_identity(tmp_path: Path):
    root_a = tmp_path / "client_a"
    root_b = tmp_path / "client_b"
    _make_fixture_client(root_a)
    _make_fixture_client(root_b)

    evidence_a = collect_client_build_evidence(
        root_a,
        client_library_id="fixture-library",
        build_id="0.5.3.3368",
        required_relative_paths=["Data"],
        reader_identity="wowviewer.core.io.alpha_wdt_reader",
        archive_glob="Data/*.MPQ",
    )
    evidence_b = collect_client_build_evidence(
        root_b,
        client_library_id="fixture-library",
        build_id="0.5.3.3368",
        required_relative_paths=["Data"],
        reader_identity="wowviewer.core.io.alpha_wdt_reader",
        archive_glob="Data/*.MPQ",
    )

    assert evidence_a.archive_catalog_identity == evidence_b.archive_catalog_identity
