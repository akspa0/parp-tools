from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest


def _load_packager():
    path = Path(__file__).parents[2] / "scripts" / "package_spec103_runpod.py"
    spec = importlib.util.spec_from_file_location("package_spec103_rights_tests", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_private_byod_does_not_turn_no_raw_files_into_a_clean_data_claim(tmp_path: Path) -> None:
    packager = _load_packager()

    rights = packager._resolve_bundle_rights(tmp_path, "private_byod")

    assert rights["contains_raw_game_client_files"] is False
    assert rights["contains_client_derived_training_data"] is True
    assert rights["rights_class"] == "private_byod"
    assert rights["legal_status"] == "not_a_legal_determination"
    assert '"scipy>=1.13"' in packager._BUNDLE_PYPROJECT
    assert "scipy>=1.13" in packager._BUNDLE_REQUIREMENTS.splitlines()


def test_clean_synthetic_fails_closed_without_verified_store_contract(tmp_path: Path) -> None:
    packager = _load_packager()

    with pytest.raises(ValueError, match="contract.json"):
        packager._resolve_bundle_rights(tmp_path, "clean_synthetic")


def test_clean_synthetic_preserves_operator_declared_rights_evidence(tmp_path: Path) -> None:
    packager = _load_packager()
    contract = {
        "rights_class": "clean_synthetic",
        "contains_raw_game_client_files": False,
        "contains_client_derived_training_data": False,
        "distribution_policy": "operator_declared_license_only",
        "source_license_summary": ["CC0-1.0"],
        "source_rights_assertion_summary": ["operator_authored"],
    }
    (tmp_path / "contract.json").write_text(json.dumps(contract), encoding="utf-8")

    rights = packager._resolve_bundle_rights(tmp_path, "clean_synthetic")

    assert rights["rights_class"] == "clean_synthetic"
    assert rights["source_license_summary"] == ["CC0-1.0"]
    assert len(rights["source_contract_sha256"]) == 64


def _write_identity_fixture(tmp_path: Path) -> tuple[Path, Path]:
    store = tmp_path / "source.zarr"
    store.mkdir()
    rows = [
        {
            "tile_id": 42,
            "build": "3.3.5",
            "map": "Kalimdor",
            "tile_x": 8,
            "tile_y": 9,
        }
    ]
    pq.write_table(pa.Table.from_pylist(rows), store / "index.parquet")
    curation = tmp_path / "curation"
    curation.mkdir()
    pq.write_table(
        pa.Table.from_pylist([{**rows[0], "keep": True, "partition": "train"}]),
        curation / "curation_manifest.parquet",
    )
    return store, curation


def test_manifest_store_identity_binds_rows_and_available_index_digest(
    tmp_path: Path,
) -> None:
    packager = _load_packager()
    store, curation = _write_identity_fixture(tmp_path)
    index_sha256 = hashlib.sha256((store / "index.parquet").read_bytes()).hexdigest()
    (curation / "curation_summary.json").write_text(
        json.dumps({"schema": "test", "index_sha256": index_sha256}),
        encoding="utf-8",
    )

    manifest, report = packager._validate_manifest_store_identity(
        store, curation / "curation_manifest.parquet"
    )

    assert manifest["tile_id"] == [42]
    assert report["status"] == "verified"
    assert report["index_digest_status"] == "matched"
    assert report["source_index_sha256"] == index_sha256

    bad = pq.read_table(curation / "curation_manifest.parquet").to_pylist()
    bad[0]["tile_x"] = 99
    pq.write_table(pa.Table.from_pylist(bad), curation / "curation_manifest.parquet")
    with pytest.raises(ValueError, match="row identity"):
        packager._validate_manifest_store_identity(
            store, curation / "curation_manifest.parquet"
        )


def test_manifest_store_identity_rejects_mismatched_declared_index_digest(
    tmp_path: Path,
) -> None:
    packager = _load_packager()
    store, curation = _write_identity_fixture(tmp_path)
    (curation / "curation_summary.json").write_text(
        json.dumps({"schema": "test", "index_sha256": "0" * 64}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="index digest"):
        packager._validate_manifest_store_identity(
            store, curation / "curation_manifest.parquet"
        )


def test_clean_package_skips_adjacent_evidence_without_index_binding(
    tmp_path: Path,
) -> None:
    packager = _load_packager()
    store, curation = _write_identity_fixture(tmp_path)
    _, identity_report = packager._validate_manifest_store_identity(
        store, curation / "curation_manifest.parquet"
    )
    (curation / "curation_summary.json").write_text(
        json.dumps({"schema": "unbound-summary"}), encoding="utf-8"
    )
    pq.write_table(
        pa.Table.from_pylist(
            [
                {
                    "tile_id": 999,
                    "build": "unknown",
                    "map": "Unrelated",
                    "tile_x": 1,
                    "tile_y": 2,
                }
            ]
        ),
        curation / "pattern_evidence_ledger.parquet",
    )

    report = packager._package_curation_evidence(
        curation,
        tmp_path / "bundle" / "data" / "curation",
        {42: 0},
        store_source=store,
        identity_report=identity_report,
        require_verified_adjacent=True,
    )

    assert set(report["source"]) == {"curation_manifest.parquet"}
    assert report["skipped"]["curation_summary.json"].startswith("clean_synthetic")
    assert report["skipped"]["pattern_evidence_ledger.parquet"].startswith(
        "clean_synthetic"
    )
