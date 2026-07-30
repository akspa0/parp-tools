"""Cross-language contract test for Spec 122's curation manifest.

The fixture Parquet files under ``tests/fixtures/spec122_curation_manifest/`` are written by the C#
side (``WowViewer.Core.Curation.Tests.CrossLanguageFixtureGeneratorTests.GenerateFixture_
ForPythonCrossLanguageReadTest``) -- regenerate them with:

    dotnet test wow-viewer/tests/WowViewer.Core.Curation.Tests -c Debug \\
      --filter FullyQualifiedName~GenerateFixture

This test proves the real contract: a manifest written by the C# writer is readable, with the exact
documented column names/dtypes, by ``pyarrow`` on the Python side (data-model.md's actual proof
requirement -- not just that each side round-trips its own output).
"""

from __future__ import annotations

import sys
from pathlib import Path

import pyarrow.compute as pc
import pyarrow.parquet as pq
import pytest

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from harvester.curation_store import (  # noqa: E402
    load_curation_findings,
    load_curation_manifest,
    resolve_curation_run_id,
)

_FIXTURE_STORE = Path(__file__).resolve().parent / "fixtures" / "spec122_curation_manifest"
_RUN_ID = "0_5_3_3368-fixture-20260730T000000000Z"


def _run_dir() -> Path:
    run_dir = _FIXTURE_STORE / "curation" / _RUN_ID
    if not run_dir.exists():
        raise FileNotFoundError(
            f"Fixture not found at {run_dir}. Regenerate it by running the C# fixture generator "
            "(see this module's docstring) before running this test."
        )
    return run_dir


def test_manifest_written_by_csharp_is_pyarrow_readable() -> None:
    table = pq.read_table(str(_run_dir() / "curation_manifest.parquet"))

    expected_columns = {
        "build", "map", "tile_x", "tile_y", "tile_id",
        "difficulty_bucket", "coverage_bucket", "lighting_bucket",
        "synthetic_fidelity_status", "synthetic_fidelity_score",
        "finding_count", "curation_run_id",
    }
    assert expected_columns.issubset(set(table.column_names))
    assert table.num_rows == 3

    rows = table.to_pylist()
    by_tile_id = {row["tile_id"]: row for row in rows}

    assert by_tile_id[0]["difficulty_bucket"] == "easy"
    assert by_tile_id[0]["coverage_bucket"] == "well_covered"
    assert by_tile_id[0]["synthetic_fidelity_status"] == "evaluated"
    assert by_tile_id[0]["synthetic_fidelity_score"] == pytest.approx(0.91, abs=1e-4)

    assert by_tile_id[1]["difficulty_bucket"] == "pathological"
    assert by_tile_id[1]["coverage_bucket"] == "blank"
    assert by_tile_id[1]["synthetic_fidelity_status"] == "not_evaluable"
    assert by_tile_id[1]["synthetic_fidelity_score"] is None
    assert by_tile_id[1]["finding_count"] == 2

    for row in rows:
        assert row["build"] == "alpha"
        assert row["map"] == "Kalimdor"
        assert row["curation_run_id"] == _RUN_ID


def test_findings_written_by_csharp_is_pyarrow_readable() -> None:
    table = pq.read_table(str(_run_dir() / "curation_findings.parquet"))

    expected_columns = {
        "build", "map", "tile_x", "tile_y", "tile_id",
        "category", "severity", "reason", "evaluability", "signal", "curation_run_id",
    }
    assert expected_columns.issubset(set(table.column_names))
    assert table.num_rows == 2

    rows = table.to_pylist()
    categories = {row["category"] for row in rows}
    assert categories == {"height_normal_mismatch", "non_finite_value"}
    for row in rows:
        assert row["tile_id"] == 1
        assert row["evaluability"] == "evaluated"
        assert row["severity"] == "high"


def test_run_record_json_matches_the_written_manifest() -> None:
    import json

    payload = json.loads((_run_dir() / "curation_run.json").read_text(encoding="utf-8"))
    assert payload["schema"] == "v50-curation-run-v1"
    assert payload["tile_count"] == 3
    assert payload["finding_counts"]["height_normal_mismatch"] == 1
    assert payload["finding_counts"]["non_finite_value"] == 1


def test_latest_pointer_resolves_to_the_fixture_run_id() -> None:
    pointer = (_FIXTURE_STORE / "curation" / "latest").read_text(encoding="utf-8").strip()
    assert pointer == _RUN_ID


# --- harvester.curation_store loader tests (User Story 2: FR-009) ---------------------------------


def test_load_curation_manifest_resolves_latest_by_default() -> None:
    table = load_curation_manifest(_FIXTURE_STORE)
    assert table.num_rows == 3
    assert resolve_curation_run_id(_FIXTURE_STORE) == _RUN_ID


def test_load_curation_manifest_accepts_an_explicit_run_id() -> None:
    table = load_curation_manifest(_FIXTURE_STORE, curation_run_id=_RUN_ID)
    assert table.num_rows == 3


def test_load_curation_manifest_missing_store_raises_a_clear_error() -> None:
    with pytest.raises(FileNotFoundError):
        load_curation_manifest(_FIXTURE_STORE.parent / "does_not_exist")


def test_querying_a_non_clean_bucket_is_the_same_operation_as_the_clean_bucket() -> None:
    """FR-009 / US2: filtering 'blank' vs 'well_covered' is the identical column-filter operation,
    with equal completeness -- there is no separate, harder-to-reach path for the bad bucket."""
    manifest = load_curation_manifest(_FIXTURE_STORE)

    clean = manifest.filter(pc.equal(manifest["coverage_bucket"], "well_covered"))
    blank = manifest.filter(pc.equal(manifest["coverage_bucket"], "blank"))

    assert clean.num_rows == 1
    assert blank.num_rows == 1
    # Same table, same method, same completeness -- neither is a degraded/unsupported path.
    assert set(clean.column_names) == set(blank.column_names) == set(manifest.column_names)


def test_querying_a_mismatch_finding_category_returns_full_completeness() -> None:
    findings = load_curation_findings(_FIXTURE_STORE)
    mismatched = findings.filter(pc.equal(findings["category"], "height_normal_mismatch"))
    assert mismatched.num_rows == 1
    assert mismatched.to_pylist()[0]["tile_id"] == 1
