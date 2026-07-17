"""Spec 111 T008 coverage: the lighting-bucket distribution report reconciles every counted tile
into exactly one of bucket_counts / not_evaluated_count / low_confidence_count, and never folds a
tile missing the shading-match field entirely into "not evaluated"."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from harvester.spec111.lighting_buckets import build_report

REQUIRED_BUILD = "0.5.3.3368"


def _write_decoded_metadata(store_path: Path, rows: list[dict]) -> None:
    store_path.mkdir(parents=True, exist_ok=True)
    table = pa.table(
        {
            "tile_id": [row["tile_id"] for row in rows],
            "decoded_metadata_json": [json.dumps(row["metadata"]) for row in rows],
        }
    )
    pq.write_table(table, store_path / "decoded_metadata.parquet")


def _minimap_lighting(status: str, hours: float | None = None, build: str | None = REQUIRED_BUILD) -> dict:
    return {
        "shading_match_status": status,
        "shading_matched_time_of_day_hours": hours,
        "shading_match_build_fingerprint": build,
    }


def test_build_report_reconciles_matched_low_confidence_and_not_evaluated(tmp_path: Path):
    rows = [
        {"tile_id": 0, "metadata": {"map_name": "Kalimdor", "minimap_lighting": _minimap_lighting("matched", 6.5)}},
        {"tile_id": 1, "metadata": {"map_name": "Kalimdor", "minimap_lighting": _minimap_lighting("matched", 13.0)}},
        {"tile_id": 2, "metadata": {"map_name": "Kalimdor", "minimap_lighting": _minimap_lighting("low_confidence_ambiguous")}},
        {"tile_id": 3, "metadata": {"map_name": "Kalimdor", "minimap_lighting": _minimap_lighting("not_evaluated")}},
        {"tile_id": 4, "metadata": {"map_name": "Azeroth", "minimap_lighting": _minimap_lighting("matched", 6.9)}},
    ]
    _write_decoded_metadata(tmp_path, rows)

    report = build_report(tmp_path, only_map=None)

    assert report["overall"]["total_eligible_tiles"] == 4  # everything except the not_evaluated row
    assert report["overall"]["bucket_counts"] == {"06-09": 2, "12-15": 1}
    assert report["overall"]["low_confidence_count"] == 1
    assert report["overall"]["not_evaluated_count"] == 0  # excluded from eligible count, not double-counted

    map_names = {row["map_name"] for row in report["per_map"]}
    assert map_names == {"Kalimdor", "Azeroth"}


def test_build_report_filters_to_a_single_map(tmp_path: Path):
    rows = [
        {"tile_id": 0, "metadata": {"map_name": "Kalimdor", "minimap_lighting": _minimap_lighting("matched", 12.0)}},
        {"tile_id": 1, "metadata": {"map_name": "Azeroth", "minimap_lighting": _minimap_lighting("matched", 12.0)}},
    ]
    _write_decoded_metadata(tmp_path, rows)

    report = build_report(tmp_path, only_map="Kalimdor")

    assert report["overall"]["total_eligible_tiles"] == 1
    assert len(report["per_map"]) == 1
    assert report["per_map"][0]["map_name"] == "Kalimdor"


def test_build_report_counts_tiles_missing_the_shading_match_field_separately(tmp_path: Path):
    rows = [
        {"tile_id": 0, "metadata": {"map_name": "Kalimdor", "minimap_lighting": {"inference_status": "unlit_or_unclassified"}}},
        {"tile_id": 1, "metadata": {"map_name": "Kalimdor"}},
    ]
    _write_decoded_metadata(tmp_path, rows)

    report = build_report(tmp_path, only_map=None)

    # Neither row carries shading_match_status at all (a pre-spec-111 store, or a row whose
    # minimap_lighting analysis never ran the shading-match step). These must not be silently
    # folded into not_evaluated, which has a specific meaning (evaluated, and the answer was "no").
    assert report["tiles_without_shading_match_field"] == 2
    assert report["overall"]["total_eligible_tiles"] == 0


def test_build_report_raises_if_store_path_has_no_decoded_metadata(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        build_report(tmp_path, only_map=None)
