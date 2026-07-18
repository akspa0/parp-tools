"""Spec 112 T010: the coverage auditor must name every unexplained zero-coverage signal, classify
recorded unavailability by the reason vocabulary, and report exact 256/1024 minimap row parity."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import zarr

from harvester.v50.coverage_audit import audit_store


def _write_store(
    path: Path,
    *,
    rows: int = 4,
    with_1024_gap: bool = False,
    unexplained_zero: bool = False,
    era_record: bool = True,
) -> None:
    group = zarr.open_group(str(path), mode="w")
    rng = np.random.default_rng(7)

    minimap = rng.integers(1, 255, size=(rows, 4, 4, 3), dtype=np.uint8)
    minimap_1024 = rng.integers(1, 255, size=(rows, 8, 8, 3), dtype=np.uint8)
    if with_1024_gap:
        minimap_1024[1] = 0  # row 1 lost its 1024px synthesis
    height = rng.random((rows, 5, 5), dtype=np.float32)
    dead = np.zeros((rows, 4, 4), dtype=np.float32)

    group.create_array("minimap_rgb", data=minimap)
    group.create_array("minimap_rgb_1024", data=minimap_1024)
    group.create_array("height_257", data=height)
    signals = [
        {"name": "minimap_rgb"},
        {"name": "minimap_rgb_1024"},
        {"name": "height_257"},
    ]
    unavailable = []
    if unexplained_zero:
        group.create_array("mcnk_flags_16", data=dead)
        signals.append({"name": "mcnk_flags_16"})
    if era_record:
        unavailable.append({"name": "mccv_rgb", "reason": "era_unavailable: MCCV introduced WotLK+ (build 0_5_3_3368)"})

    group.attrs.update(
        {
            "schema": "v50-complete-store-v1",
            "build_id": "0_5_3_3368",
            "row_count": rows,
            "signals": signals,
            "unavailable_signals": unavailable,
        }
    )
    index = [{"tile_id": i, "build": "0_5_3_3368", "map": "Kalimdor", "tile_x": i, "tile_y": 0} for i in range(rows)]
    pq.write_table(pa.Table.from_pylist(index), path / "index.parquet")


def test_clean_store_passes_with_parity_and_era_classification(tmp_path: Path):
    _write_store(tmp_path / "s.zarr")

    report = audit_store(tmp_path / "s.zarr")

    by_name = {entry["name"]: entry for entry in report["signals"]}
    assert by_name["minimap_rgb"]["status"] == "populated"
    assert by_name["minimap_rgb"]["population_fraction"] == 1.0
    assert by_name["mccv_rgb"]["status"] == "era_unavailable"
    assert by_name["mccv_rgb"]["declared"] is False
    assert report["minimap_resolution_parity"]["parity"] is True
    assert report["map"] == "Kalimdor"


def test_1024_gap_breaks_parity_and_names_the_exact_rows(tmp_path: Path):
    _write_store(tmp_path / "s.zarr", with_1024_gap=True)

    report = audit_store(tmp_path / "s.zarr")

    parity = report["minimap_resolution_parity"]
    assert parity["parity"] is False
    assert parity["rows_only_in_256"] == [1]
    assert parity["rows_only_in_1024"] == []


def test_declared_all_zero_signal_without_reason_is_unexplained(tmp_path: Path):
    _write_store(tmp_path / "s.zarr", unexplained_zero=True)

    report = audit_store(tmp_path / "s.zarr")

    by_name = {entry["name"]: entry for entry in report["signals"]}
    assert by_name["mcnk_flags_16"]["status"] == "zero_coverage_unexplained"
    assert by_name["mcnk_flags_16"]["populated_rows"] == 0


def test_report_conforms_to_the_contract_schema(tmp_path: Path):
    jsonschema = pytest.importorskip("jsonschema")
    _write_store(tmp_path / "s.zarr")
    schema_path = (
        Path(__file__).parents[3] / "specs" / "112-v50-height-model" / "contracts" / "coverage-audit-report.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    report = audit_store(tmp_path / "s.zarr")

    jsonschema.validate(report, schema)
