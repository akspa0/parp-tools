"""Unit tests for Spec 120 Sidecar Exporter (T011)."""

from __future__ import annotations

import json
from pathlib import Path

import pyarrow.parquet as pq

from harvester.spec120.sidecar_exporter import export_tile_sidecar


def test_export_tile_sidecar_json_and_parquet(tmp_path: Path) -> None:
    """Verify sidecar exporter produces valid JSON and Parquet files with correct schema."""
    mock_detections = [
        {
            "px": 128.0,
            "py": 128.0,
            "w_px": 32.0,
            "h_px": 32.0,
            "angle_deg": 45.0,
            "conf": 0.95,
            "coarse_class": "wmo",
            "asset_path": "World/wmo/Azeroth/Castle.wmo",
            "world_z": 10.0,
        },
        {
            "px": 64.0,
            "py": 64.0,
            "w_px": 16.0,
            "h_px": 16.0,
            "angle_deg": 0.0,
            "conf": 0.88,
            "coarse_class": "m2",
            "asset_path": "World/doodads/tree.m2",
            "world_z": 5.0,
        },
    ]

    json_path = tmp_path / "sidecar.json"
    parquet_path = tmp_path / "sidecar.parquet"

    items = export_tile_sidecar(
        detections=mock_detections,
        tile_x=32,
        tile_y=32,
        output_json_path=json_path,
        output_parquet_path=parquet_path,
    )

    assert len(items) == 2
    assert json_path.exists()
    assert parquet_path.exists()

    # JSON contents check
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    assert len(data) == 2
    assert data[0]["instance_id"] == 1
    assert data[0]["coarse_class"] == "wmo"
    assert data[0]["position_px"] == [128.0, 128.0]
    assert data[0]["world_position"] == [-266.67, -266.67, 10.0]

    # Parquet contents check
    table = pq.read_table(parquet_path)
    assert table.num_rows == 2
    assert "instance_id" in table.column_names
    assert "retrieved_asset" in table.column_names
