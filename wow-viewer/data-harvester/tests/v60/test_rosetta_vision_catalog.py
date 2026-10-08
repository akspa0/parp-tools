"""Unit tests for RosettaVisionCatalog."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from harvester.v60.rosetta_vision_catalog import (
    OverheadExhibit,
    OverheadPlacementBox,
    RosettaVisionCatalog,
)


def test_load_and_query_catalog(tmp_path: Path):
    # Create sample exported catalog JSON
    sample_data = {
        "MapName": "RosettaAlpha",
        "TotalExhibits": 1,
        "TotalPlacements": 2,
        "Exhibits": [
            {
                "AssetId": "objlib_123456",
                "AssetPath": "World/Generic/Human/Passive Doodads/Crates/Crate01.m2",
                "Kind": "Model",
                "ExtentX": 4.0,
                "ExtentY": 4.0,
                "ExtentZ": 4.0,
                "FootprintArea": 16.0,
                "BoundsMin": {"X": -2.0, "Y": -2.0, "Z": 0.0},
                "BoundsMax": {"X": 2.0, "Y": 2.0, "Z": 4.0},
                "Placements": [
                    {
                        "TileX": 32,
                        "TileY": 32,
                        "pos": {"X": 100.0, "Y": 200.0, "Z": 10.0},
                        "CellU": 0.25,
                        "CellV": 0.5,
                        "CellSize": 133.33,
                        "NormalizedMinU": 0.2,
                        "NormalizedMinV": 0.45,
                        "NormalizedMaxU": 0.3,
                        "NormalizedMaxV": 0.55,
                        "PixelMinX": 51,
                        "PixelMinY": 115,
                        "PixelMaxX": 77,
                        "PixelMaxY": 141,
                        "UniqueId": 101,
                    },
                    {
                        "TileX": 33,
                        "TileY": 32,
                        "pos": {"X": 600.0, "Y": 200.0, "Z": 10.0},
                        "CellU": 0.1,
                        "CellV": 0.1,
                        "CellSize": 133.33,
                        "NormalizedMinU": 0.08,
                        "NormalizedMinV": 0.08,
                        "NormalizedMaxU": 0.12,
                        "NormalizedMaxV": 0.12,
                        "PixelMinX": 20,
                        "PixelMinY": 20,
                        "PixelMaxX": 31,
                        "PixelMaxY": 31,
                        "UniqueId": 102,
                    },
                ],
            }
        ],
    }

    json_file = tmp_path / "rosetta_catalog.json"
    json_file.write_text(json.dumps(sample_data), encoding="utf-8")

    catalog = RosettaVisionCatalog.load_json(json_file)
    assert catalog.map_name == "RosettaAlpha"
    assert len(catalog.exhibits) == 1

    # Query tile (32, 32)
    tile_placements = catalog.get_placements_for_tile(32, 32)
    assert len(tile_placements) == 1
    exhibit, box = tile_placements[0]
    assert exhibit.asset_id == "objlib_123456"
    assert box.tile_x == 32
    assert box.pixel_min_x == 51

    # ComfyUI box format
    comfy_boxes = catalog.get_tile_prompt_boxes(32, 32)
    assert len(comfy_boxes) == 1
    assert comfy_boxes[0]["x"] == 51
    assert comfy_boxes[0]["width"] == 26

    # Centroid points
    points = catalog.get_tile_point_prompts(32, 32)
    assert len(points) == 1
    assert points[0]["x"] == 64
    assert points[0]["y"] == 128
