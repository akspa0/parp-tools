from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


def _load_builder(script_name: str):
    path = ROOT / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(f"test_{script_name[:-3]}", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _tile_blob(*, surface_quads: bool, above_terrain: bool, basic_type: bool) -> dict[str, np.ndarray]:
    signals = ["wl_liquid_mask", "wl_liquid_height"]
    if surface_quads:
        signals.append("wl_liquid_surface_quads_v1")
    if above_terrain:
        signals.append("wl_liquid_above_terrain_v1")
    if basic_type:
        signals.append("wl_liquid_basic_type_header_v1")
    return {
        "minimap_rgb_256": np.zeros((256, 256, 3), dtype=np.uint8),
        "height_257": np.zeros((257, 257), dtype=np.float32),
        "wl_liquid_mask": np.ones((257, 257), dtype=np.float32),
        "wl_liquid_height": np.full((257, 257), 7.5, dtype=np.float32),
        "metadata.json": np.frombuffer(
            json.dumps({"available_signals": signals}).encode("utf-8"), dtype=np.uint8
        ),
    }


@pytest.mark.parametrize("script_name", ["build_v16_dataset.py", "build_v18_dataset.py"])
def test_wl_fallback_requires_contiguous_visible_and_typed_provenance(script_name: str) -> None:
    builder = _load_builder(script_name)

    assert builder._process_tile_data(
        _tile_blob(surface_quads=False, above_terrain=False, basic_type=False)
    ) is None
    assert builder._process_tile_data(
        _tile_blob(surface_quads=True, above_terrain=False, basic_type=True)
    ) is None
    assert builder._process_tile_data(
        _tile_blob(surface_quads=True, above_terrain=True, basic_type=False)
    ) is None

    result = builder._process_tile_data(
        _tile_blob(surface_quads=True, above_terrain=True, basic_type=True)
    )
    assert result is not None
    arrays, signals = result
    assert signals["liquid_mask"] is True
    assert np.all(arrays["liquid_mask"] == 1.0)
    assert np.all(arrays["liquid_height"] == 7.5)
