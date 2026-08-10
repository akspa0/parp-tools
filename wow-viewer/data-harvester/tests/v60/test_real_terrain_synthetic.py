from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harvester.v60.clean_signal_corpus import validate_clean_signal_corpus
from harvester.v60.real_terrain_synthetic import (
    build_real_terrain_synthetic_corpus,
    real_terrain_synthetic_build_plan,
)


def _write_source_row(root: Path, tile_x: int, tile_y: int) -> None:
    y, x = np.mgrid[0:257, 0:257]
    height = (x * 0.25 + y * 0.5 + tile_x + tile_y).astype(np.float32)
    shadow = np.clip((x[:256, :256] + y[:256, :256]) / 768.0 + 0.2, 0.0, 1.0).astype(np.float32)
    metadata = {
        "tile_name": f"Azeroth_alpha-tile({tile_x},{tile_y})",
        "map_name": "Azeroth",
        "tile_x": tile_x,
        "tile_y": tile_y,
        "build_key": "alpha",
    }
    path = root / f"Azeroth_{tile_x}_{tile_y}_harvest.npz"
    np.savez(path, height_257=height, terrain_shadow_256=shadow, **{"metadata.json": json.dumps(metadata).encode()})


def test_real_terrain_synthetic_plan_is_no_write(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_source_row(inputs, 0, 0)
    _write_source_row(inputs, 1, 0)

    plan = real_terrain_synthetic_build_plan(inputs)

    assert plan["dry_run"] is True
    assert plan["source_kind"] == "real_terrain_synthetic"
    assert plan["source_npz_count"] == 2
    assert plan["families"] == ["alpha:Azeroth"]


def test_real_terrain_synthetic_builder_publishes_valid_corpus(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_source_row(inputs, 0, 0)
    _write_source_row(inputs, 1, 0)
    output = tmp_path / "clean"

    result = build_real_terrain_synthetic_corpus(inputs, output)

    assert result["dry_run"] is False
    assert result["row_count"] == 2
    report = validate_clean_signal_corpus(output)
    assert report["valid"] is True
    assert report["source_counts"] == {"real_terrain_synthetic": 2}
    manifest = json.loads((output / "clean_signal_manifest.json").read_text(encoding="utf-8"))
    assert {row["split"] for row in manifest["rows"]} == {"train", "validation"}
    assert all(row["observation_provenance"]["inference_target_reads"] == [] for row in manifest["rows"])


def test_real_terrain_synthetic_builder_requires_pair(tmp_path: Path) -> None:
    inputs = tmp_path / "inputs"
    inputs.mkdir()
    _write_source_row(inputs, 0, 0)
    output = tmp_path / "clean"

    with pytest.raises(ValueError, match="at least two rows"):
        build_real_terrain_synthetic_corpus(inputs, output)
