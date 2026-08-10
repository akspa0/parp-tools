from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from harvester.v60.clean_signal_corpus import validate_clean_signal_corpus
from harvester.v60.real_minimap_rgb import (
    build_real_minimap_rgb_corpus,
    real_minimap_rgb_build_plan,
)


def _write_store(root: Path) -> Path:
    root.mkdir()
    count = 5
    rgb = np.zeros((count, 256, 256, 3), dtype=np.uint8)
    heights = np.zeros((count, 257, 257), dtype=np.float32)
    rows = []
    maps = ["Azeroth", "Kalimdor", "Azeroth", "Kalimdor", "Azeroth"]
    for index, map_name in enumerate(maps):
        rgb[index, ..., 0] = 40 + index
        rgb[index, ..., 1] = 80 + index
        rgb[index, ..., 2] = 120 + index
        heights[index] = np.arange(257 * 257, dtype=np.float32).reshape(257, 257) + index
        rows.append(
            {
                "minimap_source": "synthetic" if index != 4 else "authored",
                "source_group_id": f"group-{index}",
                "build": "alpha",
                "map": map_name,
                "tile_x": index,
                "tile_y": 10 - index,
                "height_regime": "rolling",
            }
        )
    group = zarr.open_group(str(root), mode="w")
    group.create_array("minimap_rgb", data=rgb)
    group.create_array("height_257", data=heights)
    pq.write_table(pa.Table.from_pylist(rows), root / "index.parquet")
    return root


def test_raw_rgb_plan_reports_observable_source_and_gate_status(tmp_path: Path) -> None:
    store = _write_store(tmp_path / "store.zarr")

    plan = real_minimap_rgb_build_plan(
        store,
        source_filter="synthetic",
        validation_map="Azeroth",
    )

    assert plan["dry_run"] is True
    assert plan["source_row_count"] == 4
    assert plan["map_counts"] == {"Azeroth": 2, "Kalimdor": 2}
    assert plan["input_signal"] == "minimap_rgb"
    assert plan["preparation"] == "raw_luma_v1"
    assert plan["albedo_gate_status"] == "not_run"


def test_raw_rgb_builder_publishes_observation_without_target_input(tmp_path: Path) -> None:
    store = _write_store(tmp_path / "store.zarr")
    output = tmp_path / "corpus"

    result = build_real_minimap_rgb_corpus(
        store,
        output,
        source_filter="synthetic",
        validation_map="Azeroth",
    )

    assert result["dry_run"] is False
    report = validate_clean_signal_corpus(output)
    assert report["valid"] is True
    assert report["row_count"] == 4
    assert report["source_counts"] == {"real_minimap_diagnostic": 4}
    manifest = json.loads((output / "clean_signal_manifest.json").read_text(encoding="utf-8"))
    first = manifest["rows"][0]
    assert first["confidence_status"] == "absent_explicit"
    assert first["observation_provenance"]["source_signal"] == "minimap_rgb"
    assert first["observation_provenance"]["inference_target_reads"] == []
    with np.load(output / first["npz"], allow_pickle=False) as payload:
        index = first["observation_provenance"]["source_row_index"]
        expected = (0.2126 * (40 + index) + 0.7152 * (80 + index) + 0.0722 * (120 + index)) / 255.0
        assert np.allclose(payload["clean_observation_luma_256"], expected)
        assert np.all(payload["clean_observation_confidence_256"] == 0.0)
