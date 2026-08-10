from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import zarr

from harvester.v60.clean_signal_corpus import validate_clean_signal_corpus
from harvester.v60.real_terrain_synthetic_zarr import (
    build_zarr_real_terrain_synthetic_corpus,
    zarr_real_terrain_synthetic_build_plan,
)


def _write_store(root: Path) -> Path:
    root.mkdir()
    count = 5
    shadows = np.zeros((count, 256, 256), dtype=np.float32)
    heights = np.zeros((count, 257, 257), dtype=np.float32)
    rows = []
    maps = ["Azeroth", "Kalimdor", "Azeroth", "Kalimdor", "Azeroth"]
    for index, map_name in enumerate(maps):
        shadows[index] = 0.2 + index * 0.1
        heights[index] = np.arange(257 * 257, dtype=np.float32).reshape(257, 257) + index
        rows.append(
            {
                "minimap_source": "synthetic" if index != 4 else "authored",
                "source_group_id": f"group-{index}",
                "build": "0_5_3_3368",
                "map": map_name,
                "tile_x": index,
                "tile_y": 10 - index,
                "height_regime": "rolling",
            }
        )
    group = zarr.open_group(str(root), mode="w")
    group.create_array("terrain_shadow_256", data=shadows)
    group.create_array("height_257", data=heights)
    pq.write_table(pa.Table.from_pylist(rows), root / "index.parquet")
    return root


def test_zarr_bridge_plan_preserves_map_counts_and_source_filter(tmp_path: Path) -> None:
    store = _write_store(tmp_path / "store.zarr")

    plan = zarr_real_terrain_synthetic_build_plan(store, validation_map="Azeroth")

    assert plan["dry_run"] is True
    assert plan["source_row_count"] == 4
    assert plan["map_counts"] == {"Azeroth": 2, "Kalimdor": 2}
    assert plan["train_row_count"] == 2
    assert plan["validation_row_count"] == 2


def test_zarr_bridge_uses_original_zarr_indices_and_publishes_valid_corpus(tmp_path: Path) -> None:
    store = _write_store(tmp_path / "store.zarr")
    output = tmp_path / "corpus"

    result = build_zarr_real_terrain_synthetic_corpus(
        store,
        output,
        validation_map="Azeroth",
    )

    assert result["dry_run"] is False
    report = validate_clean_signal_corpus(output)
    assert report["valid"] is True
    assert report["row_count"] == 4
    assert report["split_counts"] == {"train": 2, "validation": 2}
    assert report["source_counts"] == {"real_terrain_synthetic": 4}

    manifest = json.loads((output / "clean_signal_manifest.json").read_text(encoding="utf-8"))
    source_indices = {row["map"]: [] for row in manifest["rows"]}
    for row in manifest["rows"]:
        source_indices[row["map"]].append(row["observation_provenance"]["source_row_index"])
    assert source_indices["Azeroth"] == [2, 0]
    assert source_indices["Kalimdor"] == [3, 1]
