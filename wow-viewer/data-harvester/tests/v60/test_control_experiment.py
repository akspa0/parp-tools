from __future__ import annotations

from pathlib import Path

import numpy as np

from harvester.v60.control_corpus import CONTROL_FAMILY_BUCKETS
from harvester.v60.control_experiment import (
    fixed_validation_rows,
    load_control_rows,
    select_training_rows,
    select_training_schedule,
    split_summary,
    tile_mean_baseline,
)


def _write_row(root: Path, row_id: str, family: str, split: str, value: float) -> dict:
    shadow = np.full((256, 256), value, dtype=np.float32)
    height = np.full((257, 257), value * 10, dtype=np.float32)
    name = f"{row_id}.npz"
    np.savez(root / name, terrain_shadow_256=shadow, height_257=height)
    return {
        "row_id": row_id,
        "control_family": family,
        "complexity_bucket": CONTROL_FAMILY_BUCKETS[family],
        "variant": 0,
        "split": split,
        "npz": name,
    }


def _write_manifest(root: Path, rows: list[dict]) -> None:
    (root / "control_manifest.json").write_text(
        __import__("json").dumps({"schema": "v60-control-corpus-v1", "row_count": len(rows), "rows": rows}),
        encoding="utf-8",
    )


def test_loader_keeps_manifest_family_holdout_and_selection_is_deterministic(tmp_path: Path) -> None:
    rows = [
        _write_row(tmp_path, "ridge-v00", "ridge", "train", 0.4),
        _write_row(tmp_path, "ridge-v01", "ridge", "train", 0.5),
        _write_row(tmp_path, "ridge-v02", "ridge", "train", 0.6),
        _write_row(tmp_path, "noise-v00", "noise", "validation", 0.7),
    ]
    _write_manifest(tmp_path, rows)
    _, loaded = load_control_rows(tmp_path)
    validation = fixed_validation_rows(loaded)
    selected_a = select_training_rows(loaded, 2, 6001)
    selected_b = select_training_rows(loaded, 2, 6001)

    assert [row.row_id for row in selected_a] == [row.row_id for row in selected_b]
    assert {row.control_family for row in validation} == {"noise"}
    assert split_summary(selected_a, validation)["family_overlap"] == []


def test_tile_mean_baseline_marks_flat_controls_ambiguous(tmp_path: Path) -> None:
    rows = [
        _write_row(tmp_path, "ridge-v00", "ridge", "train", 0.4),
        _write_row(tmp_path, "noise-v00", "noise", "validation", 0.7),
    ]
    _write_manifest(tmp_path, rows)
    _, loaded = load_control_rows(tmp_path)
    baseline = tile_mean_baseline(fixed_validation_rows(loaded))

    assert baseline["mae"] == 0.0
    assert baseline["ambiguous_rows"] == ["noise-v00"]


def test_training_schedule_is_nested(tmp_path: Path) -> None:
    rows = [
        _write_row(tmp_path, f"ridge-v0{index}", "ridge", "train", 0.4 + index / 10)
        for index in range(4)
    ]
    rows.append(_write_row(tmp_path, "noise-v00", "noise", "validation", 0.7))
    _write_manifest(tmp_path, rows)
    _, loaded = load_control_rows(tmp_path)

    schedule = select_training_schedule(loaded, [2, 3, 4], 6001)
    ids = {size: {row.row_id for row in selected} for size, selected in schedule.items()}
    assert ids[2] < ids[3] < ids[4]
    assert [row.row_id for row in select_training_rows(loaded, 2, 6001)] == sorted(ids[2])
