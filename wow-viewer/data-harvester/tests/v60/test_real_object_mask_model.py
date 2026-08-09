from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import zarr

from harvester.v60.real_object_mask_model import (
    RealObjectMaskNet,
    normalize_target_names,
    project_mask_257_to_256,
    real_object_mask_loss,
)
from harvester.v60.real_synthetic_pairs import load_pair_rows, pair_domain_report


def _trainer_module():
    path = Path(__file__).resolve().parents[2] / "scripts" / "v60_train_real_object_masks.py"
    spec = importlib.util.spec_from_file_location("v60_train_real_object_masks", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load trainer module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_project_mask_uses_any_corner_without_inventing_cells() -> None:
    mask = np.zeros((257, 257), dtype=np.float32)
    mask[1, 1] = 1.0
    projected = project_mask_257_to_256(mask)

    assert projected.shape == (256, 256)
    assert projected[0, 0] == 1.0
    assert float(projected.sum()) == 4.0


@pytest.mark.parametrize("targets", [("object_precise_mask",), ("object_mask",), ("object_precise_mask", "object_mask")])
def test_real_object_model_keeps_requested_targets_independent(targets: tuple[str, ...]) -> None:
    model = RealObjectMaskNet(in_channels=3, target_names=targets, base_channels=8)
    inputs = torch.rand(2, 3, 32, 32)
    target = torch.zeros(2, len(targets), 32, 32)
    target[:, :, 8:16, 8:16] = 1.0

    logits = model(inputs)
    loss, components = real_object_mask_loss(logits, target, targets)

    assert logits.shape == target.shape
    assert torch.isfinite(loss)
    assert set(components) == {f"{name}_loss" for name in targets}


def test_real_object_model_accepts_rgb_plus_edge() -> None:
    model = RealObjectMaskNet(in_channels=4, target_names=("object_mask",), base_channels=8)
    inputs = torch.rand(2, 4, 32, 32)
    target = torch.zeros(2, 1, 32, 32)
    target[:, :, 8:16, 8:16] = 1.0

    logits = model(inputs)

    assert logits.shape == target.shape
    assert torch.isfinite(real_object_mask_loss(logits, target, ("object_mask",))[0])


def test_target_names_reject_duplicate_or_unknown_signals() -> None:
    with pytest.raises(ValueError):
        normalize_target_names(("object_mask", "object_mask"))
    with pytest.raises(ValueError):
        normalize_target_names(("object_geometry_visible_mask_257",))


def test_real_loader_keeps_source_groups_out_of_both_splits(tmp_path: Path) -> None:
    rows = [
        {
            "map": "Kalimdor", "tile_x": 24, "tile_y": 40, "source_group_id": "g-train",
            "minimap_source": "authored", "split": "train", "source_store": "0_5_3_3368-Kalimdor.zarr",
        },
        {
            "map": "Azeroth", "tile_x": 29, "tile_y": 24, "source_group_id": "g-val",
            "minimap_source": "authored", "split": "val", "source_store": "0_5_3_3368-Azeroth.zarr",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "index.parquet")
    trainer = _trainer_module()

    selected = trainer._load_rows(tmp_path, "authored", "map_holdout", "Azeroth")

    assert [row.split for row in selected] == ["train", "val"]
    assert [row.source_group_id for row in selected] == ["g-train", "g-val"]

    rows.append({**rows[0], "map": "Azeroth", "source_group_id": "g-train"})
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "index.parquet")
    with pytest.raises(ValueError, match="source groups"):
        trainer._load_rows(tmp_path, "authored", "map_holdout", "Azeroth")


def test_pair_loader_keeps_same_tile_authored_and_synthetic_rows_together(tmp_path: Path) -> None:
    rows = [
        {
            "map": "Kalimdor", "tile_x": 24, "tile_y": 40, "source_group_id": "g-train",
            "minimap_source": "authored", "split": "train", "source_store": "0_5_3_3368-Kalimdor.zarr",
        },
        {
            "map": "Kalimdor", "tile_x": 24, "tile_y": 40, "source_group_id": "g-train",
            "minimap_source": "synthetic", "split": "train", "source_store": "0_5_3_3368-Kalimdor.zarr",
        },
        {
            "map": "Azeroth", "tile_x": 29, "tile_y": 24, "source_group_id": "g-val",
            "minimap_source": "authored", "split": "val", "source_store": "0_5_3_3368-Azeroth.zarr",
        },
        {
            "map": "Azeroth", "tile_x": 29, "tile_y": 24, "source_group_id": "g-val",
            "minimap_source": "synthetic", "split": "val", "source_store": "0_5_3_3368-Azeroth.zarr",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "index.parquet")

    pairs, selection = load_pair_rows(
        tmp_path,
        split_policy="manifest",
        val_map="Azeroth",
        validation_limit=1,
    )

    assert [(pair.authored_row_index, pair.synthetic_row_index, pair.split) for pair in pairs] == [
        (0, 1, "train"),
        (2, 3, "val"),
    ]
    assert selection["validation_pairs_after_limit"] == 1


def test_pair_domain_report_uses_flat_absdiff_for_optional_shadow_calibration(tmp_path: Path) -> None:
    rows = [
        {
            "map": "Kalimdor", "tile_x": 24, "tile_y": 40, "source_group_id": "g-train",
            "minimap_source": "authored", "split": "train", "source_store": "Kalimdor.zarr",
        },
        {
            "map": "Kalimdor", "tile_x": 24, "tile_y": 40, "source_group_id": "g-train",
            "minimap_source": "synthetic", "split": "train", "source_store": "Kalimdor.zarr",
        },
        {
            "map": "Azeroth", "tile_x": 29, "tile_y": 24, "source_group_id": "g-val",
            "minimap_source": "authored", "split": "val", "source_store": "Azeroth.zarr",
        },
        {
            "map": "Azeroth", "tile_x": 29, "tile_y": 24, "source_group_id": "g-val",
            "minimap_source": "synthetic", "split": "val", "source_store": "Azeroth.zarr",
        },
    ]
    pq.write_table(pa.Table.from_pylist(rows), tmp_path / "index.parquet")
    authored = np.zeros((4, 256, 256, 3), dtype=np.uint8)
    authored[2, :128, :, :] = 255
    group = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    group.create_array("minimap_rgb", data=authored)
    shadow = np.zeros((256, 256), dtype=np.float32)
    shadow[:128, :] = 1.0
    shadow_dir = tmp_path / "shadow"
    shadow_dir.mkdir()
    np.savez(
        shadow_dir / "Azeroth_29_24_harvest.npz",
        minimap_rgb_256=np.zeros((256, 256, 3), dtype=np.uint8),
        terrain_shadow_256=shadow,
    )

    pairs, _ = load_pair_rows(
        tmp_path,
        split_policy="manifest",
        val_map="Azeroth",
        validation_limit=1,
    )
    report = pair_domain_report(group, pairs, shadow_dir)

    assert report["diagnostic_role"] == "flat_synthetic_absdiff_and_optional_fixed_shadow_calibration"
    assert report["pair_count"] == 1
    assert report["mean_mae"] == pytest.approx(0.5)
    assert report["mean_fixed_shadow_vs_abs_diff_luma_correlation"] == pytest.approx(1.0)
    assert report["mean_fixed_shadow_vs_inverse_abs_diff_luma_correlation"] == pytest.approx(-1.0)
