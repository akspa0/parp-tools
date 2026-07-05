from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

from harvester.v23.checkpoint import load_checkpoint

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import train_v23_height  # noqa: E402
from tests.v23.support import make_synthetic_v22_store, write_curation_manifest, write_model_config_json  # noqa: E402

pytestmark = pytest.mark.v23


def _state_dict_equal(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> bool:
    if left.keys() != right.keys():
        return False
    return all(torch.equal(left[key], right[key]) for key in left)


def test_train_v23_height_smoke_is_deterministic(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    build = "3_3_5_12340"
    make_synthetic_v22_store(dataset_dir / f"{build}.zarr", build=build, tile_count=6)
    config_json = write_model_config_json(tmp_path / "tiny_config.json")

    run_a = tmp_path / "run_a"
    train_v23_height.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--builds",
            build,
            "--epochs",
            "2",
            "--train-max-tiles",
            "4",
            "--val-max-tiles",
            "2",
            "--device",
            "cpu",
            "--deterministic",
            "--seed",
            "42",
            "--run-name",
            "smoke_a",
            "--output-dir",
            str(run_a),
            "--model-config-json",
            str(config_json),
            "--no-pretrained",
            "--bias-free-mask-ratio",
            "0.15",
        ]
    )

    ckpt_a = run_a / "checkpoints" / "v23_height_last.pt"
    preview_a = run_a / "val_preview_2" / "tile_0.png"
    assert ckpt_a.exists()
    assert preview_a.exists()

    payload_a = load_checkpoint(ckpt_a)
    assert payload_a.config["seed"] == 42
    assert "commit_sha" in payload_a.config
    assert payload_a.config["input_mode"] == "full"
    assert payload_a.config["gpct_weight"] == 0.0
    assert payload_a.config["bias_free_mask_ratio"] == 0.15

    run_b = tmp_path / "run_b"
    train_v23_height.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--builds",
            build,
            "--epochs",
            "2",
            "--train-max-tiles",
            "4",
            "--val-max-tiles",
            "2",
            "--device",
            "cpu",
            "--deterministic",
            "--seed",
            "12345",
            "--run-name",
            "smoke_b",
            "--output-dir",
            str(run_b),
            "--model-config-json",
            str(config_json),
            "--no-pretrained",
            "--bias-free-mask-ratio",
            "0.15",
        ]
    )

    payload_b = load_checkpoint(run_b / "checkpoints" / "v23_height_last.pt")
    assert _state_dict_equal(payload_a.model_state, payload_b.model_state)


def test_train_v23_height_records_requested_maps_and_logs_progress(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    build = "3_3_5_12340"
    make_synthetic_v22_store(dataset_dir / f"{build}.zarr", build=build, tile_count=8, maps=["Azeroth", "Northrend"])
    config_json = write_model_config_json(tmp_path / "tiny_config.json")
    run_dir = tmp_path / "run_maps"

    train_v23_height.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--builds",
            build,
            "--maps",
            "Northrend",
            "--epochs",
            "1",
            "--train-max-tiles",
            "2",
            "--val-max-tiles",
            "1",
            "--device",
            "cpu",
            "--deterministic",
            "--seed",
            "42",
            "--output-dir",
            str(run_dir),
            "--model-config-json",
            str(config_json),
            "--no-pretrained",
            "--log-interval",
            "1",
            "--autotune-batch-size",
            "--autotune-batch-candidates",
            "1",
            "2",
            "4",
        ]
    )

    payload = load_checkpoint(run_dir / "checkpoints" / "v23_height_last.pt")
    assert payload.config["maps"] == ["Northrend"]
    output = capsys.readouterr().out
    assert "[v23] run=" in output
    assert "autotune skipped reason=non_cuda_device" in output
    assert "loss_history path=" in output
    assert "phase=train status=start step=1 batch=1/2" in output
    assert "phase=train status=done step=1 batch=1/2" in output
    assert "phase=val status=done step=1 batch=1/1" in output
    assert "samples=" in output
    assert "pct=" in output
    assert "elapsed=" in output
    assert "eta=" in output
    assert "optimizer_step=" in output
    assert "loss=" in output
    assert "affine_loss=" in output
    assert "gradient_loss=" in output
    assert "val_loss=" in output
    assert "checkpoint last=" in output
    assert "metrics path=" in output

    loss_history = run_dir / "loss_history.jsonl"
    assert loss_history.exists()
    events = [json.loads(line) for line in loss_history.read_text(encoding="utf-8").splitlines()]
    assert any(event["type"] == "batch" and event["phase"] == "train" and "loss" in event for event in events)
    assert any(event["type"] == "batch" and event["phase"] == "val" and "loss" in event for event in events)
    assert any(event["type"] == "epoch" and "train_loss" in event and "val_loss" in event for event in events)
    autotune_evidence = json.loads((run_dir / "batch_autotune.json").read_text(encoding="utf-8"))
    assert autotune_evidence["skipped"] is True
    assert autotune_evidence["reason"] == "non_cuda_device"


def test_train_v23_height_honors_val_interval(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    build = "3_3_5_12340"
    make_synthetic_v22_store(dataset_dir / f"{build}.zarr", build=build, tile_count=6, maps=["Northrend"])
    config_json = write_model_config_json(tmp_path / "tiny_config.json")
    run_dir = tmp_path / "run_val_interval"

    train_v23_height.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--builds",
            build,
            "--maps",
            "Northrend",
            "--epochs",
            "2",
            "--train-max-tiles",
            "2",
            "--val-max-tiles",
            "1",
            "--val-interval",
            "2",
            "--device",
            "cpu",
            "--deterministic",
            "--seed",
            "42",
            "--output-dir",
            str(run_dir),
            "--model-config-json",
            str(config_json),
            "--no-pretrained",
            "--log-interval",
            "1",
        ]
    )

    output = capsys.readouterr().out
    assert "epoch=1/2 phase=val skipped val_interval=2" in output
    assert "epoch=2/2 phase=val status=done step=1 batch=1/1" in output
    events = [json.loads(line) for line in (run_dir / "loss_history.jsonl").read_text(encoding="utf-8").splitlines()]
    epoch_events = [event for event in events if event["type"] == "epoch"]
    assert epoch_events[0]["validation_skipped"] is True
    assert epoch_events[0]["val_loss"] is None
    assert epoch_events[1]["validation_skipped"] is False
    assert epoch_events[1]["val_loss"] is not None


def test_train_v23_height_uses_curation_manifest_for_validation_mismatch(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    build = "3_3_5_12340"
    make_synthetic_v22_store(dataset_dir / f"{build}.zarr", build=build, tile_count=6, maps=["Northrend"])
    manifest = write_curation_manifest(
        tmp_path / "kept_tiles.parquet",
        [
            {
                "build": build,
                "tile_id": idx,
                "keep": True,
                "quality_score": 0.5 + (idx * 0.01),
                "usefulness_score": 0.5 + (idx * 0.01),
                "difficulty_score": 0.2 + (idx * 0.1),
                "difficulty_bucket": "hard" if idx >= 4 else "medium",
                "difficulty_rank": 2 if idx >= 4 else 1,
                "score_terrain_validity": 0.9,
                "score_minimap_target_usefulness": 0.5 + (idx * 0.05),
                "score_painted_signal": 1.0,
                "liquid_cov": 0.0,
                "what_plate": False,
                "normal_edge_iou": 0.9 - (idx * 0.1),
                "normal_edge_f1": 0.9 - (idx * 0.1),
            }
            for idx in range(6)
        ],
    )

    train_dataset, val_dataset = train_v23_height._split_datasets(
        dataset_dir,
        [build],
        maps=["Northrend"],
        input_mode="full",
        tileset_prune_table=None,
        curation_manifest=str(manifest),
        curation_min_terrain_validity=0.20,
        curation_min_minimap_usefulness=0.10,
        curation_max_liquid_coverage=0.05,
        curation_reject_what_plate=True,
        train_max_tiles=3,
        val_max_tiles=2,
    )

    assert len(train_dataset) == 3
    assert len(val_dataset) == 2
    assert [int(val_dataset[idx]["tile_id"]) for idx in range(len(val_dataset))] == [5, 4]
