from __future__ import annotations

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
from tests.v23.support import make_synthetic_v22_store, write_model_config_json  # noqa: E402

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
