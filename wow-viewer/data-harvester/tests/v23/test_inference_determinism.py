from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

import infer_v23_height  # noqa: E402
import train_v23_height  # noqa: E402
from tests.v23.support import make_synthetic_v22_store, write_model_config_json  # noqa: E402

pytestmark = pytest.mark.v23


def test_infer_v23_height_is_seed_invariant_under_deterministic_mode(tmp_path: Path) -> None:
    dataset_dir = tmp_path / "datasets"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    build = "3_3_5_12340"
    make_synthetic_v22_store(dataset_dir / f"{build}.zarr", build=build, tile_count=4)
    config_json = write_model_config_json(tmp_path / "tiny_config.json")
    run_dir = tmp_path / "train_run"

    train_v23_height.main(
        [
            "--dataset-dir",
            str(dataset_dir),
            "--builds",
            build,
            "--epochs",
            "1",
            "--train-max-tiles",
            "3",
            "--val-max-tiles",
            "1",
            "--device",
            "cpu",
            "--deterministic",
            "--run-name",
            "infer_smoke",
            "--output-dir",
            str(run_dir),
            "--model-config-json",
            str(config_json),
            "--no-pretrained",
        ]
    )

    checkpoint = run_dir / "checkpoints" / "v23_height_last.pt"
    out_a = tmp_path / "infer_a"
    out_b = tmp_path / "infer_b"
    infer_v23_height.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--v22-store",
            str(dataset_dir / f"{build}.zarr"),
            "--build",
            build,
            "--tiles",
            "0",
            "--output-dir",
            str(out_a),
            "--deterministic",
            "--seed",
            "42",
            "--device",
            "cpu",
        ]
    )
    infer_v23_height.main(
        [
            "--checkpoint",
            str(checkpoint),
            "--v22-store",
            str(dataset_dir / f"{build}.zarr"),
            "--build",
            build,
            "--tiles",
            "0",
            "--output-dir",
            str(out_b),
            "--deterministic",
            "--seed",
            "12345",
            "--device",
            "cpu",
        ]
    )

    pred_a = np.load(out_a / "prediction.npz")["metric_height"]
    pred_b = np.load(out_b / "prediction.npz")["metric_height"]
    assert np.array_equal(pred_a, pred_b)
