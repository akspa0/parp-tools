"""Tests for spec 077 height-only prior dataset + training script."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn.functional as F
import zarr
import zarr.codecs
import zarr.storage

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
_SCRIPTS_DIR = _REPO_ROOT / "scripts"
for _entry in (_REPO_ROOT, _SRC_DIR, _SCRIPTS_DIR):
    _entry_str = str(_entry)
    if _entry_str not in sys.path:
        sys.path.insert(0, _entry_str)

from harvester.height_only_prior_dataset import (  # noqa: E402
    HeightOnlyPriorDataset,
    dataset_summary,
)
import train_height_only_prior  # noqa: E402
from train_height_only_prior import (  # noqa: E402
    _gradient_magnitude_257,
    _masked_mean,
    _multi_scale_l1,
    compute_height_loss,
)

CODEC = zarr.codecs.BloscCodec(cname="zstd", clevel=1, shuffle="bitshuffle")


def _make_prior_store(path: Path, n_tiles: int = 3) -> None:
    if path.exists():
        import shutil
        shutil.rmtree(path)
    store = zarr.storage.LocalStore(str(path), read_only=False)
    root = zarr.group(store=store)
    prior = np.zeros((n_tiles, 256, 256, 5), dtype=np.uint8)
    mask = np.zeros((n_tiles, 256, 256), dtype=np.uint8)
    for i in range(n_tiles):
        prior[i, :, :, 0] = (i + 1) * 30
        prior[i, :, :, 1] = 50
        prior[i, :, :, 2] = 200
        prior[i, :, :, 3] = 0
        prior[i, :, :, 4] = 255
        if i == 1:
            mask[i, 60:120, 60:120] = 1
            prior[i, :, :, 3] = (mask[i] * 255).astype(np.uint8)
    root.create_array("processed_minimap_prior_256", data=prior, chunks=(n_tiles, 256, 256, 5), compressors=CODEC)
    root.create_array("teacher_object_mask_256", data=mask, chunks=(n_tiles, 256, 256), compressors=CODEC)
    root.attrs.update({"schema": "spec-077-teacher-prior", "build": "test_build"})

    import pyarrow as pa
    table = pa.table({
        "build": ["test_build"] * n_tiles,
        "map_name": ["Test", "Test", "Test"],
        "map": ["Test", "Test", "Test"],
        "tile_id": list(range(n_tiles)),
        "tile_x": [0, 1, 2],
        "tile_y": [0, 1, 2],
        "raw_minimap_key": [f"raw_minimap_rgb_256/{i}" for i in range(n_tiles)],
        "teacher_object_mask_key": [f"teacher_object_mask_256/{i}" for i in range(n_tiles)],
        "teacher_object_confidence_key": [f"teacher_object_confidence_256/{i}" for i in range(n_tiles)],
        "processed_prior_key": [f"processed_minimap_prior_256/{i}" for i in range(n_tiles)],
        "has_teacher_objects": [False, True, False],
        "teacher_object_cov": [0.0, 0.0625, 0.0],
        "filtered_mask_source": ["none", "object_filtered_mask", "none"],
    })
    pq.write_table(table, str(path / "tiles.parquet"))


def _make_v18_store(path: Path, n_tiles: int = 3) -> None:
    if path.exists():
        import shutil
        shutil.rmtree(path)
    store = zarr.storage.LocalStore(str(path), read_only=False)
    root = zarr.group(store=store)
    height = np.linspace(100.0, 200.0, 257 * 257, dtype=np.float32).reshape(n_tiles, 257, 257)
    filtered = np.zeros((n_tiles, 257, 257), dtype=np.float32)
    filtered[1, 60:120, 60:120] = 1.0
    root.create_array("height_257", data=height, chunks=(n_tiles, 257, 257), compressors=CODEC)
    root.create_array("object_filtered_mask", data=filtered, chunks=(n_tiles, 257, 257), compressors=CODEC)


# --- HeightOnlyPriorDataset ------------------------------------------------

def test_dataset_returns_documented_sample_contract() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=3)
        _make_v18_store(v18_path, n_tiles=3)
        ds = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, height_norm=False)
        assert len(ds) == 3
        sample = ds[0]
        assert sample["input_prior"].shape == (5, 256, 256)
        assert sample["height_257"].shape == (1, 257, 257)
        assert sample["weight_257"].shape == (1, 257, 257)
        assert sample["meta_build"] == "test_build"
        assert sample["meta_map"] == "Test"
        assert sample["meta_tile_id"] == 0


def test_dataset_weight_zeros_out_filtered_pixels() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=3)
        _make_v18_store(v18_path, n_tiles=3)
        ds = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, height_norm=False)
        sample = ds[1]
        weight = sample["weight_257"][0].numpy()
        # tile 1 has filtered mask at [60:120, 60:120]; weight must be 0 there.
        assert float(weight[80, 80]) == 0.0
        assert float(weight[0, 0]) > 0.0


def test_dataset_height_normalization_zeros_mean() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=1)
        _make_v18_store(v18_path, n_tiles=1)
        ds = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, height_norm=True)
        sample = ds[0]
        h = sample["height_257"][0].numpy()
        assert abs(h.mean()) < 1e-4
        assert h.std() > 0.0


def test_dataset_handles_missing_v18() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        _make_prior_store(prior_path, n_tiles=2)
        ds = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=None, height_norm=False)
        sample = ds[0]
        # Without V18, height and weight are zeros; only the prior carries data.
        assert sample["height_257"].abs().sum() == 0
        assert sample["weight_257"].sum() == 0  # no weight signal without V18
        assert sample["input_prior"].shape == (5, 256, 256)


def test_dataset_summary_returns_minimal_keys() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        _make_prior_store(prior_path, n_tiles=2)
        summary = dataset_summary(prior_path)
        assert summary["build"] == "test_build"
        assert "processed_minimap_prior_256" in summary["arrays"]


# --- training script -------------------------------------------------------

def test_multi_scale_l1_returns_zero_for_perfect_prediction() -> None:
    pred = torch.zeros(1, 1, 257, 257)
    target = torch.zeros(1, 1, 257, 257)
    weight = torch.ones(1, 1, 257, 257)
    loss, metrics = _multi_scale_l1(pred, target, weight, ms_weight=0.2)
    assert loss.item() == 0.0
    assert "l1_257" in metrics
    assert "l1_16" in metrics
    # 5 scales
    assert sum(1 for k in metrics if k.startswith("l1_")) == 5


def test_multi_scale_l1_falls_back_to_single_scale_when_disabled() -> None:
    pred = torch.zeros(1, 1, 257, 257)
    target = torch.ones(1, 1, 257, 257) * 2.0
    weight = torch.ones(1, 1, 257, 257)
    loss, metrics = _multi_scale_l1(pred, target, weight, ms_weight=0.0)
    # Single 257-px masked L1
    assert abs(loss.item() - 2.0) < 1e-5
    assert metrics["height"] == pytest.approx(2.0, abs=1e-5)
    assert "l1_257" not in metrics


def test_compute_height_loss_aggregates_auxiliary_terms() -> None:
    pred = torch.zeros(1, 1, 257, 257)
    target = torch.ones(1, 1, 257, 257) * 3.0
    weight = torch.ones(1, 1, 257, 257)
    # No auxiliary terms: should equal multi-scale L1 only.
    loss, metrics = compute_height_loss(
        pred, target, weight, ms_weight=0.2, grad_weight=0.0, nc_weight=0.0
    )
    assert metrics["height"] == pytest.approx(loss.item(), abs=1e-5)
    assert "grad_loss" not in metrics
    assert "nc_loss" not in metrics
    # With auxiliary terms: must add to the base.
    loss_aux, metrics_aux = compute_height_loss(
        pred, target, weight, ms_weight=0.2, grad_weight=0.1, nc_weight=0.1
    )
    assert loss_aux.item() > loss.item()
    assert "grad_loss" in metrics_aux
    assert "nc_loss" in metrics_aux


def test_gradient_magnitude_257_zero_for_constant_input() -> None:
    x = torch.zeros(1, 1, 257, 257)
    g = _gradient_magnitude_257(x)
    assert torch.allclose(g, torch.zeros_like(g))


def test_train_height_only_prior_smoke_runs_and_writes_artifacts() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        out_dir = root / "out"
        _make_prior_store(prior_path, n_tiles=2)
        _make_v18_store(v18_path, n_tiles=2)
        exit_code = train_height_only_prior.main_with_args(
            [
                "--prior", str(prior_path),
                "--v18", str(v18_path),
                "--output-dir", str(out_dir),
                "--run-name", "smoke",
                "--steps", "2",
                "--val-steps", "1",
                "--batch-size", "1",
                "--learning-rate", "1e-3",
                "--device", "cpu",
                "--max-tiles", "2",
                "--no-amp",
                "--no-compile",
            ]
        )
        assert exit_code == 0, f"train exit code was {exit_code}"
        assert (out_dir / "smoke_metrics.json").exists()
        assert (out_dir / "smoke_model.pt").exists()
        assert (out_dir / "smoke_preview.png").exists()
        metrics = json.loads((out_dir / "smoke_metrics.json").read_text(encoding="utf-8"))
        assert metrics["schema"] == "spec-077-height-only-prior"
        assert metrics["step_count"] == 2
        assert len(metrics["train_metrics"]) == 2
        assert len(metrics["val_metrics"]) == 1
        assert metrics["model_parameter_count"] > 0
        # Optimizations surface
        assert "compile_status" in metrics
        assert "amp_enabled" in metrics
        assert "num_workers" in metrics
        # Throughput reported
        for entry in metrics["train_metrics"]:
            assert "tiles_per_sec" in entry
            assert "l1_257" in entry
            assert "l1_16" in entry


def test_train_resumes_from_checkpoint() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        out_dir = root / "out"
        _make_prior_store(prior_path, n_tiles=2)
        _make_v18_store(v18_path, n_tiles=2)
        # First run writes a checkpoint
        first_exit = train_height_only_prior.main_with_args(
            [
                "--prior", str(prior_path),
                "--v18", str(v18_path),
                "--output-dir", str(out_dir),
                "--run-name", "resume",
                "--steps", "1",
                "--val-steps", "0",
                "--batch-size", "1",
                "--learning-rate", "1e-3",
                "--device", "cpu",
                "--max-tiles", "2",
                "--no-amp",
                "--no-compile",
            ]
        )
        assert first_exit == 0
        ckpt = out_dir / "resume_model.pt"
        assert ckpt.exists()
        # Second run resumes from the same checkpoint
        second_exit = train_height_only_prior.main_with_args(
            [
                "--prior", str(prior_path),
                "--v18", str(v18_path),
                "--output-dir", str(out_dir),
                "--run-name", "resume",
                "--steps", "1",
                "--val-steps", "0",
                "--batch-size", "1",
                "--learning-rate", "1e-3",
                "--device", "cpu",
                "--max-tiles", "2",
                "--resume-checkpoint", str(ckpt),
                "--no-amp",
                "--no-compile",
            ]
        )
        assert second_exit == 0
        metrics = json.loads((out_dir / "resume_metrics.json").read_text(encoding="utf-8"))
        # Second run should have started past step 0
        assert metrics["train_metrics"][0]["step"] >= 1
