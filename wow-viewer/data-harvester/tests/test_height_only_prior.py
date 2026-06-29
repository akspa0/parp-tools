"""Tests for spec 077 height-only prior dataset + training script."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import zarr
import zarr.codecs
import zarr.storage
from PIL import Image

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
from harvester.terrain_augment import ALL_TRANSFORMS  # noqa: E402
from harvester.v18_models import V18HeightModel  # noqa: E402
import train_height_only_prior  # noqa: E402
import build_albedo_dataset  # noqa: E402
from train_height_only_prior import (  # noqa: E402
    _gradient_magnitude_257,
    _masked_mean,
    _multi_scale_l1,
    _PANEL_LABEL_HEIGHT,
    _PANEL_SIZE,
    _save_deconstruction_preview,
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
        "map_name": ["Test"] * n_tiles,
        "map": ["Test"] * n_tiles,
        "tile_id": list(range(n_tiles)),
        "tile_x": list(range(n_tiles)),
        "tile_y": list(range(n_tiles)),
        "raw_minimap_key": [f"raw_minimap_rgb_256/{i}" for i in range(n_tiles)],
        "teacher_object_mask_key": [f"teacher_object_mask_256/{i}" for i in range(n_tiles)],
        "teacher_object_confidence_key": [f"teacher_object_confidence_256/{i}" for i in range(n_tiles)],
        "processed_prior_key": [f"processed_minimap_prior_256/{i}" for i in range(n_tiles)],
        "has_teacher_objects": [i == 1 for i in range(n_tiles)],
        "teacher_object_cov": [0.0625 if i == 1 else 0.0 for i in range(n_tiles)],
        "filtered_mask_source": ["object_filtered_mask" if i == 1 else "none" for i in range(n_tiles)],
    })
    pq.write_table(table, str(path / "tiles.parquet"))


def _make_v18_store(path: Path, n_tiles: int = 3) -> None:
    if path.exists():
        import shutil
        shutil.rmtree(path)
    store = zarr.storage.LocalStore(str(path), read_only=False)
    root = zarr.group(store=store)
    height = np.linspace(100.0, 200.0, n_tiles * 257 * 257, dtype=np.float32).reshape(n_tiles, 257, 257)
    filtered = np.zeros((n_tiles, 257, 257), dtype=np.float32)
    normals = np.zeros((n_tiles, 257, 257, 3), dtype=np.float32)
    normals[:, :, :, 2] = 1.0
    normal_mask = np.ones((n_tiles, 257, 257), dtype=np.float32)
    # MCAL alpha pack: (n_tiles, 256, 256, 4) float32 per-layer blend weights.
    # Tile 1 also varies MCLY texture IDs per 16x16 cell, proving albedo does
    # not collapse to the old fixed layer-0 cyan placeholder when alpha is flat.
    alpha = np.zeros((n_tiles, 256, 256, 4), dtype=np.float32)
    if n_tiles > 1:
        alpha[1, 0:128, 0:128, 0] = np.linspace(0.0, 0.8, 128 * 128, dtype=np.float32).reshape(128, 128)
    mcly_ids = np.full((n_tiles, 16, 16, 4), -1, dtype=np.int32)
    mcly_mask = np.zeros((n_tiles, 16, 16, 4), dtype=np.float32)
    for tile_idx in range(n_tiles):
        mcly_ids[tile_idx, :, :, 0] = 100 + tile_idx
        mcly_mask[tile_idx, :, :, 0] = 1.0
    if n_tiles > 1:
        yy16, xx16 = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
        mcly_ids[1, :, :, 0] = 200 + yy16 * 16 + xx16
    if n_tiles > 1:
        filtered[1, 60:120, 60:120] = 1.0
    root.create_array("height_257", data=height, chunks=(n_tiles, 257, 257), compressors=CODEC)
    root.create_array("object_filtered_mask", data=filtered, chunks=(n_tiles, 257, 257), compressors=CODEC)
    root.create_array("normal_xyz", data=normals, chunks=(n_tiles, 257, 257, 3), compressors=CODEC)
    root.create_array("normal_mask", data=normal_mask, chunks=(n_tiles, 257, 257), compressors=CODEC)
    root.create_array("alpha_256", data=alpha, chunks=(n_tiles, 256, 256, 4), compressors=CODEC)
    root.create_array("mcly_texture_ids", data=mcly_ids, chunks=(n_tiles, 16, 16, 4), compressors=CODEC)
    root.create_array("mcly_layer_mask", data=mcly_mask, chunks=(n_tiles, 16, 16, 4), compressors=CODEC)


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
        assert sample["raw_minimap_rgb"].shape == (3, 256, 256)
        assert sample["teacher_object_mask"].shape == (1, 256, 256)
        assert sample["teacher_object_confidence"].shape == (1, 256, 256)
        assert sample["height_257"].shape == (1, 257, 257)
        assert sample["normal_xyz"].shape == (3, 257, 257)
        assert sample["normal_mask"].shape == (1, 257, 257)
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


def test_dataset_tile_filter_preserves_original_tensor_row_mapping() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=3)
        _make_v18_store(v18_path, n_tiles=3)
        ds = HeightOnlyPriorDataset(
            prior_path=prior_path,
            v18_path=v18_path,
            tile_filter=[2],
            height_norm=False,
        )
        sample = ds[0]
        assert sample["meta_tile_id"] == 2
        assert sample["meta_prior_row"] == 2
        assert sample["meta_v18_row"] == 2
        # Prior channel 0 is set to (tile_index + 1) * 30 in the fixture.
        assert float(sample["input_prior"][0, 0, 0]) == pytest.approx(90.0 / 255.0)


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


# --- albedo channel --------------------------------------------------------


def test_dataset_albedo_off_by_default() -> None:
    """Without include_albedo, the sample dict has no albedo_rgb key."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=3)
        _make_v18_store(v18_path, n_tiles=3)
        ds = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path)
        sample = ds[0]
        assert "albedo_rgb" not in sample


def test_dataset_albedo_emits_3ch_256x256() -> None:
    """include_albedo=True produces albedo_rgb (3, 256, 256) float32 in [0, 1]."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=3)
        _make_v18_store(v18_path, n_tiles=3)
        ds = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, include_albedo=True)
        sample = ds[0]
        assert "albedo_rgb" in sample
        albedo = sample["albedo_rgb"]
        assert albedo.shape == (3, 256, 256)
        assert albedo.dtype == torch.float32
        assert float(albedo.min()) >= 0.0
        assert float(albedo.max()) <= 1.0


def test_dataset_albedo_tile_with_blend_differs_from_flat_tile() -> None:
    """Tile 1 (spatially varying alpha) must produce a different albedo than tile 0 (flat)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=3)
        _make_v18_store(v18_path, n_tiles=3)
        ds = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, include_albedo=True)
        albedo0 = ds[0]["albedo_rgb"]
        albedo1 = ds[1]["albedo_rgb"]
        # Tile 0 has all-zero alpha → pure layer-0 colour (spatially constant per channel).
        # Tile 1 has a blend gradient in the top-left quadrant → spatially varying.
        # Check spatial std per channel (not global std, which includes cross-channel variation).
        spatial_std0 = albedo0.std(dim=(1, 2))  # (3,) per-channel spatial std
        spatial_std1 = albedo1.std(dim=(1, 2))  # (3,) per-channel spatial std
        assert float(spatial_std0.max()) < 1e-4  # flat per channel
        assert float(spatial_std1.max()) > 1e-4  # varying in at least one channel


def test_dataset_albedo_zeros_without_alpha_array() -> None:
    """If alpha_256 is missing from the V18 store, albedo_rgb is zeros (not an error)."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=2)
        _make_v18_store(v18_path, n_tiles=2)
        # Remove the alpha_256 array to simulate a store without it.
        import shutil
        shutil.rmtree(v18_path / "alpha_256")
        ds = HeightOnlyPriorDataset(prior_path=prior_path, v18_path=v18_path, include_albedo=True)
        sample = ds[0]
        assert "albedo_rgb" in sample
        assert float(sample["albedo_rgb"].abs().sum()) == 0.0


def test_build_albedo_dataset_writes_precomputed_store() -> None:
    """The precompute CLI helper writes albedo_rgb_256 sidecars for training."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        v18_path = root / "v18.zarr"
        out_root = root / "albedo"
        _make_v18_store(v18_path, n_tiles=2)

        output_path = build_albedo_dataset.build_albedo_store(
            v18_path=v18_path,
            output_root=out_root,
            overwrite=False,
        )

        store = zarr.storage.LocalStore(str(output_path), read_only=True)
        zroot = zarr.open_group(store, mode="r")
        assert "albedo_rgb_256" in zroot
        assert zroot["albedo_rgb_256"].shape == (2, 256, 256, 3)
        assert zroot["albedo_rgb_256"].dtype == np.dtype("uint8")
        assert dict(zroot.attrs)["schema"] == "spec-077-albedo-guidance"
        assert dict(zroot.attrs)["compositor"] == "harvester.compositor.composite_texture_identity_albedo"
        tile1 = zroot["albedo_rgb_256"][1].astype(np.float32) / 255.0
        assert float(tile1.std(axis=(0, 1)).max()) > 1e-3
        assert (output_path / "tiles.parquet").exists()
        assert (output_path / "metadata.json").exists()


def test_dataset_prefers_precomputed_albedo_store() -> None:
    """include_albedo=True reads albedo_rgb_256 sidecar before lazy alpha compositing."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        albedo_root = root / "albedo"
        _make_prior_store(prior_path, n_tiles=2)
        _make_v18_store(v18_path, n_tiles=2)
        albedo_path = build_albedo_dataset.build_albedo_store(
            v18_path=v18_path,
            output_root=albedo_root,
            overwrite=False,
        )

        # Mutate V18 alpha after precompute. If the dataset is really using
        # the sidecar, the sample remains equal to the precomputed bytes.
        store = zarr.storage.LocalStore(str(v18_path), read_only=False)
        zroot = zarr.open_group(store, mode="a")
        zroot["alpha_256"][1] = np.ones((256, 256, 4), dtype=np.float32)

        ds = HeightOnlyPriorDataset(
            prior_path=prior_path,
            v18_path=v18_path,
            albedo_path=albedo_path,
            include_albedo=True,
        )
        sample = ds[1]
        precomputed = zarr.open_group(
            zarr.storage.LocalStore(str(albedo_path), read_only=True),
            mode="r",
        )["albedo_rgb_256"][1].astype(np.float32) / 255.0
        expected = torch.from_numpy(np.transpose(precomputed, (2, 0, 1)))
        assert torch.allclose(sample["albedo_rgb"], expected, atol=1.0 / 255.0)


def test_deconstruction_preview_includes_albedo_panel() -> None:
    """Validation previews must show the albedo signal when it is fed to the model."""
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "preview.png"
        batch = {
            "input_prior": torch.zeros(1, 5, 256, 256),
            "raw_minimap_rgb": torch.zeros(1, 3, 256, 256),
            "teacher_object_mask": torch.zeros(1, 1, 256, 256),
            "teacher_object_confidence": torch.ones(1, 1, 256, 256),
            "albedo_rgb": torch.ones(1, 3, 256, 256) * 0.5,
            "height_257": torch.zeros(1, 1, 257, 257),
            "weight_257": torch.ones(1, 1, 257, 257),
            "meta_tile_id": torch.tensor([7]),
        }
        pred = torch.zeros(1, 1, 257, 257)
        _save_deconstruction_preview(
            batch=batch,
            pred=pred,
            out_path=out_path,
            max_samples=1,
            expect_albedo=True,
        )
        with Image.open(out_path) as img:
            assert img.size == (_PANEL_SIZE * 9, _PANEL_SIZE + _PANEL_LABEL_HEIGHT)


def test_dataset_albedo_augmented_consistently() -> None:
    """Albedo is a plain image: explicit D4 augmentation transforms it like the prior."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=3)
        _make_v18_store(v18_path, n_tiles=3)
        ds = HeightOnlyPriorDataset(
            prior_path=prior_path, v18_path=v18_path,
            include_albedo=True, augment=True, augment_seed=42,
            augment_transforms=ALL_TRANSFORMS,
        )
        sample = ds[1]  # tile 1 has spatially varying albedo
        assert "albedo_rgb" in sample
        assert sample["albedo_rgb"].shape == (3, 256, 256)
        # The augmented albedo should still be in [0, 1] (flips/rotations preserve range).
        assert float(sample["albedo_rgb"].min()) >= 0.0
        assert float(sample["albedo_rgb"].max()) <= 1.0


def test_v18_height_model_accepts_6_channels() -> None:
    """V18HeightModel(in_channels=6) forward-passes a 6-channel batch."""
    model = V18HeightModel(in_channels=6)
    x = torch.randn(2, 6, 256, 256)
    out = model(x)
    assert out.shape == (2, 1, 257, 257)


def test_v18_height_model_default_3_channels_unchanged() -> None:
    """Default in_channels=3 still works (backward compat with existing checkpoints)."""
    model = V18HeightModel()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    assert out.shape == (1, 1, 257, 257)
    assert model.in_channels == 3
    assert model.norm == "batch"
    assert model.decoder_upsample == "bilinear"


def test_v18_height_model_group_norm_nearest_decoder_forward_passes() -> None:
    """Fresh anti-grid runs can opt into GroupNorm and nearest decoder upsampling."""
    model = V18HeightModel(in_channels=6, norm="group", decoder_upsample="nearest")
    x = torch.randn(1, 6, 256, 256)
    out = model(x)
    assert out.shape == (1, 1, 257, 257)
    assert model.norm == "group"
    assert model.decoder_upsample == "nearest"
    assert any(isinstance(module, nn.GroupNorm) for module in model.modules())
    assert not any(isinstance(module, nn.BatchNorm2d) for module in model.modules())


def test_train_height_only_prior_albedo_flag_widens_model() -> None:
    """--albedo flag produces a 6-channel model and writes albedo to metrics JSON."""
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        albedo_path = root / "albedo" / "v18.zarr"
        _make_prior_store(prior_path, n_tiles=3)
        _make_v18_store(v18_path, n_tiles=3)
        build_albedo_dataset.build_albedo_store(
            v18_path=v18_path,
            output_root=root / "albedo",
            overwrite=False,
        )
        out_dir = root / "out"
        exit_code = train_height_only_prior.main_with_args([
            "--prior", str(prior_path),
            "--v18", str(v18_path),
            "--albedo-path", str(albedo_path),
            "--output-dir", str(out_dir),
            "--run-name", "albedo_smoke",
            "--epochs", "1",
            "--steps", "2",
            "--val-steps", "1",
            "--batch-size", "2",
            "--albedo",
            "--model-norm", "group",
            "--decoder-upsample", "nearest",
            "--no-compile",
            "--no-amp",
            "--num-workers", "0",
        ])
        assert exit_code == 0
        metrics_path = out_dir / "albedo_smoke_metrics.json"
        assert metrics_path.exists()
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert metrics["albedo"] is True
        assert metrics["model_in_channels"] == 6
        assert metrics["model_norm"] == "group"
        assert metrics["decoder_upsample"] == "nearest"
        assert metrics["augment_policy"] == "shadow-safe"
        assert metrics["augment_transforms"] == ["identity"]
        assert metrics["albedo_paths"] == [str(albedo_path)]
        with Image.open(out_dir / "albedo_smoke_preview.png") as img:
            assert img.size[0] == _PANEL_SIZE * 9


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
    ramp = torch.linspace(0.0, 3.0, 257).view(1, 1, 1, 257)
    target = ramp.expand(1, 1, 257, 257).clone()
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


def test_compute_height_loss_can_use_normal_guidance() -> None:
    pred = torch.zeros(1, 1, 257, 257)
    target = torch.zeros(1, 1, 257, 257)
    weight = torch.ones(1, 1, 257, 257)
    target_normals = torch.zeros(1, 3, 257, 257)
    target_normals[:, 0, :, :] = 1.0
    normal_mask = torch.ones(1, 1, 257, 257)
    base, base_metrics = compute_height_loss(
        pred, target, weight,
        ms_weight=0.0, grad_weight=0.0, nc_weight=0.0,
    )
    guided, guided_metrics = compute_height_loss(
        pred, target, weight,
        ms_weight=0.0, grad_weight=0.0, nc_weight=0.0,
        normal_guidance_weight=0.5,
        target_normals=target_normals,
        normal_guidance_mask=normal_mask,
    )
    assert base.item() == pytest.approx(0.0)
    assert "normal_guidance_loss" not in base_metrics
    assert guided.item() > base.item()
    assert guided_metrics["normal_guidance_loss"] > 0.0


def test_compute_height_loss_can_use_hard_error_weighting() -> None:
    pred = torch.zeros(1, 1, 4, 4)
    target = torch.zeros(1, 1, 4, 4)
    target[:, :, 0, 0] = 10.0
    target[:, :, 1:, 1:] = 1.0
    weight = torch.ones(1, 1, 4, 4)
    base, base_metrics = compute_height_loss(
        pred, target, weight,
        ms_weight=0.0, grad_weight=0.0, nc_weight=0.0,
    )
    hard, hard_metrics = compute_height_loss(
        pred, target, weight,
        ms_weight=0.0, grad_weight=0.0, nc_weight=0.0,
        hard_error_weight=0.5,
        hard_error_power=1.0,
        hard_error_max_multiplier=4.0,
    )
    assert hard.item() > base.item()
    assert "hard_error_loss" in hard_metrics
    assert hard_metrics["hard_error_weight_max"] <= 4.0
    assert "hard_error_loss" not in base_metrics


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
        assert (out_dir / "smoke_latest.pt").exists()
        assert (out_dir / "smoke_best.pt").exists()
        assert (out_dir / "smoke_preview.png").exists()
        metrics = json.loads((out_dir / "smoke_metrics.json").read_text(encoding="utf-8"))
        assert metrics["schema"] == "spec-077-height-only-prior"
        assert metrics["step_count"] == 2
        assert metrics["global_step"] == 2
        assert metrics["epoch_count"] >= 1
        assert metrics["model_norm"] == "batch"
        assert metrics["decoder_upsample"] == "bilinear"
        assert metrics["steps_per_epoch"] >= 1
        assert metrics["latest_checkpoint_path"].endswith("smoke_latest.pt")
        assert metrics["best_checkpoint_path"].endswith("smoke_best.pt")
        assert len(metrics["train_metrics"]) == 2
        assert len(metrics["val_metrics"]) == metrics["epoch_count"]
        assert len(metrics["history"]) >= 1
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


def test_train_height_only_prior_autotune_defaults_skip_on_cpu() -> None:
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
                "--run-name", "autotune_cpu",
                "--steps", "1",
                "--val-steps", "0",
                "--batch-size", "1",
                "--learning-rate", "1e-3",
                "--device", "cpu",
                "--autotune-batch-size",
                "--target-vram-gb", "12",
                "--no-amp",
                "--no-compile",
            ]
        )
        assert exit_code == 0


def test_train_height_only_prior_accepts_multiple_sources() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_a = root / "prior_a.zarr"
        prior_b = root / "prior_b.zarr"
        v18_a = root / "v18_a.zarr"
        v18_b = root / "v18_b.zarr"
        out_dir = root / "out"
        _make_prior_store(prior_a, n_tiles=2)
        _make_prior_store(prior_b, n_tiles=2)
        _make_v18_store(v18_a, n_tiles=2)
        _make_v18_store(v18_b, n_tiles=2)
        exit_code = train_height_only_prior.main_with_args(
            [
                "--prior", str(prior_a), str(prior_b),
                "--v18", str(v18_a), str(v18_b),
                "--output-dir", str(out_dir),
                "--run-name", "multi",
                "--steps", "2",
                "--val-steps", "1",
                "--batch-size", "1",
                "--learning-rate", "1e-3",
                "--device", "cpu",
                "--max-tiles", "4",
                "--no-amp",
                "--no-compile",
            ]
        )
        assert exit_code == 0
        metrics = json.loads((out_dir / "multi_metrics.json").read_text(encoding="utf-8"))
        assert metrics["source_count"] == 2
        assert len(metrics["prior_paths"]) == 2
        assert len(metrics["v18_paths"]) == 2
        assert len(metrics["train_metrics"]) == 2


def test_train_height_only_prior_honors_curation_manifest() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        curation_dir = root / "curation"
        out_dir = root / "out"
        curation_dir.mkdir()
        _make_prior_store(prior_path, n_tiles=3)
        _make_v18_store(v18_path, n_tiles=3)
        kept = pa.table({"build": ["prior"], "tile_id": [2], "keep": [True]})
        pq.write_table(kept, str(curation_dir / "kept_tiles.parquet"))
        exit_code = train_height_only_prior.main_with_args(
            [
                "--prior", str(prior_path),
                "--v18", str(v18_path),
                "--output-dir", str(out_dir),
                "--run-name", "curated",
                "--steps", "1",
                "--val-steps", "0",
                "--batch-size", "1",
                "--learning-rate", "1e-3",
                "--device", "cpu",
                "--curation-manifest", str(curation_dir),
                "--no-amp",
                "--no-compile",
            ]
        )
        assert exit_code == 0
        metrics = json.loads((out_dir / "curated_metrics.json").read_text(encoding="utf-8"))
        assert metrics["curation_manifest"] == str(curation_dir)
        assert metrics["train_metrics"][0]["tile_id"] == 2


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


def test_train_epoch_mode_writes_best_and_latest_checkpoints() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        prior_path = root / "prior.zarr"
        v18_path = root / "v18.zarr"
        out_dir = root / "out"
        _make_prior_store(prior_path, n_tiles=4)
        _make_v18_store(v18_path, n_tiles=4)
        exit_code = train_height_only_prior.main_with_args(
            [
                "--prior", str(prior_path),
                "--v18", str(v18_path),
                "--output-dir", str(out_dir),
                "--run-name", "epoch",
                "--epochs", "2",
                "--steps", "0",
                "--val-steps", "1",
                "--batch-size", "1",
                "--learning-rate", "1e-3",
                "--lr-plateau-patience", "1",
                "--device", "cpu",
                "--no-amp",
                "--no-compile",
            ]
        )
        assert exit_code == 0
        assert (out_dir / "epoch_latest.pt").exists()
        assert (out_dir / "epoch_best.pt").exists()
        latest = torch.load(out_dir / "epoch_latest.pt", map_location="cpu", weights_only=False)
        assert latest["epoch"] == 2
        assert latest["step"] >= 2
        metrics = json.loads((out_dir / "epoch_metrics.json").read_text(encoding="utf-8"))
        assert metrics["requested_epochs"] == 2
        assert metrics["requested_steps"] == 0
        assert metrics["epoch_count"] == 2
        assert metrics["lr_plateau_enabled"] is True
        assert metrics["lr_plateau_patience"] == 1
        assert len(metrics["history"]) == 2
        assert all(row["val_batches"] == 1 for row in metrics["history"])
        assert all("learning_rate" in row for row in metrics["history"])


def test_save_checkpoint_falls_back_when_target_is_locked(monkeypatch: pytest.MonkeyPatch) -> None:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        target = root / "locked_latest.pt"
        model = torch.nn.Conv2d(1, 1, kernel_size=1)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        scaler = torch.amp.GradScaler("cuda", enabled=False)
        original_replace = train_height_only_prior.os.replace

        def _locked_replace(src, dst):
            if Path(dst) == target:
                raise PermissionError("simulated locked checkpoint")
            return original_replace(src, dst)

        monkeypatch.setattr(train_height_only_prior.os, "replace", _locked_replace)
        written = train_height_only_prior._save_checkpoint(
            target,
            model=model,
            optimizer=optimizer,
            scaler=scaler,
            use_amp=False,
            epoch=3,
            global_step=123,
            best_val=0.5,
            args=train_height_only_prior.argparse.Namespace(output_dir=root),
            history=[{"epoch": 3}],
        )
        assert written != target
        assert written.exists()
        assert "epoch0003_step0000123" in written.name
