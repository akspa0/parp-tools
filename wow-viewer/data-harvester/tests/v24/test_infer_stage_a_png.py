"""Tests for infer_v24_stage_a_png.py (Spec 096 / Slice 2)."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

SCRIPTS = Path(__file__).resolve().parents[2] / "scripts"
SCRIPT = SCRIPTS / "infer_v24_stage_a_png.py"
SRC = Path(__file__).resolve().parents[2] / "src"


def _write_test_png(path: Path, size: int = 256, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    arr = (rng.random((size, size, 3)) * 255.0).astype(np.uint8)
    Image.fromarray(arr, mode="RGB").save(str(path))


def _train_minimal_checkpoint(path: Path, base: int = 8) -> None:
    """Train a tiny StageAMinimapOnly for one step and save with the new config.

    Used to give the tests a real, on-disk checkpoint without depending on the
    50-epoch training run that lives under output/v24_validation/.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "stage_a_mod", SRC / "harvester" / "v24" / "stage_a.py"
    )
    assert spec is not None and spec.loader is not None
    stage_a_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage_a_mod)

    torch.manual_seed(0)
    model = stage_a_mod.StageAMinimapOnly(base=base)
    # One training step so the state_dict has non-default weights.
    x = torch.randn(2, stage_a_mod.IN_CHANNELS_MINIMAP_ONLY, 64, 64)
    tgt_outer = torch.zeros(2, 17, 17)
    tgt_inner = torch.zeros(2, 16, 16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    po, pi = model(x)
    loss = (po - tgt_outer).abs().mean() + (pi - tgt_inner).abs().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "base": base,
                "in_channels": stage_a_mod.IN_CHANNELS_MINIMAP_ONLY,
                "minimap_only": True,
            },
            "height_scale": stage_a_mod.HEIGHT_SCALE,
            "seed": 0,
            "epoch": 1,
            "val_l1": float(loss.item() * stage_a_mod.HEIGHT_SCALE),
        },
        str(path),
    )


def _train_cheat_checkpoint(path: Path, base: int = 8) -> None:
    """Train a tiny StageA (full 13-channel) for one step and save.

    Used to verify the strict-checkpoint refusal path.
    """
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "stage_a_mod", SRC / "harvester" / "v24" / "stage_a.py"
    )
    assert spec is not None and spec.loader is not None
    stage_a_mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(stage_a_mod)

    torch.manual_seed(0)
    model = stage_a_mod.StageAModel(base=base)
    x = torch.randn(2, stage_a_mod.IN_CHANNELS, 64, 64)
    q = torch.randn(2, 33, 33)
    tgt_outer = torch.zeros(2, 17, 17)
    tgt_inner = torch.zeros(2, 16, 16)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    po, pi = model(x, q)
    loss = (po - tgt_outer).abs().mean() + (pi - tgt_inner).abs().mean()
    opt.zero_grad()
    loss.backward()
    opt.step()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "config": {
                "base": base,
                "in_channels": stage_a_mod.IN_CHANNELS,
                "minimap_only": False,
            },
            "height_scale": stage_a_mod.HEIGHT_SCALE,
            "seed": 0,
            "epoch": 1,
            "val_l1": float(loss.item() * stage_a_mod.HEIGHT_SCALE),
        },
        str(path),
    )


@pytest.mark.v24
def test_infer_png_runs_end_to_end(tmp_path: Path) -> None:
    """Write a 256x256 RGB PNG, run the script, assert the NPZ shape."""
    png = tmp_path / "minimap.png"
    ckpt = tmp_path / "stage_a.pt"
    out_npz = tmp_path / "prior.npz"
    _write_test_png(png)
    _train_minimal_checkpoint(ckpt)

    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--checkpoint", str(ckpt),
            "--image", str(png),
            "--output", str(out_npz),
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode == 0, f"stderr:\n{proc.stderr}\nstdout:\n{proc.stdout}"

    with np.load(str(out_npz), allow_pickle=True) as data:
        assert data["outer"].shape == (17, 17)
        assert data["inner"].shape == (16, 16)
        assert bool(data["prior_unavailable"].item()) is False
        assert data["outer"].dtype == np.float32
        assert data["inner"].dtype == np.float32


@pytest.mark.v24
def test_infer_png_deterministic_across_seeds(tmp_path: Path) -> None:
    """Two runs with different seeds produce bit-identical outputs."""
    png = tmp_path / "minimap.png"
    ckpt = tmp_path / "stage_a.pt"
    out_a = tmp_path / "a.npz"
    out_b = tmp_path / "b.npz"
    _write_test_png(png)
    _train_minimal_checkpoint(ckpt)

    for seed, out in ((11, out_a), (22, out_b)):
        proc = subprocess.run(
            [
                sys.executable, str(SCRIPT),
                "--checkpoint", str(ckpt),
                "--image", str(png),
                "--output", str(out),
                "--seed", str(seed),
            ],
            capture_output=True, text=True, check=False,
        )
        assert proc.returncode == 0, f"seed={seed} stderr:\n{proc.stderr}"

    with np.load(str(out_a)) as a, np.load(str(out_b)) as b:
        assert np.array_equal(a["outer"], b["outer"])
        assert np.array_equal(a["inner"], b["inner"])


@pytest.mark.v24
def test_infer_png_refuses_cheat_checkpoint(tmp_path: Path) -> None:
    """A 13-channel cheat-regime checkpoint must be refused with a clear error."""
    png = tmp_path / "minimap.png"
    ckpt = tmp_path / "cheat.pt"
    out_npz = tmp_path / "prior.npz"
    _write_test_png(png)
    _train_cheat_checkpoint(ckpt)

    proc = subprocess.run(
        [
            sys.executable, str(SCRIPT),
            "--checkpoint", str(ckpt),
            "--image", str(png),
            "--output", str(out_npz),
        ],
        capture_output=True, text=True, check=False,
    )
    assert proc.returncode != 0
    combined = (proc.stderr or "") + (proc.stdout or "")
    assert "in_channels" in combined
    assert "minimap-only" in combined or "minimap_only" in combined
    assert not out_npz.exists()
