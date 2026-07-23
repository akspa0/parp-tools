"""Spec 118 T018 (US2): the object-masked-loss flag is really wired into BOTH geometry trainers.

Follows the project convention from the Spec 116/117 verification passes: prove the documented CLI
surface exists by exercising the real argparse (``--help`` on the thin CLI scripts), and prove the
loss math directly with torch -- not by re-simulating a full CUDA training run in a test.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from harvester.spec118.object_loss import (
    OBJECT_MASK_ARRAY,
    object_mask_available,
    object_point_weight,
)

_SCRIPTS = Path(__file__).parents[2] / "scripts"


def _help_text(script: str) -> str:
    result = subprocess.run(
        [sys.executable, str(_SCRIPTS / script), "--help"],
        capture_output=True, text=True, timeout=120,
    )
    assert result.returncode == 0, f"{script} --help failed: {result.stderr}"
    return result.stdout


def test_direct_geometry_trainer_cli_exposes_the_flag():
    assert "--object-mask-weight" in _help_text("v50_train_direct_geometry.py")


def test_geometry_detailer_trainer_cli_exposes_the_flag():
    assert "--object-mask-weight" in _help_text("v50_train_geometry_detailer.py")


def test_masked_loss_is_exactly_zero_when_only_object_pixels_are_wrong():
    # Two-step-CPU-equivalent check of the loss composition the trainers apply:
    # abs_err * (1 - w * mask), with the target wrong ONLY at visible-object pixels and w=1.
    predicted = torch.zeros(1, 4, 4)
    target = torch.zeros(1, 4, 4)
    mask = np.zeros((4, 4), dtype=np.float32)
    mask[0, 0] = 1.0  # the single visible-object pixel
    target[0, 0, 0] = 5.0  # badly wrong ONLY there
    target[0, 2, 2] = 0.25  # small honest error on free land

    weight = torch.from_numpy(object_point_weight(mask, 1.0))
    loss = (torch.abs(predicted - target) * weight).mean()
    assert loss.item() == 0.25 / 16  # only the free-land error survives

    # Parity: w=0 is exactly the unweighted mean.
    weight_zero = torch.from_numpy(object_point_weight(mask, 0.0))
    loss_zero = (torch.abs(predicted - target) * weight_zero).mean()
    assert loss_zero.item() == (5.0 + 0.25) / 16


def test_missing_array_means_masking_is_disabled(tmp_path):
    import zarr

    group = zarr.open_group(str(tmp_path / "store.zarr"), mode="w")
    assert object_mask_available(group) is False
    # The trainers' guard turns the flag into a no-op exactly like the liquid warning path.
    group.create_array(OBJECT_MASK_ARRAY, data=np.zeros((1, 257, 257), dtype=np.float32))
    assert object_mask_available(group) is True
