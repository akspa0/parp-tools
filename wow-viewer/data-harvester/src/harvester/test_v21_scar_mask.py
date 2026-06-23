from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from harvester.v21_scar_dataset import make_scar_mask
from harvester.v21_scar_model import V21ScarMaskModel


def _load_train_script():
    root = Path(__file__).resolve().parents[2]
    path = root / "scripts" / "train_v21_scar_mask.py"
    spec = importlib.util.spec_from_file_location("train_v21_scar_mask_test", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_make_scar_mask_uses_selected_layers() -> None:
    alpha = torch.zeros((8, 8, 4), dtype=torch.float32).numpy()
    alpha[1:3, 1:3, 0] = 1.0
    alpha[3:5, 3:5, 2] = 0.2

    mask = make_scar_mask(alpha, layers=(1, 2, 3), threshold=0.05)

    assert mask.shape == (8, 8)
    assert mask[1:3, 1:3].sum() == 0.0
    assert mask[3:5, 3:5].sum() == 4.0


def test_v21_scar_mask_model_shape() -> None:
    model = V21ScarMaskModel(base_channels=8)
    x = torch.randn((2, 3, 256, 256), dtype=torch.float32)

    y = model(x)

    assert y.shape == (2, 1, 256, 256)


def test_scar_loss_is_finite() -> None:
    scar_loss = _load_train_script().scar_loss

    logits = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    target = torch.zeros((1, 1, 16, 16), dtype=torch.float32)
    target[:, :, 4:8, 4:8] = 1.0

    loss, metrics = scar_loss(logits, target)

    assert torch.isfinite(loss)
    assert metrics["bce"] > 0.0
    assert 0.0 <= metrics["f1"] <= 1.0
