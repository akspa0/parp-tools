"""Regression checks for the universal V25 WDL-prior training contract."""

from __future__ import annotations

import inspect
from pathlib import Path

import torch
import pytest


def _trainer_module():
    import importlib.util

    script = Path(__file__).resolve().parents[2] / "scripts" / "train_v25_decompiler.py"
    spec = importlib.util.spec_from_file_location("v25_trainer_contract", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_stage_b_has_no_ground_truth_prior_input():
    trainer = _trainer_module()

    assert "prior_33" not in inspect.signature(trainer.V25Pipeline.forward).parameters
    assert "h_257 = self.stage_b(h_33_pred, dec[\"clean_rgb\"])" in inspect.getsource(trainer.V25Pipeline.forward)


def test_invalid_spec102_trainer_is_fail_closed():
    trainer = _trainer_module()

    with pytest.raises(RuntimeError, match="one-model/one-residual"):
        trainer.assert_training_authorized()


def test_epoch_runner_calls_pipeline_with_minimap_only():
    trainer = _trainer_module()

    class Pipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.scale = torch.nn.Parameter(torch.tensor(1.0))
            self.calls = 0

        def forward(self, minimap):
            self.calls += 1
            return {"h_257": self.scale.expand(minimap.shape[0], 257, 257)}

    class Loss:
        def __call__(self, predictions, _targets, minimap):
            value = predictions["h_257"].mean()
            return {"loss": value, "height": value}

    pipeline = Pipeline()
    optimizer = torch.optim.SGD(pipeline.parameters(), lr=0.1)
    batch = {
        "minimap": torch.zeros(1, 3, 256, 256),
        "mask": torch.zeros(1, 1, 256, 256),
        "clean_rgb": torch.zeros(1, 3, 256, 256),
        "h_257": torch.zeros(1, 257, 257),
        "h_33": torch.zeros(1, 33, 33),
        "height_mask": torch.ones(1, 257, 257),
        "h_33_mask": torch.ones(1, 33, 33),
        "class_ids": torch.zeros(1, 1, dtype=torch.long),
        "coords": torch.zeros(1, 1, 3),
        "rotations": torch.zeros(1, 1, 3),
        "exist": torch.zeros(1, 1),
        "mtex_labels": torch.zeros(1, 1),
        "mcly_labels": torch.zeros(1, 1, 1, 1, dtype=torch.long),
        "alpha": torch.zeros(1, 4, 256, 256),
    }

    trainer.run_epoch(
        [batch], pipeline, Loss(), optimizer, None, torch.device("cpu"), None,
        train=True, log_interval=0, epoch=1,
    )

    assert pipeline.calls == 1
    assert pipeline.scale.item() != 1.0
