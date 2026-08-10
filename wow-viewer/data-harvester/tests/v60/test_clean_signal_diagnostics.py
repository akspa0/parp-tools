from __future__ import annotations

# The test reuses the compact corpus fixture from the neighboring trainer test.
# ruff: noqa: I001

from pathlib import Path

import torch

from harvester.v60.clean_signal_diagnostics import (
    DIAGNOSTIC_SCHEMA,
    diagnose_clean_signal_checkpoint,
)
from harvester.v60.clean_signal_losses import get_clean_signal_loss_config
from harvester.v60.clean_signal_model import build_clean_signal_model
from harvester.v60.clean_signal_train import CHECKPOINT_SCHEMA

from test_clean_signal_train import _rows


def test_checkpoint_diagnostic_writes_prediction_rows_and_atlases(tmp_path: Path, monkeypatch) -> None:
    rows = _rows(tmp_path / "corpus")
    monkeypatch.setattr(
        "harvester.v60.clean_signal_diagnostics.load_clean_signal_rows",
        lambda _corpus: (tmp_path / "corpus", rows),
    )
    model, identity = build_clean_signal_model("unet_lite_v2", profile="tiny")
    checkpoint = tmp_path / "checkpoint_best.pt"
    torch.save(
        {
            "schema": CHECKPOINT_SCHEMA,
            "epoch": 1,
            "architecture": "unet_lite_v2",
            "model_identity": identity,
            "loss_profile": get_clean_signal_loss_config("v7_structural_v1").as_dict(),
            "split": {
                "mode": "within_family",
                "seed": 7137,
                "validation_row_ids": [row.row_id for row in rows],
            },
            "model_state_dict": model.state_dict(),
        },
        checkpoint,
    )
    output = tmp_path / "diagnostic"
    report = diagnose_clean_signal_checkpoint(checkpoint, tmp_path / "corpus", output, device="cpu", batch_size=2)

    assert report["schema"] == DIAGNOSTIC_SCHEMA
    assert report["validation_row_count"] == 4
    assert len(report["by_family"]) == 2
    assert (output / "diagnostic_report.json").is_file()
    assert (output / "validation-diagnostic-atlas.png").is_file()
    assert len(list((output / "predictions").glob("*.npz"))) == 4
