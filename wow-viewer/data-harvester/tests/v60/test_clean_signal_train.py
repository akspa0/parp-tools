from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from harvester.v60.clean_signal_model import CleanSignalPredictions
from harvester.v60.clean_signal_train import (
    CHECKPOINT_SCHEMA,
    CleanSignalRow,
    CleanSignalTrainConfig,
    CleanSignalTrainError,
    build_clean_signal_split,
    evaluate_clean_signal_model,
    train_clean_signal_model,
)


def _rows(tmp_path: Path, family_count: int = 2, variants: int = 2) -> list[CleanSignalRow]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    rows: list[CleanSignalRow] = []
    size = 17
    for family_index in range(family_count):
        family = f"family-{family_index}"
        for variant in range(variants):
            value = 0.1 * family_index + 0.05 * variant
            luma = np.full((256, 256), value, dtype=np.float32)
            gradient = np.zeros((2, 256, 256), dtype=np.float32)
            confidence = np.ones((256, 256), dtype=np.float32)
            yy, xx = np.mgrid[:size, :size].astype(np.float32)
            relative = np.zeros((257, 257), dtype=np.float32)
            relative[:size, :size] = (xx + yy) / max(1.0, float(2 * (size - 1)))
            coarse = relative.copy()
            detail = np.zeros_like(relative)
            npz_path = tmp_path / f"{family}-{variant}.npz"
            np.savez(
                npz_path,
                clean_observation_luma_256=luma,
                clean_observation_gradient_256=gradient,
                clean_observation_confidence_256=confidence,
                relative_height_257=relative,
                coarse_relief_257=coarse,
                detail_residual_257=detail,
            )
            rows.append(
                CleanSignalRow(
                    row_id=f"{family}-v{variant}",
                    source_group_id=f"{family}-g{variant}",
                    family=family,
                    complexity_bucket="smooth",
                    variant=variant,
                    split="train",
                    npz_path=npz_path,
                )
            )
    return rows


class _TinyCleanModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bias = nn.Parameter(torch.tensor(0.25))

    def forward(self, inputs: torch.Tensor) -> CleanSignalPredictions:
        base = torch.nn.functional.interpolate(inputs[:, :1], size=(257, 257), mode="bilinear", align_corners=False)
        base = base.squeeze(1) + self.bias
        coarse = torch.sigmoid(base)
        detail = torch.zeros_like(coarse) + self.bias * 0.0
        return CleanSignalPredictions(coarse, detail, torch.clamp(coarse + detail, 0.0, 1.0))


def _tiny_builder(_architecture: str, *, profile: str) -> tuple[nn.Module, dict[str, object]]:
    assert profile == "tiny"
    return _TinyCleanModel(), {"architecture": "tiny-test", "parameter_count": 1}


def test_split_identity_is_stable_and_modes_have_expected_family_overlap(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    within_a = build_clean_signal_split(rows, mode="within_family", seed=7137)
    within_b = build_clean_signal_split(rows, mode="within_family", seed=7137)
    complete = build_clean_signal_split(
        [replace(row, split="validation" if row.family == "family-1" else "train") for row in rows],
        mode="complete_family",
        seed=7137,
    )

    assert within_a.identity == within_b.identity
    assert within_a.as_dict() == within_b.as_dict()
    assert {row.family for row in within_a.train_rows} == {"family-0", "family-1"}
    assert {row.family for row in within_a.validation_rows} == {"family-0", "family-1"}
    assert not ({row.family for row in complete.train_rows} & {row.family for row in complete.validation_rows})
    with pytest.raises(CleanSignalTrainError, match="at least two variants"):
        build_clean_signal_split(_rows(tmp_path / "single", family_count=1, variants=1), mode="within_family", seed=1)


def test_evaluator_reports_per_signal_family_bucket_and_baseline(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    model = _TinyCleanModel()

    report = evaluate_clean_signal_model(model, rows, profile="parity", batch_size=2, device="cpu")

    assert report["row_count"] == 4
    assert set(report["by_family"]) == {"family-0", "family-1"}
    assert set(report["by_complexity_bucket"]) == {"smooth"}
    assert set(report["loss_components"]) >= {"point", "gradient", "total"}
    assert report["final_height_mae"] >= 0.0
    assert all("coarse_mae" in value and "detail_mae" in value for value in report["by_family"].values())


def test_trainer_selects_best_checkpoint_and_binds_split_loss_and_identity(tmp_path: Path) -> None:
    rows = _rows(tmp_path)
    output = tmp_path / "run"
    split = build_clean_signal_split(rows, mode="within_family", seed=7137)

    report = train_clean_signal_model(
        split.train_rows,
        split.validation_rows,
        architecture="unet_lite_v2",
        profile="v7_structural_v1",
        output=output,
        config=CleanSignalTrainConfig(epochs=2, batch_size=2, patience=2, device="cpu"),
        split=split,
        model_builder=_tiny_builder,
    )

    assert report["schema"] == "v7-clean-signal-training-report-v1"
    assert report["best_epoch"] in {1, 2}
    assert report["split"]["identity"] == split.identity
    assert report["loss_profile"]["name"] == "v7_structural_v1"
    assert (output / "checkpoint_best.pt").is_file()
    assert (output / "checkpoint_last.pt").is_file()
    payload = torch.load(output / "checkpoint_best.pt", map_location="cpu", weights_only=False)
    assert payload["schema"] == CHECKPOINT_SCHEMA
    assert payload["split"]["identity"] == split.identity
    saved_report = json.loads((output / "training_report.json").read_text(encoding="utf-8"))
    assert saved_report["best_epoch"] == report["best_epoch"]
    with pytest.raises(FileExistsError, match="overwrite"):
        train_clean_signal_model(
            split.train_rows,
            split.validation_rows,
            architecture="unet_lite_v2",
            output=output,
            config=CleanSignalTrainConfig(epochs=1, device="cpu"),
            split=split,
            model_builder=_tiny_builder,
        )
