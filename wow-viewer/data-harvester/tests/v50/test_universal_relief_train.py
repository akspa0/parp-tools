from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import nn

from harvester.v50 import universal_relief_train as train_module
from harvester.v50.universal_relief_contract import normalize_relief
from harvester.v50.universal_relief_train import (
    _luminance_baseline,
    augment_relief_sample,
    build_model_stage_summary,
    build_training_plan,
    compute_universal_relief_loss,
    evaluate_universal_relief,
    save_validation_sheet,
    update_ema,
)


def _loss_inputs(batch: int = 2):
    target = torch.linspace(0.0, 1.0, 224 * 224).reshape(1, 224, 224).repeat(batch, 1, 1)
    normals = torch.zeros(batch, 3, 224, 224)
    normals[:, 2] = 1.0
    return {
        "target": target,
        "point_weight": torch.ones_like(target),
        "authority_weight": torch.ones(batch),
        "target_normals": normals,
        "normal_mask": torch.zeros_like(target),
        "height_range": torch.ones(batch),
    }


def test_exact_prediction_has_zero_guided_loss() -> None:
    inputs = _loss_inputs()
    loss, components = compute_universal_relief_loss(inputs["target"].clone(), **inputs)
    assert float(loss) == pytest.approx(0.0, abs=1e-7)
    assert all(float(value) == pytest.approx(0.0, abs=1e-7) for value in components.values())


def test_wrong_flat_prediction_is_penalized_at_multiple_terms() -> None:
    inputs = _loss_inputs()
    predicted = torch.full_like(inputs["target"], 0.5)
    loss, components = compute_universal_relief_loss(predicted, **inputs)
    assert float(loss) > 0.0
    assert float(components["point"]) > 0.0
    assert float(components["gradient"]) > 0.0
    assert float(components["hard"]) > 0.0


def test_pseudo_authority_downweights_point_loss() -> None:
    inputs = _loss_inputs(batch=2)
    predicted = inputs["target"].clone()
    predicted[1].zero_()
    exact, _ = compute_universal_relief_loss(predicted, **inputs)
    inputs["authority_weight"] = torch.tensor([1.0, 0.5])
    pseudo, _ = compute_universal_relief_loss(predicted, **inputs)
    assert float(pseudo) < float(exact)


def test_normal_guidance_only_acts_where_mask_is_valid() -> None:
    inputs = _loss_inputs(batch=1)
    predicted = torch.zeros_like(inputs["target"])
    unmasked, components_unmasked = compute_universal_relief_loss(predicted, **inputs)
    inputs["target_normals"][:, 0] = 1.0
    inputs["target_normals"][:, 2] = 0.0
    inputs["normal_mask"] = torch.ones_like(inputs["target"])
    masked, components_masked = compute_universal_relief_loss(predicted, **inputs)
    assert float(components_unmasked["normal"]) == 0.0
    assert float(components_masked["normal"]) > 0.0
    assert float(masked) > float(unmasked)


def test_d4_augmentation_transforms_all_spatial_signals_and_keeps_bounds(monkeypatch) -> None:
    sample = {
        "rgb": torch.rand(3, 224, 224),
        "target": torch.rand(224, 224),
        "point_weight": torch.ones(224, 224),
        "normals": torch.zeros(3, 224, 224),
        "normal_mask": torch.ones(224, 224),
    }
    monkeypatch.setattr(torch, "randint", lambda *args, **kwargs: torch.tensor(1))
    augmented = augment_relief_sample(sample)
    assert augmented["rgb"].shape == sample["rgb"].shape
    assert augmented["target"].shape == sample["target"].shape
    assert torch.isfinite(augmented["rgb"]).all()
    assert torch.all((augmented["rgb"] >= 0.0) & (augmented["rgb"] <= 1.0))


def test_luminance_baseline_is_stable_for_constant_and_spans_gradient() -> None:
    constant = _luminance_baseline(torch.ones(1, 3, 8, 8))
    assert torch.equal(constant, torch.zeros_like(constant))
    gradient = torch.linspace(0.0, 1.0, 8).view(1, 1, 1, 8).expand(1, 3, 8, 8)
    baseline = _luminance_baseline(gradient)
    assert float(baseline.min()) == 0.0
    assert float(baseline.max()) == 1.0


def test_ema_updates_parameters_and_copies_buffers() -> None:
    model = nn.BatchNorm2d(3)
    ema = copy.deepcopy(model)
    with torch.no_grad():
        model.weight.fill_(2.0)
        model.running_mean.fill_(5.0)
    update_ema(ema, model, 0.5)
    assert torch.allclose(ema.weight, torch.full_like(ema.weight, 1.5))
    assert torch.allclose(ema.running_mean, torch.full_like(ema.running_mean, 5.0))


def test_normalization_helper_remains_consistent_with_curriculum_targets() -> None:
    values = np.arange(16, dtype=np.float32).reshape(4, 4)
    normalized = normalize_relief(values)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0


def test_dry_training_plan_records_full_recipe_without_writing_output(tmp_path, monkeypatch) -> None:
    summary = {
        "curriculum_id": "sha256:" + "a" * 64,
        "visual_families": {
            "wow_minimap": 10,
            "aerial": 3,
            "photos": 3,
            "paintings": 3,
            "drawings": 3,
        },
        "held_out_families": ["paintings"],
        "row_count": 22,
    }
    rows = [
        {"width": 256, "height": 256, "split": "train"},
        {"width": 300, "height": 224, "split": "validation"},
        {"width": 224, "height": 224, "split": "compatibility"},
    ]
    monkeypatch.setattr(train_module, "_validate_curriculum", lambda _path: (summary, rows))
    output = tmp_path / "run"

    plan = build_training_plan(
        curriculum=tmp_path / "curriculum.zarr",
        output=output,
        batch_size=8,
        epochs=50,
        workers=0,
        seed=114,
        overlap=28,
        learning_rate=2e-4,
        weight_decay=1e-4,
        pseudo_weight=0.5,
        freeze_backbone=True,
        gradient_weight=0.07,
        normal_weight=0.11,
        hard_error_weight=0.03,
        hard_error_max_multiplier=3.0,
        ema_decay=0.998,
        warmup_fraction=0.08,
        grad_clip=0.75,
        use_amp=False,
    )

    assert not output.exists()
    assert plan["training_recipe"] == "v114.3-universal-guided-onecycle-ema"
    assert plan["deployment_inputs"] == ["rgb"]
    assert plan["loss"]["gradient_weight"] == 0.07
    assert plan["loss"]["normal_weight"] == 0.11
    assert plan["loss"]["hard_error_weight"] == 0.03
    assert plan["loss"]["hard_error_max_multiplier"] == 3.0
    assert plan["optimization"]["amp"] is False
    assert plan["optimization"]["ema_decay"] == 0.998
    assert plan["optimization"]["warmup_fraction"] == 0.08
    assert plan["optimization"]["gradient_clip"] == 0.75
    assert plan["optimization"]["schedule"] == "OneCycleLR warmup+cosine"
    assert plan["optimization"]["sampler"] == "family_balanced_weighted_replacement"
    assert plan["validation"]["baselines"] == [
        "constant_relief",
        "direct_luminance_relief",
    ]


def test_validation_sheet_writes_fixed_scale_review_artifact(tmp_path) -> None:
    preview = {
        "rgb": np.zeros((16, 16, 3), dtype=np.float32),
        "truth": np.zeros((16, 16), dtype=np.float32),
        "prediction": np.ones((16, 16), dtype=np.float32),
        "luminance": np.zeros((16, 16), dtype=np.float32),
    }
    output = tmp_path / "sheet.png"
    save_validation_sheet([preview], output, "fixture")
    assert output.is_file()
    assert output.stat().st_size > 0


def test_promotion_gate_uses_only_whole_family_compatibility_tiles() -> None:
    class RedChannelRelief(nn.Module):
        def forward(self, rgb: torch.Tensor) -> torch.Tensor:
            return rgb[:, 0]

    target = torch.linspace(0.0, 1.0, 224).view(1, 1, 224).expand(2, 224, 224)
    rgb = torch.zeros(2, 3, 224, 224)
    rgb[:, 0] = target
    rgb[:, 1] = 1.0 - target
    loader = [
        {
            "rgb": rgb,
            "target": target,
            "row_id": ["validation-row", "compatibility-row"],
            "visual_family": ["wow_minimap", "heldout_painting"],
            "split": ["validation", "compatibility"],
            "tile_x": torch.tensor([0, 0]),
            "tile_y": torch.tensor([0, 0]),
        }
    ]
    summary, records, previews = evaluate_universal_relief(
        RedChannelRelief(), loader, device=torch.device("cpu"), use_amp=False
    )
    assert len(records) == 2
    assert previews
    assert summary["promotion_scope"] == "whole_family_compatibility_only"
    assert set(summary["compatibility_family_metrics"]) == {"heldout_painting"}
    assert summary["compatibility_macro"]["mae"] == pytest.approx(0.0)
    assert summary["passes_five_percent_gate"] is True


def test_model_stage_summary_validates_against_published_spec114_schema(tmp_path) -> None:
    checkpoint = tmp_path / "checkpoint_best.pt"
    checkpoint.write_bytes(b"immutable checkpoint fixture")
    student = {
        "architecture_id": "dinov2_small_relief_v1",
        "hub_id": "facebook/dinov2-small",
        "revision": "e" * 40,
        "weight_file": "model.safetensors",
        "weights_sha256": "a" * 64,
        "license": "apache-2.0",
        "output_signal": "relative_relief",
        "input_tile_size": 224,
    }
    plan = {
        "curriculum_id": "sha256:" + "b" * 64,
        "curriculum": str(tmp_path / "curriculum.zarr"),
        "student": student,
        "freeze_backbone": True,
        "loss": {"gradient_weight": 0.05},
        "optimization": {"amp": True},
    }
    validation = {
        "compatibility_macro": {
            "mae": 0.1,
            "gradient_mae": 0.02,
            "constant_mae": 0.2,
            "constant_gradient_mae": 0.04,
            "luminance_mae": 0.18,
            "luminance_gradient_mae": 0.035,
        },
        "passes_five_percent_gate": True,
    }
    summary = build_model_stage_summary(
        plan=plan,
        final_validation=validation,
        checkpoint_path=checkpoint,
        best_epoch=3,
        epochs_completed=5,
        parameter_count=123,
        peak_vram_gb=1.25,
        wall_seconds=60.0,
    )
    schema_path = (
        Path(__file__).resolve().parents[3]
        / "specs"
        / "114-direct-terrain-reconstruction"
        / "contracts"
        / "model-stage-and-curriculum.schema.json"
    )
    schema = json.loads(schema_path.read_text(encoding="utf-8"))["$defs"]["model_stage_run"]
    assert set(summary) == set(schema["required"]) | {"pretrained_source"}
    assert summary["schema"] == schema["properties"]["schema"]["const"]
    assert summary["stage"] in schema["properties"]["stage"]["enum"]
    assert summary["promotion_verdict"] in schema["properties"]["promotion_verdict"]["enum"]
    for identity in (summary["curriculum"], summary["checkpoint"]):
        assert len(identity["sha256"]) == 64
    assert summary["promotion_verdict"] == "pending"
    assert summary["visual_evidence"]["user_verdict"] == "pending"
