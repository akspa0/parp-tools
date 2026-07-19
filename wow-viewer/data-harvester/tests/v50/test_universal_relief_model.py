from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from harvester.v50.universal_relief_model import (
    INPUT_TILE_SIZE,
    OUTPUT_SIGNAL,
    UniversalReliefNet,
    student_identity,
    verify_student_weight,
)


class FakePatchBackbone(nn.Module):
    def __init__(self, hidden_size: int = 48, patch_size: int = 14) -> None:
        super().__init__()
        self.config = SimpleNamespace(hidden_size=hidden_size, patch_size=patch_size)
        self.patch_embed = nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size)
        self.cls = nn.Parameter(torch.zeros(1, 1, hidden_size))

    def forward(self, *, pixel_values: torch.Tensor):
        patches = self.patch_embed(pixel_values).flatten(2).transpose(1, 2)
        cls = self.cls.expand(pixel_values.shape[0], -1, -1)
        return SimpleNamespace(last_hidden_state=torch.cat((cls, patches), dim=1))


def test_pinned_student_identity_is_full_safe_and_single_output() -> None:
    identity = student_identity()
    assert len(identity.revision) == 40
    assert identity.weight_file == "model.safetensors"
    assert len(identity.weights_sha256) == 64
    assert identity.license == "apache-2.0"
    assert identity.output_signal == OUTPUT_SIGNAL


def test_frozen_backbone_model_emits_one_finite_bounded_relief_tile() -> None:
    model = UniversalReliefNet(FakePatchBackbone(), freeze_backbone=True)
    output = model(torch.rand(2, 3, INPUT_TILE_SIZE, INPUT_TILE_SIZE))

    assert output.shape == (2, INPUT_TILE_SIZE, INPUT_TILE_SIZE)
    assert torch.isfinite(output).all()
    assert torch.all((output >= 0.0) & (output <= 1.0))
    assert model.deployment_inputs == ("rgb",)
    assert model.output_signal == "relative_relief"


def test_frozen_backbone_stays_eval_and_only_decoder_is_trainable() -> None:
    model = UniversalReliefNet(FakePatchBackbone(), freeze_backbone=True)
    model.train()

    assert not model.backbone.training
    assert not any(parameter.requires_grad for parameter in model.backbone.parameters())
    assert model.trainable_parameter_count() > 0
    assert model.trainable_parameter_count() < model.total_parameter_count()


def test_unfrozen_ablation_allows_backbone_gradients() -> None:
    model = UniversalReliefNet(FakePatchBackbone(), freeze_backbone=False)
    output = model(torch.rand(1, 3, INPUT_TILE_SIZE, INPUT_TILE_SIZE))
    output.mean().backward()

    assert model.backbone.patch_embed.weight.grad is not None


def test_model_refuses_wrong_channels_and_non_patch_aligned_tiles() -> None:
    model = UniversalReliefNet(FakePatchBackbone(), freeze_backbone=True)
    with pytest.raises(ValueError, match="Bx3xHxW"):
        model(torch.rand(1, 1, INPUT_TILE_SIZE, INPUT_TILE_SIZE))
    with pytest.raises(ValueError, match="divisible"):
        model(torch.rand(1, 3, 223, INPUT_TILE_SIZE))


def test_student_weight_hash_mismatch_fails_closed(tmp_path) -> None:
    weight = tmp_path / "model.safetensors"
    weight.write_bytes(b"not the pinned student")
    with pytest.raises(ValueError, match="hash mismatch"):
        verify_student_weight(weight)
