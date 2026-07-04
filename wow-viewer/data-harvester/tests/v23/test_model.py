from __future__ import annotations

import pytest
import torch
from transformers import DepthAnythingConfig

from harvester.v23.model import V23HeightPredictor

pytestmark = pytest.mark.v23


def _tiny_config() -> DepthAnythingConfig:
    return DepthAnythingConfig(
        backbone_config={
            "model_type": "dinov2",
            "image_size": 56,
            "patch_size": 14,
            "num_channels": 3,
            "hidden_size": 64,
            "num_hidden_layers": 4,
            "num_attention_heads": 4,
            "mlp_ratio": 4,
            "qkv_bias": True,
            "apply_layernorm": True,
            "reshape_hidden_states": False,
            "use_mask_token": True,
            "out_features": ["stage1", "stage2", "stage3", "stage4"],
            "out_indices": [1, 2, 3, 4],
            "stage_names": ["stem", "stage1", "stage2", "stage3", "stage4"],
        },
        reassemble_hidden_size=64,
        neck_hidden_sizes=[16, 32, 48, 64],
        fusion_hidden_size=32,
        head_hidden_size=16,
        patch_size=14,
        reassemble_factors=[4, 2, 1, 0.5],
    )


def test_model_forward_returns_typed_output() -> None:
    model = V23HeightPredictor(
        in_channels=15,
        config=_tiny_config(),
        load_pretrained=False,
        schema_input_size=56,
    )
    x = torch.randn(2, 15, 56, 56)
    output = model(x)

    assert tuple(output.disparity.shape) == (2, 1, 257, 257)
    assert tuple(output.affine_anchor.shape) == (2, 2)
    assert tuple(output.metric_height.shape) == (2, 1, 257, 257)
    assert output.metric_height.dtype == torch.float32


def test_model_trainable_param_count_stays_below_budget() -> None:
    model = V23HeightPredictor(
        in_channels=15,
        config=_tiny_config(),
        load_pretrained=False,
        schema_input_size=56,
    )
    trainable = sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
    assert trainable < 8_000_000
