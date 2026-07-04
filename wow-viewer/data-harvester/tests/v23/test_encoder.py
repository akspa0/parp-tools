from __future__ import annotations

import copy

import pytest
import torch
from transformers import DepthAnythingConfig, DepthAnythingForDepthEstimation

from harvester.v23.encoder import DepthAnythingV2SmallEncoder

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


def test_encoder_freezes_base_and_trains_lora_plus_patch_embed() -> None:
    encoder = DepthAnythingV2SmallEncoder(in_channels=15, config=_tiny_config(), load_pretrained=False)

    lora_params = [parameter for name, parameter in encoder.named_parameters() if "lora_" in name]
    assert lora_params
    assert sum(parameter.numel() for parameter in lora_params) < 2_000_000

    for parameter in encoder.patch_embed_projection.parameters():
        assert parameter.requires_grad

    frozen_params = [
        parameter
        for name, parameter in encoder.named_parameters()
        if "lora_" not in name and "patch_embeddings.projection" not in name
    ]
    assert frozen_params
    assert all(not parameter.requires_grad for parameter in frozen_params)


def test_encoder_forward_returns_documented_feature_pyramid() -> None:
    encoder = DepthAnythingV2SmallEncoder(in_channels=15, config=_tiny_config(), load_pretrained=False)
    x = torch.randn(2, 15, 56, 56)
    features = encoder(x)

    assert len(features.raw_feature_maps) == 4
    assert [tuple(level.shape) for level in features.raw_feature_maps] == [
        (2, 17, 64),
        (2, 17, 64),
        (2, 17, 64),
        (2, 17, 64),
    ]
    assert [tuple(level.shape) for level in features.neck_features] == [
        (2, 32, 4, 4),
        (2, 32, 8, 8),
        (2, 32, 16, 16),
        (2, 32, 32, 32),
    ]
    assert features.patch_height == 4
    assert features.patch_width == 4


def test_encoder_disable_lora_matches_stock_model_on_three_channel_input() -> None:
    stock = DepthAnythingForDepthEstimation(_tiny_config())
    encoder = DepthAnythingV2SmallEncoder(
        in_channels=3,
        base_model=copy.deepcopy(stock),
        load_pretrained=False,
    )

    for seed in (7, 11, 29):
        torch.manual_seed(seed)
        x = torch.randn(2, 3, 56, 56)
        with torch.inference_mode():
            stock_raw = tuple(stock.backbone(x).feature_maps)
            stock_neck = tuple(stock.neck(list(stock_raw), 4, 4))
            with encoder.disable_lora():
                encoded = encoder(x)

        assert len(stock_raw) == len(encoded.raw_feature_maps)
        assert len(stock_neck) == len(encoded.neck_features)
        for expected, observed in zip(stock_raw, encoded.raw_feature_maps, strict=True):
            assert torch.allclose(expected, observed, atol=0.0, rtol=0.0)
        for expected, observed in zip(stock_neck, encoded.neck_features, strict=True):
            assert torch.allclose(expected, observed, atol=0.0, rtol=0.0)
