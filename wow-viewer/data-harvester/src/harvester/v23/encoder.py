"""DepthAnything-backed V23 encoder for Spec 089 Phase 2.

This wrapper targets the local Hugging Face ``DepthAnything`` implementation that
ships with ``transformers``. It keeps pretrained loading optional so the source
tree and tests remain runnable in offline environments while preserving the
production contract for cached HF checkpoints.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import copy
import warnings

import torch
from peft import LoraConfig, get_peft_model
from torch import nn
from transformers import AutoModelForDepthEstimation, DepthAnythingConfig, DepthAnythingForDepthEstimation

_DEFAULT_MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"


@dataclass(frozen=True)
class V23FeaturePyramid:
    """Feature pyramid emitted by :class:`DepthAnythingV2SmallEncoder`.

    Attributes:
        raw_feature_maps: Sequence-first backbone tokens from the four selected
            transformer stages. Each tensor is shaped ``[B, N, C]``.
        neck_features: Spatial feature pyramid after the DepthAnything neck.
            Shapes grow from coarse to fine, e.g. ``[B, C, H, W]``.
        patch_height: Patch-grid height for the current input.
        patch_width: Patch-grid width for the current input.
        input_height: Original input height.
        input_width: Original input width.
    """

    raw_feature_maps: tuple[torch.Tensor, ...]
    neck_features: tuple[torch.Tensor, ...]
    patch_height: int
    patch_width: int
    input_height: int
    input_width: int


def _clone_depth_anything_config(config: DepthAnythingConfig | None) -> DepthAnythingConfig:
    if config is None:
        return DepthAnythingConfig()
    return DepthAnythingConfig.from_dict(config.to_dict())


def _load_depth_anything_model(
    *,
    model_id: str,
    config: DepthAnythingConfig | None,
    load_pretrained: bool,
    local_files_only: bool,
) -> DepthAnythingForDepthEstimation:
    resolved_config = _clone_depth_anything_config(config)
    if not load_pretrained:
        return DepthAnythingForDepthEstimation(resolved_config)

    try:
        model = AutoModelForDepthEstimation.from_pretrained(model_id, local_files_only=local_files_only)
    except OSError as exc:
        warnings.warn(
            f"Falling back to random DepthAnything init because '{model_id}' was unavailable locally: {exc}",
            RuntimeWarning,
            stacklevel=2,
        )
        return DepthAnythingForDepthEstimation(resolved_config)

    if not isinstance(model, DepthAnythingForDepthEstimation):
        raise TypeError(f"Expected DepthAnythingForDepthEstimation, got {type(model).__name__}")
    return model


def _resolve_patch_projection(module: nn.Module) -> nn.Conv2d:
    if hasattr(module, "embeddings") and hasattr(module.embeddings, "patch_embeddings"):
        projection = module.embeddings.patch_embeddings.projection
        if not isinstance(projection, nn.Conv2d):
            raise TypeError("DepthAnything patch embedding projection is not a Conv2d")
        return projection

    if hasattr(module, "base_model"):
        nested = getattr(module.base_model, "model", module.base_model)
        return _resolve_patch_projection(nested)

    raise TypeError(f"Could not locate patch embedding projection on {type(module).__name__}")


def _replace_patch_projection(module: nn.Module, new_projection: nn.Conv2d) -> None:
    if hasattr(module, "embeddings") and hasattr(module.embeddings, "patch_embeddings"):
        module.embeddings.patch_embeddings.projection = new_projection
        module.embeddings.patch_embeddings.num_channels = int(new_projection.in_channels)
        return
    if hasattr(module, "base_model"):
        nested = getattr(module.base_model, "model", module.base_model)
        _replace_patch_projection(nested, new_projection)
        return
    raise TypeError(f"Could not replace patch embedding projection on {type(module).__name__}")


def _build_patch_projection(original: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    if int(original.in_channels) == int(in_channels):
        return original

    replacement = nn.Conv2d(
        in_channels=in_channels,
        out_channels=original.out_channels,
        kernel_size=original.kernel_size,
        stride=original.stride,
        padding=original.padding,
        bias=original.bias is not None,
    )
    nn.init.normal_(replacement.weight, mean=0.0, std=0.02)
    if replacement.bias is not None:
        nn.init.zeros_(replacement.bias)

    with torch.no_grad():
        preserved = min(int(original.in_channels), int(in_channels))
        replacement.weight[:, :preserved].copy_(original.weight[:, :preserved])
        if in_channels > original.in_channels:
            mean_weight = original.weight.mean(dim=1, keepdim=True)
            replacement.weight[:, preserved:].copy_(mean_weight.expand(-1, in_channels - preserved, -1, -1))
        if replacement.bias is not None and original.bias is not None:
            replacement.bias.copy_(original.bias)

    return replacement


class DepthAnythingV2SmallEncoder(nn.Module):
    """Frozen DepthAnything backbone + neck with LoRA adapters on attention projections.

    The local ``transformers`` package currently exposes the ``DepthAnything``
    model family rather than a separate ``depth_anything_v2`` module. This
    wrapper therefore binds to ``DepthAnythingForDepthEstimation`` while keeping
    the public V23 contract stable.
    """

    def __init__(
        self,
        in_channels: int,
        *,
        model_id: str = _DEFAULT_MODEL_ID,
        config: DepthAnythingConfig | None = None,
        base_model: DepthAnythingForDepthEstimation | None = None,
        load_pretrained: bool = False,
        local_files_only: bool = True,
        lora_rank: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.in_channels = int(in_channels)
        self.model_id = str(model_id)
        self.load_pretrained = bool(load_pretrained)
        self.local_files_only = bool(local_files_only)

        model = (
            copy.deepcopy(base_model)
            if base_model is not None
            else _load_depth_anything_model(
                model_id=self.model_id,
                config=config,
                load_pretrained=self.load_pretrained,
                local_files_only=self.local_files_only,
            )
        )
        if not isinstance(model, DepthAnythingForDepthEstimation):
            raise TypeError(f"Expected DepthAnythingForDepthEstimation, got {type(model).__name__}")

        self.config = _clone_depth_anything_config(model.config)
        self.config.backbone_config.num_channels = self.in_channels

        for parameter in model.backbone.parameters():
            parameter.requires_grad = False
        for parameter in model.neck.parameters():
            parameter.requires_grad = False

        patch_projection = _resolve_patch_projection(model.backbone)
        replacement_projection = _build_patch_projection(patch_projection, self.in_channels)
        _replace_patch_projection(model.backbone, replacement_projection)

        lora_config = LoraConfig(
            target_modules=["query", "key", "value", "dense"],
            r=int(lora_rank),
            lora_alpha=int(lora_alpha),
            lora_dropout=float(lora_dropout),
            bias="none",
        )
        self.backbone = get_peft_model(model.backbone, lora_config)
        self.neck = model.neck
        self.patch_embed_projection = _resolve_patch_projection(self.backbone)
        for parameter in self.patch_embed_projection.parameters():
            parameter.requires_grad = True

    def gradient_checkpointing_enable(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_enable"):
            self.backbone.gradient_checkpointing_enable()

    def gradient_checkpointing_disable(self) -> None:
        if hasattr(self.backbone, "gradient_checkpointing_disable"):
            self.backbone.gradient_checkpointing_disable()

    @contextmanager
    def disable_lora(self):
        """Temporarily disable LoRA adapters for bitwise baseline comparisons."""
        with self.backbone.disable_adapter():
            yield

    def forward(self, x: torch.Tensor) -> V23FeaturePyramid:
        outputs = self.backbone(x, output_hidden_states=False, output_attentions=False, return_dict=True)
        raw_feature_maps = tuple(outputs.feature_maps)
        patch_size = int(self.config.patch_size)
        patch_height = int(x.shape[-2] // patch_size)
        patch_width = int(x.shape[-1] // patch_size)
        neck_features = tuple(self.neck(list(raw_feature_maps), patch_height, patch_width))
        return V23FeaturePyramid(
            raw_feature_maps=raw_feature_maps,
            neck_features=neck_features,
            patch_height=patch_height,
            patch_width=patch_width,
            input_height=int(x.shape[-2]),
            input_width=int(x.shape[-1]),
        )
