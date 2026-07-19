"""Pinned general-visual student with one continuous universal-relief output (Spec 114)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

MODEL_ARCHITECTURE_ID = "dinov2_small_relief_v1"
OUTPUT_SIGNAL = "relative_relief"
INPUT_TILE_SIZE = 224
STUDENT_HUB_ID = "facebook/dinov2-small"
STUDENT_REVISION = "ed25f3a31f01632728cabb09d1542f84ab7b0056"
STUDENT_WEIGHT_FILE = "model.safetensors"
STUDENT_WEIGHTS_SHA256 = "ae1e99fcefd534ed978cdeb8326f08030c96e28b7a81ffcbc98a857c84d14be1"
STUDENT_LICENSE = "apache-2.0"
IMAGE_MEAN = (0.485, 0.456, 0.406)
IMAGE_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class StudentIdentity:
    architecture_id: str = MODEL_ARCHITECTURE_ID
    hub_id: str = STUDENT_HUB_ID
    revision: str = STUDENT_REVISION
    weight_file: str = STUDENT_WEIGHT_FILE
    weights_sha256: str = STUDENT_WEIGHTS_SHA256
    license: str = STUDENT_LICENSE
    output_signal: str = OUTPUT_SIGNAL
    input_tile_size: int = INPUT_TILE_SIZE


def student_identity() -> StudentIdentity:
    return StudentIdentity()


def student_identity_dict() -> dict[str, Any]:
    return asdict(student_identity())


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def verify_student_weight(path: str | Path, identity: StudentIdentity | None = None) -> None:
    expected = identity or student_identity()
    observed = sha256_file(path)
    if observed != expected.weights_sha256:
        raise ValueError(
            f"student weight hash mismatch: expected {expected.weights_sha256}, observed {observed}"
        )


def download_pinned_student_backbone(*, cache_dir: str | Path | None = None) -> nn.Module:
    """Download/load only the pinned safe DINOv2 backbone after the user starts a real run."""
    from huggingface_hub import snapshot_download
    from transformers import AutoModel

    identity = student_identity()
    snapshot = Path(
        snapshot_download(
            repo_id=identity.hub_id,
            revision=identity.revision,
            cache_dir=str(cache_dir) if cache_dir else None,
            allow_patterns=["*.json", "*.txt", identity.weight_file],
        )
    )
    verify_student_weight(snapshot / identity.weight_file, identity)
    return AutoModel.from_pretrained(
        snapshot,
        local_files_only=True,
        use_safetensors=True,
    )


def _decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.SiLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
        nn.GroupNorm(min(8, out_channels), out_channels),
        nn.SiLU(inplace=True),
    )


class UniversalReliefNet(nn.Module):
    """DINOv2 patch features -> one sigmoid-bounded view-axis-relief field.

    Arbitrary source dimensions are owned by ``universal_relief_contract`` tiling/stitching. This
    model consumes one 224x224 RGB tile and emits one 224x224 relief tile. The default backbone is
    frozen initially so the first experiment trains only the compact decoder; later unfreezing must
    be a recorded ablation, not an implicit behavior change.
    """

    deployment_inputs = ("rgb",)
    output_signal = OUTPUT_SIGNAL

    def __init__(self, backbone: nn.Module, *, freeze_backbone: bool = True) -> None:
        super().__init__()
        self.backbone = backbone
        self.freeze_backbone = freeze_backbone
        hidden_size = int(backbone.config.hidden_size)
        self.patch_size = int(backbone.config.patch_size)
        self.projection = nn.Conv2d(hidden_size, 192, kernel_size=1)
        self.decoder = nn.ModuleList(
            [
                _decoder_block(192, 128),
                _decoder_block(128, 96),
                _decoder_block(96, 64),
                _decoder_block(64, 32),
            ]
        )
        self.head = nn.Conv2d(32, 1, kernel_size=3, padding=1)
        self.register_buffer(
            "image_mean",
            torch.tensor(IMAGE_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        self.register_buffer(
            "image_std",
            torch.tensor(IMAGE_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=True,
        )
        if freeze_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()

    def train(self, mode: bool = True) -> UniversalReliefNet:
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def forward(self, rgb: torch.Tensor) -> torch.Tensor:
        if rgb.ndim != 4 or rgb.shape[1] != 3:
            raise ValueError("universal relief input must have shape Bx3xHxW")
        height, width = rgb.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"input dimensions must be divisible by patch size {self.patch_size}")
        normalized = (rgb - self.image_mean) / self.image_std
        if self.freeze_backbone:
            with torch.no_grad():
                encoded = self.backbone(pixel_values=normalized).last_hidden_state
        else:
            encoded = self.backbone(pixel_values=normalized).last_hidden_state
        patch_tokens = encoded[:, 1:, :]
        patch_height = height // self.patch_size
        patch_width = width // self.patch_size
        expected_tokens = patch_height * patch_width
        if patch_tokens.shape[1] != expected_tokens:
            raise ValueError(
                f"backbone returned {patch_tokens.shape[1]} patch tokens, expected {expected_tokens}"
            )
        features = patch_tokens.transpose(1, 2).reshape(
            rgb.shape[0], patch_tokens.shape[2], patch_height, patch_width
        )
        decoded = self.projection(features)
        for block in self.decoder:
            decoded = nn.functional.interpolate(
                decoded, scale_factor=2.0, mode="bilinear", align_corners=False
            )
            decoded = block(decoded)
        decoded = nn.functional.interpolate(
            decoded, size=(height, width), mode="bilinear", align_corners=False
        )
        return torch.sigmoid(self.head(decoded)).squeeze(1)

    def trainable_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters() if parameter.requires_grad)

    def total_parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())
