"""Spec 119 US1 object classifier (T010, research D-02).

A small classifier over a single object-library capture image. Supports multiple backbones:
- ``scratch``: from-scratch conv encoder (98K params @ base 16, 128 input, 128-d embedding)
- ``dinov2_vits14``: DINOv2 ViT-S/14 via transformers (21M params, 224 input, 384-d embedding)
- ``clip_vitb32``: CLIP ViT-B/32 via transformers (150M params, 224 input, 768-d embedding)
- ``timm/<model_name>``: any timm model (e.g. ``timm/efficientnet_b0``, ``timm/starnet_s1``)

Constructable from ``backbone`` + ``base`` (scratch) or ``backbone`` alone (pretrained) so
inference rebuilds the exact architecture from the checkpoint's config (D-02). The
penultimate-layer vector doubles as the US3 per-asset embedding.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from harvester.spec119.object_library_contract import COARSE_CLASS_INDEX

BACKBONE_CONFIGS: dict[str, dict] = {
    "scratch": {"input_size": 128, "embedding_dim": None, "model_id": None},
    "dinov2_vits14": {"input_size": 224, "embedding_dim": 384, "model_id": "facebook/dinov2-small"},
    "clip_vitb32": {"input_size": 224, "embedding_dim": 768, "model_id": "openai/clip-vit-base-patch32"},
}


def _scratch_encoder(base: int) -> nn.Module:
    """From-scratch conv encoder 128->64->32->16->8."""
    def _block(in_c: int, out_c: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )
    return nn.Sequential(
        _block(3, base),          # 128 -> 64
        _block(base, base * 2),   # 64 -> 32
        _block(base * 2, base * 4),  # 32 -> 16
        _block(base * 4, base * 8),  # 16 -> 8
    )


def _load_pretrained_backbone(backbone: str):
    """Load a pretrained vision backbone, return (model, embed_dim, input_size)."""
    if backbone.startswith("timm/"):
        import timm
        model_name = backbone[len("timm/"):]
        model = timm.create_model(model_name, pretrained=True, num_classes=0)
        embed_dim = model.num_features if hasattr(model, "num_features") else (
            model.head_hidden_size if hasattr(model, "head_hidden_size") else
            model.embed_dim if hasattr(model, "embed_dim") else
            model.config.hidden_size if hasattr(model, "config") else 512
        )
        # Get input size from model config
        cfg = model.default_cfg
        input_size = cfg.get("input_size", (3, 224, 224))[1]
        return model, embed_dim, input_size

    from transformers import AutoModel, CLIPModel

    cfg = BACKBONE_CONFIGS[backbone]
    embed_dim = cfg["embedding_dim"]
    input_size = cfg["input_size"]

    if backbone.startswith("dinov2"):
        model = AutoModel.from_pretrained(cfg["model_id"])
        return model, embed_dim, input_size

    if backbone.startswith("clip"):
        model = CLIPModel.from_pretrained(cfg["model_id"])
        return model.vision_model, embed_dim, input_size

    raise ValueError(f"Unknown pretrained backbone: {backbone}")


class ObjectClassifier(nn.Module):
    """Classifier over a single object-library capture image.

    ``forward`` returns class logits; ``embedding`` returns the penultimate fixed-length
    vector (the US3 per-asset embedding, FR-009).
    """

    def __init__(
        self,
        backbone: str = "scratch",
        base: int = 16,
        num_classes: int = len(COARSE_CLASS_INDEX),
    ) -> None:
        super().__init__()
        if backbone not in BACKBONE_CONFIGS and not backbone.startswith("timm/"):
            raise ValueError(f"Unknown backbone {backbone!r}; options: {sorted(BACKBONE_CONFIGS)} or timm/<model>")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2; got {num_classes}")

        self.backbone_name = backbone
        self.base = int(base)
        self.num_classes = int(num_classes)

        if backbone == "scratch":
            if base < 1:
                raise ValueError(f"base must be positive; got {base}")
            self.input_size = BACKBONE_CONFIGS["scratch"]["input_size"]
            self.encoder = _scratch_encoder(base)
            self.pool = nn.AdaptiveAvgPool2d(1)
            embed_dim = base * 8
        else:
            self.encoder, embed_dim, self.input_size = _load_pretrained_backbone(backbone)
            self.pool = nn.Identity()  # pretrained models have their own pooling

        self.embed_dim = embed_dim
        self.head = nn.Linear(embed_dim, num_classes)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate-layer vector ``(B, embed_dim)`` — the US3 per-asset embedding."""
        # Ensure input is on the same device as the model's parameters.
        param = next(self.parameters())
        x = x.to(param.device, dtype=param.dtype)

        if self.backbone_name == "scratch":
            return self.pool(self.encoder(x)).flatten(1)

        if self.backbone_name.startswith("timm/"):
            # timm models with num_classes=0 return pooled features directly
            return self.encoder(x)

        if self.backbone_name.startswith("dinov2"):
            # DINOv2: [CLS] token is first in last_hidden_state
            out = self.encoder(pixel_values=x)
            return out.last_hidden_state[:, 0, :]

        if self.backbone_name.startswith("clip"):
            # CLIP vision model: pooler_output is the [CLS] after projection
            out = self.encoder(x)
            return out.pooler_output

        raise ValueError(f"Unknown backbone: {self.backbone_name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(x))


def compute_class_weights(labels: Sequence[int], num_classes: int = len(COARSE_CLASS_INDEX)) -> np.ndarray:
    """Inverse-frequency class weights (FR-007), mean-normalized to 1.0."""
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    weights = np.zeros(num_classes, dtype=np.float64)
    present = [c for c in range(num_classes) if counts.get(c, 0) > 0]
    for c in present:
        weights[c] = total / counts[c]
    if present:
        weights[present] /= weights[present].mean()
    return weights


def majority_class_baseline(labels: Sequence[int]) -> float:
    """Held-out accuracy of a model that always predicts the training majority class (FR-005)."""
    counts = Counter(int(label) for label in labels)
    if not counts:
        return 0.0
    return max(counts.values()) / sum(counts.values())


def per_class_precision_recall(
    predictions: Sequence[int], targets: Sequence[int], num_classes: int
) -> dict[int, dict[str, float | None]]:
    """Per-class precision/recall (FR-007); None when the class has no support/predictions."""
    out: dict[int, dict[str, float | None]] = {}
    for c in range(num_classes):
        tp = sum(1 for p, t in zip(predictions, targets, strict=True) if p == c and t == c)
        pred_c = sum(1 for p in predictions if p == c)
        true_c = sum(1 for t in targets if t == c)
        out[c] = {
            "precision": (tp / pred_c) if pred_c else None,
            "recall": (tp / true_c) if true_c else None,
            "support": int(true_c),
        }
    return out


__all__ = [
    "BACKBONE_CONFIGS",
    "ObjectClassifier",
    "compute_class_weights",
    "majority_class_baseline",
    "per_class_precision_recall",
]
