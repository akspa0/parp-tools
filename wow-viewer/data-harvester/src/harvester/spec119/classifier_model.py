"""Spec 119 US1 object classifier (T010, research D-02).

A small from-scratch conv encoder + global pool + linear head over a single 128x128x3
object-library capture image. No pretrained backbone (FR-003/SC-005). Constructable from
``base`` alone so inference rebuilds the exact architecture from the checkpoint's config
(D-02, mirroring the Spec 117/118 bridge pattern). The penultimate-layer (global-pooled)
vector doubles as the US3 per-asset embedding.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence

import numpy as np
import torch
from torch import nn

from harvester.spec119.object_library_contract import COARSE_CLASS_INDEX


def _block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    )


class ObjectClassifier(nn.Module):
    """Conv encoder 128->64->32->16->8, global pool, linear head.

    ``forward`` returns class logits; ``embedding`` returns the penultimate fixed-length
    vector (the US3 per-asset embedding, FR-009).
    """

    def __init__(self, base: int = 16, num_classes: int = len(COARSE_CLASS_INDEX)) -> None:
        super().__init__()
        if base < 1:
            raise ValueError(f"base must be positive; got {base}")
        if num_classes < 2:
            raise ValueError(f"num_classes must be >= 2; got {num_classes}")
        self.base = int(base)
        self.num_classes = int(num_classes)
        width = base
        self.encoder = nn.Sequential(
            _block(3, width),          # 128 -> 64
            _block(width, width * 2),  # 64 -> 32
            _block(width * 2, width * 4),  # 32 -> 16
            _block(width * 4, width * 8),  # 16 -> 8
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(width * 8, num_classes)

    def embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Penultimate-layer vector ``(B, base*8)`` — the US3 per-asset embedding."""
        return self.pool(self.encoder(x)).flatten(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(x))


def compute_class_weights(labels: Sequence[int], num_classes: int = len(COARSE_CLASS_INDEX)) -> np.ndarray:
    """Inverse-frequency class weights (FR-007), mean-normalized to 1.0.

    A class absent from the training labels gets weight 0.0 (it cannot be learned, and a
    nonzero weight would only distort the loss scale).
    """
    counts = Counter(int(label) for label in labels)
    total = sum(counts.values())
    weights = np.zeros(num_classes, dtype=np.float64)
    present = [c for c in range(num_classes) if counts.get(c, 0) > 0]
    for c in present:
        weights[c] = total / counts[c]
    if present:
        weights[present] /= weights[present].mean()  # mean-normalize present classes to 1.0
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
    "ObjectClassifier",
    "compute_class_weights",
    "majority_class_baseline",
    "per_class_precision_recall",
]
