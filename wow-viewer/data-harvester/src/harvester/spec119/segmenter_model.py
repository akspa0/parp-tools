"""Spec 119 US2 object segmenter (T016, research D-02).

A small from-scratch U-Net-lite over a single 128x128x3 object-library capture image, predicting
one binary foreground channel — the "reproduce the renderer's silhouette from RGB alone"
learnability test (US2). Mirrors ``ObjectSegmentNet``'s shape (Spec 118) but at 128x128 and
binary, and is a SEPARATE independently checkpointed specialist (Rule 7). Constructable from
``base`` alone (D-02) so inference rebuilds the exact architecture from the checkpoint config.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import torch
from torch import nn


class _DoubleConv(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )


class ObjectSegmenter(nn.Module):
    """U-Net-lite: encoder 128->64->32->16, skip decoder back to 128, one foreground channel."""

    def __init__(self, base: int = 16) -> None:
        super().__init__()
        if base < 1:
            raise ValueError(f"base must be positive; got {base}")
        self.base = int(base)
        b = base
        self.inc = _DoubleConv(3, b)
        self.down1 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(b, b * 2))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(b * 2, b * 4))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), _DoubleConv(b * 4, b * 8))
        self.up1 = nn.ConvTranspose2d(b * 8, b * 4, kernel_size=2, stride=2)
        self.conv1 = _DoubleConv(b * 8, b * 4)
        self.up2 = nn.ConvTranspose2d(b * 4, b * 2, kernel_size=2, stride=2)
        self.conv2 = _DoubleConv(b * 4, b * 2)
        self.up3 = nn.ConvTranspose2d(b * 2, b, kernel_size=2, stride=2)
        self.conv3 = _DoubleConv(b * 2, b)
        self.outc = nn.Conv2d(b, 1, kernel_size=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Foreground probability map ``(B, 1, 128, 128)`` in [0, 1]."""
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        y = self.conv1(torch.cat([self.up1(x4), x3], dim=1))
        y = self.conv2(torch.cat([self.up2(y), x2], dim=1))
        y = self.conv3(torch.cat([self.up3(y), x1], dim=1))
        return torch.sigmoid(self.outc(y))


def binary_iou(prediction: np.ndarray, target: np.ndarray) -> float:
    """Per-image foreground IoU; both args binary {0,1}. All-background-vs-all-background = 1.0."""
    p = np.asarray(prediction) > 0
    t = np.asarray(target) > 0
    union = (p | t).sum()
    if union == 0:
        return 1.0
    return float((p & t).sum() / union)


def trivial_iou_baselines(targets: Sequence[np.ndarray]) -> dict[str, float]:
    """The SC-002 trivial baselines over a set of binary targets.

    ``all_foreground`` IoU is exactly the mean coverage; ``all_background`` IoU is 1.0 only for
    all-background targets and 0.0 otherwise (blank captures are excluded upstream, D-04, so in
    practice it is ~0). Both are recorded so "beats the trivial baseline" is checkable.
    """
    if not targets:
        return {"all_foreground": 0.0, "all_background": 0.0}
    foreground = []
    background = []
    for target in targets:
        t = np.asarray(target) > 0
        coverage = float(t.mean())
        foreground.append(coverage)
        background.append(0.0 if t.any() else 1.0)
    return {
        "all_foreground": float(np.mean(foreground)),
        "all_background": float(np.mean(background)),
    }


COVERAGE_BUCKETS: tuple[tuple[float, float], ...] = ((0.0, 0.05), (0.05, 0.2), (0.2, 0.5), (0.5, 1.01))


def per_coverage_bucket_iou(
    ious: Sequence[float], coverages: Sequence[float]
) -> dict[str, dict[str, float | int | None]]:
    """Held-out IoU stratified by ground-truth mask-coverage bucket (US2 acceptance 2, SC-002)."""
    out: dict[str, dict[str, float | int | None]] = {}
    for low, high in COVERAGE_BUCKETS:
        bucket = [iou for iou, cov in zip(ious, coverages, strict=True) if low <= cov < high]
        out[f"[{low:.2f},{high:.2f})"] = {
            "count": len(bucket),
            "mean_iou": float(np.mean(bucket)) if bucket else None,
        }
    return out


__all__ = [
    "COVERAGE_BUCKETS",
    "ObjectSegmenter",
    "binary_iou",
    "per_coverage_bucket_iou",
    "trivial_iou_baselines",
]
