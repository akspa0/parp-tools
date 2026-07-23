"""Spec 118 US3: the from-scratch visible-object segmenter and its target contract.

``ObjectSegmentNet`` maps one 256x256 minimap tile to per-pixel visible-object class logits
(``none``/``doodad``/``building``, data-model.md Segmentation Target). It is deliberately a small
U-Net-lite in the same family as the project's other independently checkpointed specialists: a
strided double-conv encoder (256->128->64->32->16) with a skip-connected decoder back to full
resolution -- no pretrained backbone, no DepthAnything-family anything (FR-010, constitution IV),
and constructable from ``base`` alone so ``object_feature_bridge.py`` reconstructs it unchanged
from the checkpoint's ``object_config``.

The supervision target derives from ``object_geometry_visible_source_257`` (the strict, already
visibility-corrected per-pixel class from US1), cropped 257->256 with the same top-left convention
as the C# ``Crop257To256`` used for the liquid arrays. Semantic class segmentation is the first
target; per-instance separation from the minimap is an explicit stretch goal (spec Assumptions).
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from harvester.spec118.object_contract import CLASS_COUNT, CLASS_NAMES

INPUT_SIZE = 256


class ObjectSegmentError(ValueError):
    """Raised when an object-segmentation target or input violates its declared contract."""


def derive_class_target(object_mask_257: np.ndarray, *, threshold: float = 0.5) -> np.ndarray:
    """Map a raw ``object_mask`` (or ``object_precise_mask``) tile to the (256,256) int64 target.

    Binary vocabulary (0 none / 1 object): the v18 placement-footprint mask is a float in [0, 1]
    (hard 1.0 for `object_mask`, soft edges for `object_precise_mask`); a pixel is class 1 wherever
    the footprint exceeds ``threshold``. Cropped 257->256 with the top-left convention the C#
    ``Crop257To256`` uses. Raises on the wrong shape so a mis-fed array fails closed.
    """
    mask = np.asarray(object_mask_257, dtype=np.float32)
    if mask.shape != (257, 257):
        raise ObjectSegmentError(f"expected (257,257) object mask, got {mask.shape}")
    cropped = mask[:INPUT_SIZE, :INPUT_SIZE]
    return (cropped > threshold).astype(np.int64)


def per_class_iou_recall(predicted: np.ndarray, target: np.ndarray) -> dict[str, dict[str, float]]:
    """Per-class IoU and recall between two (…,H,W) class-id arrays.

    A class absent from BOTH prediction and target reports ``None`` (not 0.0 -- an absent class is
    not a failure); a class in the target but never predicted reports recall 0.0 honestly.
    """
    pred = np.asarray(predicted).reshape(-1)
    truth = np.asarray(target).reshape(-1)
    if pred.shape != truth.shape:
        raise ObjectSegmentError(f"shape mismatch: predicted {pred.shape} vs target {truth.shape}")
    out: dict[str, dict[str, float]] = {}
    for class_id, name in enumerate(CLASS_NAMES):
        p = pred == class_id
        t = truth == class_id
        intersection = int((p & t).sum())
        union = int((p | t).sum())
        out[name] = {
            "iou": (intersection / union) if union else None,
            "recall": (intersection / int(t.sum())) if t.sum() else None,
            "support": int(t.sum()),
        }
    return out


def visible_object_iou(predicted: np.ndarray, target: np.ndarray) -> float | None:
    """IoU of the UNION of object classes (doodad+building) -- the D-07 gate metric."""
    pred = np.asarray(predicted) > 0
    truth = np.asarray(target) > 0
    union = int((pred | truth).sum())
    if union == 0:
        return None
    return int((pred & truth).sum()) / union


def compute_class_weights(targets: list[np.ndarray], *, none_cap: float = 1.0) -> torch.Tensor:
    """Inverse-frequency class weights over a target list, with ``none`` capped.

    Background dominates every tile, so an uncapped inverse frequency would let ``none`` swamp the
    object classes. The cap keeps the background weight at or below ``none_cap`` while the rare
    object classes keep their full inverse frequency.
    """
    counts = np.zeros(CLASS_COUNT, dtype=np.float64)
    for target in targets:
        counts += np.bincount(np.asarray(target).reshape(-1), minlength=CLASS_COUNT)
    total = counts.sum()
    weights = np.where(counts > 0, total / (CLASS_COUNT * np.maximum(counts, 1.0)), 0.0)
    # A class absent from the TRAIN targets gets the rarest-present weight, not 0.0: zero weight
    # would silently ignore that class's pixels wherever they do occur (e.g. a rare class present
    # only in a handful of tiles), and a never-trainable weight is exactly the kind of quiet
    # failure this project's contracts exist to prevent.
    present = weights[counts > 0]
    if present.size:
        weights[counts == 0] = present.max()
    weights[0] = min(weights[0], none_cap)
    return torch.from_numpy(weights.astype(np.float32))


def _block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    """Conv3x3 + GroupNorm + SiLU -- the same block the other v50 specialists use."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1),
        nn.GroupNorm(min(8, out_ch), out_ch),
        nn.SiLU(inplace=True),
    )


def _double_block(in_ch: int, out_ch: int, stride: int = 1) -> nn.Sequential:
    return nn.Sequential(
        _block(in_ch, out_ch, stride=stride),
        _block(out_ch, out_ch, stride=1),
    )


class ObjectSegmentNet(nn.Module):
    """U-Net-lite pixel classifier: 256x256x3 minimap RGB -> (B, 3, 256, 256) class logits.

    Strided double-conv encoder 256->128->64->32->16, then a skip-connected decoder back to full
    resolution (the Spec 117 v2 lesson: a bottleneck-only head cannot localize). Constructable
    from ``base`` alone; independently weighted from every other stage (constitution IV / FR-010).
    """

    def __init__(self, base: int = 24) -> None:
        super().__init__()
        self.enc1 = _double_block(3, base)                        # 256, b
        self.enc2 = _double_block(base, base * 2, stride=2)       # 128, 2b
        self.enc3 = _double_block(base * 2, base * 4, stride=2)   # 64, 4b
        self.enc4 = _double_block(base * 4, base * 8, stride=2)   # 32, 8b
        self.enc5 = _double_block(base * 8, base * 8, stride=2)   # 16, 8b
        self.dec4 = _double_block(base * 16, base * 4)            # 32 (cat e5↑ + e4)
        self.dec3 = _double_block(base * 8, base * 2)             # 64 (cat d4↑ + e3)
        self.dec2 = _double_block(base * 4, base)                 # 128 (cat d3↑ + e2)
        self.dec1 = _double_block(base * 2, base)                 # 256 (cat d2↑ + e1)
        self.head = nn.Conv2d(base, CLASS_COUNT, 1)

    @staticmethod
    def _upsample(x: torch.Tensor, like: torch.Tensor) -> torch.Tensor:
        return nn.functional.interpolate(x, size=like.shape[-2:], mode="nearest")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 4 or x.shape[1] != 3:
            raise ObjectSegmentError(f"ObjectSegmentNet consumes (B, 3, H, W); got {tuple(x.shape)}")
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        e5 = self.enc5(e4)
        d4 = self.dec4(torch.cat([self._upsample(e5, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._upsample(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._upsample(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._upsample(d2, e1), e1], dim=1))
        return self.head(d1)


__all__ = [
    "ObjectSegmentError",
    "INPUT_SIZE",
    "derive_class_target",
    "per_class_iou_recall",
    "visible_object_iou",
    "compute_class_weights",
    "ObjectSegmentNet",
]
