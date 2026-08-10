"""Deterministic training-only coarse/detail targets for the clean-signal v60 lane."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

TARGET_DECOMPOSITION_VERSION = "v7-clean-signal-target-v1"
LOW_PASS_VERSION = "box9-edge-replicate-v1"
LOW_PASS_KERNEL_SIZE = 9
RANGE_FLOOR = 1.0
TARGET_SHAPE = (257, 257)


class CleanTargetError(ValueError):
    """Raised when a structural target cannot be represented safely."""


def encode_relative_height(height: Any, *, require_target_shape: bool = False) -> tuple[np.ndarray, float, float]:
    """Encode a finite height grid with the v112-compatible per-tile range floor."""

    array = np.asarray(height, dtype=np.float64)
    if array.ndim != 2 or array.size == 0 or not np.isfinite(array).all():
        raise CleanTargetError("height must be a non-empty finite 2D array")
    if require_target_shape and array.shape != TARGET_SHAPE:
        raise CleanTargetError(f"height shape {array.shape} != {TARGET_SHAPE}")
    tile_min = float(array.min())
    tile_max = float(array.max())
    denominator = max(tile_max - tile_min, RANGE_FLOOR)
    normalized = np.clip((array - tile_min) / denominator, 0.0, 1.0).astype(np.float32)
    return normalized, tile_min, tile_max


def _box_blur(array: np.ndarray, kernel_size: int = LOW_PASS_KERNEL_SIZE) -> np.ndarray:
    if kernel_size < 1 or kernel_size % 2 == 0:
        raise CleanTargetError("low-pass kernel_size must be a positive odd integer")
    pad = kernel_size // 2
    padded = np.pad(array.astype(np.float64), ((pad, pad), (pad, pad)), mode="edge")
    integral = np.pad(
        np.cumsum(np.cumsum(padded, axis=0), axis=1),
        ((1, 0), (1, 0)),
    )
    height, width = array.shape
    summed = (
        integral[kernel_size : kernel_size + height, kernel_size : kernel_size + width]
        - integral[:height, kernel_size : kernel_size + width]
        - integral[kernel_size : kernel_size + height, :width]
        + integral[:height, :width]
    )
    return (summed / float(kernel_size * kernel_size)).astype(np.float32)


@dataclass(frozen=True)
class StructuralTarget:
    """Exact target decomposition; all fields are training/evaluation side only."""

    height_257: np.ndarray
    relative_height_257: np.ndarray
    coarse_relief_257: np.ndarray
    detail_residual_257: np.ndarray
    tile_min: float
    tile_max: float
    decomposition_version: str = TARGET_DECOMPOSITION_VERSION
    low_pass_version: str = LOW_PASS_VERSION

    def __post_init__(self) -> None:
        arrays = (
            self.height_257,
            self.relative_height_257,
            self.coarse_relief_257,
            self.detail_residual_257,
        )
        for name, array in zip(
            ("height_257", "relative_height_257", "coarse_relief_257", "detail_residual_257"),
            arrays,
            strict=True,
        ):
            value = np.asarray(array, dtype=np.float32)
            if value.shape != TARGET_SHAPE:
                raise CleanTargetError(f"{name} shape {value.shape} != {TARGET_SHAPE}")
            if not np.isfinite(value).all():
                raise CleanTargetError(f"{name} contains non-finite values")
            object.__setattr__(self, name, np.ascontiguousarray(value))
        if not np.allclose(
            self.relative_height_257,
            self.coarse_relief_257 + self.detail_residual_257,
            atol=2e-6,
            rtol=0.0,
        ):
            raise CleanTargetError("coarse_relief_257 + detail_residual_257 does not recompose target")

    @property
    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "height_257": self.height_257,
            "relative_height_257": self.relative_height_257,
            "coarse_relief_257": self.coarse_relief_257,
            "detail_residual_257": self.detail_residual_257,
        }


def decompose_relative_height(
    height_257: Any,
    *,
    kernel_size: int = LOW_PASS_KERNEL_SIZE,
) -> StructuralTarget:
    """Create the versioned relative-height, low-pass coarse, and signed detail fields."""

    height = np.asarray(height_257, dtype=np.float32)
    normalized, tile_min, tile_max = encode_relative_height(height, require_target_shape=True)
    coarse = _box_blur(normalized, kernel_size=kernel_size)
    detail = np.asarray(normalized - coarse, dtype=np.float32)
    low_pass_version = LOW_PASS_VERSION if kernel_size == LOW_PASS_KERNEL_SIZE else f"box{kernel_size}-edge-replicate-unversioned"
    return StructuralTarget(
        height_257=height,
        relative_height_257=normalized,
        coarse_relief_257=coarse,
        detail_residual_257=detail,
        tile_min=tile_min,
        tile_max=tile_max,
        low_pass_version=low_pass_version,
    )


def recompose_height(coarse_relief_257: Any, detail_residual_257: Any) -> np.ndarray:
    """Recompose a published height field before any output-boundary clamp."""

    coarse = np.asarray(coarse_relief_257, dtype=np.float32)
    detail = np.asarray(detail_residual_257, dtype=np.float32)
    if coarse.shape != TARGET_SHAPE or detail.shape != TARGET_SHAPE:
        raise CleanTargetError(
            f"coarse/detail shapes {coarse.shape}/{detail.shape} must both equal {TARGET_SHAPE}"
        )
    if not np.isfinite(coarse).all() or not np.isfinite(detail).all():
        raise CleanTargetError("coarse/detail fields must be finite")
    return np.ascontiguousarray(coarse + detail, dtype=np.float32)


__all__ = [
    "CleanTargetError",
    "LOW_PASS_KERNEL_SIZE",
    "LOW_PASS_VERSION",
    "RANGE_FLOOR",
    "StructuralTarget",
    "TARGET_DECOMPOSITION_VERSION",
    "TARGET_SHAPE",
    "decompose_relative_height",
    "encode_relative_height",
    "recompose_height",
]
