"""SAM 3.1 Minimap Sieve for Object & Contamination Masking.

Leverages the ComfyUI orchestration layer to segment elevated structures, buildings,
and placed 3D doodads from top-down minimap tiles.

NOTE: Roads, trails, cobblestone, and paths are 2D alpha texture splats (MCLY/MCAL)
painted directly on the terrain heightmap. They are NOT placed 3D objects or Rosetta
assets, and must NEVER be masked out as objects.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from PIL import Image
from scipy import ndimage

from harvester.v60.comfyui_orchestrator import ComfyUIConnectionError, ComfyUIOrchestrator

logger = logging.getLogger(__name__)


def compute_mask_iou(mask1: np.ndarray, mask2: np.ndarray) -> float:
    """Compute Intersection-over-Union (IoU) between two binary masks."""
    b1 = np.asarray(mask1, dtype=bool)
    b2 = np.asarray(mask2, dtype=bool)

    intersection = np.logical_and(b1, b2).sum()
    union = np.logical_or(b1, b2).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0
    return float(intersection / union)


def refine_mask_morphology(
    mask: np.ndarray,
    dilate_radius: int = 2,
    min_component_size: int = 16,
) -> np.ndarray:
    """Refine raw SAM mask with dilation and small island cleanup.

    Ensures object borders and anti-aliasing edges are fully occluded
    without excessively eroding fine terrain ridges.
    """
    m = (np.asarray(mask) > 127).astype(np.uint8)

    # 1. Remove tiny salt-and-pepper noise components
    if min_component_size > 0:
        labeled, num_features = ndimage.label(m)
        if num_features > 0:
            sizes = ndimage.sum(m, labeled, range(num_features + 1))
            mask_sizes = sizes < min_component_size
            remove_pixel = mask_sizes[labeled]
            m[remove_pixel] = 0

    # 2. Gentle dilation with circular kernel
    if dilate_radius > 0:
        y, x = np.ogrid[-dilate_radius : dilate_radius + 1, -dilate_radius : dilate_radius + 1]
        kernel = (x * x + y * y) <= (dilate_radius * dilate_radius)
        m = ndimage.binary_dilation(m, structure=kernel).astype(np.uint8)

    return (m * 255).astype(np.uint8)


class SamMinimapSieve:
    """Minimap object and structure sieve powered by SAM 3.1."""

    def __init__(
        self,
        orchestrator: Optional[ComfyUIOrchestrator] = None,
        base_url: str = "http://127.0.0.1:8199",
    ):
        self.orchestrator = orchestrator or ComfyUIOrchestrator(base_url=base_url)

    def sieve_objects(
        self,
        image_data: Union[bytes, np.ndarray, Image.Image, Path, str],
        threshold: float = 0.35,
        refine_iterations: int = 2,
        prompt_text: str = "building, house, roof, tower, castle, tent, elevated 3d doodad",
        dilate_radius: int = 2,
        timeout_seconds: float = 60.0,
    ) -> np.ndarray:
        """Sieve objects from a minimap image crop. Returns refined binary mask (H, W) uint8."""
        raw_mask = self.orchestrator.segment_objects(
            image_data=image_data,
            threshold=threshold,
            refine_iterations=refine_iterations,
            prompt_text=prompt_text,
            timeout_seconds=timeout_seconds,
        )
        return refine_mask_morphology(raw_mask, dilate_radius=dilate_radius)

    @staticmethod
    def heuristic_color_sieve(
        image_rgb: np.ndarray,
        roof_red_threshold: float = 0.25,
        roof_blue_threshold: float = 0.2,
    ) -> np.ndarray:
        """Fallback / synthetic heuristic sieve based on high-contrast chroma anomalies."""
        arr = np.asarray(image_rgb, dtype=np.float32)
        if arr.max() > 1.0:
            arr /= 255.0

        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        # Identify strong unnatural saturation or synthetic colored roofs
        red_diff = r - np.maximum(g, b)
        blue_diff = b - np.maximum(r, g)
        mask = (red_diff > roof_red_threshold) | (blue_diff > roof_blue_threshold)
        return refine_mask_morphology(mask.astype(np.uint8) * 255, dilate_radius=2)
