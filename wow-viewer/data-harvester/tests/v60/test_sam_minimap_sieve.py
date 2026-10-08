"""Unit tests for SAM 3.1 minimap sieve."""

from __future__ import annotations

import numpy as np
import pytest

from harvester.v60.sam_minimap_sieve import (
    SamMinimapSieve,
    compute_mask_iou,
    refine_mask_morphology,
)


def test_compute_mask_iou_exact():
    m1 = np.zeros((32, 32), dtype=np.uint8)
    m2 = np.zeros((32, 32), dtype=np.uint8)

    m1[10:20, 10:20] = 255
    m2[10:20, 10:20] = 255

    assert compute_mask_iou(m1, m2) == 1.0


def test_compute_mask_iou_partial():
    m1 = np.zeros((32, 32), dtype=np.uint8)
    m2 = np.zeros((32, 32), dtype=np.uint8)

    m1[10:20, 10:20] = 255  # 100 pixels
    m2[15:25, 10:20] = 255  # 100 pixels, overlap 50 pixels

    # intersection = 50, union = 150 -> iou = 50 / 150 = 1/3
    iou = compute_mask_iou(m1, m2)
    assert abs(iou - (1.0 / 3.0)) < 1e-4


def test_refine_mask_morphology_removes_noise_and_dilates():
    raw = np.zeros((64, 64), dtype=np.uint8)

    # 1. Tiny isolated noise island (< 16 pixels)
    raw[5, 5] = 255
    raw[5, 6] = 255

    # 2. Substantial object block (10x10 = 100 pixels)
    raw[30:40, 30:40] = 255

    refined = refine_mask_morphology(raw, dilate_radius=2, min_component_size=16)

    # Noise island should be removed
    assert refined[5, 5] == 0
    assert refined[5, 6] == 0

    # Substantial block should be preserved and dilated
    assert refined[35, 35] == 255
    # Dilation by radius 2 should expand beyond 30..40
    assert refined[29, 35] == 255
    assert refined[41, 35] == 255


def test_heuristic_color_sieve():
    # Synthetic terrain tile (green/brown) with red building roof
    img = np.zeros((32, 32, 3), dtype=np.float32)
    img[..., 1] = 0.5  # Green terrain

    # Red roof block
    img[12:20, 12:20, 0] = 0.95
    img[12:20, 12:20, 1] = 0.2
    img[12:20, 12:20, 2] = 0.1

    mask = SamMinimapSieve.heuristic_color_sieve(img)
    assert mask.shape == (32, 32)
    assert mask[15, 15] == 255
    assert mask[0, 0] == 0
