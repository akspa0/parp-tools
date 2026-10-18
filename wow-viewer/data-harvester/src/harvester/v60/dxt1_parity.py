"""Fast DXT1 / BC1 block quantization parity codec (Spec 274).

Replicates the exact DXT1 4x4 block compression and RGB565 color endpoint
quantization cycle implemented in `Dxt1TileCodec.cs` using vectorized NumPy.

Used to eliminate the codec confound when generating synthetic hillshade
training targets, ensuring they share the exact block-quantization noise
floor of real authored DXT1 minimap tiles.
"""

from __future__ import annotations

import numpy as np


def quantize_rgb565(rgb: np.ndarray) -> np.ndarray:
    """Quantize RGB float or uint8 in [0, 255] to RGB565 representation in [0, 255]."""
    r = np.round(rgb[..., 0] * (31.0 / 255.0)) * (255.0 / 31.0)
    g = np.round(rgb[..., 1] * (63.0 / 255.0)) * (255.0 / 63.0)
    b = np.round(rgb[..., 2] * (31.0 / 255.0)) * (255.0 / 31.0)
    return np.round(np.stack([r, g, b], axis=-1)).astype(np.uint8)


def rgb565_to_uint16(rgb: np.ndarray) -> np.ndarray:
    """Convert RGB565 array (H, W, 3) to 16-bit integer representation."""
    r5 = (np.round(rgb[..., 0] * (31.0 / 255.0)).astype(np.uint16) & 0x1F) << 11
    g6 = (np.round(rgb[..., 1] * (63.0 / 255.0)).astype(np.uint16) & 0x3F) << 5
    b5 = np.round(rgb[..., 2] * (31.0 / 255.0)).astype(np.uint16) & 0x1F
    return r5 | g6 | b5


def dxt1_encode_decode_rgb(image: np.ndarray) -> np.ndarray:
    """Apply DXT1 encode/decode degradation cycle to an RGB image (H, W, 3) uint8 or float.

    Performs 4x4 block bounding-box RGB565 endpoint estimation and 4-color nearest-neighbor
    index quantization, matching the client DXT1 block banding.

    Args:
        image: Array of shape (H, W, 3) with H and W multiples of 4.
               If float in [0, 1], scaled to [0, 255]. Returns uint8 (H, W, 3).

    Returns:
        Decoded image array of shape (H, W, 3) dtype uint8.
    """
    if image.ndim == 2:
        image = np.stack([image, image, image], axis=-1)

    if np.issubdtype(image.dtype, np.floating):
        if image.max() <= 1.01:
            img = np.clip(image * 255.0, 0.0, 255.0).astype(np.uint8)
        else:
            img = np.clip(image, 0.0, 255.0).astype(np.uint8)
    else:
        img = np.clip(image, 0, 255).astype(np.uint8)

    h, w, c = img.shape
    assert h % 4 == 0 and w % 4 == 0, f"Image dimensions ({h}, {w}) must be multiples of 4"
    assert c == 3, f"Image must have 3 channels, got {c}"

    bh = h // 4
    bw = w // 4

    # Reshape into (bh, bw, 16, 3)
    blocks = (
        img.reshape(bh, 4, bw, 4, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(bh, bw, 16, 3)
        .astype(np.float32)
    )

    # Find min and max colors per block
    min_rgb = blocks.min(axis=2)  # (bh, bw, 3)
    max_rgb = blocks.max(axis=2)  # (bh, bw, 3)

    # Quantize to RGB565
    e0 = quantize_rgb565(max_rgb).astype(np.float32)  # (bh, bw, 3)
    e1 = quantize_rgb565(min_rgb).astype(np.float32)  # (bh, bw, 3)

    # Encode as uint16 to determine 4-color mode ordering (c0 > c1)
    c0 = rgb565_to_uint16(e0)
    c1 = rgb565_to_uint16(e1)

    # Swap where c0 <= c1 to enforce 4-color mode (c0 > c1)
    swap_mask = c0 <= c1
    temp_e = e0[swap_mask].copy()
    e0[swap_mask] = e1[swap_mask]
    e1[swap_mask] = temp_e

    # Palette endpoints:
    # e2 = (2 * e0 + e1) / 3
    # e3 = (e0 + 2 * e1) / 3
    e2 = np.round((2.0 * e0 + e1) / 3.0)
    e3 = np.round((e0 + 2.0 * e1) / 3.0)

    # Palette shape: (bh, bw, 4, 3)
    palette = np.stack([e0, e1, e2, e3], axis=2)

    # Compute distances: blocks is (bh, bw, 16, 1, 3), palette is (bh, bw, 1, 4, 3)
    diff = blocks[:, :, :, None, :] - palette[:, :, None, :, :]  # (bh, bw, 16, 4, 3)
    dist_sq = np.sum(diff * diff, axis=-1)  # (bh, bw, 16, 4)

    # Nearest index in 0..3 for each pixel
    nearest_idx = np.argmin(dist_sq, axis=-1)  # (bh, bw, 16)

    # Gather decoded pixels: palette is (bh, bw, 4, 3), index is (bh, bw, 16)
    bh_idx = np.arange(bh)[:, None, None]
    bw_idx = np.arange(bw)[None, :, None]
    decoded_blocks = palette[bh_idx, bw_idx, nearest_idx]  # (bh, bw, 16, 3)

    # Reshape back to (h, w, 3)
    decoded = (
        decoded_blocks.reshape(bh, bw, 4, 4, 3)
        .transpose(0, 2, 1, 3, 4)
        .reshape(h, w, 3)
        .astype(np.uint8)
    )

    return decoded
