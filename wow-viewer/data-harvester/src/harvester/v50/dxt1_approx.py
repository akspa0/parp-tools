"""Vectorized DXT1 (BC1) encode/decode round-trip — the codec-noise layer for training inputs.

WHY THIS EXISTS
---------------
Authored 0.5.3 minimaps are BLP2/DXTC/DXT1: each 4x4 block stores two RGB565 endpoints plus 2-bit
per-pixel indices, so a block holds at most 4 colours, all on a line between those endpoints. Our
synthesizer emits pristine 24-bit RGB. A model trained on pristine minimaps and deployed on authored
tiles therefore meets a domain shift the training loss never saw — block banding, colours collapsed
onto per-block lines, and RGB565 quantisation that is twice as coarse in red/blue (5 bits, step 8) as
in green (6 bits, step 4).

Closing that gap belongs in the DATASET, not the model: degrade our synthetic minimaps the way the
codec would, and train on the degraded image. No BLP container is ever written — this is an in-memory
pixels-to-pixels transform.

PARITY WITH THE C# CODEC
------------------------
This mirrors ``WowViewer.Core.IO.Blp.Dxt1TileCodec`` exactly, so the Python dataset path and the C#
``--dxt1-parity`` companion output produce identical pixels:

- endpoints are the per-channel bounding box (independent min/max per channel), not a PCA fit;
- ``c0`` is built from the max corner, ``c1`` from the min corner, swapped if ``c0 <= c1``;
- RGB565 -> RGB8 expands by bit replication (``r |= r >> 5``, ``g |= g >> 6``);
- the two interpolants use truncating integer division, ``(2*e0+e1)//3`` and ``(e0+2*e1)//3``;
- indices pick the nearest palette entry by squared RGB distance, ties going to the lowest index;
- decode honours DXT1's mode bit: when ``c0 == c1`` (a flat block) the decoder takes the 3-colour
  branch, which is why flat blocks survive the round-trip unchanged.

``dxt1_round_trip`` is the only function callers need.
"""

from __future__ import annotations

import numpy as np

BLOCK = 4


def _rgb565(color: np.ndarray) -> np.ndarray:
    """Pack an (..., 3) uint8 array into RGB565 as uint16, matching Dxt1TileCodec.Rgb565."""
    r = color[..., 0].astype(np.uint16) >> 3
    g = color[..., 1].astype(np.uint16) >> 2
    b = color[..., 2].astype(np.uint16) >> 3
    return (r << 11) | (g << 5) | b


def _rgb565_to_rgb(packed: np.ndarray) -> np.ndarray:
    """Unpack RGB565 to (..., 3) int16 with bit replication, matching Dxt1TileCodec.Rgb565ToRgb."""
    r = ((packed >> 11) & 0x1F).astype(np.int16) << 3
    g = ((packed >> 5) & 0x3F).astype(np.int16) << 2
    b = (packed & 0x1F).astype(np.int16) << 3
    r = r | (r >> 5)
    g = g | (g >> 6)
    b = b | (b >> 5)
    return np.stack([r, g, b], axis=-1)


def _to_blocks(image: np.ndarray) -> np.ndarray:
    """(H, W, 3) -> (n_blocks, 16, 3), row-major within each 4x4 block."""
    h, w = image.shape[:2]
    tiled = image.reshape(h // BLOCK, BLOCK, w // BLOCK, BLOCK, 3)
    return tiled.transpose(0, 2, 1, 3, 4).reshape(-1, BLOCK * BLOCK, 3)


def _from_blocks(blocks: np.ndarray, h: int, w: int) -> np.ndarray:
    """Inverse of ``_to_blocks``."""
    tiled = blocks.reshape(h // BLOCK, w // BLOCK, BLOCK, BLOCK, 3)
    return tiled.transpose(0, 2, 1, 3, 4).reshape(h, w, 3)


def dxt1_round_trip(image: np.ndarray) -> np.ndarray:
    """Apply one DXT1 encode/decode cycle to an (H, W, 3) uint8 image.

    Returns a uint8 array of the same shape carrying the codec's degradation: at most four colours
    per 4x4 block, all on the line between two RGB565-quantised endpoints.
    """
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError(f"expected an (H, W, 3) image, got {array.shape}")
    h, w = array.shape[:2]
    if h % BLOCK or w % BLOCK:
        raise ValueError(f"image dimensions must be multiples of {BLOCK}, got {h}x{w}")
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)

    blocks = _to_blocks(array)                       # (n, 16, 3) uint8
    low = blocks.min(axis=1)                         # (n, 3)
    high = blocks.max(axis=1)                        # (n, 3)

    c0 = _rgb565(high)
    c1 = _rgb565(low)
    # Force 4-colour ordering exactly as the C# encoder does; equal endpoints stay equal.
    swap = c0 <= c1
    c0, c1 = np.where(swap, c1, c0), np.where(swap, c0, c1)

    e0 = _rgb565_to_rgb(c0).astype(np.int32)         # (n, 3)
    e1 = _rgb565_to_rgb(c1).astype(np.int32)

    # The encoder always fits against the 4-colour palette, regardless of the eventual mode bit.
    encode_palette = np.stack([e0, e1, (2 * e0 + e1) // 3, (e0 + 2 * e1) // 3], axis=1)  # (n, 4, 3)
    distance = ((blocks[:, :, None, :].astype(np.int32) - encode_palette[:, None, :, :]) ** 2).sum(-1)
    indices = distance.argmin(axis=-1)               # ties -> lowest index, matching NearestIndex

    # The decoder picks its palette from the mode bit: c0 > c1 is 4-colour, otherwise 3-colour.
    four_colour = (c0 > c1)[:, None]
    e2 = np.where(four_colour, (2 * e0 + e1) // 3, (e0 + e1) // 2)
    e3 = np.where(four_colour, (e0 + 2 * e1) // 3, np.zeros_like(e0))
    decode_palette = np.stack([e0, e1, e2, e3], axis=1)

    decoded = np.take_along_axis(decode_palette, indices[:, :, None], axis=1)
    return _from_blocks(decoded.astype(np.uint8), h, w)


def unique_colour_count(image: np.ndarray) -> int:
    """Distinct RGB triples in an image — the statistic that identified authored tiles as DXT1."""
    flat = np.asarray(image).reshape(-1, 3).astype(np.uint32)
    return int(np.unique((flat[:, 0] << 16) | (flat[:, 1] << 8) | flat[:, 2]).size)


def block_edge_ratio(image: np.ndarray) -> float:
    """Mean absolute step across 4-pixel-aligned block boundaries divided by the same across
    interior columns. A block codec pushes this above 1; a smooth render sits near 1.

    This is the Python counterpart of ``MeasureBlockEdgeRatio`` — the right detector for a block
    codec, where a 3x3-median noise test is the wrong one.
    """
    array = np.asarray(image).astype(np.float64)
    steps = np.abs(np.diff(array, axis=1)).mean(axis=2)  # (H, W-1) step between column x and x+1
    columns = np.arange(steps.shape[1])
    on_boundary = (columns % BLOCK) == (BLOCK - 1)       # the step that crosses into the next block
    boundary = steps[:, on_boundary].mean()
    interior = steps[:, ~on_boundary].mean()
    return float(boundary / interior) if interior > 1e-9 else 0.0
