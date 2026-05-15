"""Python port of TerrainMinimapCompositor — bit-exact with C# implementation.

Composites tileset textures + MCAL alpha weights into a 256x256x3 RGB synthetic minimap.
This is the inverse of D1 decomposition — given MCAL/MCLY data, produce the
expected minimap appearance so we can compute the residual:

    residual = minimap_rgb_256 - composite(textures, alpha)

The compositor uses fixed placeholder colours per layer, matching the C# reference.
"""

import numpy as np
from numpy.typing import NDArray

PLACEHOLDER_COLORS = np.array(
    [
        [0.549, 0.706, 0.784],  # Layer 0: BGR(200,180,140) → RGB(140,180,200) /255
        [0.392, 0.549, 0.627],  # Layer 1: BGR(160,140,100) → RGB(100,140,160) /255
        [0.431, 0.510, 0.471],  # Layer 2: BGR(120,130,110) → RGB(110,130,120) /255
        [0.510, 0.471, 0.392],  # Layer 3: BGR(100,120,130) → RGB(130,120,100) /255
    ],
    dtype=np.float32,
)


def compute_compositor_weights(
    alpha_pack: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute per-layer compositor blend weights from MCAL alpha values.

    Args:
        alpha_pack: (H, W, 4) float32 array of raw MCAL alpha values (0-1).

    Returns:
        (H, W, 4) float32 array of compositor blend weights, row-major.
        Weights sum to 1.0 per pixel (or 0.0 where no layers contribute).
    """
    a1 = alpha_pack[..., 0]
    a2 = alpha_pack[..., 1]
    a3 = alpha_pack[..., 2]
    a4 = alpha_pack[..., 3]

    w0 = 1.0 - a1
    w1 = a1 * (1.0 - a2)
    w2 = a1 * a2 * (1.0 - a3)
    w3 = a1 * a2 * a3 * (1.0 - a4)

    weights = np.stack([w0, w1, w2, w3], axis=-1)
    total = weights.sum(axis=-1, keepdims=True)
    mask = total > 1e-6
    weights = np.where(mask, weights / np.where(mask, total, 1.0), 0.0)
    return weights


def composite_synthetic_minimap(
    alpha_pack: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Composite a synthetic 256x256x3 RGB minimap using placeholder colours and MCAL alphas.

    Args:
        alpha_pack: (256, 256, 4) float32 array of raw MCAL alpha values (0-1).

    Returns:
        (256, 256, 3) float32 synthetic minimap in [0, 1].
    """
    weights = compute_compositor_weights(alpha_pack)
    synthetic = np.tensordot(weights, PLACEHOLDER_COLORS, axes=([2], [0]))
    return synthetic.clip(0.0, 1.0).astype(np.float32)


def compute_residual(
    real_minimap: NDArray[np.float32],
    synthetic_minimap: NDArray[np.float32],
) -> NDArray[np.float32]:
    """Compute residual between real and synthetic minimap.

    residual = real_minimap - synthetic_minimap
    """
    return (real_minimap - synthetic_minimap).astype(np.float32)


def compute_d1_targets(
    alpha_pack: NDArray[np.float32],
    minimap: NDArray[np.float32],
) -> tuple[NDArray[np.float32], NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]]:
    """Compute D1 ground-truth targets from MCAL alpha and minimap.

    D1 predicts:
      - tileset_layer_1 (256,256,3): minimap contribution of combined layers 0+1
      - tileset_layer_2 (256,256,3): minimap contribution of combined layers 2+3
      - alpha_mask_1 (256,256): combined alpha weight for layers 0+1
      - alpha_mask_2 (256,256): combined alpha weight for layers 2+3

    Ground truth is computed by grouping the 4 compositor layers into 2 groups.
    """
    weights = compute_compositor_weights(alpha_pack)  # (H, W, 4)

    # Group 1: layers 0+1, Group 2: layers 2+3
    w_group1 = weights[..., 0] + weights[..., 1]
    w_group2 = weights[..., 2] + weights[..., 3]
    w_total = w_group1 + w_group2 + 1e-8

    w1_norm = w_group1 / w_total
    w2_norm = w_group2 / w_total

    tileset_1 = minimap * w1_norm[..., None]
    tileset_2 = minimap * w2_norm[..., None]

    alpha_1 = alpha_pack[..., 0].copy()
    alpha_2 = alpha_pack[..., 1].copy()

    return tileset_1, tileset_2, alpha_1, alpha_2
