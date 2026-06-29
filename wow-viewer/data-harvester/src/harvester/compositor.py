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


def _resize_hwc_nearest(arr: NDArray, target_h: int, target_w: int) -> NDArray:
    """Nearest-neighbor resize for HWC arrays."""
    h, w = arr.shape[0], arr.shape[1]
    if h == target_h and w == target_w:
        return arr
    ys = np.linspace(0, h - 1, target_h).astype(np.int64)
    xs = np.linspace(0, w - 1, target_w).astype(np.int64)
    return arr[ys[:, None], xs[None, :], :]


def texture_id_palette(texture_ids: NDArray[np.int32]) -> NDArray[np.float32]:
    """Map terrain texture file-data IDs to deterministic display colours.

    These are not decoded BLP average colours. They are stable pseudo-colours
    that encode terrain texture identity, so two different MCLY texture IDs do
    not collapse to the same placeholder layer colour.
    """
    ids = np.asarray(texture_ids, dtype=np.int64)
    u = np.maximum(ids, 0).astype(np.uint64, copy=False)
    r = ((u * np.uint64(1103515245) + np.uint64(12345)) >> np.uint64(16)) & np.uint64(255)
    g = ((u * np.uint64(1664525) + np.uint64(1013904223)) >> np.uint64(16)) & np.uint64(255)
    b = ((u * np.uint64(22695477) + np.uint64(1)) >> np.uint64(16)) & np.uint64(255)
    rgb = np.stack([r, g, b], axis=-1).astype(np.float32) / 255.0
    # Keep colours away from black/white extremes so preview panels stay readable.
    rgb = 0.20 + 0.70 * rgb
    return np.where(ids[..., None] >= 0, rgb, 0.0).astype(np.float32)


def compute_identity_albedo_weights(
    alpha_pack: NDArray[np.float32],
    mcly_layer_mask: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Return per-layer weights for texture-identity albedo previews.

    V18 documents ``alpha_256`` as per-layer blend weights. For rows where the
    alpha pack is empty (common on single-layer terrain), fall back to the
    first active MCLY layer, or layer 0 when no mask exists.
    """
    alpha = np.clip(np.asarray(alpha_pack, dtype=np.float32), 0.0, 1.0)
    if alpha.ndim != 3 or alpha.shape[-1] != 4:
        raise ValueError(f"alpha_pack must be HxWx4, got {alpha.shape}")
    h, w = alpha.shape[0], alpha.shape[1]
    weights = alpha.copy()

    active = None
    if mcly_layer_mask is not None:
        active = np.asarray(mcly_layer_mask, dtype=np.float32)
        if active.ndim == 3 and active.shape[-1] == 4:
            active = _resize_hwc_nearest(active, h, w) > 0.0
            if active.any():
                weights = np.where(active, weights, 0.0)
        else:
            active = None

    total = weights.sum(axis=-1, keepdims=True)
    weights = np.where(total > 1e-6, weights / np.where(total > 1e-6, total, 1.0), 0.0)

    empty = weights.sum(axis=-1) <= 1e-6
    if np.any(empty):
        fallback_layer = np.zeros((h, w), dtype=np.int64)
        if active is not None:
            has_active = active.any(axis=-1)
            fallback_layer = np.argmax(active, axis=-1).astype(np.int64)
            fallback_layer = np.where(has_active, fallback_layer, 0)
        weights[empty] = 0.0
        yy, xx = np.nonzero(empty)
        weights[yy, xx, fallback_layer[yy, xx]] = 1.0
    return weights.astype(np.float32)


def composite_texture_identity_albedo(
    alpha_pack: NDArray[np.float32],
    mcly_texture_ids: NDArray[np.int32] | None = None,
    mcly_layer_mask: NDArray[np.float32] | None = None,
) -> NDArray[np.float32]:
    """Composite a texture-identity albedo preview from V18 terrain layers.

    Uses `alpha_256` for per-pixel layer weights and MCLY texture IDs for the
    layer colours. If texture IDs are unavailable, falls back to the old fixed
    placeholder colours.
    """
    alpha = np.clip(np.asarray(alpha_pack, dtype=np.float32), 0.0, 1.0)
    weights = compute_identity_albedo_weights(alpha, mcly_layer_mask)
    h, w = alpha.shape[0], alpha.shape[1]

    if mcly_texture_ids is not None:
        tex = np.asarray(mcly_texture_ids, dtype=np.int32)
        if tex.ndim == 3 and tex.shape[-1] == 4:
            tex = _resize_hwc_nearest(tex, h, w)
            colours = texture_id_palette(tex)
            fallback = np.broadcast_to(PLACEHOLDER_COLORS, colours.shape)
            colours = np.where(tex[..., None] >= 0, colours, fallback)
        else:
            colours = np.broadcast_to(PLACEHOLDER_COLORS, (h, w, 4, 3))
    else:
        colours = np.broadcast_to(PLACEHOLDER_COLORS, (h, w, 4, 3))

    albedo = (weights[..., None] * colours).sum(axis=2)
    return albedo.clip(0.0, 1.0).astype(np.float32)


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
