"""V23 loss stack for Spec 089 Phase 4."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torch.nn.functional as F


def _as_spatial_mask(mask: torch.Tensor | None, reference: torch.Tensor) -> torch.Tensor:
    if mask is None:
        return torch.ones_like(reference, dtype=reference.dtype)
    if mask.ndim == reference.ndim - 1:
        mask = mask.unsqueeze(1)
    return mask.to(device=reference.device, dtype=reference.dtype)


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return torch.zeros((), device=reference.device, dtype=reference.dtype)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1e-8)


def _estimate_affine(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = pred.shape[0]
    pred_flat = pred.reshape(batch_size, -1)
    target_flat = target.reshape(batch_size, -1)
    mask_flat = mask.reshape(batch_size, -1)

    weight_sum = mask_flat.sum(dim=1).clamp_min(1e-8)
    pred_mean = (pred_flat * mask_flat).sum(dim=1) / weight_sum
    target_mean = (target_flat * mask_flat).sum(dim=1) / weight_sum

    pred_centered = pred_flat - pred_mean[:, None]
    target_centered = target_flat - target_mean[:, None]
    covariance = (mask_flat * pred_centered * target_centered).sum(dim=1)
    variance = (mask_flat * pred_centered.square()).sum(dim=1).clamp_min(1e-8)
    scale = covariance / variance
    shift = target_mean - scale * pred_mean
    return scale, shift


def affine_invariant_lssi(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Least-squares scale/shift invariant loss on ``pred`` against ``target``."""
    mask_tensor = _as_spatial_mask(mask, pred)
    scale, shift = _estimate_affine(pred, target, mask_tensor)
    aligned = pred * scale[:, None, None, None] + shift[:, None, None, None]
    return _masked_mean((aligned - target).square(), mask_tensor)


def _sobel_filters(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]],
        device=device,
        dtype=dtype,
    ).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3).contiguous()
    return sobel_x, sobel_y


def gradient_matching_lgm(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Gradient-matching loss on aligned metric predictions."""
    mask_tensor = _as_spatial_mask(mask, pred)
    sobel_x, sobel_y = _sobel_filters(pred.device, pred.dtype)
    pred_dx = F.conv2d(pred, sobel_x, padding=1)
    pred_dy = F.conv2d(pred, sobel_y, padding=1)
    target_dx = F.conv2d(target, sobel_x, padding=1)
    target_dy = F.conv2d(target, sobel_y, padding=1)
    loss_map = (pred_dx - target_dx).abs() + (pred_dy - target_dy).abs()
    return _masked_mean(loss_map, mask_tensor)


def spatial_distance_constraint(
    features_pred: torch.Tensor,
    features_target: torch.Tensor,
    patch_size: int = 16,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Patch-level relative-geometry constraint.

    This uses pooled patch descriptors and penalizes disagreement between the
    pairwise patch-distance matrices of prediction and target.
    """
    if patch_size <= 0:
        raise ValueError("patch_size must be positive")
    pooled_pred = F.avg_pool2d(features_pred, kernel_size=patch_size, stride=patch_size, ceil_mode=False)
    pooled_target = F.avg_pool2d(features_target, kernel_size=patch_size, stride=patch_size, ceil_mode=False)
    batch_size = pooled_pred.shape[0]
    pred_flat = pooled_pred.flatten(2)
    target_flat = pooled_target.flatten(2)
    pred_dist = torch.cdist(pred_flat.transpose(1, 2), pred_flat.transpose(1, 2), p=2)
    target_dist = torch.cdist(target_flat.transpose(1, 2), target_flat.transpose(1, 2), p=2)
    loss = (pred_dist - target_dist).abs()
    if mask is None:
        return loss.mean()

    pooled_mask = F.avg_pool2d(_as_spatial_mask(mask, features_pred), kernel_size=patch_size, stride=patch_size, ceil_mode=False)
    patch_valid = (pooled_mask.flatten(2).squeeze(1) > 0.5).to(loss.dtype)
    pair_mask = patch_valid.unsqueeze(1) * patch_valid.unsqueeze(2)
    return (loss * pair_mask).sum() / pair_mask.sum().clamp_min(1e-8)


def _overlap_slices(coord: Sequence[int]) -> tuple[int, int, slice, slice, slice, slice]:
    if len(coord) != 10:
        raise ValueError("overlap coords must have 10 integers")
    left_idx, right_idx, left_y, left_x, right_y, right_x, height, width, _, _ = [int(v) for v in coord]
    return (
        left_idx,
        right_idx,
        slice(left_y, left_y + height),
        slice(left_x, left_x + width),
        slice(right_y, right_y + height),
        slice(right_x, right_x + width),
    )


def gpct_overlap_consistency(
    sub_tile_preds: Sequence[torch.Tensor],
    sub_tile_features: Sequence[torch.Tensor] | None,
    overlap_coords: Sequence[Sequence[int]],
    *,
    feature_loss: bool = True,
) -> torch.Tensor:
    """Grouped patch consistency loss across overlapping predictions."""
    if not sub_tile_preds:
        raise ValueError("sub_tile_preds cannot be empty")
    reference = sub_tile_preds[0]
    total = _zero_like(reference)
    count = 0
    for coord in overlap_coords:
        left_idx, right_idx, left_y, left_x, right_y, right_x = _overlap_slices(coord)
        pred_left = sub_tile_preds[left_idx][..., left_y, left_x]
        pred_right = sub_tile_preds[right_idx][..., right_y, right_x]
        total = total + F.mse_loss(pred_left, pred_right)
        count += 1
        if feature_loss and sub_tile_features is not None:
            feat_left = sub_tile_features[left_idx][..., left_y, left_x]
            feat_right = sub_tile_features[right_idx][..., right_y, right_x]
            total = total + 0.5 * F.mse_loss(feat_left, feat_right)
            count += 1
    if count == 0:
        return _zero_like(reference)
    return total / float(count)


def apply_bias_free_masking(
    input_tensor: torch.Tensor,
    ratio: float = 0.15,
    generator: torch.Generator | None = None,
    *,
    patch_size: int = 16,
    channels: slice | Sequence[int] = slice(0, 3),
) -> tuple[torch.Tensor, torch.Tensor]:
    """Mask patch blocks by replacing them with the per-sample channel mean."""
    def _generator_device(value: torch.Generator | None) -> torch.device:
        if value is None:
            return input_tensor.device
        return torch.device(str(value.device))

    if ratio <= 0.0:
        empty = torch.zeros(
            input_tensor.shape[0],
            input_tensor.shape[-2] // patch_size,
            input_tensor.shape[-1] // patch_size,
            device=input_tensor.device,
            dtype=torch.bool,
        )
        return input_tensor, empty

    masked = input_tensor.clone()
    if isinstance(channels, slice):
        channel_indices = list(range(*channels.indices(input_tensor.shape[1])))
    else:
        channel_indices = [int(value) for value in channels]
    if not channel_indices:
        raise ValueError("channels cannot be empty")

    batch_size, _, height, width = input_tensor.shape
    grid_h = height // patch_size
    grid_w = width // patch_size
    rand_device = _generator_device(generator)
    patch_mask = (
        torch.rand((batch_size, grid_h, grid_w), device=rand_device, generator=generator) < float(ratio)
    ).to(device=input_tensor.device)
    spatial_mask = patch_mask.repeat_interleave(patch_size, dim=1).repeat_interleave(patch_size, dim=2)
    selected = masked[:, channel_indices]
    channel_means = selected.mean(dim=(2, 3), keepdim=True)
    expanded_mask = spatial_mask.unsqueeze(1)
    masked[:, channel_indices] = torch.where(expanded_mask, channel_means, selected)
    return masked, patch_mask


def compute_v23_loss(
    outputs: Mapping[str, torch.Tensor] | object,
    target: torch.Tensor,
    weights: Mapping[str, float],
    *,
    valid_mask: torch.Tensor | None = None,
    sub_tile_preds: Sequence[torch.Tensor] | None = None,
    sub_tile_features: Sequence[torch.Tensor] | None = None,
    overlap_coords: Sequence[Sequence[int]] | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the weighted V23 loss breakdown."""
    disparity = getattr(outputs, "disparity", None)
    metric_height = getattr(outputs, "metric_height", None)
    if disparity is None or metric_height is None:
        if not isinstance(outputs, Mapping):
            raise TypeError("outputs must be a V23ModelOutput-like object or mapping")
        disparity = outputs["disparity"]
        metric_height = outputs["metric_height"]

    mask_tensor = _as_spatial_mask(valid_mask, target)
    components = {
        "affine": affine_invariant_lssi(disparity, target, mask_tensor),
        "gradient": gradient_matching_lgm(metric_height, target, mask_tensor),
        "sdc": spatial_distance_constraint(metric_height, target, patch_size=16, mask=mask_tensor),
    }

    if float(weights.get("gpct", 0.0)) > 0.0 and sub_tile_preds and overlap_coords:
        components["gpct"] = gpct_overlap_consistency(
            sub_tile_preds,
            sub_tile_features,
            overlap_coords,
            feature_loss=bool(weights.get("gpct_feature", 1.0)),
        )
    else:
        components["gpct"] = _zero_like(target)

    total = _zero_like(target)
    total = total + float(weights.get("affine", 1.0)) * components["affine"]
    total = total + float(weights.get("gradient", 0.5)) * components["gradient"]
    total = total + float(weights.get("sdc", 0.1)) * components["sdc"]
    total = total + float(weights.get("gpct", 0.0)) * components["gpct"]
    components["total"] = total
    return total, components


__all__ = [
    "affine_invariant_lssi",
    "apply_bias_free_masking",
    "compute_v23_loss",
    "gpct_overlap_consistency",
    "gradient_matching_lgm",
    "spatial_distance_constraint",
]
