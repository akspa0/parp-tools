"""v7 losses, ported verbatim from gillijimproject_refactor (read-only reference).

Source: `gillijimproject_refactor/src/WoWMapConverter/scripts/v7_losses.py`. The channel
assumptions inside `derive_recovery_mask_from_inputs` (ch 9 liquid, ch 11 object, ch 12 brush)
are part of the pinned 13-channel contract — see research-v7-contract.md §1.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F

LOSS_WEIGHTS = {
    "heightmap_global": 0.08,
    "heightmap_local": 0.14,
    "detail_aux": 0.08,
    "bounds": 0.04,
    "ssim": 0.05,
    "gradient": 0.10,
    "edge": 0.12,
    "frequency": 0.08,
    "adversarial": 0.12,
    "laplacian": 0.12,
    "transition": 0.10,
    "tile_edge": 0.12,
    "recovery": 0.16,
}

EDGE_FOCUS_WIDTH = 12
TRANSITION_FOCUS_GAIN = 3.0
RECOVERY_FOCUS_GAIN = 3.5


def weighted_l1_loss(predicted: torch.Tensor, target: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    weighted_error = (predicted - target).abs() * weights
    return weighted_error.sum() / torch.clamp(weights.sum(), min=1e-6)


def build_recovery_mask(
    object_mask: torch.Tensor,
    liquid_mask: torch.Tensor,
    brush_mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    recovery_mask = torch.maximum(object_mask, liquid_mask)
    if brush_mask is not None:
        recovery_mask = torch.maximum(recovery_mask, brush_mask * 0.5)
    if bool(torch.any(recovery_mask > 0)):
        recovery_mask = F.max_pool2d(recovery_mask, kernel_size=5, stride=1, padding=2)
    return torch.clamp(recovery_mask, 0.0, 1.0)


def derive_recovery_mask_from_inputs(inputs: torch.Tensor) -> torch.Tensor:
    liquid_mask = inputs[:, 9:10] if inputs.shape[1] > 9 else torch.zeros_like(inputs[:, 0:1])
    object_mask = inputs[:, 11:12] if inputs.shape[1] > 11 else torch.zeros_like(inputs[:, 0:1])
    brush_mask = inputs[:, 12:13] if inputs.shape[1] > 12 else None
    return build_recovery_mask(object_mask=object_mask, liquid_mask=liquid_mask, brush_mask=brush_mask)


def ssim_loss(predicted: torch.Tensor, target: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    predicted = predicted.float()
    target = target.float()
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2

    def gaussian_window(size: int, sigma: float = 1.5) -> torch.Tensor:
        coords = torch.arange(size, dtype=torch.float32) - size // 2
        gaussian = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
        return gaussian / gaussian.sum()

    gaussian = gaussian_window(window_size).to(predicted.device)
    window = (gaussian[:, None] @ gaussian[None, :]).unsqueeze(0).unsqueeze(0)
    window = window.expand(predicted.shape[1], 1, window_size, window_size)

    mu_pred = F.conv2d(predicted, window, padding=window_size // 2, groups=predicted.shape[1])
    mu_target = F.conv2d(target, window, padding=window_size // 2, groups=predicted.shape[1])
    mu_pred_sq = mu_pred.pow(2)
    mu_target_sq = mu_target.pow(2)
    mu_pred_target = mu_pred * mu_target

    sigma_pred_sq = torch.clamp(
        F.conv2d(predicted * predicted, window, padding=window_size // 2, groups=predicted.shape[1]) - mu_pred_sq,
        min=0.0,
    )
    sigma_target_sq = torch.clamp(
        F.conv2d(target * target, window, padding=window_size // 2, groups=predicted.shape[1]) - mu_target_sq,
        min=0.0,
    )
    sigma_pred_target = F.conv2d(predicted * target, window, padding=window_size // 2, groups=predicted.shape[1]) - mu_pred_target

    numerator = (2 * mu_pred_target + c1) * (2 * sigma_pred_target + c2)
    denominator = torch.clamp((mu_pred_sq + mu_target_sq + c1) * (sigma_pred_sq + sigma_target_sq + c2), min=1e-8)
    ssim_map = torch.clamp(numerator / denominator, min=-1.0, max=1.0)
    return torch.clamp(1 - ssim_map.mean(), min=0.0)


def edge_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=predicted.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    def compute(tensor: torch.Tensor) -> torch.Tensor:
        edges = torch.zeros_like(tensor)
        for channel in range(tensor.shape[1]):
            current = tensor[:, channel:channel + 1]
            edges[:, channel:channel + 1] = F.conv2d(current, sobel_x, padding=1).abs() + F.conv2d(current, sobel_y, padding=1).abs()
        return edges

    return F.l1_loss(compute(predicted[:, :2]), compute(target[:, :2]))


def frequency_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_float = predicted[:, :2].float()
    target_float = target[:, :2].float()
    pred_fft = torch.fft.rfft2(pred_float)
    target_fft = torch.fft.rfft2(target_float)
    pred_mag = torch.log1p(pred_fft.abs())
    target_mag = torch.log1p(target_fft.abs())
    return F.l1_loss(pred_mag, target_mag)


def laplacian_loss(predicted: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    kernel = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=predicted.device).view(1, 1, 3, 3)

    def apply_laplacian(tensor: torch.Tensor) -> torch.Tensor:
        result = torch.zeros_like(tensor)
        for ch in range(tensor.shape[1]):
            result[:, ch : ch + 1] = F.conv2d(tensor[:, ch : ch + 1], kernel, padding=1)
        return result

    return F.l1_loss(apply_laplacian(predicted[:, :2]), apply_laplacian(target[:, :2]))


def recovery_focus_loss(predicted: torch.Tensor, target: torch.Tensor, recovery_mask: torch.Tensor) -> torch.Tensor:
    if not bool(torch.any(recovery_mask > 0)):
        return torch.zeros((), dtype=predicted.dtype, device=predicted.device)
    recovery_mask = recovery_mask.expand(predicted.shape[0], 2, predicted.shape[2], predicted.shape[3])
    weights = 1.0 + RECOVERY_FOCUS_GAIN * recovery_mask
    return weighted_l1_loss(predicted[:, :2], target[:, :2], weights)


def transition_focus_loss(predicted: torch.Tensor, target: torch.Tensor, gain: float = TRANSITION_FOCUS_GAIN) -> torch.Tensor:
    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=target.device).view(1, 1, 3, 3)
    sobel_y = sobel_x.transpose(2, 3)

    weight_maps = []
    for channel in range(target.shape[1]):
        current = target[:, channel:channel + 1]
        grad_x = F.conv2d(current, sobel_x, padding=1)
        grad_y = F.conv2d(current, sobel_y, padding=1)
        magnitude = torch.sqrt(torch.clamp(grad_x * grad_x + grad_y * grad_y, min=0.0))
        normalized = magnitude / (magnitude.mean(dim=(2, 3), keepdim=True) + 1e-6)
        weight_maps.append(1.0 + gain * torch.clamp(normalized, min=0.0, max=1.0))

    weights = torch.cat(weight_maps, dim=1)
    return weighted_l1_loss(predicted[:, :2], target[:, :2], weights)


def tile_edge_loss(predicted: torch.Tensor, target: torch.Tensor, edge_width: int = EDGE_FOCUS_WIDTH) -> torch.Tensor:
    if edge_width <= 0:
        return torch.zeros((), dtype=predicted.dtype, device=predicted.device)

    _, channels, height, width = predicted[:, :2].shape
    border_mask = torch.zeros((1, 1, height, width), dtype=predicted.dtype, device=predicted.device)
    border_mask[:, :, :edge_width, :] = 1.0
    border_mask[:, :, -edge_width:, :] = 1.0
    border_mask[:, :, :, :edge_width] = 1.0
    border_mask[:, :, :, -edge_width:] = 1.0
    border_mask = border_mask.expand(predicted.shape[0], channels, height, width)
    return weighted_l1_loss(predicted[:, :2], target[:, :2], border_mask)


def combined_loss(
    predicted_heightmap: torch.Tensor,
    predicted_bounds: torch.Tensor,
    target_heightmap: torch.Tensor,
    target_bounds: torch.Tensor,
    input_context: Optional[torch.Tensor] = None,
    adv_loss: Optional[torch.Tensor] = None,
    adversarial_scale: float = 1.0,
    detail_head_weight: float = 0.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    predicted_heightmap = predicted_heightmap.float()
    predicted_bounds = predicted_bounds.float()
    target_heightmap = target_heightmap.float()
    target_bounds = target_bounds.float()

    global_loss = F.l1_loss(predicted_heightmap[:, 0:1], target_heightmap[:, 0:1])
    local_loss = F.l1_loss(predicted_heightmap[:, 1:2], target_heightmap[:, 1:2])
    detail_aux_component = torch.zeros((), dtype=predicted_heightmap.dtype, device=predicted_heightmap.device)
    if detail_head_weight > 0.0 and predicted_heightmap.shape[1] > 2:
        target_detail = target_heightmap[:, 1:2] - target_heightmap[:, 0:1]
        predicted_detail = predicted_heightmap[:, 2:3]
        detail_aux_component = F.l1_loss(predicted_detail, target_detail)
    bounds_loss = F.mse_loss(predicted_bounds, target_bounds)

    def get_gradient(tensor: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return tensor[:, :, :, 1:] - tensor[:, :, :, :-1], tensor[:, :, 1:, :] - tensor[:, :, :-1, :]

    predicted_dx, predicted_dy = get_gradient(predicted_heightmap[:, :2])
    target_dx, target_dy = get_gradient(target_heightmap[:, :2])
    gradient_component = F.l1_loss(predicted_dx, target_dx) + F.l1_loss(predicted_dy, target_dy)

    ssim_component = ssim_loss(predicted_heightmap[:, :2], target_heightmap[:, :2])
    edge_component = edge_loss(predicted_heightmap, target_heightmap)
    frequency_component = frequency_loss(predicted_heightmap, target_heightmap)
    laplacian_component = laplacian_loss(predicted_heightmap, target_heightmap)
    transition_component = transition_focus_loss(predicted_heightmap, target_heightmap)
    tile_edge_component = tile_edge_loss(predicted_heightmap, target_heightmap)
    recovery_component = torch.zeros((), dtype=predicted_heightmap.dtype, device=predicted_heightmap.device)
    if input_context is not None:
        recovery_mask = derive_recovery_mask_from_inputs(input_context.float())
        recovery_component = recovery_focus_loss(predicted_heightmap, target_heightmap, recovery_mask)

    total = (
        LOSS_WEIGHTS["heightmap_global"] * global_loss
        + LOSS_WEIGHTS["heightmap_local"] * local_loss
        + (LOSS_WEIGHTS["detail_aux"] * detail_head_weight) * detail_aux_component
        + LOSS_WEIGHTS["bounds"] * bounds_loss
        + LOSS_WEIGHTS["gradient"] * gradient_component
        + LOSS_WEIGHTS["ssim"] * ssim_component
        + LOSS_WEIGHTS["edge"] * edge_component
        + LOSS_WEIGHTS["frequency"] * frequency_component
        + LOSS_WEIGHTS["laplacian"] * laplacian_component
        + LOSS_WEIGHTS["transition"] * transition_component
        + LOSS_WEIGHTS["tile_edge"] * tile_edge_component
        + LOSS_WEIGHTS["recovery"] * recovery_component
    )

    adv_value = 0.0
    if adv_loss is not None:
        adv_loss = adv_loss.float()
        total = total + (LOSS_WEIGHTS["adversarial"] * adversarial_scale) * adv_loss
        adv_value = float(adv_loss.item())

    return total, {
        "heightmap_global": float(global_loss.item()),
        "heightmap_local": float(local_loss.item()),
        "detail_aux": float(detail_aux_component.item()),
        "bounds": float(bounds_loss.item()),
        "gradient": float(gradient_component.item()),
        "ssim": float(ssim_component.item()),
        "edge": float(edge_component.item()),
        "frequency": float(frequency_component.item()),
        "laplacian": float(laplacian_component.item()),
        "transition": float(transition_component.item()),
        "tile_edge": float(tile_edge_component.item()),
        "recovery": float(recovery_component.item()),
        "adversarial": adv_value,
    }
