"""V19 loss functions for height regression.

Multi-component loss: L1 + normal consistency + edge (Sobel).
All functions are self-contained — no imports from gillijimproject_refactor.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def height_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked L1 loss on heightmap."""
    diff = (pred - target).abs()
    return (diff * mask).sum() / mask.sum().clamp_min(1e-8)


def normal_consistency_loss(
    pred_height: torch.Tensor,
    target_normals: torch.Tensor,
    mask: torch.Tensor,
    weight: float = 0.1,
) -> torch.Tensor:
    """Cosine similarity between predicted normals (from height) and target normals."""
    if weight <= 0.0:
        return torch.tensor(0.0, device=pred_height.device)

    dzdx = pred_height[:, :, :, 2:] - pred_height[:, :, :, :-2]
    dzdy = pred_height[:, :, 2:, :] - pred_height[:, :, :-2, :]
    dzdx = F.pad(dzdx * 0.5, (1, 1, 0, 0), mode="replicate")
    dzdy = F.pad(dzdy * 0.5, (0, 0, 1, 1), mode="replicate")
    nx = -dzdx
    ny = -dzdy
    nz = torch.ones_like(nx)
    pred_normals = F.normalize(torch.cat([nx, ny, nz], dim=1), dim=1, eps=1e-6)

    target_n = F.normalize(target_normals, dim=1, eps=1e-6)
    cosine = 1.0 - (pred_normals * target_n).sum(dim=1, keepdim=True)
    loss = (cosine * mask).sum() / mask.sum().clamp_min(1e-8)
    return weight * loss


def edge_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    weight: float = 0.05,
) -> torch.Tensor:
    """Sobel edge loss on heightmap."""
    if weight <= 0.0:
        return torch.tensor(0.0, device=pred.device)

    sobel_x = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32, device=pred.device)
    sobel_y = sobel_x.T
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)

    def apply_sobel(x: torch.Tensor) -> torch.Tensor:
        edges = torch.zeros_like(x)
        for c in range(x.shape[1]):
            ch = x[:, c:c+1]
            edges[:, c:c+1] = F.conv2d(ch, sobel_x, padding=1).abs() + F.conv2d(ch, sobel_y, padding=1).abs()
        return edges

    pred_edges = apply_sobel(pred)
    target_edges = apply_sobel(target)
    loss = ((pred_edges - target_edges).abs() * mask).sum() / mask.sum().clamp_min(1e-8)
    return weight * loss


def v19_combined_loss(
    pred_global: torch.Tensor,
    pred_local: torch.Tensor,
    pred_bounds: torch.Tensor,
    target_height: torch.Tensor,
    terrain_valid_mask: torch.Tensor,
    normals: torch.Tensor | None = None,
    normal_mask: torch.Tensor | None = None,
    height_mean: torch.Tensor | None = None,
    height_std: torch.Tensor | None = None,
    nc_weight: float = 0.1,
    edge_weight: float = 0.05,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Combined V19 loss: L1(global) + L1(local) + normal consistency + edge."""
    mask = terrain_valid_mask.clamp(0.0, 1.0)

    height_mask = mask
    if height_mean is not None and height_std is not None:
        h_mean = height_mean.view(-1, 1, 1, 1)
        h_std = height_std.view(-1, 1, 1, 1)
        target_unnorm = target_height * h_std + h_mean
        pred_global_unnorm = pred_global * h_std + h_mean
        pred_local_unnorm = pred_local * h_std + h_mean
    else:
        target_unnorm = target_height
        pred_global_unnorm = pred_global
        pred_local_unnorm = pred_local

    l1_global = height_l1_loss(pred_global, target_height, height_mask)
    l1_local = height_l1_loss(pred_local, target_height, height_mask)
    l1_loss = l1_global + l1_local

    nc_loss = torch.tensor(0.0, device=pred_global.device)
    if normals is not None and normal_mask is not None and nc_weight > 0.0:
        nc_mask = (height_mask * normal_mask).clamp(0.0, 1.0)
        nc_loss = normal_consistency_loss(pred_global_unnorm, normals, nc_mask, weight=nc_weight)

    e_loss = torch.tensor(0.0, device=pred_global.device)
    if edge_weight > 0.0:
        e_loss = edge_loss(pred_global, target_height, height_mask, weight=edge_weight)

    total = l1_loss + nc_loss + e_loss

    metrics = {
        "total": float(total.item()),
        "l1_global": float(l1_global.item()),
        "l1_local": float(l1_local.item()),
        "nc": float(nc_loss.item()),
        "edge": float(e_loss.item()),
        "mask_cov": float(mask.mean().item()),
    }
    return total, metrics
