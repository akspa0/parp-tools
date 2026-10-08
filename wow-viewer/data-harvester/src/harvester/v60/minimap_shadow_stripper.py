"""Minimap Shadow Stripper and Multi-Scale Terrain Inpainter.

Removes high-contrast object albedo and seamlessly inpaints masked object
regions using multi-scale Laplacian diffusion, preserving the low-frequency
and directional terrain shadow signals (AC-004).
"""

from __future__ import annotations

import logging
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage

logger = logging.getLogger(__name__)


def compute_relative_high_frequency_energy(
    signal: np.ndarray,
    mask: np.ndarray,
) -> float:
    """Compute total high-frequency gradient energy within the masked region."""
    arr = np.asarray(signal, dtype=np.float32)
    m = np.asarray(mask, dtype=bool)
    if not np.any(m):
        return 0.0

    # Sobel / central difference gradients
    gx = ndimage.sobel(arr, axis=1, mode="reflect")
    gy = ndimage.sobel(arr, axis=0, mode="reflect")
    grad_mag_sq = gx * gx + gy * gy

    return float(np.sum(grad_mag_sq[m]))


def multiscale_laplacian_inpaint(
    image: np.ndarray,
    mask: np.ndarray,
    num_scales: int = 3,
    iterations_per_scale: int = 40,
) -> np.ndarray:
    """Inpaint masked regions using multi-scale hierarchical Laplacian heat diffusion.

    Iteratively solves Laplace's equation:
        \\nabla^2 u = 0  within mask
    subject to Dirichlet boundary conditions matching valid surrounding terrain pixels.
    """
    img = np.asarray(image, dtype=np.float32).copy()
    m = (np.asarray(mask) > 0).astype(bool)

    if not np.any(m):
        return img

    h, w = img.shape[:2]

    # Build image and mask pyramid
    pyramid_imgs = [img]
    pyramid_masks = [m]

    for s in range(1, num_scales):
        scale_factor = 0.5
        down_img = ndimage.zoom(pyramid_imgs[-1], (scale_factor, scale_factor), order=1)
        down_mask = ndimage.zoom(pyramid_masks[-1].astype(np.float32), (scale_factor, scale_factor), order=0) > 0.5
        pyramid_imgs.append(down_img)
        pyramid_masks.append(down_mask)

    # 4-connected discrete Laplacian stencil kernel
    # [ 0, 1/4, 0 ]
    # [ 1/4, 0, 1/4 ]
    # [ 0, 1/4, 0 ]
    laplace_kernel = np.array([
        [0.0, 0.25, 0.0],
        [0.25, 0.0, 0.25],
        [0.0, 0.25, 0.0],
    ], dtype=np.float32)

    # Coarsest level initialization
    current_u = pyramid_imgs[-1].copy()
    current_m = pyramid_masks[-1]

    # Fill masked regions at coarsest scale with mean of unmasked boundary pixels
    valid_pixels = current_u[~current_m]
    fill_val = float(np.mean(valid_pixels)) if valid_pixels.size > 0 else 0.5
    current_u[current_m] = fill_val

    for s in reversed(range(num_scales)):
        if s < num_scales - 1:
            # Upsample solution from coarser scale
            target_h, target_w = pyramid_imgs[s].shape[:2]
            zh = target_h / current_u.shape[0]
            zw = target_w / current_u.shape[1]
            current_u = ndimage.zoom(current_u, (zh, zw), order=1)
            # Ensure shape matches exactly
            current_u = current_u[:target_h, :target_w]

        scale_orig = pyramid_imgs[s]
        scale_mask = pyramid_masks[s]

        # Reset unmasked boundary condition pixels to known ground truth
        current_u[~scale_mask] = scale_orig[~scale_mask]

        # Jacobi / SOR diffusion iterations
        for _ in range(iterations_per_scale):
            diffused = ndimage.convolve(current_u, laplace_kernel, mode="reflect")
            # Update only masked pixels
            current_u[scale_mask] = diffused[scale_mask]

    return current_u


class MinimapShadowStripper:
    """Strips albedo and inpaints minimap tiles to extract pure terrain shadow residual signals."""

    def __init__(self, inpaint_scales: int = 3, iterations: int = 50):
        self.inpaint_scales = inpaint_scales
        self.iterations = iterations

    def strip_and_inpaint(
        self,
        minimap_rgb: np.ndarray,
        object_mask: np.ndarray,
        normalize_albedo: bool = True,
        alpha_mask: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, float]:
        """Strip albedo and inpaint masked regions.

        Parameters:
            minimap_rgb: RGB minimap image array in [0, 1] or [0, 255].
            object_mask: Binary mask of rooftops, doodads, and buildings to inpaint.
            normalize_albedo: Whether to divide out base texture albedo.
            alpha_mask: Optional texture splat alpha map (from _tex0.adt MCAL/MCLY). When provided,
                isolates genuine terrain shadows by decoupling texture splat boundaries.

        Returns:
            (stripped_shadow, energy_attenuation_ratio)
            - stripped_shadow: float32 array in [0, 1] representing the clean terrain shadow.
            - energy_attenuation_ratio: percentage of object high-frequency energy removed (>= 0.90 target).
        """
        arr = np.asarray(minimap_rgb, dtype=np.float32)
        if arr.max() > 1.0:
            arr /= 255.0

        mask = (np.asarray(object_mask) > 0).astype(bool)

        # 1. Convert to grayscale luminance
        if arr.ndim == 3 and arr.shape[-1] >= 3:
            luma = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        else:
            luma = arr.squeeze()

        # 2. Diffuse albedo normalization
        if normalize_albedo:
            if alpha_mask is not None and np.any(alpha_mask > 1e-4):
                # Alpha mask encodes texture splat blending boundaries
                # Fit linear albedo model against alpha mask to isolate pure terrain shadow
                a_flat = alpha_mask.flatten()
                l_flat = luma.flatten()
                var_a = float(np.var(a_flat))
                if var_a > 1e-4:
                    p = np.polyfit(a_flat, l_flat, 1)
                    albedo_floor = np.maximum(p[0] * alpha_mask + p[1], 0.15)
                else:
                    baseline_albedo = ndimage.gaussian_filter(luma, sigma=4.0)
                    albedo_floor = np.maximum(baseline_albedo, 0.15)
                shadow_candidate = np.clip(luma / albedo_floor, 0.0, 1.5)
                s_min = float(np.min(shadow_candidate))
                s_max = float(np.max(shadow_candidate))
                shadow_candidate = (shadow_candidate - s_min) / max(1e-5, s_max - s_min)
            else:
                baseline_albedo = ndimage.gaussian_filter(luma, sigma=4.0)
                albedo_floor = np.maximum(baseline_albedo, 0.15)
                shadow_candidate = np.clip(luma / albedo_floor, 0.0, 1.0)
        else:
            shadow_candidate = luma

        # 3. Measure initial energy in masked object region
        initial_energy = compute_relative_high_frequency_energy(shadow_candidate, mask)

        # 4. Multi-scale hierarchical Laplacian inpainting
        inpainted_shadow = multiscale_laplacian_inpaint(
            image=shadow_candidate,
            mask=mask,
            num_scales=self.inpaint_scales,
            iterations_per_scale=self.iterations,
        )
        inpainted_shadow = np.clip(inpainted_shadow, 0.0, 1.0)

        # 5. Measure residual energy in masked region after inpainting
        final_energy = compute_relative_high_frequency_energy(inpainted_shadow, mask)

        if initial_energy > 1e-6:
            attenuation = max(0.0, 1.0 - (final_energy / initial_energy))
        else:
            attenuation = 1.0

        return inpainted_shadow, float(attenuation)
