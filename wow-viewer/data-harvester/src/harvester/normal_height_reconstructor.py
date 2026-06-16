"""Reconstruct height fields from normal vectors via Frankot-Chellappa integration.

Given normal vectors (nx, ny, nz) on a regular grid, this module recovers
a height field z(x,y) by integrating the surface gradients in the Fourier domain.

Surface gradients:
    p = dz/dx = -nx / nz
    q = dz/dy = -ny / nz

Frankot-Chellappa (1988) integration in the Fourier domain:
    Z(u,v) = (-j * u * P - j * v * Q) / (u^2 + v^2)
for (u,v) != (0,0), with Z(0,0) = 0.

where u, v are angular frequencies in [-pi, pi] and P, Q are DFTs of p, q.
"""

from __future__ import annotations

import numpy as np


def reconstruct_height_from_normals(
    normals: np.ndarray,
    normal_mask: np.ndarray | None = None,
    *,
    nz_clip: float = 0.05,
    apply_window: bool = True,
) -> np.ndarray:
    if normals.ndim != 3 or normals.shape[2] != 3:
        raise ValueError(f"normals must have shape (H, W, 3), got {normals.shape}")
    h, w = normals.shape[0], normals.shape[1]

    nx = normals[:, :, 0].astype(np.float64)
    ny = normals[:, :, 1].astype(np.float64)
    nz = normals[:, :, 2].astype(np.float64)

    nz_safe = np.where(np.abs(nz) < nz_clip, np.sign(nz) * nz_clip if np.any(nz != 0) else nz_clip, nz)
    p = -nx / nz_safe
    q = -ny / nz_safe

    if normal_mask is not None:
        mask = normal_mask.astype(np.float64)
        p = p * mask
        q = q * mask

    if apply_window:
        wh = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(h) / max(h - 1, 1)))
        ww = 0.5 * (1.0 - np.cos(2.0 * np.pi * np.arange(w) / max(w - 1, 1)))
        win = np.outer(wh, ww).astype(np.float64)
        p = p * win
        q = q * win

    P = np.fft.fft2(p)
    Q = np.fft.fft2(q)

    u = np.fft.fftfreq(w) * 2.0 * np.pi
    v = np.fft.fftfreq(h) * 2.0 * np.pi
    U, V = np.meshgrid(u, v)

    denom = U**2 + V**2
    denom[0, 0] = 1.0

    Z = (-1j * U * P - 1j * V * Q) / denom
    Z[0, 0] = 0.0

    heights = np.fft.ifft2(Z).real.astype(np.float32)
    return heights


def anchor_heights(
    reconstructed: np.ndarray,
    original: np.ndarray,
    *,
    normal_mask: np.ndarray | None = None,
) -> np.ndarray:
    reconstructed = reconstructed.astype(np.float32, copy=True)

    if normal_mask is not None:
        mask = normal_mask.astype(bool)
        orig_valid = original[mask]
        if len(orig_valid) > 0:
            orig_mean = float(orig_valid.mean())
            rec_valid = reconstructed[mask]
            rec_mean = float(rec_valid.mean())
            offset = orig_mean - rec_mean
            reconstructed = reconstructed + offset
            return reconstructed

    orig_mean = float(original.mean())
    rec_mean = float(reconstructed.mean())
    offset = orig_mean - rec_mean
    reconstructed = reconstructed + offset
    return reconstructed
