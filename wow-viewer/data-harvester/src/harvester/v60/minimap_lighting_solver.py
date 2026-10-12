"""Automated photometric minimap lighting and specular calibration solver.

Calibrates solar azimuth, elevation, ambient/diffuse balance, and specular reflectance
against real 0.5.3 whiteplate or untextured minimap observations.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import minimize


@dataclass(frozen=True, slots=True)
class LightingCalibrationProfile:
    build: str
    map_name: str
    solar_azimuth_rad: float
    solar_elevation_rad: float
    ambient_intensity: float
    diffuse_intensity: float
    specular_intensity: float
    specular_power: float
    photometric_mae: float
    converged: bool
    iterations: int
    provenance_hash: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: Path | str) -> None:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with p.open("w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def render_synthetic_shadow(
    normals: np.ndarray,
    solar_azimuth: float,
    solar_elevation: float,
    ambient: float,
    diffuse: float,
    specular_intensity: float = 0.0,
    specular_power: float = 8.0,
    shadow_mask: np.ndarray | None = None,
) -> np.ndarray:
    """Render terrain shading given surface normals and photometric lighting parameters."""
    norm = np.asarray(normals, dtype=np.float32)
    if norm.ndim != 3 or norm.shape[-1] != 3:
        raise ValueError(f"Expected normals with shape (H, W, 3), got {norm.shape}")

    cos_el = np.cos(solar_elevation, dtype=np.float32)
    sin_el = np.sin(solar_elevation, dtype=np.float32)
    light_dir = np.array(
        [cos_el * np.cos(solar_azimuth, dtype=np.float32), cos_el * np.sin(solar_azimuth, dtype=np.float32), sin_el],
        dtype=np.float32,
    )
    light_dir = light_dir / (np.linalg.norm(light_dir) + 1e-7)

    # Top-down orthographic camera vector pointing from surface to camera (+Z)
    view_dir = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    half_vec = light_dir + view_dir
    half_norm = np.linalg.norm(half_vec)
    half_vec = half_vec / (half_norm if half_norm > 1e-7 else 1.0)

    # Lambertian term: N · L
    lambert = np.clip(np.sum(norm * light_dir, axis=-1), 0.0, 1.0)

    # Specular term: (N · H)^p
    if specular_intensity > 0.0 and specular_power > 0.0:
        n_dot_h = np.clip(np.sum(norm * half_vec, axis=-1), 0.0, 1.0)
        spec = (n_dot_h**specular_power) * specular_intensity
    else:
        spec = 0.0

    vis = 1.0
    if shadow_mask is not None:
        vis = 1.0 - np.clip(np.asarray(shadow_mask, dtype=np.float32), 0.0, 1.0)

    lit = ambient + (diffuse * lambert * vis) + (diffuse * spec * vis)
    return np.clip(lit, 0.0, 1.0)


def compute_photometric_mae(
    params: np.ndarray,
    normals: np.ndarray,
    observed_luma: np.ndarray,
    shadow_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
) -> float:
    """Compute mean absolute error between observed luminance and synthetic shadow render."""
    az, el, amb, diff, spec_int, spec_pow = params
    synth = render_synthetic_shadow(
        normals=normals,
        solar_azimuth=az,
        solar_elevation=el,
        ambient=amb,
        diffuse=diff,
        specular_intensity=spec_int,
        specular_power=spec_pow,
        shadow_mask=shadow_mask,
    )

    diff_map = np.abs(synth - observed_luma)
    if valid_mask is not None:
        v = np.asarray(valid_mask, dtype=bool)
        if np.any(v):
            return float(np.mean(diff_map[v]))
    return float(np.mean(diff_map))


def solve_minimap_lighting(
    normals: np.ndarray,
    observed_minimap: np.ndarray,
    shadow_mask: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    build: str = "0.5.3.3368",
    map_name: str = "Azeroth",
    initial_guess: tuple[float, float, float, float, float, float] | None = None,
) -> LightingCalibrationProfile:
    """Solve for optimal lighting parameters using two-stage coarse grid and L-BFGS-B refinement."""
    obs = np.asarray(observed_minimap, dtype=np.float32)
    if obs.ndim == 3 and obs.shape[-1] >= 3:
        # Convert RGB to normalized luminance [0, 1]
        obs_luma = (0.299 * obs[..., 0] + 0.587 * obs[..., 1] + 0.114 * obs[..., 2]) / (
            255.0 if obs.max() > 1.0 else 1.0
        )
    else:
        obs_luma = obs / (255.0 if obs.max() > 1.0 else 1.0)

    # Initial parameter vector: [azimuth, elevation, ambient, diffuse, spec_intensity, spec_power]
    if initial_guess is None:
        # Stage 1: Coarse azimuth grid sweep (8 directions)
        best_x0 = np.array([np.pi * 0.75, np.pi / 4.0, 0.25, 0.75, 0.05, 8.0], dtype=np.float32)
        best_mae = float("inf")
        for az in np.linspace(0, 2 * np.pi, 8, endpoint=False):
            candidate = np.array([az, np.pi / 4.5, 0.25, 0.75, 0.05, 8.0], dtype=np.float32)
            mae = compute_photometric_mae(candidate, normals, obs_luma, shadow_mask, valid_mask)
            if mae < best_mae:
                best_mae = mae
                best_x0 = candidate
    else:
        best_x0 = np.array(initial_guess, dtype=np.float32)

    bounds = [
        (0.0, 2.0 * np.pi),  # azimuth
        (np.pi / 12.0, np.pi / 2.0),  # elevation [15, 90 deg]
        (0.05, 0.60),  # ambient
        (0.20, 1.00),  # diffuse
        (0.00, 0.50),  # specular intensity
        (1.0, 64.0),  # specular power
    ]

    res = minimize(
        compute_photometric_mae,
        best_x0,
        args=(normals, obs_luma, shadow_mask, valid_mask),
        method="L-BFGS-B",
        bounds=bounds,
        options={"maxiter": 40, "ftol": 1e-4},
    )

    final_params = res.x
    final_mae = float(res.fun)

    prov_bytes = f"{build}|{map_name}|{final_params.tolist()}|{final_mae:.6f}".encode("utf-8")
    prov_hash = hashlib.sha256(prov_bytes).hexdigest()

    return LightingCalibrationProfile(
        build=build,
        map_name=map_name,
        solar_azimuth_rad=float(final_params[0]),
        solar_elevation_rad=float(final_params[1]),
        ambient_intensity=float(final_params[2]),
        diffuse_intensity=float(final_params[3]),
        specular_intensity=float(final_params[4]),
        specular_power=float(final_params[5]),
        photometric_mae=final_mae,
        converged=bool((res.success or final_mae < 0.05) and final_mae < 0.20),
        iterations=int(res.nit),
        provenance_hash=prov_hash,
    )
