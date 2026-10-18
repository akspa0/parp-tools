"""Minimap Hillshade Correlation & Terrain Implication Evaluator (Spec 274).

Bridges translated clean hillshades to terrain elevation targets:
- Evaluates domain translation fidelity (SSIM, L1 error against synthetic DXT1 ground truth)
- Evaluates terrain correlation metrics (Pearson r, MAE in yards, Max Error)
- Chains: Real Minimap RGB -> Domain Translation -> Clean Hillshade -> Terrain Elevation
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats

from harvester.v60.domain_translation_model import DomainTranslationUNet

logger = logging.getLogger(__name__)


def compute_elevation_metrics(
    pred_z: np.ndarray,
    target_z: np.ndarray,
    mask: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Compute terrain reconstruction metrics: Pearson r, MAE, RMSE, Max Error in yards."""
    p = pred_z.flatten()
    t = target_z.flatten()

    if mask is not None:
        m = mask.flatten().astype(bool)
        p = p[m]
        t = t[m]

    if len(p) == 0:
        return {"pearson_r": 0.0, "mae_yards": 0.0, "rmse_yards": 0.0, "max_error_yards": 0.0}

    # Pearson correlation r
    if np.std(p) > 1e-6 and np.std(t) > 1e-6:
        r, _ = stats.pearsonr(p, t)
    else:
        r = 0.0

    err = np.abs(p - t)
    mae = float(np.mean(err))
    rmse = float(np.sqrt(np.mean((p - t) ** 2)))
    max_err = float(np.max(err))

    return {
        "pearson_r": float(r),
        "mae_yards": mae,
        "rmse_yards": rmse,
        "max_error_yards": max_err,
    }


class MinimapHillshadeCorrelator:
    """Evaluates the end-to-end correlation pipeline from minimap tile to terrain elevation."""

    def __init__(
        self,
        domain_model_path: Optional[Path | str] = None,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.device = torch.device(device)
        self.domain_model: Optional[DomainTranslationUNet] = None

        if domain_model_path and Path(domain_model_path).exists():
            ckpt = torch.load(domain_model_path, map_location=self.device)
            self.domain_model = DomainTranslationUNet(in_channels=3, out_channels=3).to(self.device)
            state_dict = ckpt.get("model_state_dict", ckpt)
            self.domain_model.load_state_dict(state_dict)
            self.domain_model.eval()
            logger.info(f"Loaded DomainTranslationUNet from {domain_model_path}")

    @torch.no_grad()
    def translate_minimap(self, minimap_rgb: np.ndarray) -> np.ndarray:
        """Translate a real textured minimap (256, 256, 3) uint8 or float -> clean hillshade."""
        if self.domain_model is None:
            raise RuntimeError("DomainTranslationUNet checkpoint not loaded.")

        if minimap_rgb.max() > 1.01:
            img = (minimap_rgb.astype(np.float32) / 255.0)
        else:
            img = minimap_rgb.astype(np.float32)

        tensor = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        pred = self.domain_model(tensor).squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip(pred, 0.0, 1.0)
