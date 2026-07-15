"""Visual reconstruction helpers for the paired WDL lattice (review only, not full terrain)."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def reconstruct_wdl_pair(outer_17: np.ndarray, inner_16: np.ndarray, size: int = 257) -> np.ndarray:
    """Interpolate the 17x17 outer grid, inject 16x16 inner samples, then resample to a height grid."""
    outer = np.asarray(outer_17, dtype=np.float32)
    inner = np.asarray(inner_16, dtype=np.float32)
    if outer.shape != (17, 17) or inner.shape != (16, 16):
        raise ValueError(f"expected paired WDL shapes (17,17)/(16,16), got {outer.shape}/{inner.shape}")
    coarse = F.interpolate(torch.from_numpy(outer).view(1, 1, 17, 17), size=(33, 33), mode="bilinear", align_corners=True)[0, 0]
    coarse[::2, ::2] = torch.from_numpy(outer)
    coarse[1::2, 1::2] = torch.from_numpy(inner)
    return F.interpolate(coarse.view(1, 1, 33, 33), size=(size, size), mode="bilinear", align_corners=True)[0, 0].numpy()
