"""FR-003/FR-004: Python wrapper over the C# terrain->WDL path.

Thin wrapper around WdlWriter.ExtractTileHeightsFromAlpha (via the
WowViewer.Tool.WdlRead shim). No new WDL algorithm lives in Python.
"""

from __future__ import annotations

import numpy as np

from harvester.v24 import shim


def build_synth_wdl(
    height_257: np.ndarray,
    liquid_mask_256: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build the synthetic WDL for one tile.

    Returns ``(outer (17,17) float32, inner (16,16) float32)`` — the C# reader's
    grid shape. Liquid-covered lattice sample points are re-sampled from the
    nearest non-liquid pixel inside the shim.
    """
    return shim.build_synth_wdl_batch(height_257, liquid_mask_256)


def build_synth_wdl_batch(
    height_257: np.ndarray,
    liquid_mask_256: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Batch variant: (N,257,257) heights -> ((N,17,17), (N,16,16)) grids."""
    return shim.build_synth_wdl_batch(height_257, liquid_mask_256)
