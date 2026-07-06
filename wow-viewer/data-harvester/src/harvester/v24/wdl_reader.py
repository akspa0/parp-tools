"""FR-001: Python wrapper over the C# WDL reader (real staged-client WDLs).

The C# reader (WdlSummaryReader, wrapped by WowViewer.Tool.WdlRead) is the
source of truth for the WDL grid shape: 17x17 outer + 16x16 inner int16 per
MARE, converted to float32 at the shim boundary. MAHO is not exposed by the
C# reader, so the second tuple element is always ``None`` (spec amendment A1).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from harvester.v24 import shim


def read_wdl_mare(
    wdl_path: Path | str,
    tile_x: int,
    tile_y: int,
) -> tuple[tuple[np.ndarray, np.ndarray], None] | None:
    """Read one MARE from a loose ``.wdl`` file.

    Returns ``((outer (17,17) float32, inner (16,16) float32), None)`` — the
    ``None`` is the MAHO slot, unexposed by the C# reader — or ``None`` when
    the WDL has no entry for the tile.
    """
    tiles = shim.read_wdl_map(wdl_path=wdl_path)
    if tiles is None:
        return None
    grids = tiles.get((int(tile_x), int(tile_y)))
    if grids is None:
        return None
    return grids, None


def read_wdl_map_tiles(
    client_root: Path | str,
    map_name: str,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None:
    """Read every MARE tile for a map from a staged client (batch entry point).

    Returns ``{(tile_x, tile_y): (outer, inner)}`` or ``None`` when the map has
    no WDL in the staged client.
    """
    return shim.read_wdl_map(client_root=client_root, map_name=map_name)
