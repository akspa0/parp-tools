"""Subprocess bridge to the C# WDL shim (WowViewer.Tool.WdlRead).

The shim wraps the unmodified C# WDL reader (WdlSummaryReader) and the
unmodified terrain->WDL path (WdlWriter.ExtractTileHeightsFromAlpha).
All game-data parsing stays in C#; this module only moves NPZ files.
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

DATA_HARVESTER_ROOT = Path(__file__).resolve().parents[3]
WOW_VIEWER_ROOT = DATA_HARVESTER_ROOT.parent
_SHIM_PROJECT = WOW_VIEWER_ROOT / "tools" / "wdl-read" / "WowViewer.Tool.WdlRead"
_SHIM_DLL_CANDIDATES = (
    _SHIM_PROJECT / "bin" / "Release" / "net10.0" / "WowViewer.Tool.WdlRead.dll",
    _SHIM_PROJECT / "bin" / "Debug" / "net10.0" / "WowViewer.Tool.WdlRead.dll",
)

OUTER_SHAPE = (17, 17)
INNER_SHAPE = (16, 16)

EXIT_OK = 0
EXIT_NO_WDL = 2
EXIT_TILE_ABSENT = 3


def find_shim_dll() -> Path:
    """Locate the built shim DLL (env override, then Release, then Debug)."""
    override = os.environ.get("WOWVIEWER_WDLREAD_DLL")
    if override:
        path = Path(override)
        if path.exists():
            return path
        raise RuntimeError(f"WOWVIEWER_WDLREAD_DLL points at a missing file: {override}")

    for candidate in _SHIM_DLL_CANDIDATES:
        if candidate.exists():
            return candidate

    raise RuntimeError(
        "WowViewer.Tool.WdlRead is not built. Build it first:\n"
        f"  dotnet build {_SHIM_PROJECT} -c Debug"
    )


def run_shim(args: list[str]) -> subprocess.CompletedProcess[str]:
    """Run the shim with the given arguments and return the completed process."""
    cmd = ["dotnet", str(find_shim_dll()), *args]
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def read_wdl_map(
    client_root: Path | str | None = None,
    map_name: str | None = None,
    wdl_path: Path | str | None = None,
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] | None:
    """Read every MARE tile of a map's WDL via the C# reader.

    Returns ``{(tile_x, tile_y): (outer (17,17) float32, inner (16,16) float32)}``,
    or ``None`` when the map has no WDL at all.
    """
    with tempfile.TemporaryDirectory(prefix="wdlread_") as tmp:
        output = Path(tmp) / "wdl.npz"
        if wdl_path is not None:
            args = ["read", "--wdl", str(wdl_path), "--output", str(output)]
        else:
            if client_root is None or map_name is None:
                raise ValueError("read_wdl_map needs either wdl_path or client_root + map_name")
            args = [
                "read",
                "--client-root", str(client_root),
                "--map", str(map_name),
                "--output", str(output),
            ]

        proc = run_shim(args)
        if proc.returncode == EXIT_NO_WDL:
            return None
        if proc.returncode != EXIT_OK:
            raise RuntimeError(
                f"WdlRead read failed (exit {proc.returncode}): {proc.stderr or proc.stdout}"
            )

        with np.load(output) as data:
            tile_xy = data["tile_xy"]
            outer = data["outer"].astype(np.float32)
            inner = data["inner"].astype(np.float32)

    result: dict[tuple[int, int], tuple[np.ndarray, np.ndarray]] = {}
    for i in range(tile_xy.shape[0]):
        key = (int(tile_xy[i, 0]), int(tile_xy[i, 1]))
        result[key] = (outer[i], inner[i])
    return result


def build_synth_wdl_batch(
    height_257: np.ndarray,
    liquid_mask_256: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build synthetic WDL grids from stacked heightmaps via the C# terrain->WDL path.

    ``height_257`` is (N, 257, 257) float32 (a single (257, 257) tile is promoted).
    ``liquid_mask_256`` is (N, 256, 256) with values > 0.5 treated as liquid.
    Returns ``(outer (N, 17, 17) float32, inner (N, 16, 16) float32)``.
    """
    heights = np.asarray(height_257, dtype=np.float32)
    single = heights.ndim == 2
    if single:
        heights = heights[None]
    if heights.shape[1:] != (257, 257):
        raise ValueError(f"height_257 must be (N, 257, 257); got {heights.shape}")

    liquids = None
    if liquid_mask_256 is not None:
        liquids = np.asarray(liquid_mask_256, dtype=np.float32)
        if single and liquids.ndim == 2:
            liquids = liquids[None]
        if liquids.shape != (heights.shape[0], 256, 256):
            raise ValueError(
                f"liquid_mask_256 must be (N, 256, 256) matching heights; got {liquids.shape}"
            )

    with tempfile.TemporaryDirectory(prefix="wdlsynth_") as tmp:
        height_npz = Path(tmp) / "height.npz"
        output = Path(tmp) / "synth.npz"
        np.savez(height_npz, height_257=heights)
        args = ["synth", "--height", str(height_npz), "--output", str(output)]
        if liquids is not None:
            liquid_npz = Path(tmp) / "liquid.npz"
            np.savez(liquid_npz, liquid_mask=liquids)
            args.extend(["--liquid", str(liquid_npz)])

        proc = run_shim(args)
        if proc.returncode != EXIT_OK:
            raise RuntimeError(
                f"WdlRead synth failed (exit {proc.returncode}): {proc.stderr or proc.stdout}"
            )

        with np.load(output) as data:
            outer = data["outer"].astype(np.float32)
            inner = data["inner"].astype(np.float32)

    if single:
        return outer[0], inner[0]
    return outer, inner
