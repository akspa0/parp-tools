"""MCCV Terrain Shadow Comparator and Synthesis Bridge (Spec 262 / WoW: Forever 1.60).

Compares 1.12.1 / 0.5.3 minimap bare residual shadow signals against 1.60 MCCV
terrain vertex shadow data, and provides bidirectional conversion to synthesize
1.60-compliant MCCV vertex color chunks (145 vertices BGRA) from residual shadows.
"""

from __future__ import annotations

import logging
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy import interpolate, ndimage

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class MccvComparisonMetrics:
    normalized_cross_correlation: float
    mean_absolute_error: float
    structural_similarity: float
    dynamic_range_ratio: float
    correlation_passed: bool

    def to_dict(self) -> Dict[str, float]:
        return {
            "normalized_cross_correlation": self.normalized_cross_correlation,
            "mean_absolute_error": self.mean_absolute_error,
            "structural_similarity": self.structural_similarity,
            "dynamic_range_ratio": self.dynamic_range_ratio,
            "correlation_passed": float(self.correlation_passed),
        }


def get_chunk_vertex_uvs() -> np.ndarray:
    """Generate normalized (u, v) in [0, 1] for the 145 vertices of a WoW MCNK chunk.

    145 vertices consists of:
      - 9x9 outer grid (81 vertices): u, v in {0, 1/8, 2/8, ..., 1}
      - 8x8 inner grid (64 vertices): u, v in {1/16, 3/16, ..., 15/16}
    """
    uvs = np.zeros((145, 2), dtype=np.float32)

    # 9x9 Outer grid (index 0..80)
    idx = 0
    for y in range(9):
        for x in range(9):
            uvs[idx] = [x / 8.0, y / 8.0]
            idx += 1

    # 8x8 Inner grid (index 81..144)
    for y in range(8):
        for x in range(8):
            uvs[idx] = [(x + 0.5) / 8.0, (y + 0.5) / 8.0]
            idx += 1

    return uvs


def rasterize_mccv_tile_to_grid(
    chunk_mccv_colors: Dict[Tuple[int, int], np.ndarray] | np.ndarray,
    tile_res: int = 256,
) -> np.ndarray:
    """Rasterize a tile's 16x16 chunk MCCV vertex colors (145 vertices each) into a 256x256 shadow map.

    In 1.60 modern terrain rendering, MCCV acts as a terrain self-shadow / ambient occlusion
    multiplier where 127 (0.5 in float) is neutral (1.0x light multiplier), lower values darken,
    and higher values brighten.
    """
    raster = np.zeros((tile_res, tile_res), dtype=np.float32)
    chunk_pixels = tile_res // 16  # 16 pixels per chunk for 256x256
    vertex_uvs = get_chunk_vertex_uvs()

    # Precompute Delaunay triangulation or interpolation grid for a single chunk
    grid_y, grid_x = np.mgrid[0:chunk_pixels, 0:chunk_pixels]
    sample_uvs = np.column_stack([grid_x.ravel() / (chunk_pixels - 1), grid_y.ravel() / (chunk_pixels - 1)])

    for cy in range(16):
        for cx in range(16):
            if isinstance(chunk_mccv_colors, dict):
                mccv_raw = chunk_mccv_colors.get((cx, cy), None)
            elif isinstance(chunk_mccv_colors, np.ndarray) and chunk_mccv_colors.ndim == 3:
                mccv_raw = chunk_mccv_colors[cy, cx]
            else:
                mccv_raw = None

            if mccv_raw is None or len(mccv_raw) < 145:
                # Default neutral shadow (0.5)
                chunk_vals = np.full((chunk_pixels, chunk_pixels), 0.5, dtype=np.float32)
            else:
                # Extract luminance from BGRA / RGBA
                if mccv_raw.ndim == 2 and mccv_raw.shape[-1] >= 3:
                    luma = (0.299 * mccv_raw[:, 2] + 0.587 * mccv_raw[:, 1] + 0.114 * mccv_raw[:, 0]) / 255.0
                elif mccv_raw.dtype == np.uint8:
                    # Flat 580-byte BGRA array
                    bgra = mccv_raw.reshape((145, 4))
                    luma = (0.299 * bgra[:, 2] + 0.587 * bgra[:, 1] + 0.114 * bgra[:, 0]) / 255.0
                else:
                    luma = mccv_raw.astype(np.float32)
                    if luma.max() > 1.0:
                        luma /= 255.0

                # Interpolate 145 vertex values onto 16x16 chunk pixel grid
                interp = interpolate.LinearNDInterpolator(vertex_uvs, luma, fill_value=0.5)
                chunk_vals = interp(sample_uvs).reshape((chunk_pixels, chunk_pixels))

            px = cx * chunk_pixels
            py = cy * chunk_pixels
            raster[py : py + chunk_pixels, px : px + chunk_pixels] = chunk_vals

    return raster


def synthesize_mccv_from_residual(
    residual_shadow_256: np.ndarray,
    neutral_level: float = 0.5,
    contrast_scale: float = 0.8,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Sample a 256x256 residual shadow map into authentic 1.60 MCCV vertex color chunks (145 vertices BGRA).

    Allows injecting extracted 1.12.1 / 0.5.3 minimap residual shadows directly into
    WoW: Forever ADT files to restore authentic terrain depth and shading hints.
    """
    res = np.asarray(residual_shadow_256, dtype=np.float32)
    if res.max() > 1.0:
        res /= 255.0

    chunk_pixels = res.shape[0] // 16
    vertex_uvs = get_chunk_vertex_uvs()
    mccv_dict: Dict[Tuple[int, int], np.ndarray] = {}

    for cy in range(16):
        for cx in range(16):
            px0 = cx * chunk_pixels
            py0 = cy * chunk_pixels
            chunk_slice = res[py0 : py0 + chunk_pixels, px0 : px0 + chunk_pixels]

            # Sample chunk slice at the 145 vertex coordinates
            coords_x = vertex_uvs[:, 0] * (chunk_pixels - 1)
            coords_y = vertex_uvs[:, 1] * (chunk_pixels - 1)

            sampled_luma = ndimage.map_coordinates(chunk_slice, [coords_y, coords_x], order=1, mode="nearest")

            byte_vals = np.clip(sampled_luma * 255.0, 0.0, 255.0).astype(np.uint8)

            # Pack into 145 * 4 BGRA format (B=G=R, A=255)
            bgra = np.zeros((145, 4), dtype=np.uint8)
            bgra[:, 0] = byte_vals  # Blue
            bgra[:, 1] = byte_vals  # Green
            bgra[:, 2] = byte_vals  # Red
            bgra[:, 3] = 255        # Alpha

            mccv_dict[(cx, cy)] = bgra.ravel()

    return mccv_dict


def compare_residual_to_mccv(
    residual_shadow_256: np.ndarray,
    mccv_raster_256: np.ndarray,
) -> MccvComparisonMetrics:
    """Compare 1.12.1 minimap residual shadow against 1.60 MCCV rasterized shadow."""
    s1 = np.asarray(residual_shadow_256, dtype=np.float32)
    s2 = np.asarray(mccv_raster_256, dtype=np.float32)

    if s1.max() > 1.0:
        s1 = s1 / 255.0
    if s2.max() > 1.0:
        s2 = s2 / 255.0

    # Mean center
    s1_c = s1 - np.mean(s1)
    s2_c = s2 - np.mean(s2)

    # Normalized Cross-Correlation (NCC)
    norm1 = np.linalg.norm(s1_c)
    norm2 = np.linalg.norm(s2_c)
    if norm1 > 1e-6 and norm2 > 1e-6:
        ncc = float(np.sum(s1_c * s2_c) / (norm1 * norm2))
    else:
        ncc = 0.0

    mae = float(np.mean(np.abs(s1 - s2)))

    # Simplified structural similarity (luminance + contrast correlation)
    var1 = np.var(s1)
    var2 = np.var(s2)
    cov = np.mean(s1_c * s2_c)
    c1 = (0.01) ** 2
    c2 = (0.03) ** 2
    ssim = float(
        (2.0 * np.mean(s1) * np.mean(s2) + c1)
        * (2.0 * cov + c2)
        / ((np.mean(s1) ** 2 + np.mean(s2) ** 2 + c1) * (var1 + var2 + c2))
    )

    dr_ratio = float(np.std(s1) / max(1e-6, np.std(s2)))

    return MccvComparisonMetrics(
        normalized_cross_correlation=ncc,
        mean_absolute_error=mae,
        structural_similarity=ssim,
        dynamic_range_ratio=dr_ratio,
        correlation_passed=bool(ncc >= 0.70 or ssim >= 0.75),
    )


def read_mccv_from_adt(adt_source: bytes | str | Path) -> Dict[Tuple[int, int], np.ndarray]:
    """Extract 16x16 chunk MCCV vertex colors (145 vertices BGRA) from a root ADT file.

    Walks MCIN / MCNK chunks and parses MCCV subchunks.
    Returns dict mapping (cx, cy) -> np.ndarray of shape (145, 4) uint8.
    """
    if isinstance(adt_source, (str, Path)):
        data = Path(adt_source).read_bytes()
    else:
        data = bytes(adt_source)

    chunks: Dict[Tuple[int, int], np.ndarray] = {}
    total_len = len(data)

    # First search for MCIN chunk to locate MCNK chunks
    mcin_offset = -1
    pos = 0
    while pos + 8 <= total_len:
        magic = data[pos : pos + 4]
        size = struct.unpack_from("<I", data, pos + 4)[0]
        if magic in (b"MCIN", b"NICM"):
            mcin_offset = pos + 8
            break
        pos += 8 + size
        if size % 2 != 0:
            pos += 1

    mcnk_offsets: List[int] = []
    if mcin_offset >= 0 and mcin_offset + (256 * 16) <= total_len:
        for i in range(256):
            off = struct.unpack_from("<I", data, mcin_offset + (i * 16))[0]
            if off > 0 and off + 8 <= total_len:
                mcnk_offsets.append(off)

    if not mcnk_offsets:
        # Fallback: scan for MCNK chunks sequentially
        pos = 0
        while pos + 8 <= total_len:
            magic = data[pos : pos + 4]
            size = struct.unpack_from("<I", data, pos + 4)[0]
            if magic in (b"MCNK", b"KNCM"):
                mcnk_offsets.append(pos)
            pos += 8 + size
            if size % 2 != 0:
                pos += 1

    for mcnk_off in mcnk_offsets:
        if mcnk_off + 8 + 128 > total_len:
            continue
        mcnk_size = struct.unpack_from("<I", data, mcnk_off + 4)[0]
        mcnk_payload = mcnk_off + 8

        # MCNK header: IndexX at +0x04, IndexY at +0x08, ofsMccv at +0x74
        cx = struct.unpack_from("<I", data, mcnk_payload + 4)[0]
        cy = struct.unpack_from("<I", data, mcnk_payload + 8)[0]
        ofs_mccv = struct.unpack_from("<I", data, mcnk_payload + 0x74)[0]

        mccv_data: Optional[bytes] = None

        if ofs_mccv > 0 and mcnk_payload + ofs_mccv + 8 + 580 <= total_len:
            sub_magic = data[mcnk_payload + ofs_mccv : mcnk_payload + ofs_mccv + 4]
            if sub_magic in (b"MCCV", b"VCCM"):
                mccv_data = data[mcnk_payload + ofs_mccv + 8 : mcnk_payload + ofs_mccv + 8 + 580]

        if mccv_data is None:
            # Subchunk scan inside MCNK
            sub_pos = mcnk_payload + 128
            mcnk_end = min(total_len, mcnk_payload + mcnk_size)
            while sub_pos + 8 <= mcnk_end:
                sub_magic = data[sub_pos : sub_pos + 4]
                sub_size = struct.unpack_from("<I", data, sub_pos + 4)[0]
                if sub_magic in (b"MCCV", b"VCCM") and sub_size >= 580:
                    mccv_data = data[sub_pos + 8 : sub_pos + 8 + 580]
                    break
                sub_pos += 8 + sub_size

        if mccv_data is not None and len(mccv_data) == 580:
            chunks[(cx, cy)] = np.frombuffer(mccv_data, dtype=np.uint8).reshape((145, 4))

    return chunks
