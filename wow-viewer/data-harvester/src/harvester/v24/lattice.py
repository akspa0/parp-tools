"""WDL lattice geometry (spec amendment A6).

The 17x17 outer grid samples height_257 at (16r, 16c); the 16x16 inner grid
samples at (16r+8, 16c+8). Together they interleave as a quincunx on a 33x33
half-step lattice, and because (33-1)*8+1 = 257, a bilinear upsample of the
33x33 lattice with align_corners semantics is exact at every WDL sample point.
"""

from __future__ import annotations

import numpy as np

OUTER_DIM = 17
INNER_DIM = 16
QUINCUNX_DIM = 33
FULL_DIM = 257


def sample_lattice_from_height(height_257: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Point-sample a heightmap at the WDL lattice positions (baseline / A1 rule).

    Unlike the C# path this keeps float precision (no int16 rounding); it is the
    trivial "no learning" baseline used by SC-002.
    """
    heights = np.asarray(height_257, dtype=np.float32)
    if heights.shape[-2:] != (FULL_DIM, FULL_DIM):
        raise ValueError(f"height_257 must end in (257, 257); got {heights.shape}")

    outer = heights[..., ::16, ::16]
    inner = heights[..., 8::16, 8::16]
    return np.ascontiguousarray(outer), np.ascontiguousarray(inner)


def quincunx_33(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Interleave outer (…,17,17) + inner (…,16,16) into the 33x33 quincunx.

    Positions (2i, 2j) carry outer values, (2i+1, 2j+1) carry inner values, and
    the remaining half-step positions are filled with the mean of their valid
    4-neighbours (all of which are lattice points).
    """
    outer = np.asarray(outer, dtype=np.float32)
    inner = np.asarray(inner, dtype=np.float32)
    if outer.shape[-2:] != (OUTER_DIM, OUTER_DIM) or inner.shape[-2:] != (INNER_DIM, INNER_DIM):
        raise ValueError(f"expected (...,17,17) and (...,16,16); got {outer.shape} / {inner.shape}")

    lead = outer.shape[:-2]
    q = np.zeros((*lead, QUINCUNX_DIM, QUINCUNX_DIM), dtype=np.float32)
    q[..., ::2, ::2] = outer
    q[..., 1::2, 1::2] = inner

    # (even row, odd col): horizontal outer neighbours + vertical inner neighbours.
    total = q[..., ::2, :-2:2] + q[..., ::2, 2::2]
    count = np.full(total.shape, 2.0, dtype=np.float32)
    inner_above = np.zeros_like(total)
    inner_above[..., 1:, :] = q[..., 1:-1:2, 1::2]
    has_above = np.zeros_like(total)
    has_above[..., 1:, :] = 1.0
    inner_below = np.zeros_like(total)
    inner_below[..., :-1, :] = q[..., 1::2, 1::2]
    has_below = np.zeros_like(total)
    has_below[..., :-1, :] = 1.0
    q[..., ::2, 1::2] = (total + inner_above + inner_below) / (count + has_above + has_below)

    # (odd row, even col): vertical outer neighbours + horizontal inner neighbours.
    total = q[..., :-2:2, ::2] + q[..., 2::2, ::2]
    count = np.full(total.shape, 2.0, dtype=np.float32)
    inner_left = np.zeros_like(total)
    inner_left[..., :, 1:] = q[..., 1::2, 1:-1:2]
    has_left = np.zeros_like(total)
    has_left[..., :, 1:] = 1.0
    inner_right = np.zeros_like(total)
    inner_right[..., :, :-1] = q[..., 1::2, 1::2]
    has_right = np.zeros_like(total)
    has_right[..., :, :-1] = 1.0
    q[..., 1::2, ::2] = (total + inner_left + inner_right) / (count + has_left + has_right)

    return q


def upsample_prior_257(outer: np.ndarray, inner: np.ndarray) -> np.ndarray:
    """Deterministic bilinear upsample of the WDL prior to 257x257.

    Exact at every WDL lattice point (align_corners bilinear, integer scale 8).
    """
    q = quincunx_33(outer, inner)
    return upsample_quincunx_257(q)


def upsample_quincunx_257(quincunx: np.ndarray) -> np.ndarray:
    """Bilinear 33 -> 257 with align_corners semantics, pure NumPy."""
    q = np.asarray(quincunx, dtype=np.float32)
    if q.shape[-2:] != (QUINCUNX_DIM, QUINCUNX_DIM):
        raise ValueError(f"expected (...,33,33); got {q.shape}")

    coords = np.arange(FULL_DIM, dtype=np.float32) / 8.0
    idx0 = np.minimum(coords.astype(np.int32), QUINCUNX_DIM - 2)
    frac = coords - idx0

    rows0 = q[..., idx0, :]
    rows1 = q[..., idx0 + 1, :]
    fy = frac.reshape((1,) * (q.ndim - 2) + (FULL_DIM, 1))
    interp_rows = rows0 * (1.0 - fy) + rows1 * fy

    cols0 = interp_rows[..., :, idx0]
    cols1 = interp_rows[..., :, idx0 + 1]
    fx = frac.reshape((1,) * (q.ndim - 2) + (1, FULL_DIM))
    return cols0 * (1.0 - fx) + cols1 * fx
