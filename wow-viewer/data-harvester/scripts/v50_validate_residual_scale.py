"""Test the Spec 125 residual hypotheses against the ground-truth heightmap.

The user's scale hypothesis (2026-08-02): the residual may be the heightmap scaled down by ~33.334x —
the same weak-signal band around +/- 2.778 Z that the weak-signal amplifier was built to restore. The
residuals look, visually and in the game data, like the weak signals.

The first run (683 Azeroth pairs) REFUTED the linear-scale hypothesis: correlation ~0.20, best-fit
scale ~-0.0003. That is expected — the residual is a *shading* signal (Lambert N·L + cast shadows),
which is a nonlinear function of the heightmap through its normals, not a linear scale.

This script therefore tests the correct hypotheses:

  1. residual ~= height / 33.334            (the user's scale hypothesis)
  2. residual ~= height / 33.334, clipped to +/- 2.778 Z  (the weak-signal band)
  3. residual ~= a * height + b              (best-fit linear, to see how close 1/33.334 is)
  4. residual ~= normalized relative height  (the Spec 112 target, as a baseline)
  5. residual ~= hillshade(height)           (Lambert N·L from height gradients — the synthesizer's
                                             own shading model; the physically correct comparison)
  6. per-tile best-fit scale                 (does a consistent per-tile scale exist even if no
                                             global one does? the user notes the weak-signal scale
                                             was not always 33.334 but some fraction of it)

For each candidate it reports Pearson correlation, so the hypotheses are confirmed or refuted on real
data rather than by eye.

Usage (USER runs):
  uv run python scripts/v50_validate_residual_scale.py \
      --residual-dir <residual-output>/tiles \
      --height-store <v50-store> \
      --map Azeroth
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

RESIDUAL_RE = re.compile(r"^(?P<map>.+?)_(?P<tx>\d{2})_(?P<ty>\d{2})_residual\.png$")
SCALE_33_334 = 33.334
WEAK_BAND = 2.778  # +/- 2.778 Z


def _load_residual(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        arr = np.asarray(img.convert("L"), dtype=np.float32)
    return arr / 255.0  # 0..1


def _load_height(store: Path, map_name: str, tx: int, ty: int) -> np.ndarray | None:
    import zarr

    group = zarr.open_group(str(store), mode="r")
    if "height_257" not in group:
        return None
    table = pq.read_table(store / "index.parquet")
    rows = table.to_pylist()
    for i, row in enumerate(rows):
        if str(row.get("map", "")) == map_name and int(row.get("tile_x", -1)) == tx and int(row.get("tile_y", -1)) == ty:
            return np.asarray(group["height_257"][i], dtype=np.float32)
    return None


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    if a.size == 0 or b.size == 0:
        return 0.0
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


def _best_linear_scale(a: np.ndarray, b: np.ndarray) -> float:
    """Least-squares scalar k minimizing ||a - k*b||^2."""
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    denom = float((b * b).sum())
    return float((a * b).sum() / denom) if denom > 1e-12 else 0.0


def _hillshade(height: np.ndarray) -> np.ndarray:
    """Lambert N·L hillshade of a height field, matching the synthesizer's shading model.

    Computes unit normals from height gradients and dots them with a fixed NW light direction
    (the traced 1.0.0 bearing, 45 deg from +X toward +Y). Returns a 0..1 shading field.
    """
    h = np.asarray(height, dtype=np.float64)
    gy, gx = np.gradient(h)
    # Normal = (-gx, -gy, 1), normalized.
    nz = np.ones_like(gx)
    norm = np.sqrt(gx * gx + gy * gy + nz * nz)
    nx, ny, nz = -gx / norm, -gy / norm, nz / norm
    # Fixed NW light: normalize(-0.5, -0.5, 0.72) — the traced source bearing.
    lx, ly, lz = -0.5, -0.5, 0.72
    lnorm = np.sqrt(lx * lx + ly * ly + lz * lz)
    lx, ly, lz = lx / lnorm, ly / lnorm, lz / lnorm
    shade = np.clip(nx * lx + ny * ly + nz * lz, 0.0, 1.0)
    return shade.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Test the Spec 125 residual-scale hypothesis")
    ap.add_argument("--residual-dir", required=True, type=Path)
    ap.add_argument("--height-store", required=True, type=Path)
    ap.add_argument("--map", required=True)
    args = ap.parse_args()

    if not args.residual_dir.is_dir():
        raise SystemExit(f"residual dir not found: {args.residual_dir}")
    if not args.height_store.is_dir():
        raise SystemExit(f"height store not found: {args.height_store}")

    residuals: list[np.ndarray] = []
    heights: list[np.ndarray] = []
    for path in sorted(args.residual_dir.glob("*_residual.png")):
        m = RESIDUAL_RE.match(path.name)
        if not m:
            continue
        tx, ty = int(m.group("tx")), int(m.group("ty"))
        residual = _load_residual(path)
        height = _load_height(args.height_store, args.map, tx, ty)
        if height is None:
            continue
        # Residual is 256x256; height_257 is 257x257. Crop the height to the residual's grid so the
        # two fields are pixel-aligned (the 257th row/col is the shared tile-edge vertex).
        if height.shape[0] == residual.shape[0] + 1 and height.shape[1] == residual.shape[1] + 1:
            height = height[: residual.shape[0], : residual.shape[1]]
        if height.shape != residual.shape:
            print(f"  skip {path.name}: residual {residual.shape} vs height {height.shape} mismatch")
            continue
        residuals.append(residual)
        heights.append(height)

    if not residuals:
        raise SystemExit("no matched residual/height pairs")

    print(f"Matched {len(residuals)} residual/height pairs for {args.map}.\n")

    # Aggregate all pixels across tiles for a single robust correlation.
    R = np.concatenate([r.ravel() for r in residuals])
    H = np.concatenate([h.ravel() for h in heights])

    # Candidate 1: residual ~= height / 33.334
    cand_scale = H / SCALE_33_334
    # Candidate 2: residual ~= height/33.334 clipped to +/- 2.778 Z band
    cand_band = np.clip(H / SCALE_33_334, -WEAK_BAND / SCALE_33_334, WEAK_BAND / SCALE_33_334)
    # Candidate 3: best-fit linear residual ~= k*height
    k_fit = _best_linear_scale(R, H)
    cand_fit = k_fit * H
    # Candidate 4: normalized relative height (Spec 112 target)
    hmin, hmax = H.min(), H.max()
    cand_rel = (H - hmin) / max(hmax - hmin, 1.0)
    # Candidate 5: hillshade(height) — the synthesizer's own Lambert N·L shading model.
    cand_shade = np.concatenate([_hillshade(h).ravel() for h in heights])

    print(f"{'candidate':<46} {'corr':>8} {'scale':>10}")
    print("-" * 70)
    print(f"{'residual ~= height / 33.334':<46} {_pearson(R, cand_scale):8.4f} {'1/33.334':>10}")
    print(f"{'residual ~= height/33.334 clipped +/-2.778Z':<46} {_pearson(R, cand_band):8.4f} {'band':>10}")
    print(f"{'residual ~= k*height (best-fit)':<46} {_pearson(R, cand_fit):8.4f} {k_fit:10.4f}")
    print(f"{'residual ~= normalized relative height':<46} {_pearson(R, cand_rel):8.4f} {'rel':>10}")
    print(f"{'residual ~= hillshade(height) [N.L]':<46} {_pearson(R, cand_shade):8.4f} {'N.L':>10}")

    print()
    print(f"Best-fit scale k = {k_fit:.4f}  (hypothesis predicts 1/33.334 = {1/SCALE_33_334:.6f})")
    print(f"Ratio k / (1/33.334) = {k_fit / (1/SCALE_33_334):.3f}x")
    print()

    # Per-tile scale analysis: does a consistent per-tile scale exist even if no global one does?
    # The user notes the weak-signal scale was not always 33.334 but some fraction of it.
    per_tile_k = [_best_linear_scale(r, h) for r, h in zip(residuals, heights)]
    per_tile_k = [k for k in per_tile_k if abs(k) > 1e-9]
    if per_tile_k:
        k_arr = np.asarray(per_tile_k)
        print(f"Per-tile best-fit scale k: mean={k_arr.mean():.6f} std={k_arr.std():.6f} "
              f"min={k_arr.min():.6f} max={k_arr.max():.6f} (n={len(k_arr)})")
        print(f"  fraction of tiles with |k| > 1e-3: {(np.abs(k_arr) > 1e-3).mean():.3f}")
        print(f"  fraction of tiles with |k| > 1e-2: {(np.abs(k_arr) > 1e-2).mean():.3f}")
    print()
    print("How to read this: a high correlation for the hillshade [N.L] candidate confirms the")
    print("residual is the synthesizer's own shading model (the physically correct comparison). A")
    print("high per-tile scale fraction with a consistent per-tile k would suggest a per-tile scale")
    print("even though no global one exists. If none correlate, the residual is a learned (nonlinear)")
    print("shading signal, not a direct heightmap transform.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
