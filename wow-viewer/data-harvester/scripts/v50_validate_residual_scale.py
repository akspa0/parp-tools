"""Test the Spec 125 scale hypothesis: is the textureless residual the heightmap scaled by ~33.334x?

The user's hypothesis (2026-08-02): the residual may be the heightmap scaled down by ~33.334x — the
same weak-signal band around +/- 2.778 Z that the weak-signal amplifier was built to restore. The
residuals look, visually and in the game data, like the weak signals.

This script compares each harvested residual PNG against the ground-truth heightmap (height_257 from
an existing v50 store) under several candidate transforms and reports which best explains the
residual:

  1. residual ~= height / 33.334            (the user's scale hypothesis)
  2. residual ~= height / 33.334, clipped to +/- 2.778 Z  (the weak-signal band)
  3. residual ~= a * height + b              (best-fit linear, to see how close 1/33.334 is)
  4. residual ~= normalized relative height  (the Spec 112 target, as a baseline)

For each candidate it reports Pearson correlation and a scale-fit error, so the hypothesis is
confirmed or refuted on real data rather than by eye.

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

    print(f"{'candidate':<46} {'corr':>8} {'scale':>10}")
    print("-" * 70)
    print(f"{'residual ~= height / 33.334':<46} {_pearson(R, cand_scale):8.4f} {'1/33.334':>10}")
    print(f"{'residual ~= height/33.334 clipped +/-2.778Z':<46} {_pearson(R, cand_band):8.4f} {'band':>10}")
    print(f"{'residual ~= k*height (best-fit)':<46} {_pearson(R, cand_fit):8.4f} {k_fit:10.4f}")
    print(f"{'residual ~= normalized relative height':<46} {_pearson(R, cand_rel):8.4f} {'rel':>10}")

    print()
    print(f"Best-fit scale k = {k_fit:.4f}  (hypothesis predicts 1/33.334 = {1/SCALE_33_334:.6f})")
    print(f"Ratio k / (1/33.334) = {k_fit / (1/SCALE_33_334):.3f}x")
    print()
    print("How to read this: a high correlation for the 1/33.334 candidate (and a best-fit k near")
    print("1/33.334) confirms the residual is a near-linear scaled heightmap. If the clipped-band")
    print("candidate wins, the residual is the weak-signal band specifically. If none correlate,")
    print("the residual is a learned (nonlinear) shading signal, not a direct heightmap transform.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
