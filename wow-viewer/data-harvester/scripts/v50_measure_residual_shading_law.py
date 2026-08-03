"""Measure the actual relationship between the textureless residual and the heightmap (Spec 125).

WHY THIS REPLACES THE GRADIENT HILLSHADE IN ``v50_validate_residual_scale.py``
-----------------------------------------------------------------------------
That script's hillshade candidate derives normals with ``np.gradient(height)`` in *pixel* units and
dots them against a hardcoded light. Two things make it structurally unable to confirm the
hypothesis even if the hypothesis is true:

1. **Pixel-space gradients.** An ADT tile spans 533.333 world units across 256 pixels, so
   ``dh/dpixel`` is ~2.083x ``dh/dworld``. The normal is a *nonlinear* function of the gradient, so
   that factor does not cancel in a Pearson correlation — it models terrain twice as steep as
   reality and flattens N.L toward its saturated ends.
2. **A guessed sun.** The light is pinned at ``(-0.5, -0.5, 0.72)`` -> ~45 deg elevation, but the
   traced client sun is low (20-37 deg). A wrong sun vector alone can drive the correlation to noise.

Both are avoidable: the v50 store already carries ``normal_xyz`` (257x257x3, MCNR normals streamed
from the client), so we do not have to derive normals or guess the world scale at all. And rather
than assume a sun, we SWEEP azimuth/elevation and report the best-fitting one — which both measures
the relationship and independently recovers the sun the compositor used.

THE MODEL BEING TESTED
----------------------
The residual is the compositor's output with a neutral-white albedo, i.e. Lambert shading plus cast
shadows and ambient. So the law under test is affine in N.L:

    residual ~= ambient + gain * clamp(dot(N, L), 0, 1)          (cast-shadowed pixels excluded)

We report Pearson r and the affine fit's R^2 for the best (azimuth, elevation). Because cast shadows
are a separate multiplicative visibility term we cannot compute here, we also report the fit on the
brightest-N% of pixels, which are the ones least likely to be in cast shadow.

``--self-test`` renders a synthetic residual from known normals and a known sun, then checks that
the sweep recovers that sun and reports r ~ 1. Run it first: a detector that cannot find a planted
signal cannot be trusted to report its absence.

Usage (USER runs):
  uv run python scripts/v50_measure_residual_shading_law.py --self-test
  uv run python scripts/v50_measure_residual_shading_law.py \
      --residual-dir <residual-output>/tiles \
      --store <v50-store> \
      --map Azeroth
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import zarr

_SRC = Path(__file__).resolve().parent.parent / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

RESIDUAL_RE = re.compile(r"^(?P<map>.+?)_(?P<tx>\d{2})_(?P<ty>\d{2})_residual\.png$")

# An ADT tile spans 533.33333 world units. Only needed for the gradient-normal comparison arm.
TILE_WORLD_SIZE = 533.33333


def _load_residual(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as img:
        return np.asarray(img.convert("L"), dtype=np.float32) / 255.0


def _sun_vector(azimuth_deg: float, elevation_deg: float) -> np.ndarray:
    """Unit vector toward the sun. Azimuth is degrees CCW from +X in the XY plane."""
    az = math.radians(azimuth_deg)
    el = math.radians(elevation_deg)
    horizontal = math.cos(el)
    return np.array(
        [horizontal * math.cos(az), horizontal * math.sin(az), math.sin(el)], dtype=np.float64
    )


def _lambert(normals: np.ndarray, sun: np.ndarray) -> np.ndarray:
    """clamp(dot(N, L), 0, 1) over an (..., 3) array of unit normals."""
    return np.clip(normals @ sun, 0.0, 1.0)


def _normals_from_height(height: np.ndarray) -> np.ndarray:
    """Derive unit normals from a height field using WORLD-unit gradients (the arm the old script
    got wrong). Included so the two normal sources can be compared side by side."""
    h = np.asarray(height, dtype=np.float64)
    spacing = TILE_WORLD_SIZE / (h.shape[0] - 1)
    gy, gx = np.gradient(h, spacing, spacing)
    normals = np.stack([-gx, -gy, np.ones_like(gx)], axis=-1)
    return normals / np.linalg.norm(normals, axis=-1, keepdims=True)


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    a = a.ravel().astype(np.float64)
    b = b.ravel().astype(np.float64)
    a = a - a.mean()
    b = b - b.mean()
    denominator = float(np.sqrt((a * a).sum()) * np.sqrt((b * b).sum()))
    return float((a * b).sum() / denominator) if denominator > 1e-12 else 0.0


def _affine_r2(shade: np.ndarray, residual: np.ndarray) -> tuple[float, float, float]:
    """Least-squares fit residual ~= gain * shade + ambient. Returns (gain, ambient, r2)."""
    x = shade.ravel().astype(np.float64)
    y = residual.ravel().astype(np.float64)
    design = np.stack([x, np.ones_like(x)], axis=1)
    (gain, ambient), *_ = np.linalg.lstsq(design, y, rcond=None)
    predicted = design @ np.array([gain, ambient])
    total = float(((y - y.mean()) ** 2).sum())
    r2 = 1.0 - float(((y - predicted) ** 2).sum()) / total if total > 1e-12 else 0.0
    return float(gain), float(ambient), float(r2)


def _sweep(normals: np.ndarray, residual: np.ndarray, *, coarse: int = 12) -> dict:
    """Find the (azimuth, elevation) whose Lambert shading best explains the residual."""
    best = {"r": -2.0}
    for azimuth in np.arange(0.0, 360.0, coarse):
        for elevation in np.arange(5.0, 90.0, coarse / 2.0):
            shade = _lambert(normals, _sun_vector(azimuth, elevation))
            r = _pearson(shade, residual)
            if r > best["r"]:
                best = {"r": r, "azimuth": float(azimuth), "elevation": float(elevation)}
    # Refine around the coarse winner.
    for azimuth in np.arange(best["azimuth"] - coarse, best["azimuth"] + coarse, 1.0):
        for elevation in np.arange(max(1.0, best["elevation"] - coarse), best["elevation"] + coarse, 1.0):
            if not 0.0 < elevation < 90.0:
                continue
            shade = _lambert(normals, _sun_vector(azimuth, elevation))
            r = _pearson(shade, residual)
            if r > best["r"]:
                best = {"r": r, "azimuth": float(azimuth % 360.0), "elevation": float(elevation)}
    shade = _lambert(normals, _sun_vector(best["azimuth"], best["elevation"]))
    gain, ambient, r2 = _affine_r2(shade, residual)
    best.update({"gain": gain, "ambient": ambient, "r2": r2})
    # Cast shadows only ever darken, so refit on the brightest 60% — the pixels least likely to be
    # occluded. A large jump here means cast shadows, not the shading law, are the residual term.
    lit = residual >= np.quantile(residual, 0.40)
    gain_lit, ambient_lit, r2_lit = _affine_r2(shade[lit], residual[lit])
    best.update({"r_lit": _pearson(shade[lit], residual[lit]), "r2_lit": r2_lit,
                 "gain_lit": gain_lit, "ambient_lit": ambient_lit})
    return best


def _self_test() -> int:
    """Plant a known sun in a synthetic residual and confirm the sweep recovers it."""
    rng = np.random.default_rng(125)
    # A smooth random terrain, so normals vary over a realistic range.
    coarse = rng.random((17, 17)) * 120.0
    yy, xx = np.mgrid[0:257, 0:257] / 256.0 * 16.0
    height = np.zeros((257, 257), dtype=np.float64)
    for i in range(16):
        for j in range(16):
            block = (np.clip(1 - np.abs(xx - j), 0, 1) * np.clip(1 - np.abs(yy - i), 0, 1))
            height += block * coarse[i, j]
    normals = _normals_from_height(height)[:256, :256]

    truth_azimuth, truth_elevation = 225.0, 30.0
    planted = 0.25 + 0.7 * _lambert(normals, _sun_vector(truth_azimuth, truth_elevation))
    # Quantize to 8-bit like a real residual PNG.
    planted = np.round(np.clip(planted, 0, 1) * 255.0) / 255.0

    found = _sweep(normals, planted)
    azimuth_error = abs((found["azimuth"] - truth_azimuth + 180.0) % 360.0 - 180.0)
    elevation_error = abs(found["elevation"] - truth_elevation)
    print(json.dumps({"planted": {"azimuth": truth_azimuth, "elevation": truth_elevation},
                      "recovered": found,
                      "azimuth_error_deg": azimuth_error,
                      "elevation_error_deg": elevation_error}, indent=2))
    ok = found["r"] > 0.99 and azimuth_error <= 3.0 and elevation_error <= 5.0
    print("\nSELF-TEST " + ("PASS: the sweep recovers a planted sun; a null result from it is meaningful."
                            if ok else "FAIL: do NOT trust a null result from this detector."), flush=True)
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description="Measure the residual's shading law against ground-truth normals")
    ap.add_argument("--self-test", action="store_true", help="verify detector power on a planted signal")
    ap.add_argument("--residual-dir", type=Path)
    ap.add_argument("--store", type=Path, help="v50 store with normal_xyz and height_257")
    ap.add_argument("--map")
    ap.add_argument("--limit", type=int, default=64, help="tiles to pool (sweep cost scales with this)")
    ap.add_argument("--output", type=Path, help="optional JSON report path")
    args = ap.parse_args()

    if args.self_test:
        return _self_test()
    if not (args.residual_dir and args.store and args.map):
        raise SystemExit("--residual-dir, --store and --map are required (or pass --self-test)")

    group = zarr.open_group(str(args.store), mode="r")
    for name in ("normal_xyz", "height_257"):
        if name not in group:
            raise SystemExit(f"store has no {name} array: {args.store}")
    lookup: dict[tuple[str, int, int], int] = {}
    for i, row in enumerate(pq.read_table(args.store / "index.parquet").to_pylist()):
        lookup.setdefault((str(row.get("map", "")), int(row.get("tile_x", -1)), int(row.get("tile_y", -1))), i)

    residual_parts: list[np.ndarray] = []
    mcnr_parts: list[np.ndarray] = []
    gradient_parts: list[np.ndarray] = []
    for path in sorted(args.residual_dir.glob("*_residual.png")):
        if len(residual_parts) >= args.limit:
            break
        m = RESIDUAL_RE.match(path.name)
        if not m:
            continue
        row = lookup.get((args.map, int(m.group("tx")), int(m.group("ty"))))
        if row is None:
            continue
        residual = _load_residual(path)
        normals = np.asarray(group["normal_xyz"][row], dtype=np.float64)[:256, :256]
        norms = np.linalg.norm(normals, axis=-1, keepdims=True)
        if not np.all(norms > 1e-6):
            continue
        residual_parts.append(residual.ravel())
        mcnr_parts.append((normals / norms).reshape(-1, 3))
        gradient_parts.append(
            _normals_from_height(np.asarray(group["height_257"][row], dtype=np.float64))[:256, :256].reshape(-1, 3)
        )

    if len(residual_parts) < 8:
        raise SystemExit(f"only {len(residual_parts)} tiles matched; need >= 8")

    residual = np.concatenate(residual_parts)
    report = {
        "tiles": len(residual_parts),
        "pixels": int(residual.size),
        "mcnr_normals": _sweep(np.concatenate(mcnr_parts), residual),
        "gradient_normals_world_units": _sweep(np.concatenate(gradient_parts), residual),
    }
    print(json.dumps(report, indent=2), flush=True)
    print(
        "\nHow to read this: 'mcnr_normals.r' is the real answer — the correlation between the\n"
        "residual and Lambert shading of the client's own normals under the best-fit sun. A high r\n"
        "with a low-elevation best-fit sun confirms the residual is a shading field over the terrain\n"
        "normals, i.e. it encodes the height GRADIENT, not height itself. Compare 'r' to 'r_lit':\n"
        "a large gap means cast-shadow occlusion is the dominant unexplained term.",
        flush=True,
    )
    if args.output:
        args.output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
