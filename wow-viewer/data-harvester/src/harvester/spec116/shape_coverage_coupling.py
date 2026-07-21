"""Spec 116 US2: settle whether layer masks derive from terrain shape.

A practitioner determines whether layer masks are **derived from the terrain surface** (elevation
and slope) or authored independently of it. The working hypothesis (masks distilled from
higher-resolution source artwork with hand fix-ups) predicts a **bimodal** coupling across tiles
-- strong where automated, weak where hand-edited.

For each tile and each detail layer (1..3; the base layer 0 is always opaque and carries no
coverage to fit) we fit a non-linear mapping ``{elevation, slope} -> coverage`` with a
``GradientBoostingRegressor`` and report explained variance (R^2). We then test the
explained-variance distribution for bimodality two ways and report both:

- the SAS **bimodality coefficient** ``BC = (skew^2 + 1) / (excess_kurtosis + 3)`` (``>~0.555``
  suggests bimodality), via ``scipy.stats``;
- a 1-vs-2 component Gaussian-mixture **BIC** comparison via ``sklearn.mixture``.

The report states whether a distinct high-coupling population exists and its tile share, and --
because a prior linear analysis found weak coupling with no bimodality -- explicitly records that a
linear test was **underpowered for threshold relationships** rather than treating disagreement as
noise (spec US2 acceptance 3). No model is trained.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from harvester.spec116.structure_contract import sha256_file, validate_analysis_report

CHUNKS_PER_AXIS = 16
PIXELS_PER_CHUNK = 16
DETAIL_LAYERS = (1, 2, 3)  # base layer 0 is opaque; no coverage to fit
HIGH_COUPLING_VARIANCE = 0.50  # a tile/layer is "high-coupling" at/above this explained variance
HIGH_COUPLING_POPULATION_SHARE = 0.20  # a distinct high-coupling population exists at/above this share
BIMODALITY_COEFFICIENT_THRESHOLD = 5.0 / 9.0  # ~0.555, the standard SAS bimodality cutoff
ANALYSIS_REPORT_SCHEMA = "v116-analysis-report-v1"
TAXONOMY_REVISION = "v115.1"  # corpus labeling context (US2 does not consume texture names)


class ShapeCoverageCouplingError(ValueError):
    """Raised when the shape->coverage coupling measurement cannot be produced honestly."""


def _chunk_elevation_and_slope(height_257: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Downsample a (257, 257) height field to per-chunk (16, 16) elevation and slope.

    Elevation is the mean height over each 16x16 pixel block (the last vertex row/col is ignored
    so the grid aligns with the 16x16 chunk lattice). Slope is the gradient magnitude of that
    per-chunk elevation surface.
    """
    h = np.asarray(height_257, dtype=np.float32)
    if h.shape != (257, 257):
        raise ShapeCoverageCouplingError(f"height_257 must be (257, 257), got {h.shape}")
    block = h[: CHUNKS_PER_AXIS * PIXELS_PER_CHUNK, : CHUNKS_PER_AXIS * PIXELS_PER_CHUNK]
    # mean over each 16x16 block -> (16, 16)
    elevation = block.reshape(CHUNKS_PER_AXIS, PIXELS_PER_CHUNK, CHUNKS_PER_AXIS, PIXELS_PER_CHUNK).mean(axis=(1, 3))
    gy, gx = np.gradient(elevation)
    slope = np.sqrt(gy * gy + gx * gx)
    return elevation.astype(np.float32), slope.astype(np.float32)


def _explained_variance(features: np.ndarray, target: np.ndarray) -> float | None:
    """Non-linear R^2 of {elevation, slope} -> coverage via GradientBoostingRegressor.

    Returns None when the target has zero variance (a layer with no coverage anywhere has nothing
    to explain); such (tile, layer) pairs are skipped, not scored as zero.
    """
    from sklearn.ensemble import GradientBoostingRegressor

    y = np.asarray(target, dtype=np.float32).reshape(-1)
    if y.std() <= 1e-8:
        return None  # constant target -> explained variance undefined
    model = GradientBoostingRegressor(
        n_estimators=50, max_depth=3, learning_rate=0.1, random_state=0,
    )
    model.fit(features, y)
    return float(model.score(features, y))


def _bimodality_coefficient(values: np.ndarray) -> float:
    """SAS bimodality coefficient ``BC = (skew^2 + 1) / (excess_kurtosis + 3)``.

    ``>~0.555`` (5/9) suggests bimodality. Returns 0.0 for a degenerate (single-value) sample.
    """
    from scipy.stats import kurtosis, skew

    v = np.asarray(values, dtype=np.float64).reshape(-1)
    if v.size < 3 or v.std() <= 1e-12:
        return 0.0
    g1 = float(skew(v, bias=False))
    g2 = float(kurtosis(v, fisher=True, bias=False))  # excess kurtosis
    denom = g2 + 3.0
    if denom <= 1e-12:
        return 0.0
    return (g1 * g1 + 1.0) / denom


def _mixture_bic(values: np.ndarray) -> dict:
    """1-vs-2 component Gaussian-mixture BIC; ``two_preferred`` is True when 2 components win."""
    from sklearn.mixture import GaussianMixture

    v = np.asarray(values, dtype=np.float64).reshape(-1, 1)
    if v.size < 4 or v.std() <= 1e-12:
        return {"one_component": 0.0, "two_component": 0.0, "two_preferred": False}
    try:
        bic1 = float(GaussianMixture(n_components=1, random_state=0, n_init=1).fit(v).bic(v))
        bic2 = float(GaussianMixture(n_components=2, random_state=0, n_init=1).fit(v).bic(v))
    except Exception:  # noqa: BLE001 - EM can fail to converge on tiny/degenerate samples
        return {"one_component": 0.0, "two_component": 0.0, "two_preferred": False}
    return {"one_component": bic1, "two_component": bic2, "two_preferred": bool(bic2 < bic1)}


def measure_shape_coverage_coupling(
    *,
    store: Path,
    build_id: str = "",
) -> dict:
    """Measure shape->coverage coupling and return the ``v116-analysis-report-v1`` artifact."""
    import pyarrow.parquet as pq
    import zarr

    group = zarr.open_group(str(store), mode="r")
    for required in ("height_257", "mcly_layer_mask"):
        if required not in group:
            raise ShapeCoverageCouplingError(f"store is missing {required!r}: {store}")
    index_path = store / "index.parquet"
    if not index_path.exists():
        raise ShapeCoverageCouplingError(f"store has no index.parquet: {store}")
    index_rows = pq.read_table(index_path).to_pylist()

    height = group["height_257"]
    mask = group["mcly_layer_mask"]
    row_count = int(height.shape[0])
    if row_count != len(index_rows):
        raise ShapeCoverageCouplingError(
            f"index rows ({len(index_rows)}) != height_257 rows ({row_count})"
        )

    per_tile_layer: list[dict] = []
    explained_variances: list[float] = []
    skipped_zero_coverage = 0

    for row in range(row_count):
        elevation, slope = _chunk_elevation_and_slope(np.asarray(height[row]))
        features = np.stack([elevation.reshape(-1), slope.reshape(-1)], axis=1)
        tile_mask = np.asarray(mask[row], dtype=np.float32)  # (16, 16, 4)
        for layer in DETAIL_LAYERS:
            coverage = tile_mask[:, :, layer].reshape(-1)
            ev = _explained_variance(features, coverage)
            if ev is None:
                skipped_zero_coverage += 1
                continue
            per_tile_layer.append({"tile_row": row, "layer": layer, "explained_variance": ev})
            explained_variances.append(ev)

    if not explained_variances:
        raise ShapeCoverageCouplingError("no (tile, layer) produced a fittable coverage target")

    ev_array = np.asarray(explained_variances, dtype=np.float64)
    high_coupling_share = float(np.mean(ev_array >= HIGH_COUPLING_VARIANCE))
    bc = _bimodality_coefficient(ev_array)
    bic = _mixture_bic(ev_array)
    high_coupling_population_exists = high_coupling_share >= HIGH_COUPLING_POPULATION_SHARE
    decision_value = "coverage_derivable" if high_coupling_population_exists else "coverage_independent"

    linear_note = (
        "A prior linear analysis found weak shape->coverage coupling with no bimodality. The "
        "non-linear GradientBoosting fit can detect threshold/quadratic relationships a linear "
        "regression averages away, so a disagreement here means the linear test was underpowered "
        "for threshold relationships, not that the coupling is absent."
    )

    report = {
        "schema": ANALYSIS_REPORT_SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "report_kind": "shape_coverage_coupling",
        "identity": {
            "store": {"path": str(store.resolve()), "sha256": sha256_file(index_path)},
            "taxonomy_revision": TAXONOMY_REVISION,
        },
        "shape_coverage_coupling": {
            "per_tile_layer_explained_variance": per_tile_layer,
            "bimodality_coefficient": bc,
            "mixture_bic": bic,
            "high_coupling_tile_share": high_coupling_share,
            "linear_underpowered_note": linear_note,
        },
        "decision": {"kind": "derivability", "value": decision_value},
        # Provenance:
        "row_count": row_count,
        "fittable_tile_layer_count": len(explained_variances),
        "skipped_zero_coverage": skipped_zero_coverage,
        "high_coupling_variance_threshold": HIGH_COUPLING_VARIANCE,
        "bimodality_coefficient_threshold": BIMODALITY_COEFFICIENT_THRESHOLD,
        "build_id": build_id,
    }
    validate_analysis_report(report)
    return report


def decision_from_report(report: dict) -> str:
    """Read the durable derivability decision out of a US2 report (consumed by US3)."""
    return str(report["decision"]["value"])


__all__ = [
    "ShapeCoverageCouplingError",
    "measure_shape_coverage_coupling",
    "decision_from_report",
    "HIGH_COUPLING_VARIANCE",
    "BIMODALITY_COEFFICIENT_THRESHOLD",
    "ANALYSIS_REPORT_SCHEMA",
]
