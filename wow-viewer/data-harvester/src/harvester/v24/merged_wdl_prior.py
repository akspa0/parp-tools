"""FR-005: merged WDL prior coverage (Stage 0 merge rules).

Pure NumPy: the caller supplies the real WDL grids (from the C# reader via
``wdl_reader``) and the synthetic grids (from the C# terrain->WDL path via
``synth_wdl``); this module only merges them per the spec's rules.

Sources: 0 = real (agrees with synthetic within the threshold),
1 = synthetic, 2 = learned-fill (audit-empty tile).
Confidence: 1.0 real-agreeing, 0.7 synthetic-only, 0.4 synthetic-disagreeing,
0.0 learned-fill. The threshold comparison is inclusive (<=) because the real
client WDL is int16-quantized and sits up to exactly 1.0 from the float
synthetic value (see docs/architecture/wdl-reader-shape-audit-2026-07-06.md).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

SOURCE_REAL = 0
SOURCE_SYNTHETIC = 1
SOURCE_LEARNED_FILL = 2

CONFIDENCE_REAL = 1.0
CONFIDENCE_SYNTHETIC = 0.7
CONFIDENCE_DISAGREE = 0.4
CONFIDENCE_LEARNED_FILL = 0.0


@dataclass(frozen=True)
class MergedPrior:
    outer: np.ndarray
    inner: np.ndarray
    source_outer: np.ndarray
    source_inner: np.ndarray
    confidence_outer: np.ndarray
    confidence_inner: np.ndarray
    disagree_ratio: float

    @property
    def disagrees_with_real(self) -> bool:
        return self.disagree_ratio > 0.0


def _merge_grid(
    synth: np.ndarray,
    real: np.ndarray | None,
    disagree_threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    synth = np.asarray(synth, dtype=np.float32)
    prior = synth.copy()
    source = np.full(synth.shape, SOURCE_SYNTHETIC, dtype=np.uint8)
    confidence = np.full(synth.shape, CONFIDENCE_SYNTHETIC, dtype=np.float32)
    disagree_count = 0

    if real is not None:
        real = np.asarray(real, dtype=np.float32)
        if real.shape != synth.shape:
            raise ValueError(f"real grid {real.shape} does not match synthetic {synth.shape}")
        agrees = np.abs(real - synth) <= disagree_threshold
        prior[agrees] = real[agrees]
        source[agrees] = SOURCE_REAL
        confidence[agrees] = CONFIDENCE_REAL
        confidence[~agrees] = CONFIDENCE_DISAGREE
        disagree_count = int((~agrees).sum())

    return prior, source, confidence, disagree_count


def build_merged_wdl_prior(
    height_257: np.ndarray,
    real_wdl: tuple[np.ndarray, np.ndarray] | None,
    real_wdl_available: bool,
    synth_wdl: tuple[np.ndarray, np.ndarray],
    audit_empty: bool = False,
    disagree_threshold: float = 1.0,
) -> MergedPrior:
    """Merge real + synthetic WDL grids into the per-tile prior.

    ``real_wdl`` / ``synth_wdl`` are ``(outer (17,17), inner (16,16))`` pairs.
    ``audit_empty`` marks tiles whose V18 ``height_257`` is audit-empty; they
    get a flat per-tile-mean prior with learned-fill source and 0 confidence.
    """
    heights = np.asarray(height_257, dtype=np.float32)

    if audit_empty:
        mean = float(heights.mean()) if heights.size else 0.0
        outer = np.full((17, 17), mean, dtype=np.float32)
        inner = np.full((16, 16), mean, dtype=np.float32)
        source_outer = np.full((17, 17), SOURCE_LEARNED_FILL, dtype=np.uint8)
        source_inner = np.full((16, 16), SOURCE_LEARNED_FILL, dtype=np.uint8)
        conf_outer = np.zeros((17, 17), dtype=np.float32)
        conf_inner = np.zeros((16, 16), dtype=np.float32)
        return MergedPrior(outer, inner, source_outer, source_inner, conf_outer, conf_inner, 0.0)

    synth_outer, synth_inner = synth_wdl
    real_outer = real_inner = None
    if real_wdl_available and real_wdl is not None:
        real_outer, real_inner = real_wdl

    outer, source_outer, conf_outer, disagree_outer = _merge_grid(
        synth_outer, real_outer, disagree_threshold
    )
    inner, source_inner, conf_inner, disagree_inner = _merge_grid(
        synth_inner, real_inner, disagree_threshold
    )

    total_cells = outer.size + inner.size
    disagree_ratio = (disagree_outer + disagree_inner) / float(total_cells)
    return MergedPrior(
        outer, inner, source_outer, source_inner, conf_outer, conf_inner, disagree_ratio
    )
