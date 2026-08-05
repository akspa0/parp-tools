"""Three-tier terrain signal classification (Spec 132 US1).

The WoW terrain was built with fractal brushes that simultaneously affected the heightmap
(MCVT/MCNK), the alpha layers (MCAL), and the texture tilesets. When textures were re-done the
alpha layers were replaced but the heightmap brush scars remained -- a broken relationship between
3D shape and 2D texture. This module classifies every tile into one of three signal tiers so that
the "normal" (missing middle) class stops falling through the gap between "usable" and "weak".

Three tiers (spec.md "Three signal classes"):
  - ``strong``  -- full-height terrain with intact brush-texture relationship.
  - ``normal``  -- visible relief but compressed or partially re-textured (the degraded middle).
  - ``weak``    -- near-flat, sub-metre relief, often abandoned work.

The criteria are published and deterministic (FR-001, FR-002, FR-006): the same inputs always
produce the same tier, and the evidence says *why*. ``alpha_texture_correlation`` is optional -- a
tile with no alpha data reports ``None`` and the height/levels criteria decide the tier on their own
(FR-007): we never fabricate a score for a tile that has no alpha data to measure.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

# --- Published criteria (spec.md US1 + plan Phase 1) -------------------------------------------
# A tile is "weak" when its whole-map height range is sub-metre: near-flat, often abandoned work.
WEAK_MAX_RANGE = 5.0
# A tile is "normal" (degraded middle) when its height range sits in the 5-50 unit band, OR its
# surviving height levels are compressed to 8-64, OR its alpha<->height brush correlation is low.
NORMAL_MAX_RANGE = 50.0
NORMAL_MIN_LEVELS = 8
NORMAL_MAX_LEVELS = 64
# An "intact" brush-texture relationship correlates heightmap scars with alpha patterns >= 0.3;
# below this the alpha layers were re-done without re-sculpting (broken relationship).
LOW_CORRELATION = 0.3
# Strong tiles exceed the normal band on both amplitude and shape.
STRONG_MIN_LEVELS = NORMAL_MAX_LEVELS + 1


class SignalTier(str, Enum):
    """The three signal tiers. ``NA`` marks a tile with no measurable height at all."""

    STRONG = "strong"
    NORMAL = "normal"
    WEAK = "weak"
    NA = "na"


@dataclass(frozen=True)
class TierResult:
    """One tile's three-tier classification plus the evidence string that produced it."""

    tier: SignalTier
    height_range: float
    surviving_levels: int
    alpha_texture_correlation: float | None
    evidence: str


def compute_signal_tier(
    *,
    height_range: float,
    surviving_levels: int,
    alpha_texture_correlation: float | None = None,
) -> TierResult:
    """Classify one tile into strong/normal/weak with published, deterministic criteria.

    Order of checks (first match wins, so the evidence names the deciding feature):

    1. No relief at all (range <= 0) -> ``NA``: there is no signal to tier.
    2. Sub-metre relief -> ``weak`` regardless of level count -- a squeezed tile is still a weak
       tile, even if its shape survives (amplitude is the tier's defining feature).
    3. The "normal" middle: either the amplitude is compressed (5-50 units), or the surviving
       shape is compressed (8-64 levels), or the brush<->alpha correlation is low (< 0.3).
    4. Otherwise -> ``strong``: full relief and full shape with an intact relationship.
    """
    height_range = float(height_range)
    surviving_levels = int(surviving_levels)
    if alpha_texture_correlation is not None:
        alpha_texture_correlation = float(alpha_texture_correlation)

    if height_range <= 0.0:
        return TierResult(
            tier=SignalTier.NA,
            height_range=height_range,
            surviving_levels=surviving_levels,
            alpha_texture_correlation=alpha_texture_correlation,
            evidence="no height relief (range<=0)",
        )

    if height_range < WEAK_MAX_RANGE:
        return TierResult(
            tier=SignalTier.WEAK,
            height_range=height_range,
            surviving_levels=surviving_levels,
            alpha_texture_correlation=alpha_texture_correlation,
            evidence=f"sub-metre relief (range={height_range:.3f}<{WEAK_MAX_RANGE})",
        )

    reasons: list[str] = []
    if height_range <= NORMAL_MAX_RANGE:
        reasons.append(f"compressed amplitude (range={height_range:.1f}<= {NORMAL_MAX_RANGE})")
    if NORMAL_MIN_LEVELS <= surviving_levels <= NORMAL_MAX_LEVELS:
        reasons.append(f"compressed shape ({surviving_levels} levels in 8..64)")
    if alpha_texture_correlation is not None and alpha_texture_correlation < LOW_CORRELATION:
        reasons.append(
            f"low brush<->alpha correlation ({alpha_texture_correlation:.2f}<{LOW_CORRELATION})"
        )

    if reasons:
        return TierResult(
            tier=SignalTier.NORMAL,
            height_range=height_range,
            surviving_levels=surviving_levels,
            alpha_texture_correlation=alpha_texture_correlation,
            evidence="; ".join(reasons),
        )

    return TierResult(
        tier=SignalTier.STRONG,
        height_range=height_range,
        surviving_levels=surviving_levels,
        alpha_texture_correlation=alpha_texture_correlation,
        evidence=(
            f"full relief (range={height_range:.1f}>{NORMAL_MAX_RANGE}) and "
            f"full shape ({surviving_levels} levels>{NORMAL_MAX_LEVELS})"
        ),
    )