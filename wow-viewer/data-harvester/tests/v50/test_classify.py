"""Unit tests for the three-tier terrain signal classifier (Spec 132 US1).

Covers the published criteria in spec.md US1 / plan.md Phase 1: a tile is weak when its range is
sub-metre, normal when its amplitude or shape is compressed (or its brush<->alpha correlation is
low), and strong otherwise. Also asserts determinism (FR-006) and graceful no-alpha handling
(FR-007).
"""

from __future__ import annotations

import pytest

from harvester.v50.classify import (
    LOW_CORRELATION,
    SignalTier,
    compute_signal_tier,
)


def test_acceptance_strong_tile():
    """Given a tile with full height range + intact alpha/texture -> strong."""
    result = compute_signal_tier(
        height_range=120.0, surviving_levels=2000, alpha_texture_correlation=0.9
    )
    assert result.tier == SignalTier.STRONG


def test_acceptance_normal_retextured_tile():
    """Given a tile with visible relief but re-textured alpha (low correlation) -> normal."""
    result = compute_signal_tier(
        height_range=120.0, surviving_levels=2000, alpha_texture_correlation=0.1
    )
    assert result.tier == SignalTier.NORMAL
    assert "correlation" in result.evidence


def test_acceptance_weak_tile():
    """Given a tile with sub-metre height range -> weak."""
    result = compute_signal_tier(height_range=1.0, surviving_levels=64)
    assert result.tier == SignalTier.WEAK


def test_normal_by_compressed_amplitude():
    """A 5-50 unit range is the degraded middle, not full terrain."""
    result = compute_signal_tier(height_range=30.0, surviving_levels=2000)
    assert result.tier == SignalTier.NORMAL
    assert "amplitude" in result.evidence


def test_normal_by_compressed_shape():
    """8-64 surviving height levels marks compressed shape even at full amplitude."""
    result = compute_signal_tier(height_range=200.0, surviving_levels=32)
    assert result.tier == SignalTier.NORMAL
    assert "shape" in result.evidence


def test_normal_boundary_low_correlation():
    """Correlation just under 0.3 flips a strong tile to normal; at/above it stays strong."""
    below = compute_signal_tier(
        height_range=200.0, surviving_levels=1000, alpha_texture_correlation=LOW_CORRELATION - 0.01
    )
    at = compute_signal_tier(
        height_range=200.0, surviving_levels=1000, alpha_texture_correlation=LOW_CORRELATION
    )
    assert below.tier == SignalTier.NORMAL
    assert at.tier == SignalTier.STRONG


def test_no_alpha_data_does_not_fabricate():
    """A tile with no alpha data (correlation=None) still tiers on height/levels, never a score."""
    result = compute_signal_tier(height_range=200.0, surviving_levels=1000)
    assert result.tier == SignalTier.STRONG
    assert result.alpha_texture_correlation is None


def test_no_relief_is_na():
    """A tile with zero height range has no signal to tier -> na."""
    result = compute_signal_tier(height_range=0.0, surviving_levels=1)
    assert result.tier == SignalTier.NA


@pytest.mark.parametrize(
    "kwargs",
    [
        {"height_range": 120.0, "surviving_levels": 2000},
        {"height_range": 30.0, "surviving_levels": 2000},
        {"height_range": 1.0, "surviving_levels": 64},
        {"height_range": 0.0, "surviving_levels": 1},
    ],
)
def test_deterministic(kwargs):
    """Same input -> same tier and evidence every time (FR-006)."""
    a = compute_signal_tier(**kwargs)
    b = compute_signal_tier(**kwargs)
    assert (a.tier, a.evidence) == (b.tier, b.evidence)


def test_weak_boundary_range():
    """Weak is strictly sub-<WEAK_MAX_RANGE>; the boundary itself is normal."""
    from harvester.v50.classify import NORMAL_MIN_LEVELS, NORMAL_MAX_LEVELS

    weak = compute_signal_tier(height_range=4.99, surviving_levels=NORMAL_MAX_LEVELS)
    assert weak.tier == SignalTier.WEAK
    boundary = compute_signal_tier(height_range=5.0, surviving_levels=NORMAL_MIN_LEVELS)
    assert boundary.tier == SignalTier.NORMAL