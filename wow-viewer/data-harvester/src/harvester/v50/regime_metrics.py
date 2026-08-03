"""Per-regime baselines and a selection metric that rewards learning where it happens.

WHY THIS EXISTS
---------------
Pooled validation MAE is the wrong thing to early-stop and checkpoint on when the validation set
spans terrain regimes whose targets have different natural scale. Measured on the Azeroth extractor
curriculum:

    rolling  own baseline MAE 0.0937   target std 0.1219
    steep    own baseline MAE 0.1243   target std 0.1475
    POOLED   baseline MAE 0.1114

A model reaching 0.0736 on rolling and 0.1087 on steep is improving 21.5% and 12.6% against the
baselines that actually apply to it. Reported as pooled MAE it is "0.0938 against 0.1114" -- 15.7%,
which credits neither regime properly. Worse, steep's achieved MAE (0.1114 at one point) coincidentally
equalled the POOLED baseline, making a regime that was improving 10% look completely dead.

Two consequences, both fixed here:

1. A regime's result is only interpretable against ITS OWN baseline. Reporting one pooled baseline
   next to per-regime errors invites exactly the wrong reading.
2. Pooled MAE is dominated by whichever regime is numerically largest and most numerous, so gains in
   an easier regime barely move it. Early stopping and best-checkpoint selection then under-reward
   real learning.

``regime_improvement`` selects on the MEAN RELATIVE IMPROVEMENT across regimes, which is scale-free
and gives each regime equal say regardless of tile count or target magnitude.
"""

from __future__ import annotations


def per_regime_baselines(
    rows: list[int],
    regime_of,
    baseline_of,
) -> dict[str, float]:
    """Mean trivial-baseline error per regime.

    ``baseline_of(row)`` returns that row's own trivial-predictor error (tile-mean prediction).
    """
    grouped: dict[str, list[float]] = {}
    for row in rows:
        grouped.setdefault(regime_of(row), []).append(float(baseline_of(row)))
    return {k: sum(v) / len(v) for k, v in sorted(grouped.items()) if v}


def regime_improvements(
    regime_mae: dict[str, float],
    regime_baseline: dict[str, float],
) -> dict[str, float]:
    """Relative improvement per regime: ``(baseline - mae) / baseline``. Positive is better."""
    out: dict[str, float] = {}
    for regime, mae in regime_mae.items():
        base = regime_baseline.get(regime)
        if base and base > 1e-12:
            out[regime] = (base - mae) / base
    return dict(sorted(out.items()))


def mean_regime_improvement(
    regime_mae: dict[str, float],
    regime_baseline: dict[str, float],
) -> float:
    """Unweighted mean of per-regime relative improvement -- the selection metric.

    Unweighted on purpose: a regime with fewer validation tiles or a smaller-magnitude target still
    gets equal say, so the model cannot win by serving only the numerically dominant regime.
    """
    improvements = regime_improvements(regime_mae, regime_baseline)
    if not improvements:
        return float("-inf")
    return sum(improvements.values()) / len(improvements)


def format_regime_line(
    regime_mae: dict[str, float],
    regime_baseline: dict[str, float],
) -> str:
    """``steep=0.108700 (base 0.124325, -12.6%)`` -- readable at a glance during training."""
    parts = []
    for regime, mae in sorted(regime_mae.items()):
        base = regime_baseline.get(regime)
        if base and base > 1e-12:
            parts.append(f"{regime}={mae:.6f} (base {base:.6f}, {100 * (mae - base) / base:+.1f}%)")
        else:
            parts.append(f"{regime}={mae:.6f}")
    return " ".join(parts)


def dead_regimes(
    regime_mae: dict[str, float],
    regime_baseline: dict[str, float],
    *,
    threshold: float = 0.01,
) -> list[str]:
    """Regimes not beating their own baseline by at least ``threshold`` relative.

    Constitution IV: a run is not a success while any signal sits at its baseline. This is what makes
    that checkable per regime rather than per output head only.
    """
    return sorted(
        regime for regime, imp in regime_improvements(regime_mae, regime_baseline).items()
        if imp < threshold
    )
