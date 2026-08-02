# Contract: Lighting Baseline Survey

**Library**: `WowViewer.Core.IO.Maps` | **File**: `MinimapLightingBaseline.cs`

## Purpose

Test the global lighting normalisation hypothesis — whether authored tiles of a map share a common
lighting baseline (FR-016).

## API

```csharp
namespace WowViewer.Core.IO.Maps;

/// <summary>Result of a per-map lighting-baseline survey.</summary>
public sealed record LightingBaselineResult(
    string Map,
    string Build,
    float MeanLuma,
    float StdLuma,
    bool BaselinePresent,
    bool BuildRecognised);

public static class MinimapLightingBaseline
{
    /// <summary>
    /// Survey a set of authored tiles for a shared lighting baseline. A baseline is present when
    /// cross-tile luma variance is small relative to within-tile variance.
    /// </summary>
    public static LightingBaselineResult Survey(
        string map,
        string build,
        IEnumerable<byte[,,]> authoredTiles,
        bool buildRecognised);

    /// <summary>
    /// Normalise a synthetic tile to the authored baseline (mean/std match) before scoring.
    /// </summary>
    public static Image<Rgba32> NormalizeToBaseline(
        Image<Rgba32> synthetic,
        LightingBaselineResult baseline);
}
```

## Behaviour

- `Survey` computes cross-tile mean/std luma. `BaselinePresent` is true when cross-tile variance is
  small relative to within-tile variance (threshold configurable, conservative default).
- `NormalizeToBaseline` rescales the synthetic tile's luma to match the authored baseline's
  mean/std, so comparison attributes only residual (non-baseline) differences.
- `BuildRecognised` reflects era-gating (FR-006); an unrecognised build is flagged, never silently
  defaulted.

## Validation

- `MinimapLightingBaselineTests`: a synthetic set with a known shared baseline is detected; a set
  with independent per-tile exposures is not; normalisation brings a shifted synthetic tile's
  mean/std to the baseline.
