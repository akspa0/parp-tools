# Contract: Encoding Survey

**Library**: `WowViewer.Core.IO.Maps` | **File**: `MinimapEncodingSurvey.cs`

## Purpose

Report, per build and map, the distribution of encodings found, so the DXT1 assumption is verified
rather than inherited (FR-013).

## API

```csharp
namespace WowViewer.Core.IO.Maps;

/// <summary>Per-build/map distribution of tile encodings.</summary>
public sealed record EncodingSurveyResult(
    string Build,
    string Map,
    IReadOnlyDictionary<string, int> EncodingDistribution, // encoding label -> tile count
    bool BuildRecognised);

public static class MinimapEncodingSurvey
{
    /// <summary>
    /// Inspect authored tiles and report the distribution of encodings (format, compression mode,
    /// colour depth, dimensions, mip count) per file (FR-001, FR-013).
    /// </summary>
    public static EncodingSurveyResult Survey(
        string build,
        string map,
        IEnumerable<byte[]> authoredBlpFiles,
        bool buildRecognised);
}
```

## Behaviour

- Detects each tile's actual encoding from the file, never assuming a build-wide default (FR-001).
- Aggregates into a per-build/map distribution (FR-013).
- `BuildRecognised` reflects era-gating (FR-006).
- Covers at least three maps across at least two build eras for the survey success criterion (SC-007).

## Validation

- Survey over a known set reports the correct distribution; an unrecognised build is flagged.
