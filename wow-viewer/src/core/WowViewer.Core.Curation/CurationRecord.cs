namespace WowViewer.Core.Curation;

/// <summary>Per-tile terrain-difficulty bucket (relief/coverage-based). Ports the four-bucket
/// vocabulary from <c>v16_curation.DIFFICULTY_BUCKETS</c>.</summary>
public static class DifficultyBucket
{
    public const string Easy = "easy";
    public const string Medium = "medium";
    public const string Hard = "hard";
    public const string Pathological = "pathological";
}

/// <summary>Per-tile painted-signal coverage bucket. Ports <c>is_blank_what_plate</c> /
/// <c>mcly_painted_coverage</c> from <c>v16_curation.py</c>.</summary>
public static class CoverageBucket
{
    public const string WellCovered = "well_covered";
    public const string LowCoverage = "low_coverage";
    public const string Blank = "blank";
}

/// <summary>Per-tile lighting/time-of-day match status. Ports the exact status vocabulary
/// <see cref="WowViewer.Core.IO.Maps.MinimapShadingMatch"/> already produces (spec111), so no
/// downstream string comparison silently breaks.</summary>
public static class LightingBucket
{
    public const string Matched = "matched";
    public const string LowConfidenceAmbiguous = "low_confidence_ambiguous";
    public const string LowConfidenceFlatTerrain = "low_confidence_flat_terrain";
    public const string NotEvaluated = "not_evaluated";
}

/// <summary>Whether the tile has both a synthesized and authored minimap to compare (User Story 3).</summary>
public static class SyntheticFidelityStatus
{
    public const string Evaluated = "evaluated";
    public const string NotEvaluable = "not_evaluable";
}

/// <summary>Mismatch/defect category a finding belongs to. Ports the reason vocabulary from
/// <c>mismatch_detector.py</c> plus the new synthetic-fidelity category (US3).</summary>
public static class MismatchCategory
{
    public const string HeightNormalMismatch = "height_normal_mismatch";
    public const string NonFiniteValue = "non_finite_value";
    public const string HasFlagMismatch = "has_flag_mismatch";
    public const string SyntheticFidelityGap = "synthetic_fidelity_gap";
}

/// <summary>Finding severity. Ports <c>mismatch_detector._severity</c>'s exact four-level vocabulary.</summary>
public static class MismatchSeverity
{
    public const string None = "none";
    public const string Low = "low";
    public const string Medium = "medium";
    public const string High = "high";
    /// <summary>Sentinel used only when <see cref="MismatchFinding.Evaluability"/> is
    /// <see cref="Evaluability.NotEvaluable"/>, so a consumer filtering on severity alone can never
    /// mistake "not checked" for "checked, no problem" (data-model.md Mismatch Finding validation
    /// rules).</summary>
    public const string NotEvaluable = "not_evaluable";
}

/// <summary>Whether a check actually ran for a given tile/finding.</summary>
public static class Evaluability
{
    public const string Evaluated = "evaluated";
    public const string NotEvaluable = "not_evaluable";
}

/// <summary>
/// The per-tile classification result (data-model.md "Tile Curation Record"). One instance per
/// tile that exists in the source store's <c>index.parquet</c> -- full coverage is mandatory
/// (spec FR-008/SC-006): a tile with zero findings and a clean bucket is still a record, never an
/// absence.
/// </summary>
public sealed record TileCurationRecord(
    string Build,
    string Map,
    int TileX,
    int TileY,
    long TileId,
    string DifficultyBucket,
    string CoverageBucket,
    string LightingBucket,
    string SyntheticFidelityStatus,
    float? SyntheticFidelityScore,
    int FindingCount,
    string CurationRunId);

/// <summary>
/// A specific detected problem on a tile (data-model.md "Mismatch Finding"). One instance per
/// (tile, finding) -- a tile with zero findings produces zero instances, a tile with three findings
/// produces three (spec FR-010: findings are never collapsed into a single label).
/// </summary>
public sealed record MismatchFinding(
    string Build,
    string Map,
    int TileX,
    int TileY,
    long TileId,
    string Category,
    string Severity,
    string Reason,
    string Evaluability,
    string? Signal,
    string CurationRunId);
