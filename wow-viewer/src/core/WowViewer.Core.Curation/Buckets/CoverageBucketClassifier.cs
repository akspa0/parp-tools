using WowViewer.Core.Maps;

namespace WowViewer.Core.Curation.Buckets;

/// <summary>
/// Ports <c>v16_curation.mcly_painted_coverage</c> plus the well/low/blank coverage vocabulary
/// implied by <c>is_blank_what_plate</c> (blank) and the painted_signal term of
/// <c>_score_row_v16_1_1</c> (well-covered vs low-coverage boundary).
/// </summary>
public static class CoverageBucketClassifier
{
    /// <summary>Below this painted-signal coverage (max of alpha/mcly coverage), a non-blank tile
    /// is "low_coverage" rather than "well_covered" -- matches the 0.60 normalization denominator
    /// <c>_score_row_v16_1_1</c>'s <c>painted_signal</c> term uses as its "fully covered" reference
    /// point, at the fraction where that term first saturates being an unreasonably high bar for
    /// "well covered"; 0.15 is a conservative fraction of that denominator chosen so a tile with
    /// meaningfully painted terrain (not just a few brush strokes) clears it.</summary>
    private const float WellCoveredThreshold = 0.15f;

    public static string Classify(TerrainTileTensorPack pack)
    {
        ArgumentNullException.ThrowIfNull(pack);

        if (BlankTileDetector.IsBlank(pack))
            return WowViewer.Core.Curation.CoverageBucket.Blank;

        float alphaCov = pack.McalAlphaPack256 is not null ? CurationMath.AlphaPaintedCoverage(pack.McalAlphaPack256) : 0f;
        float mclyCov = pack.MclyLayerMask is not null ? CurationMath.MclyPaintedCoverage(pack.MclyLayerMask) : 0f;
        float paintedSignal = MathF.Max(alphaCov, mclyCov);

        return paintedSignal >= WellCoveredThreshold
            ? WowViewer.Core.Curation.CoverageBucket.WellCovered
            : WowViewer.Core.Curation.CoverageBucket.LowCoverage;
    }
}
