using WowViewer.Core.Maps;

namespace WowViewer.Core.Curation.Mismatch;

/// <summary>
/// Reuses the correlation already computed by
/// <see cref="WowViewer.Core.IO.Maps.MinimapShadingMatch"/> (Spec 111) -- which sweeps synthesized
/// candidate renders against the authored minimap and records the best-candidate tint-invariant
/// luma correlation -- as a durable synthetic-vs-authored fidelity finding (Spec 122 User Story 3).
/// This does not reinvent the comparison: a tile's synthetic render is a poor match for its
/// authored counterpart exactly when the best candidate <see cref="MinimapShadingMatch"/> could
/// find still correlates poorly, independent of which time-of-day it best resembles.
/// </summary>
public static class SyntheticFidelityDetector
{
    /// <summary>Below this best-candidate correlation, a tile's synthetic render is flagged as a
    /// fidelity gap -- reuses the same 0.30 threshold <see cref="MinimapLightingProvenance.Infer"/>
    /// already uses for its own "mcshLikely" correlation judgement (this codebase's precedent for
    /// what counts as a meaningful shading correlation), rather than inventing a new number.</summary>
    private const float FidelityGapThreshold = 0.30f;

    public static (string Status, float? Score) Classify(TerrainTileTensorPack pack)
    {
        ArgumentNullException.ThrowIfNull(pack);

        MinimapLightingProvenance? provenance = pack.MinimapLightingProvenance;
        if (provenance is null || provenance.ShadingMatchStatus == WowViewer.Core.Curation.LightingBucket.NotEvaluated)
            return (WowViewer.Core.Curation.SyntheticFidelityStatus.NotEvaluable, null);

        return (WowViewer.Core.Curation.SyntheticFidelityStatus.Evaluated, provenance.ShadingMatchConfidence);
    }

    /// <summary>Returns null when the tile is not evaluable (no authored/synthetic pair, or the
    /// tile is out of the shading-match's build scope) or when fidelity is acceptable. Returns a
    /// finding when a tile IS evaluable but its best-fit correlation still falls below the gap
    /// threshold -- a synthetic render that does not track its authored counterpart even at its
    /// best-guess time of day.</summary>
    public static MismatchFinding? Detect(
        TerrainTileTensorPack pack,
        string build,
        string map,
        int tileX,
        int tileY,
        long tileId,
        string curationRunId)
    {
        (string status, float? score) = Classify(pack);

        if (status == WowViewer.Core.Curation.SyntheticFidelityStatus.NotEvaluable)
        {
            return new MismatchFinding(build, map, tileX, tileY, tileId,
                WowViewer.Core.Curation.MismatchCategory.SyntheticFidelityGap,
                WowViewer.Core.Curation.MismatchSeverity.NotEvaluable,
                "no_authored_synthetic_pair_or_out_of_build_scope",
                WowViewer.Core.Curation.Evaluability.NotEvaluable,
                Signal: "minimap_rgb_authored",
                curationRunId);
        }

        if (score is null || score >= FidelityGapThreshold)
            return null; // Evaluated and acceptable -- consistent with this library's "clean tiles get zero finding rows" convention.

        string severity = score < 0.10f
            ? WowViewer.Core.Curation.MismatchSeverity.High
            : WowViewer.Core.Curation.MismatchSeverity.Medium;

        return new MismatchFinding(build, map, tileX, tileY, tileId,
            WowViewer.Core.Curation.MismatchCategory.SyntheticFidelityGap,
            severity,
            "synthetic_render_shading_does_not_track_authored_minimap",
            WowViewer.Core.Curation.Evaluability.Evaluated,
            Signal: "minimap_rgb",
            curationRunId);
    }
}
