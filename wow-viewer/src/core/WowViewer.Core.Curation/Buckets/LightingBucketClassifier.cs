using WowViewer.Core.Maps;

namespace WowViewer.Core.Curation.Buckets;

/// <summary>
/// Reads the lighting/time-of-day match status already computed by
/// <see cref="WowViewer.Core.IO.Maps.MinimapShadingMatch"/> (Spec 111) off
/// <see cref="TerrainTileTensorPack.MinimapLightingProvenance"/>, and reduces it to the same
/// bucket vocabulary <c>spec111/lighting_buckets.py</c> already reports on
/// (matched/low_confidence_ambiguous/low_confidence_flat_terrain/not_evaluated) -- no new scoring
/// logic, this classifier only durably records a value that was already computed at harvest time.
/// </summary>
public static class LightingBucketClassifier
{
    public static string Classify(TerrainTileTensorPack pack)
    {
        ArgumentNullException.ThrowIfNull(pack);
        return pack.MinimapLightingProvenance?.ShadingMatchStatus
            ?? WowViewer.Core.Curation.LightingBucket.NotEvaluated;
    }
}
