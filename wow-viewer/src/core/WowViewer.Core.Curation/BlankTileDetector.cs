using WowViewer.Core.Curation.Buckets;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Curation;

/// <summary>
/// Ports <c>v16_curation.is_blank_what_plate</c>: a tile whose height has essentially zero
/// variance AND whose painted-signal coverage (alpha/mcly/liquid/object) is essentially zero is a
/// "what plate" -- a technically-present but contentless tile.
/// </summary>
public static class BlankTileDetector
{
    public static bool IsBlank(TerrainTileTensorPack pack)
    {
        ArgumentNullException.ThrowIfNull(pack);

        float[,] height256 = pack.Height257 is not null
            ? CurationMath.Crop257To256(pack.Height257)
            : new float[256, 256];

        float alphaCov = pack.McalAlphaPack256 is not null ? CurationMath.AlphaPaintedCoverage(pack.McalAlphaPack256) : 0f;
        float mclyCov = pack.MclyLayerMask is not null ? CurationMath.MclyPaintedCoverage(pack.MclyLayerMask) : 0f;
        float liquidCov = DifficultyBucketClassifier.ComputeLiquidCoverage(pack);
        float objectCov = DifficultyBucketClassifier.ComputeObjectCoverage(pack);

        return CurationMath.IsBlankWhatPlate(height256, alphaCov, mclyCov, liquidCov, objectCov);
    }
}
