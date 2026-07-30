using WowViewer.Core.Maps;

namespace WowViewer.Core.Curation.Buckets;

/// <summary>
/// Ports <c>v16_curation.DIFFICULTY_BUCKETS</c> and the weighted scoring formula from
/// <c>build_v16_curation_manifest._score_row_v16_1_1</c> (the concrete threshold/weight source
/// behind the four-bucket vocabulary) into C#, adapted to the v50 signal set:
/// <list type="bullet">
/// <item>roof coverage is always 0 -- v50 dropped roof masks as "broken/dead signals"
/// (docs/architecture/v50-clean-room-dataset-repo-audit-2026-07-15.md, Dropped Signals).</item>
/// <item>object coverage prefers the strict, occlusion-aware
/// <see cref="TerrainTileTensorPack.ObjectGeometryVisibleMask257"/> when present (ADT-builder
/// path); falls back to the alpha-builder footprint masks
/// (<see cref="TerrainTileTensorPack.ObjectPreciseMask257"/>, then
/// <see cref="TerrainTileTensorPack.ObjectMask257"/>) since the strict signals zero-fill on the
/// 0.5.3.3368 alpha corpus (same documented dual-path behavior the frozen v50 catalog records).</item>
/// </list>
/// The four-bucket boundary logic (pathology_pressure/difficulty_score/usefulness_score
/// thresholds) is reproduced exactly; only the object/roof coverage <i>sources</i> differ from the
/// V16 zarr-column originals because those exact columns do not all exist in v50.
/// </summary>
public static class DifficultyBucketClassifier
{
    private const float NormalEdgeThreshold = 0.02f;

    public static string Classify(TerrainTileTensorPack pack)
    {
        ArgumentNullException.ThrowIfNull(pack);

        float[,] height256 = pack.Height257 is not null
            ? CurationMath.Crop257To256(pack.Height257)
            : new float[256, 256];

        bool hasNormals = pack.McnrNormalXyz is not null;
        float[,] normalMask257 = hasNormals && pack.McnrMask257 is not null
            ? BoolMaskToFloat(pack.McnrMask257)
            : new float[257, 257];
        float[,] relief256 = hasNormals
            ? CurationMath.Crop257To256(CurationMath.NormalRelief(pack.McnrNormalXyz!, normalMask257))
            : new float[256, 256];
        float[,] normalEdge256 = hasNormals
            ? CurationMath.Crop257To256(CurationMath.NormalEdgeStrength(pack.McnrNormalXyz!, normalMask257))
            : new float[256, 256];

        float normalCov = CurationMath.Mean(CurationMath.Crop257To256(normalMask257));
        float normalReliefMean = CurationMath.Mean(relief256);
        float normalEdgeFrac = CurationMath.FractionAtLeast(normalEdge256, NormalEdgeThreshold);

        float alphaCov = pack.McalAlphaPack256 is not null ? CurationMath.AlphaPaintedCoverage(pack.McalAlphaPack256) : 0f;
        float mclyCov = pack.MclyLayerMask is not null ? CurationMath.MclyPaintedCoverage(pack.MclyLayerMask) : 0f;
        float paintedSignal = CurationMath.Clamp01(MathF.Max(alphaCov / 0.60f, mclyCov / 0.60f));

        float liquidCov = ComputeLiquidCoverage(pack);
        float objectCov = ComputeObjectCoverage(pack);
        const float roofCov = 0f; // v50 has no roof mask (dropped signal); never available.

        float[,] heightGrad256 = CurationMath.EdgeStrength(height256);
        float terrainDetailMean = (0.65f * CurationMath.Mean(heightGrad256)) + (0.35f * CurationMath.Mean(relief256));

        float minimapGrayStd = 0f;
        float minimapEdgeFrac = 0f;
        float normalEdgeF1 = 0f;
        if (pack.MinimapRgb256 is not null)
        {
            float[,] gray = CurationMath.MinimapGrayscale(pack.MinimapRgb256);
            minimapGrayStd = CurationMath.StdDev(gray) * 255f; // ports v16's 0-255 scale (its minimap arrays are uint8)
            float[,] minimapEdge = CurationMath.EdgeStrength(gray);
            const float minimapEdgeThreshold = 0.05f; // matches build_v16_curation_manifest's CLI default
            minimapEdgeFrac = CurationMath.FractionAtLeast(minimapEdge, minimapEdgeThreshold);
            normalEdgeF1 = ComputeF1(normalEdge256, minimapEdge, NormalEdgeThreshold, minimapEdgeThreshold);
        }

        float deformationRichness = CurationMath.Clamp01(
            (0.45f * MathF.Min(terrainDetailMean / 0.22f, 1.5f))
            + (0.35f * MathF.Min(normalReliefMean / 0.20f, 1.5f))
            + (0.20f * MathF.Min(normalEdgeFrac / 0.12f, 1.5f)));

        float normalCoverageScore = CurationMath.Clamp01((normalCov - 0.20f) / 0.60f);

        float terrainValidity = CurationMath.Clamp01(
            (0.80f * MathF.Min(1f - roofCov, 1.25f)) // roofCov always 0, so this term is a constant 0.80 -- documented v50 adaptation
            + (0.20f * (1f - MathF.Min((objectCov + roofCov + (0.85f * liquidCov)) / 0.75f, 1.0f))));

        float minimapTargetUsefulness = CurationMath.Clamp01(
            (0.55f * MathF.Min(normalEdgeF1 / 0.75f, 1.25f))
            + (0.25f * MathF.Min(minimapGrayStd / 18.0f, 1.25f))
            + (0.20f * MathF.Min(MathF.Min(normalEdgeFrac, minimapEdgeFrac) / 0.10f, 1.25f)));

        float usefulnessScore = CurationMath.Clamp01(
            (0.30f * deformationRichness)
            + (0.15f * normalCoverageScore)
            + (0.20f * terrainValidity)
            + (0.15f * paintedSignal)
            + (0.20f * minimapTargetUsefulness));

        float difficultyScore = CurationMath.Clamp01(
            (0.55f * deformationRichness)
            + (0.20f * paintedSignal)
            + (0.15f * normalCoverageScore)
            + (0.10f * minimapTargetUsefulness));

        float pathologyPressure = CurationMath.Clamp01(
            (MathF.Max(0f, 0.40f - terrainValidity) * 1.6f)
            + (MathF.Max(0f, 0.32f - minimapTargetUsefulness) * 1.2f)
            + (MathF.Max(0f, objectCov + roofCov + liquidCov - 0.55f) * 1.5f));

        if (pathologyPressure >= 0.22f && difficultyScore >= 0.35f)
            return WowViewer.Core.Curation.DifficultyBucket.Pathological;
        if (difficultyScore >= 0.62f && usefulnessScore >= 0.42f)
            return WowViewer.Core.Curation.DifficultyBucket.Hard;
        if (difficultyScore >= 0.34f || usefulnessScore >= 0.38f)
            return WowViewer.Core.Curation.DifficultyBucket.Medium;
        return WowViewer.Core.Curation.DifficultyBucket.Easy;
    }

    internal static float ComputeLiquidCoverage(TerrainTileTensorPack pack)
    {
        if (pack.UnifiedLiquidMask is null) return 0f;
        float[,] liquid256 = CurationMath.Crop257To256(pack.UnifiedLiquidMask);
        return CurationMath.Mean(ClampAll01(liquid256));
    }

    internal static float ComputeObjectCoverage(TerrainTileTensorPack pack)
    {
        float[,]? source = pack.ObjectGeometryVisibleMask257 ?? pack.ObjectPreciseMask257 ?? pack.ObjectMask257;
        if (source is null) return 0f;
        float[,] cov256 = CurationMath.Crop257To256(source);
        return CurationMath.Mean(ClampAll01(cov256));
    }

    private static float[,] ClampAll01(float[,] x)
    {
        int h = x.GetLength(0), w = x.GetLength(1);
        var result = new float[h, w];
        for (int y = 0; y < h; y++)
            for (int col = 0; col < w; col++)
                result[y, col] = CurationMath.Clamp01(x[y, col]);
        return result;
    }

    private static float[,] BoolMaskToFloat(bool[,] mask)
    {
        int h = mask.GetLength(0), w = mask.GetLength(1);
        var result = new float[h, w];
        for (int y = 0; y < h; y++)
            for (int col = 0; col < w; col++)
                result[y, col] = mask[y, col] ? 1f : 0f;
        return result;
    }

    private static float ComputeF1(float[,] a, float[,] b, float thresholdA, float thresholdB)
    {
        int h = a.GetLength(0), w = a.GetLength(1);
        int intersection = 0, sizeA = 0, sizeB = 0;
        for (int y = 0; y < h; y++)
        {
            for (int col = 0; col < w; col++)
            {
                bool va = a[y, col] >= thresholdA;
                bool vb = b[y, col] >= thresholdB;
                if (va) sizeA++;
                if (vb) sizeB++;
                if (va && vb) intersection++;
            }
        }
        int denom = sizeA + sizeB;
        return denom <= 0 ? 1f : (2f * intersection) / denom;
    }
}
