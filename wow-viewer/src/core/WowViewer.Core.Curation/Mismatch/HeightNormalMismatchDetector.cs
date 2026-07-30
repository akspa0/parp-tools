using WowViewer.Core.Maps;

namespace WowViewer.Core.Curation.Mismatch;

/// <summary>
/// Ports <c>mismatch_detector.compute_tile_mismatch_metrics</c> + <c>detect_mismatches</c> exactly
/// (same thresholds, same severity bands, same reason strings) -- detects tiles where normal
/// vectors encode significant terrain variation but the height data is suspiciously flat,
/// indicating poisoned supervision that would degrade height-model training if trained on
/// uncritically.
/// </summary>
public static class HeightNormalMismatchDetector
{
    private const float NormalReliefThreshold = 0.02f;
    private const float HeightRangeThreshold = 3.0f;
    private const float NormalCovThreshold = 0.10f;

    /// <summary>Returns null when the tile is evaluated and found consistent (flat normals,
    /// sufficient height range, or a genuine non-mismatch) -- per the query contract (data-model.md),
    /// an evaluated-and-clean tile contributes zero finding rows. Returns a
    /// <see cref="Evaluability.NotEvaluable"/> finding when the check cannot run at all (no normal
    /// data, or insufficient normal coverage to trust a relief measurement). Returns an
    /// <see cref="Evaluability.Evaluated"/> finding with a real severity when an actual mismatch is
    /// detected.</summary>
    public static MismatchFinding? Detect(
        TerrainTileTensorPack pack,
        string build,
        string map,
        int tileX,
        int tileY,
        long tileId,
        string curationRunId)
    {
        ArgumentNullException.ThrowIfNull(pack);

        if (pack.Height257 is null)
            return null; // No height ground truth at all -- a different check's concern (non-finite/missing signal), not this one.

        float[,] height257 = pack.Height257;
        (float min, float max) = CurationMath.MinMax(height257);
        float heightRange = max - min;

        if (pack.McnrNormalXyz is null)
        {
            return new MismatchFinding(build, map, tileX, tileY, tileId,
                WowViewer.Core.Curation.MismatchCategory.HeightNormalMismatch,
                WowViewer.Core.Curation.MismatchSeverity.NotEvaluable,
                "no_normal_data",
                WowViewer.Core.Curation.Evaluability.NotEvaluable,
                Signal: "normal_xyz",
                curationRunId);
        }

        float[,] normalMask = pack.McnrMask257 is not null
            ? BoolMaskToFloat(pack.McnrMask257)
            : Ones(height257.GetLength(0), height257.GetLength(1));
        float normalCov = CurationMath.Mean(normalMask);

        if (normalCov < NormalCovThreshold)
        {
            return new MismatchFinding(build, map, tileX, tileY, tileId,
                WowViewer.Core.Curation.MismatchCategory.HeightNormalMismatch,
                WowViewer.Core.Curation.MismatchSeverity.NotEvaluable,
                "insufficient_normal_coverage",
                WowViewer.Core.Curation.Evaluability.NotEvaluable,
                Signal: "normal_mask",
                curationRunId);
        }

        float[,] relief = CurationMath.NormalRelief(pack.McnrNormalXyz, normalMask);
        float reliefMean = CurationMath.Mean(relief);

        if (reliefMean < NormalReliefThreshold)
            return null; // "flat_normals": consistent (flat terrain, flat normals), not a mismatch.

        if (heightRange >= HeightRangeThreshold)
            return null; // "sufficient_height_range": normals vary but so does height -- consistent.

        string severity = Severity(reliefMean, heightRange);
        return new MismatchFinding(build, map, tileX, tileY, tileId,
            WowViewer.Core.Curation.MismatchCategory.HeightNormalMismatch,
            severity,
            "height_flat_vs_normal_varied",
            WowViewer.Core.Curation.Evaluability.Evaluated,
            Signal: "height_257",
            curationRunId);
    }

    private static string Severity(float relief, float heightRange)
    {
        float ratio = heightRange < 0.001f ? relief : relief / MathF.Max(heightRange, 1e-6f);
        if (ratio > 0.10f) return WowViewer.Core.Curation.MismatchSeverity.High;
        if (ratio > 0.03f) return WowViewer.Core.Curation.MismatchSeverity.Medium;
        return WowViewer.Core.Curation.MismatchSeverity.Low;
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

    private static float[,] Ones(int h, int w)
    {
        var result = new float[h, w];
        for (int y = 0; y < h; y++)
            for (int col = 0; col < w; col++)
                result[y, col] = 1f;
        return result;
    }
}
