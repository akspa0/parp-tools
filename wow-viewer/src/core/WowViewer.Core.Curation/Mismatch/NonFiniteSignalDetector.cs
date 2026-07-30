using WowViewer.Core.Maps;

namespace WowViewer.Core.Curation.Mismatch;

/// <summary>
/// Ports the <c>verify_v18</c>-style non-finite-value defect check (Spec 109 T018) into a durable,
/// per-tile finding: any NaN/Infinity in a tile's numeric signal arrays is an unambiguous defect,
/// independent of whether the value falls in a plausible range.
/// </summary>
public static class NonFiniteSignalDetector
{
    public static IReadOnlyList<MismatchFinding> Detect(
        TerrainTileTensorPack pack,
        string build,
        string map,
        int tileX,
        int tileY,
        long tileId,
        string curationRunId)
    {
        ArgumentNullException.ThrowIfNull(pack);

        var findings = new List<MismatchFinding>();

        CheckSignal(findings, pack.Height257, "height_257", build, map, tileX, tileY, tileId, curationRunId);
        CheckSignal(findings, pack.McnrNormalXyz, "normal_xyz", build, map, tileX, tileY, tileId, curationRunId);
        CheckSignal(findings, pack.McalAlphaPack256, "alpha_256", build, map, tileX, tileY, tileId, curationRunId);
        CheckSignal(findings, pack.MccvRgb, "mccv_rgb", build, map, tileX, tileY, tileId, curationRunId);
        CheckSignal(findings, pack.UnifiedLiquidMask, "unified_liquid_mask", build, map, tileX, tileY, tileId, curationRunId);
        CheckSignal(findings, pack.UnifiedLiquidHeight, "unified_liquid_height", build, map, tileX, tileY, tileId, curationRunId);

        return findings;
    }

    private static void CheckSignal(
        List<MismatchFinding> findings, float[,]? array, string signalName,
        string build, string map, int tileX, int tileY, long tileId, string curationRunId)
    {
        if (array is null) return;
        if (HasNonFinite(array))
            findings.Add(MakeFinding(signalName, build, map, tileX, tileY, tileId, curationRunId));
    }

    private static void CheckSignal(
        List<MismatchFinding> findings, float[,,]? array, string signalName,
        string build, string map, int tileX, int tileY, long tileId, string curationRunId)
    {
        if (array is null) return;
        if (HasNonFinite(array))
            findings.Add(MakeFinding(signalName, build, map, tileX, tileY, tileId, curationRunId));
    }

    private static MismatchFinding MakeFinding(
        string signalName, string build, string map, int tileX, int tileY, long tileId, string curationRunId) =>
        new(build, map, tileX, tileY, tileId,
            WowViewer.Core.Curation.MismatchCategory.NonFiniteValue,
            WowViewer.Core.Curation.MismatchSeverity.High,
            $"non_finite_value_in_{signalName}",
            WowViewer.Core.Curation.Evaluability.Evaluated,
            Signal: signalName,
            curationRunId);

    private static bool HasNonFinite(float[,] array)
    {
        int h = array.GetLength(0), w = array.GetLength(1);
        for (int y = 0; y < h; y++)
            for (int col = 0; col < w; col++)
                if (!float.IsFinite(array[y, col]))
                    return true;
        return false;
    }

    private static bool HasNonFinite(float[,,] array)
    {
        int h = array.GetLength(0), w = array.GetLength(1), d = array.GetLength(2);
        for (int y = 0; y < h; y++)
            for (int col = 0; col < w; col++)
                for (int c = 0; c < d; c++)
                    if (!float.IsFinite(array[y, col, c]))
                        return true;
        return false;
    }
}
