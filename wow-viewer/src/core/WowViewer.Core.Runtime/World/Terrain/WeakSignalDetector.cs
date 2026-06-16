using WowViewer.Core.Runtime.World.Terrain;

namespace WowViewer.Core.Runtime.World.Terrain;

public static class WeakSignalDetector
{
    public const float DefaultMaxHeightRange = float.MaxValue;
    public const float DefaultMinNormalRelief = 0.02f;
    public const float DefaultMinNormalCoverage = 0.10f;
    public const float MinZLimit = -8192f;
    public const float MaxZLimit = 512f;
    public const float MaxFactor = 512f;
    public const float MinFactorThreshold = 1.25f;
    public const float ClassicCompressionFactor = 33.334f;

    public static WeakSignalAnalysis Analyze(
        WorldTerrainHeightmapData heightmap,
        WeakSignalOptions? options = null)
    {
        options ??= new WeakSignalOptions();

        float heightRange = heightmap.HeightRange;
        float minHeight = heightmap.MinHeight;
        float maxHeight = heightmap.MaxHeight;

        bool hasTerrainData = heightRange > 0.001f;
        bool isCompressed = heightRange < 15f;
        bool nearZeroBand = Math.Abs(minHeight) < 50f && Math.Abs(maxHeight) < 50f;

        bool isCandidate = hasTerrainData && isCompressed && nearZeroBand;

        return new WeakSignalAnalysis
        {
            TileX = 0,
            TileY = 0,
            HeightRange = heightRange,
            MinHeight = minHeight,
            MaxHeight = maxHeight,
            IsWeakSignalCandidate = isCandidate,
            Severity = ClassifySeverity(heightRange, isCandidate)
        };
    }

    public static float EstimateAmplificationFactor(
        WorldTerrainHeightmapData heightmap,
        float referenceMin,
        float referenceMax)
    {
        return EstimateFactorFromRanges(
            heightmap.MinHeight,
            heightmap.MaxHeight,
            referenceMin,
            referenceMax);
    }

    public static float EstimateFallbackFactor(float observedMin, float observedMax)
    {
        float observedRange = Math.Max(observedMax - observedMin, 0f);

        // Tiles with full-scale terrain range (>15m compressed, >500m real) are normal terrain — skip.
        if (observedRange > 15f)
            return 1f;

        // Tiles with zero range are flat/blank — skip.
        if (observedRange < 0.001f)
            return 1f;

        float rawFactor = ClassicCompressionFactor;
        return Math.Clamp(rawFactor, 1f, MaxFactor);
    }

    public static float EstimateFactorFromRanges(
        float observedMin,
        float observedMax,
        float coarseMin,
        float coarseMax)
    {
        const float epsilon = 0.001f;
        float observedRange = Math.Max(observedMax - observedMin, 0f);
        float coarseRange = Math.Max(coarseMax - coarseMin, 0f);
        if (observedRange <= epsilon || coarseRange <= epsilon)
            return 1f;

        float rawFactor = 1f;
        if (coarseRange > observedRange * 1.15f)
            rawFactor = Math.Max(rawFactor, coarseRange / observedRange);

        float observedBelow = Math.Max(0f, -observedMin);
        float coarseBelow = Math.Max(0f, -coarseMin);
        if (observedBelow > epsilon && coarseBelow > observedBelow * 1.15f)
            rawFactor = Math.Max(rawFactor, coarseBelow / observedBelow);

        float observedAbove = Math.Max(0f, observedMax);
        float coarseAbove = Math.Max(0f, coarseMax);
        if (observedAbove > epsilon && coarseAbove > observedAbove * 1.15f)
            rawFactor = Math.Max(rawFactor, coarseAbove / observedAbove);

        return Math.Clamp(rawFactor, 1f, MaxFactor);
    }

    public static float SnapFactor(float rawFactor)
    {
        if (rawFactor <= 1f)
            return 1f;

        float[] supported = { 1f, 2f, 4f, 8f, 16f, 32f, ClassicCompressionFactor, 64f, 128f, 256f, 512f };
        foreach (float value in supported)
        {
            if (rawFactor <= value)
                return value;
        }

        return supported[^1];
    }

    public static WeakSignalChunkMask AnalyzeChunks(float[,] heightmap257)
    {
        int dim = WeakSignalChunkMask.ChunksPerDim;
        var mask = new bool[WeakSignalChunkMask.ChunksPerTile];
        var mins = new float[WeakSignalChunkMask.ChunksPerTile];
        var maxs = new float[WeakSignalChunkMask.ChunksPerTile];
        int weakCount = 0;

        for (int cy = 0; cy < dim; cy++)
        {
            for (int cx = 0; cx < dim; cx++)
            {
                float minH = float.MaxValue;
                float maxH = float.MinValue;
                int baseX = cx * 16;
                int baseY = cy * 16;

                for (int row = 0; row < 17; row++)
                {
                    bool isInner = (row & 1) != 0;
                    int cols = isInner ? 8 : 9;
                    for (int col = 0; col < cols; col++)
                    {
                        int sx = isInner ? (col * 2) + 1 : col * 2;
                        int sy = isInner ? ((row / 2) * 2) + 1 : (row / 2) * 2;
                        int px = baseX + sx;
                        int py = baseY + sy;
                        if ((uint)px >= 257 || (uint)py >= 257) continue;
                        float h = heightmap257[py, px];
                        if (h < minH) minH = h;
                        if (h > maxH) maxH = h;
                    }
                }

                int idx = cy * dim + cx;
                mins[idx] = minH;
                maxs[idx] = maxH;
                float range = maxH - minH;
                bool isWeak = range > 0.001f && range < 15f
                    && Math.Abs(minH) < 50f && Math.Abs(maxH) < 50f;
                mask[idx] = isWeak;
                if (isWeak) weakCount++;
            }
        }

        return new WeakSignalChunkMask
        {
            Mask = mask,
            ChunkMinHeights = mins,
            ChunkMaxHeights = maxs,
            WeakChunkCount = weakCount,
        };
    }

    public static float[,] AmplifyChunks(float[,] original, WeakSignalChunkMask chunkMask, float factor)
    {
        var result = new float[257, 257];

        float anchor = 0f;
        for (int i = 0; i < WeakSignalChunkMask.ChunksPerTile; i++)
        {
            if (chunkMask.Mask[i] && chunkMask.ChunkMinHeights[i] < anchor)
                anchor = chunkMask.ChunkMinHeights[i];
        }

        int dim = WeakSignalChunkMask.ChunksPerDim;
        for (int y = 0; y < 257; y++)
        {
            int cy = Math.Min(y / 16, dim - 1);
            for (int x = 0; x < 257; x++)
            {
                int cx = Math.Min(x / 16, dim - 1);
                int idx = cy * dim + cx;

                if (chunkMask.Mask[idx])
                {
                    float h = original[y, x];
                    float amplified = anchor + (h - anchor) * factor;
                    if (anchor >= 0 && amplified < 0f) amplified = 0f;
                    result[y, x] = amplified;
                }
                else
                {
                    result[y, x] = original[y, x];
                }
            }
        }

        return result;
    }

    public static float[] AmplifyHeightmapFlat(WorldTerrainHeightmapData heightmap, float factor, float? anchorHeight = null)
    {
        float anchor = anchorHeight ?? (heightmap.MinHeight < 0f ? heightmap.MinHeight : 0f);
        bool preserveNegativeFloor = anchor < 0f;
        float[] restored = new float[heightmap.Heights.Length];

        for (int i = 0; i < heightmap.Heights.Length; i++)
        {
            float sourceHeight = heightmap.Heights[i];
            float restoredHeight = anchor + ((sourceHeight - anchor) * factor);
            if (!preserveNegativeFloor && restoredHeight < 0f)
                restoredHeight = 0f;

            restored[i] = restoredHeight;
        }

        return restored;
    }

    public static float ClampZ(float value)
        => Math.Clamp(value, MinZLimit, MaxZLimit);

    private static string ClassifySeverity(float heightRange, bool isCandidate)
    {
        if (!isCandidate)
            return "none";

        if (heightRange < 0.5f)
            return "high";
        if (heightRange < 2.0f)
            return "medium";
        return "low";
    }
}
