using System.Numerics;

namespace WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

/// <summary>
/// High-throughput analyzer for computing surviving height levels, raw delta entropy,
/// and micro-relief distributions across terrain lattices without elevation bias.
/// </summary>
public static class StratigraphyLevelAnalyzer
{
    public const int TileLatticeDim = 257;
    public const int ChunksPerTileDim = 16;
    public const int ChunksPerTile = 256;
    public const int HeightsPerChunk = 145; // 9x9 outer (81) + 8x8 inner (64)

    /// <summary>
    /// Counts the exact number of distinct floating-point height levels.
    /// Uses integer bit representation for high-speed hash set deduplication.
    /// </summary>
    public static int CountSurvivingLevels(ReadOnlySpan<float> heights)
    {
        if (heights.IsEmpty)
            return 0;

        var uniqueBits = new HashSet<int>(Math.Min(heights.Length, 4096));
        for (int i = 0; i < heights.Length; i++)
        {
            float h = heights[i];
            if (float.IsNaN(h) || float.IsInfinity(h))
                continue;

            int bits = BitConverter.SingleToInt32Bits(h);
            uniqueBits.Add(bits);
        }

        return uniqueBits.Count;
    }

    /// <summary>
    /// Analyzes a 257x257 height lattice into per-chunk stratum records and tile-wide metrics.
    /// </summary>
    public static StratigraphyTileAnalysis AnalyzeTile(
        float[,] heights257,
        ushort[]? holeMasks16 = null,
        int tileX = 0,
        int tileY = 0,
        string tileName = "")
    {
        ArgumentNullException.ThrowIfNull(heights257);

        float minHeight = float.MaxValue;
        float maxHeight = float.MinValue;
        var totalLevelSet = new HashSet<int>(8192);

        var chunkRecords = new ChunkStratumRecord[ChunksPerTile];
        int activeCount = 0;
        int squeezedCount = 0;
        int holedCount = 0;
        int flatCount = 0;

        for (int cy = 0; cy < ChunksPerTileDim; cy++)
        {
            for (int cx = 0; cx < ChunksPerTileDim; cx++)
            {
                int chunkIdx = cy * ChunksPerTileDim + cx;
                ushort holeMask = holeMasks16 != null && chunkIdx < holeMasks16.Length ? holeMasks16[chunkIdx] : (ushort)0;
                int holedQuads = BitOperations.PopCount((uint)holeMask);

                float chunkMin = float.MaxValue;
                float chunkMax = float.MinValue;
                var chunkLevelSet = new HashSet<int>(HeightsPerChunk);

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
                        if ((uint)px >= TileLatticeDim || (uint)py >= TileLatticeDim)
                            continue;

                        float h = heights257[py, px];
                        if (float.IsNaN(h) || float.IsInfinity(h))
                            continue;

                        if (h < chunkMin) chunkMin = h;
                        if (h > chunkMax) chunkMax = h;
                        if (h < minHeight) minHeight = h;
                        if (h > maxHeight) maxHeight = h;

                        int bits = BitConverter.SingleToInt32Bits(h);
                        chunkLevelSet.Add(bits);
                        totalLevelSet.Add(bits);
                    }
                }

                if (chunkMin > chunkMax)
                {
                    chunkMin = 0f;
                    chunkMax = 0f;
                }

                int survivingLevels = chunkLevelSet.Count;
                float range = chunkMax - chunkMin;

                TemporalStratum stratum = ClassifyChunkStratum(chunkMin, chunkMax, survivingLevels, holeMask);
                float estFactor = EstimateChunkAmplificationFactor(stratum, range);

                switch (stratum)
                {
                    case TemporalStratum.Active_1x:
                        activeCount++;
                        break;
                    case TemporalStratum.ClassicErasure_33x:
                    case TemporalStratum.LateRevision_4x_8x:
                    case TemporalStratum.DeepProto_64x_512x:
                        squeezedCount++;
                        break;
                    case TemporalStratum.Holed_DevMesh_1x:
                        holedCount++;
                        break;
                    case TemporalStratum.BitExact_Flat:
                        flatCount++;
                        break;
                }

                chunkRecords[chunkIdx] = new ChunkStratumRecord
                {
                    ChunkX = cx,
                    ChunkY = cy,
                    MinHeight = chunkMin,
                    MaxHeight = chunkMax,
                    SurvivingLevels = survivingLevels,
                    RelativeEntropy = survivingLevels > 1 && range > 0.0001f ? (float)survivingLevels / HeightsPerChunk : 0f,
                    HoleMask = holeMask,
                    HoledQuadCount = holedQuads,
                    Stratum = stratum,
                    EstimatedAmplificationFactor = estFactor,
                    Description = FormatStratumDescription(stratum, range, survivingLevels, holedQuads),
                };
            }
        }

        if (minHeight > maxHeight)
        {
            minHeight = 0f;
            maxHeight = 0f;
        }

        SeamDiscontinuityResult seamProfile = SeamDiscontinuityProfiler.AnalyzeSeams(heights257);
        TemporalStratum dominantStratum = DetermineDominantStratum(activeCount, squeezedCount, holedCount, flatCount, chunkRecords);

        float suggestedFactor = CalculateSuggestedTileFactor(chunkRecords, dominantStratum);

        return new StratigraphyTileAnalysis
        {
            TileName = string.IsNullOrEmpty(tileName) ? $"tile_{tileX}_{tileY}" : tileName,
            TileX = tileX,
            TileY = tileY,
            MinHeight = minHeight,
            MaxHeight = maxHeight,
            TotalSurvivingLevels = totalLevelSet.Count,
            DominantStratum = dominantStratum,
            SuggestedAmplificationFactor = suggestedFactor,
            ActiveChunkCount = activeCount,
            SqueezedChunkCount = squeezedCount,
            HoledChunkCount = holedCount,
            FlatChunkCount = flatCount,
            SeamProfile = seamProfile,
            Chunks = chunkRecords,
        };
    }

    /// <summary>
    /// Classifies an individual 33.333m MCNK chunk based on physical metrics.
    /// </summary>
    public static TemporalStratum ClassifyChunkStratum(float minH, float maxH, int survivingLevels, ushort holeMask)
    {
        if (holeMask != 0 && survivingLevels >= 16)
            return TemporalStratum.Holed_DevMesh_1x;

        if (survivingLevels <= 1)
            return TemporalStratum.BitExact_Flat;

        float range = maxH - minH;

        // Submerged deep bathymetry (ocean floor beneath sea level with rich relief)
        if (maxH < -20f && range > 5f && survivingLevels >= 32)
            return TemporalStratum.Submerged_OceanFloor;

        // Normal active full-scale terrain
        if (range >= 15f && survivingLevels >= 32)
            return TemporalStratum.Active_1x;

        // Moderate compression (late editing passes)
        if (range >= 2.0f && range < 15f && survivingLevels >= 24)
            return TemporalStratum.LateRevision_4x_8x;

        // Classic 1/0.03 = 33.334x erasure compression
        if (range >= 0.02f && range < 2.0f && survivingLevels >= 8)
            return TemporalStratum.ClassicErasure_33x;

        // Sub-millimeter deep prototype trace
        if (range > 0.0001f && range < 0.02f && survivingLevels >= 4)
            return TemporalStratum.DeepProto_64x_512x;

        return range >= 0.001f ? TemporalStratum.ClassicErasure_33x : TemporalStratum.BitExact_Flat;
    }

    private static float EstimateChunkAmplificationFactor(TemporalStratum stratum, float range) => stratum switch
    {
        TemporalStratum.Active_1x => 1f,
        TemporalStratum.Holed_DevMesh_1x => 1f,
        TemporalStratum.LateRevision_4x_8x => range > 0.001f ? Math.Clamp(15f / range, 2f, 8f) : 4f,
        TemporalStratum.ClassicErasure_33x => TemporalStratigraphyOptions.DefaultClassicFactor,
        TemporalStratum.DeepProto_64x_512x => 128f,
        TemporalStratum.Submerged_OceanFloor => 1f,
        TemporalStratum.StagingArea_Untextured => 1f,
        _ => 1f,
    };

    private static TemporalStratum DetermineDominantStratum(
        int activeCount,
        int squeezedCount,
        int holedCount,
        int flatCount,
        ChunkStratumRecord[] chunks)
    {
        if (activeCount >= 128)
            return TemporalStratum.Active_1x;
        if (squeezedCount >= 64)
            return TemporalStratum.ClassicErasure_33x;
        if (holedCount >= 64)
            return TemporalStratum.Holed_DevMesh_1x;
        if (flatCount >= 200)
            return TemporalStratum.BitExact_Flat;

        // Return highest frequency stratum
        var counts = new Dictionary<TemporalStratum, int>();
        foreach (var c in chunks)
        {
            counts[c.Stratum] = counts.GetValueOrDefault(c.Stratum, 0) + 1;
        }

        return counts.OrderByDescending(kv => kv.Value).Select(kv => kv.Key).FirstOrDefault(TemporalStratum.Active_1x);
    }

    private static float CalculateSuggestedTileFactor(ChunkStratumRecord[] chunks, TemporalStratum dominant)
    {
        if (dominant == TemporalStratum.Active_1x || dominant == TemporalStratum.BitExact_Flat)
            return 1f;

        float sum = 0f;
        int count = 0;
        foreach (var c in chunks)
        {
            if (c.Stratum != TemporalStratum.Active_1x && c.Stratum != TemporalStratum.BitExact_Flat)
            {
                sum += c.EstimatedAmplificationFactor;
                count++;
            }
        }

        return count > 0 ? Math.Clamp(sum / count, 1f, TemporalStratigraphyOptions.DefaultMaxFactor) : TemporalStratigraphyOptions.DefaultClassicFactor;
    }

    private static string FormatStratumDescription(TemporalStratum stratum, float range, int levels, int holes) => stratum switch
    {
        TemporalStratum.Active_1x => $"Active terrain ({range:0.0}m range, {levels} levels)",
        TemporalStratum.LateRevision_4x_8x => $"Late revision ({range:0.00}m range, {levels} levels)",
        TemporalStratum.ClassicErasure_33x => $"Classic 1/0.03 erasure ({range:0.000}m range, {levels} levels)",
        TemporalStratum.DeepProto_64x_512x => $"Deep proto trace ({range:0.0000}m range, {levels} levels)",
        TemporalStratum.Holed_DevMesh_1x => $"Holed dev mesh ({holes} holes, {levels} levels)",
        TemporalStratum.Submerged_OceanFloor => $"Submerged ocean floor ({range:0.0}m range, {levels} levels)",
        TemporalStratum.StagingArea_Untextured => $"Dev staging area ({levels} levels)",
        TemporalStratum.BitExact_Flat => "Bit-exact flat plane",
        _ => "Unknown stratum",
    };
}
