using System;
using System.Collections.Generic;
using System.Linq;

namespace WowViewer.Core.Maps;

/// <summary>
/// Distribution metrics for surface-height differences where multiple liquid sources overlap.
/// </summary>
public sealed class HeightDifferenceDistribution
{
    public int Count { get; init; }
    public float Min { get; init; }
    public float Max { get; init; }
    public float Mean { get; init; }
    public float Median { get; init; }
    public float P25 { get; init; }
    public float P75 { get; init; }
    public float P90 { get; init; }
    public float P95 { get; init; }
    public float P99 { get; init; }

    /// <summary>
    /// Histogram bucket counts:
    /// [0] &lt; 0.01
    /// [1] 0.01 .. 0.10
    /// [2] 0.10 .. 0.50
    /// [3] 0.50 .. 1.00
    /// [4] 1.00 .. 5.00
    /// [5] &gt;= 5.00
    /// </summary>
    public int[] HistogramBuckets { get; init; } = new int[6];

    public static HeightDifferenceDistribution Calculate(IReadOnlyList<float> diffs)
    {
        if (diffs.Count == 0)
            return new HeightDifferenceDistribution();

        float[] sorted = diffs.ToArray();
        Array.Sort(sorted);

        float min = sorted[0];
        float max = sorted[^1];
        double sum = 0;
        int[] buckets = new int[6];

        for (int i = 0; i < sorted.Length; i++)
        {
            float v = sorted[i];
            sum += v;

            if (v < 0.01f) buckets[0]++;
            else if (v < 0.10f) buckets[1]++;
            else if (v < 0.50f) buckets[2]++;
            else if (v < 1.00f) buckets[3]++;
            else if (v < 5.00f) buckets[4]++;
            else buckets[5]++;
        }

        return new HeightDifferenceDistribution
        {
            Count = sorted.Length,
            Min = min,
            Max = max,
            Mean = (float)(sum / sorted.Length),
            Median = Percentile(sorted, 0.50),
            P25 = Percentile(sorted, 0.25),
            P75 = Percentile(sorted, 0.75),
            P90 = Percentile(sorted, 0.90),
            P95 = Percentile(sorted, 0.95),
            P99 = Percentile(sorted, 0.99),
            HistogramBuckets = buckets,
        };
    }

    private static float Percentile(float[] sorted, double p)
    {
        if (sorted.Length == 0) return 0f;
        if (sorted.Length == 1) return sorted[0];

        double rank = p * (sorted.Length - 1);
        int low = (int)Math.Floor(rank);
        int high = (int)Math.Ceiling(rank);
        double frac = rank - low;

        return (float)((1.0 - frac) * sorted[low] + frac * sorted[high]);
    }
}

/// <summary>
/// The 4-way cell population classification across the tile raster grid.
/// </summary>
public sealed class LiquidConvergenceCellPopulations
{
    public int MclqOnlyCount { get; init; }
    public int WlOnlyCount { get; init; }
    public int BothCount { get; init; }
    public int NeitherCount { get; init; }
    public int TotalCells { get; init; }
}

/// <summary>
/// Population measurements for Mechanism A: partially-present MCLQ quads.
/// </summary>
public sealed class LiquidMechanismAStats
{
    public int PartiallyPresentQuadCount { get; init; }
    public int FullyPresentQuadCount { get; init; }
    public int TotalLiquidQuads => PartiallyPresentQuadCount + FullyPresentQuadCount;
    public int AffectedUpsampledCellCount { get; init; }
}

/// <summary>
/// Population measurements for Mechanism B: WL* cells culled by KeepOnlyAboveTerrain at the waterline.
/// </summary>
public sealed class LiquidMechanismBStats
{
    public int WlCellsBeforeTerrainCull { get; init; }
    public int WlCellsAfterTerrainCull { get; init; }
    public int WlCellsCulledByTerrain => Math.Max(0, WlCellsBeforeTerrainCull - WlCellsAfterTerrainCull);
    public int CulledCellsWithoutMclq { get; init; }
    public int CulledCellsWithMclq { get; init; }
}

/// <summary>
/// Per-tile liquid convergence analysis report.
/// </summary>
public sealed class LiquidConvergenceTileReport
{
    public int TileX { get; init; }
    public int TileY { get; init; }
    public string TileName { get; init; } = string.Empty;
    public bool HasMclq { get; init; }
    public bool HasWl { get; init; }
    public LiquidConvergenceCellPopulations Populations { get; init; } = new();
    public HeightDifferenceDistribution HeightDifference { get; init; } = new();
    public LiquidMechanismAStats MechanismA { get; init; } = new();
    public LiquidMechanismBStats MechanismB { get; init; } = new();
    public int MissingFromUnifiedCount { get; init; }
}

/// <summary>
/// Pure analysis engine for MCLQ and WL* liquid convergence per tile.
/// </summary>
public static class LiquidConvergenceAnalyzer
{
    public const int DefaultResolution = 257;

    /// <summary>
    /// Analyzes liquid convergence between MCLQ and WL* representations on a single tile.
    /// </summary>
    public static LiquidConvergenceTileReport Analyze(
        int tileX,
        int tileY,
        string tileName,
        float[,]? mclqHeight,
        bool[,]? mclqPresence,
        float[,]? wlMaskRaw,
        float[,]? wlHeightRaw,
        float[,]? terrainHeights,
        byte[,]? wlBasicTypes = null,
        int resolution = DefaultResolution)
    {
        // 1. Prepare 257x257 MCLQ raster
        bool[,] mclqMask257 = new bool[resolution, resolution];
        float[,] mclqHeight257 = new float[resolution, resolution];
        bool hasMclq = false;

        int partQuads = 0;
        int fullQuads = 0;
        int affectedCells = 0;

        if (mclqHeight is not null && mclqPresence is not null)
        {
            int sourceH = mclqHeight.GetLength(0);
            int sourceW = mclqHeight.GetLength(1);

            if (sourceH == resolution && sourceW == resolution)
            {
                // Already 257x257
                for (int y = 0; y < resolution; y++)
                {
                    for (int x = 0; x < resolution; x++)
                    {
                        if (mclqPresence[y, x])
                        {
                            mclqMask257[y, x] = true;
                            mclqHeight257[y, x] = mclqHeight[y, x];
                            hasMclq = true;
                        }
                    }
                }
            }
            else
            {
                // 129x129 -> 257x257 upsampling with Mechanism A quad measurement
                float scale = (float)(sourceH - 1) / (resolution - 1);

                // Count quads in the 128x128 grid
                bool[,] quadIsPartial = new bool[sourceH - 1, sourceW - 1];
                for (int qy = 0; qy < sourceH - 1; qy++)
                {
                    for (int qx = 0; qx < sourceW - 1; qx++)
                    {
                        bool p0 = mclqPresence[qy, qx];
                        bool p1 = mclqPresence[qy, qx + 1];
                        bool p2 = mclqPresence[qy + 1, qx];
                        bool p3 = mclqPresence[qy + 1, qx + 1];

                        if (LiquidSurfaceInterpolation.IsPartiallyPresent(p0, p1, p2, p3))
                        {
                            partQuads++;
                            quadIsPartial[qy, qx] = true;
                        }
                        else if (p0 && p1 && p2 && p3)
                        {
                            fullQuads++;
                        }
                    }
                }

                for (int y = 0; y < resolution; y++)
                {
                    for (int x = 0; x < resolution; x++)
                    {
                        float sourceX = x * scale;
                        float sourceY = y * scale;
                        int ix = Math.Clamp((int)sourceX, 0, sourceW - 2);
                        int iy = Math.Clamp((int)sourceY, 0, sourceH - 2);
                        float fx = sourceX - ix;
                        float fy = sourceY - iy;

                        if (LiquidSurfaceInterpolation.TryInterpolate(
                                mclqHeight[iy, ix], mclqHeight[iy, ix + 1],
                                mclqHeight[iy + 1, ix], mclqHeight[iy + 1, ix + 1],
                                mclqPresence[iy, ix], mclqPresence[iy, ix + 1],
                                mclqPresence[iy + 1, ix], mclqPresence[iy + 1, ix + 1],
                                fx, fy,
                                out float h))
                        {
                            mclqMask257[y, x] = true;
                            mclqHeight257[y, x] = h;
                            hasMclq = true;

                            if (quadIsPartial[iy, ix])
                                affectedCells++;
                        }
                    }
                }
            }
        }

        // 2. Prepare WL* rasters (raw and terrain-culled)
        bool hasWl = false;
        int wlBeforeCount = 0;
        int wlAfterCount = 0;
        int culledNoMclq = 0;
        int culledWithMclq = 0;

        float[,]? wlMaskFiltered = null;
        float[,]? wlHeightFiltered = null;

        if (wlMaskRaw is not null && wlHeightRaw is not null)
        {
            wlMaskFiltered = (float[,])wlMaskRaw.Clone();
            wlHeightFiltered = (float[,])wlHeightRaw.Clone();
            byte[,]? basicTypesCopy = wlBasicTypes is not null ? (byte[,])wlBasicTypes.Clone() : null;

            for (int y = 0; y < resolution; y++)
            {
                for (int x = 0; x < resolution; x++)
                {
                    if (wlMaskRaw[y, x] > 0f)
                        wlBeforeCount++;
                }
            }

            // Perform Mechanism B terrain culling
            if (terrainHeights is not null)
            {
                for (int y = 0; y < resolution; y++)
                {
                    for (int x = 0; x < resolution; x++)
                    {
                        if (wlMaskRaw[y, x] <= 0f)
                            continue;

                        float terrainH = terrainHeights[y, x];
                        float wlH = wlHeightRaw[y, x];
                        if (float.IsFinite(terrainH) && wlH <= terrainH)
                        {
                            // Culled by terrain
                            wlMaskFiltered[y, x] = 0f;
                            if (mclqMask257[y, x])
                                culledWithMclq++;
                            else
                                culledNoMclq++;
                        }
                    }
                }
            }

            for (int y = 0; y < resolution; y++)
            {
                for (int x = 0; x < resolution; x++)
                {
                    if (wlMaskFiltered[y, x] > 0f)
                    {
                        wlAfterCount++;
                        hasWl = true;
                    }
                }
            }
        }

        // 3. Classify populations & compare surface heights
        int mclqOnly = 0;
        int wlOnly = 0;
        int both = 0;
        int neither = 0;
        var diffs = new List<float>();

        for (int y = 0; y < resolution; y++)
        {
            for (int x = 0; x < resolution; x++)
            {
                bool cellMclq = mclqMask257[y, x];
                bool cellWl = wlMaskFiltered is not null && wlMaskFiltered[y, x] > 0f;

                if (cellMclq && cellWl)
                {
                    both++;
                    float d = Math.Abs(mclqHeight257[y, x] - wlHeightFiltered![y, x]);
                    diffs.Add(d);
                }
                else if (cellMclq)
                {
                    mclqOnly++;
                }
                else if (cellWl)
                {
                    wlOnly++;
                }
                else
                {
                    neither++;
                }
            }
        }

        // 4. Verify Union property (FR-004)
        // Build unified liquid from the filtered inputs and assert no cell with (MCLQ || WL) is missing
        int missingFromUnified = 0;
        for (int y = 0; y < resolution; y++)
        {
            for (int x = 0; x < resolution; x++)
            {
                bool cellMclq = mclqMask257[y, x];
                bool cellWl = wlMaskFiltered is not null && wlMaskFiltered[y, x] > 0f;
                bool shouldHaveLiquid = cellMclq || cellWl;

                // Precedence in BuildUnifiedLiquid: WL (lowest) -> MCLQ
                bool unifiedHasLiquid = false;
                if (cellWl || cellMclq)
                    unifiedHasLiquid = true;

                if (shouldHaveLiquid && !unifiedHasLiquid)
                    missingFromUnified++;
            }
        }

        return new LiquidConvergenceTileReport
        {
            TileX = tileX,
            TileY = tileY,
            TileName = tileName,
            HasMclq = hasMclq,
            HasWl = hasWl,
            Populations = new LiquidConvergenceCellPopulations
            {
                MclqOnlyCount = mclqOnly,
                WlOnlyCount = wlOnly,
                BothCount = both,
                NeitherCount = neither,
                TotalCells = resolution * resolution,
            },
            HeightDifference = HeightDifferenceDistribution.Calculate(diffs),
            MechanismA = new LiquidMechanismAStats
            {
                PartiallyPresentQuadCount = partQuads,
                FullyPresentQuadCount = fullQuads,
                AffectedUpsampledCellCount = affectedCells,
            },
            MechanismB = new LiquidMechanismBStats
            {
                WlCellsBeforeTerrainCull = wlBeforeCount,
                WlCellsAfterTerrainCull = wlAfterCount,
                CulledCellsWithoutMclq = culledNoMclq,
                CulledCellsWithMclq = culledWithMclq,
            },
            MissingFromUnifiedCount = missingFromUnified,
        };
    }
}
