using System.Numerics;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class LiquidConvergenceAnalyzerTests
{
    [Fact]
    public void ClassifiesPopulations_Accurately_AndVerifiesUnion()
    {
        const int res = 257;
        float[,] mclqHeight = new float[res, res];
        bool[,] mclqPresence = new bool[res, res];
        float[,] wlMask = new float[res, res];
        float[,] wlHeight = new float[res, res];
        float[,] terrainHeights = new float[res, res];

        // Pixel (10, 10): Both MCLQ and WL
        mclqPresence[10, 10] = true;
        mclqHeight[10, 10] = 50.0f;
        wlMask[10, 10] = 1.0f;
        wlHeight[10, 10] = 52.5f;

        // Pixel (10, 11): MCLQ only
        mclqPresence[10, 11] = true;
        mclqHeight[10, 11] = 50.0f;

        // Pixel (10, 12): WL only
        wlMask[10, 12] = 1.0f;
        wlHeight[10, 12] = 48.0f;

        // Terrain heights safely below water so no culling occurs
        for (int y = 0; y < res; y++)
        {
            for (int x = 0; x < res; x++)
                terrainHeights[y, x] = 10.0f;
        }

        LiquidConvergenceTileReport report = LiquidConvergenceAnalyzer.Analyze(
            30, 48, "Azeroth_30_48",
            mclqHeight, mclqPresence,
            wlMask, wlHeight,
            terrainHeights,
            resolution: res);

        Assert.True(report.HasMclq);
        Assert.True(report.HasWl);
        Assert.Equal(1, report.Populations.BothCount);
        Assert.Equal(1, report.Populations.MclqOnlyCount);
        Assert.Equal(1, report.Populations.WlOnlyCount);
        Assert.Equal(res * res - 3, report.Populations.NeitherCount);

        // Difference at (10,10): |50.0 - 52.5| = 2.5
        Assert.Equal(1, report.HeightDifference.Count);
        Assert.Equal(2.5f, report.HeightDifference.Mean, tolerance: 1e-4f);
        Assert.Equal(2.5f, report.HeightDifference.Min, tolerance: 1e-4f);
        Assert.Equal(2.5f, report.HeightDifference.Max, tolerance: 1e-4f);
        // 2.5 falls in bucket 4 (1.00 .. 5.00)
        Assert.Equal(1, report.HeightDifference.HistogramBuckets[4]);

        // Union property
        Assert.Equal(0, report.MissingFromUnifiedCount);
    }

    [Fact]
    public void DetectsMechanismA_PartiallyPresentQuads()
    {
        const int quadGridSize = 129;
        float[,] mclqHeight = new float[quadGridSize, quadGridSize];
        bool[,] mclqPresence = new bool[quadGridSize, quadGridSize];

        // Isolated Single-vertex at (10, 10):
        // Affects 4 quads around it: (9,9), (9,10), (10,9), (10,10).
        // Each has exactly 1 corner present, so all 4 are partially present!
        mclqPresence[10, 10] = true;
        mclqHeight[10, 10] = 20f;

        // Fully present isolated 2x2 vertices (1 quad) at (50, 50):
        // Corners: (50,50), (50,51), (51,50), (51,51) all true
        mclqPresence[50, 50] = true; mclqHeight[50, 50] = 20f;
        mclqPresence[50, 51] = true; mclqHeight[50, 51] = 20f;
        mclqPresence[51, 50] = true; mclqHeight[51, 50] = 20f;
        mclqPresence[51, 51] = true; mclqHeight[51, 51] = 20f;
        // Quad (50, 50) is fully present (all 4 corners true).
        // The 8 border quads surrounding it have 1 or 2 corners present (partially present).

        LiquidConvergenceTileReport report = LiquidConvergenceAnalyzer.Analyze(
            30, 48, "Azeroth_30_48",
            mclqHeight, mclqPresence,
            wlMaskRaw: null, wlHeightRaw: null,
            terrainHeights: null);

        // Quad (50, 50) is the only fully present quad
        Assert.Equal(1, report.MechanismA.FullyPresentQuadCount);
        // 4 partial quads around (10,10) + 8 partial quads around (50,50) = 12 partial quads
        Assert.Equal(12, report.MechanismA.PartiallyPresentQuadCount);
        Assert.True(report.MechanismA.AffectedUpsampledCellCount > 0);
    }

    [Fact]
    public void MeasuresMechanismB_TerrainCullingAtWaterline()
    {
        const int res = 257;
        float[,] wlMask = new float[res, res];
        float[,] wlHeight = new float[res, res];
        float[,] terrain = new float[res, res];
        bool[,] mclqPresence = new bool[res, res];
        float[,] mclqHeight = new float[res, res];

        // Cell (5, 5): Water 30.0, terrain 35.0 -> CULLED (terrain rises above water)
        // No MCLQ -> creates potential waterline gap!
        wlMask[5, 5] = 1.0f;
        wlHeight[5, 5] = 30.0f;
        terrain[5, 5] = 35.0f;

        // Cell (5, 6): Water 30.0, terrain 35.0 -> CULLED
        // Has MCLQ -> MCLQ covers the culled cell
        wlMask[5, 6] = 1.0f;
        wlHeight[5, 6] = 30.0f;
        terrain[5, 6] = 35.0f;
        mclqPresence[5, 6] = true;
        mclqHeight[5, 6] = 32.0f;

        // Cell (5, 7): Water 30.0, terrain 20.0 -> SURVIVES (above terrain)
        wlMask[5, 7] = 1.0f;
        wlHeight[5, 7] = 30.0f;
        terrain[5, 7] = 20.0f;

        LiquidConvergenceTileReport report = LiquidConvergenceAnalyzer.Analyze(
            30, 48, "Azeroth_30_48",
            mclqHeight, mclqPresence,
            wlMask, wlHeight,
            terrain,
            resolution: res);

        Assert.Equal(3, report.MechanismB.WlCellsBeforeTerrainCull);
        Assert.Equal(1, report.MechanismB.WlCellsAfterTerrainCull);
        Assert.Equal(2, report.MechanismB.WlCellsCulledByTerrain);
        Assert.Equal(1, report.MechanismB.CulledCellsWithoutMclq);
        Assert.Equal(1, report.MechanismB.CulledCellsWithMclq);
    }

    [Fact]
    public void HeightDifferenceDistribution_CalculatesQuartilesAndBuckets()
    {
        float[] diffs = [0.005f, 0.05f, 0.25f, 0.75f, 2.5f, 10.0f];
        HeightDifferenceDistribution dist = HeightDifferenceDistribution.Calculate(diffs);

        Assert.Equal(6, dist.Count);
        Assert.Equal(0.005f, dist.Min);
        Assert.Equal(10.0f, dist.Max);
        Assert.Equal(1, dist.HistogramBuckets[0]); // < 0.01
        Assert.Equal(1, dist.HistogramBuckets[1]); // 0.01 .. 0.10
        Assert.Equal(1, dist.HistogramBuckets[2]); // 0.10 .. 0.50
        Assert.Equal(1, dist.HistogramBuckets[3]); // 0.50 .. 1.00
        Assert.Equal(1, dist.HistogramBuckets[4]); // 1.00 .. 5.00
        Assert.Equal(1, dist.HistogramBuckets[5]); // >= 5.00
    }
}
