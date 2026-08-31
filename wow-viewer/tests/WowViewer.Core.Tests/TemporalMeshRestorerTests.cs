using System.Numerics;
using WowViewer.Core.Runtime.World.Terrain.Stratigraphy;
using Xunit;

namespace WowViewer.Core.Tests;

public sealed class TemporalMeshRestorerTests
{
    [Fact]
    public void RestoreLattice_ClassicFactor_MultipliesHeightReliefProportionally()
    {
        float[,] source = new float[257, 257];
        for (int y = 0; y < 257; y++)
            for (int x = 0; x < 257; x++)
                source[y, x] = x * 0.001f; // 0 to 0.256m

        float[,] restored = TemporalMeshRestorer.RestoreLattice(source, factor: 33.334f);

        Assert.Equal(0f, restored[0, 0], precision: 4);
        Assert.Equal(0.256f * 33.334f, restored[0, 256], precision: 2);
    }

    [Fact]
    public void RestoreLattice_PreservesNegativeFloor_WhenBelowZero()
    {
        float[,] source = new float[257, 257];
        for (int y = 0; y < 257; y++)
            for (int x = 0; x < 257; x++)
                source[y, x] = -100f + (x * 0.001f); // -100m to -99.744m

        var opts = new TemporalStratigraphyOptions { PreserveNegativeFloor = true };
        float[,] restored = TemporalMeshRestorer.RestoreLattice(source, factor: 10f, options: opts);

        Assert.Equal(-100f, restored[0, 0], precision: 3);
        Assert.True(restored[0, 256] > -100f);
        Assert.Equal(-100f + (0.256f * 10f), restored[0, 256], precision: 2);
    }

    [Fact]
    public void SeamDiscontinuityProfiler_2x2MergeSpikeAt8_DetectedCorrectly()
    {
        float[,] lattice = new float[257, 257];
        // Inject a distinct step discontinuity at chunk boundary 8 (index 128)
        for (int y = 0; y < 257; y++)
        {
            for (int x = 0; x < 128; x++)
                lattice[y, x] = 10f;
            for (int x = 128; x < 257; x++)
                lattice[y, x] = 50f;
        }

        SeamDiscontinuityResult result = SeamDiscontinuityProfiler.AnalyzeSeams(lattice);
        Assert.True(result.Has2x2MergeSpikeAt8);
        Assert.Contains("2x2 Sub-Tile Merge", result.InferredMergeOrigin);
    }

    [Fact]
    public void FastTerrainNormalSolver_FlatPlane_ReturnsUnitUpNormals()
    {
        float[] flatHeights = new float[145];
        Array.Fill(flatHeights, 20f);

        Vector3[] normals = FastTerrainNormalSolver.ComputeChunkNormals(flatHeights);

        Assert.Equal(145, normals.Length);
        foreach (var n in normals)
        {
            Assert.True(Vector3.Dot(n, Vector3.UnitZ) > 0.999f);
        }
    }

    [Fact]
    public void BuildBoundaryFeatherWeights_BlendsEdgesTowardsZero()
    {
        float[] weights = TemporalMeshRestorer.BuildBoundaryFeatherWeights(blendNorth: true, blendSouth: false, blendWest: true, blendEast: false);

        // Top-left corner (0,0) should be 0.0
        Assert.Equal(0f, weights[0], precision: 3);

        // Center / South-East should remain 1.0
        int southEastOuterIdx = 8 * 17 + 8;
        Assert.Equal(1f, weights[southEastOuterIdx], precision: 3);
    }
}
