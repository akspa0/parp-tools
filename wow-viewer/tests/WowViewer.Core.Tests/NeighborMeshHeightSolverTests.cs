using WowViewer.Core.Runtime.World.Terrain.Stratigraphy;
using Xunit;

namespace WowViewer.Core.Tests;

public class NeighborMeshHeightSolverTests
{
    [Fact]
    public void ExtractEdge145_North_ExtractsFirstRow()
    {
        var heights = new float[145];
        for (int i = 0; i < 9; i++)
            heights[i] = 100f + i;

        Span<float> edge = stackalloc float[9];
        NeighborMeshHeightSolver.ExtractEdge145(heights, NeighborMeshHeightSolver.Direction.North, edge);

        for (int i = 0; i < 9; i++)
            Assert.Equal(100f + i, edge[i]);
    }

    [Fact]
    public void ExtractEdge145_South_ExtractsLastOuterRow()
    {
        var heights = new float[145];
        for (int i = 0; i < 9; i++)
            heights[136 + i] = 50f + i;

        Span<float> edge = stackalloc float[9];
        NeighborMeshHeightSolver.ExtractEdge145(heights, NeighborMeshHeightSolver.Direction.South, edge);

        for (int i = 0; i < 9; i++)
            Assert.Equal(50f + i, edge[i]);
    }

    [Fact]
    public void SolveFromBoundaryPairs_Exact33xScaling_RecoversFactorAndZeroRmse()
    {
        var pairs = new List<BoundaryVertexPair>();
        float scale = 33.334f;
        float baseFloor = 10f;
        float offset = 50f;

        for (int i = 0; i < 9; i++)
        {
            float uncompressed = 100f + (i * 5f);
            float compressed = baseFloor + ((uncompressed - baseFloor - offset) / scale);
            pairs.Add(new BoundaryVertexPair(uncompressed, compressed));
        }

        var result = NeighborMeshHeightSolver.SolveFromBoundaryPairs(pairs);

        Assert.True(result.FoundNeighbor);
        Assert.Equal(33.334f, result.BestFactor, 2);
        Assert.False(result.BestPolarityInverted);
        Assert.True(result.ResidualRmseMeters < 0.05f);
    }

    [Fact]
    public void SolveFromBoundaryPairs_InvertedScaling_RecoversPolarityInversion()
    {
        var pairs = new List<BoundaryVertexPair>();
        float scale = 16.0f;
        float ceiling = 200f;
        float offset = 20f;

        for (int i = 0; i < 9; i++)
        {
            float uncompressed = 100f + (i * 4f);
            float compressed = ceiling - ((uncompressed - ceiling - offset) / scale);
            pairs.Add(new BoundaryVertexPair(uncompressed, compressed));
        }

        var result = NeighborMeshHeightSolver.SolveFromBoundaryPairs(pairs);

        Assert.True(result.FoundNeighbor);
        Assert.Equal(16.0f, result.BestFactor, 2);
        Assert.True(result.BestPolarityInverted);
        Assert.True(result.ResidualRmseMeters < 0.05f);
    }
}
