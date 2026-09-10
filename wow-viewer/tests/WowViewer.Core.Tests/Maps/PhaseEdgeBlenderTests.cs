using WowViewer.Core.Runtime.World.Terrain.Stratigraphy;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Spec 232 T064: magnetic WDL edge-snap. Only footprint-boundary outer vertices blend toward
/// the base map's WDL macro lattice; interior vertices and edges shared with a contributing
/// neighbor target tile stay untouched.
/// </summary>
public sealed class PhaseEdgeBlenderTests
{
    [Fact]
    public void BlendTileBoundary_PullsWestEdgeTowardTheWdlLattice()
    {
        float[] heights = CreateFlatHeights(100f);
        short[,] wdl = CreateFlatLattice(50);

        bool modified = PhaseEdgeBlender.BlendTileBoundary(
            heights, wdl, chunkX: 0, chunkY: 5, strength: 1f,
            neighborContributes: static (_, _) => false);

        Assert.True(modified);
        // West edge outer vertices (col 0) blend fully to the WDL value.
        Assert.Equal(50f, heights[0 * 17 + 0]);
        Assert.Equal(50f, heights[8 * 17 + 0]);
        // Interior column and east edge stay at the source height.
        Assert.Equal(100f, heights[0 * 17 + 4]);
        Assert.Equal(100f, heights[0 * 17 + 8]);
        // Inner vertices are never boundary vertices.
        Assert.Equal(100f, heights[9]); // first inner vertex (row 1, col 0 of inner block)
    }

    [Fact]
    public void BlendTileBoundary_SkipsEdgesSharedWithAContributingNeighbor()
    {
        float[] heights = CreateFlatHeights(100f);
        short[,] wdl = CreateFlatLattice(50);

        // The west neighbor also receives this layer's heights: that edge is an interior seam.
        PhaseEdgeBlender.BlendTileBoundary(
            heights, wdl, chunkX: 0, chunkY: 5, strength: 1f,
            neighborContributes: (dx, _) => dx == -1);

        Assert.Equal(100f, heights[0]);
        Assert.Equal(100f, heights[8 * 17]);
    }

    [Fact]
    public void BlendTileBoundary_ZeroStrengthIsANoOp()
    {
        float[] heights = CreateFlatHeights(100f);
        short[,] wdl = CreateFlatLattice(50);

        bool modified = PhaseEdgeBlender.BlendTileBoundary(
            heights, wdl, chunkX: 0, chunkY: 0, strength: 0f,
            neighborContributes: static (_, _) => false);

        Assert.False(modified);
        Assert.Equal(100f, heights[0]);
    }

    [Fact]
    public void BlendTileBoundary_InteriorChunksAreUntouched()
    {
        float[] heights = CreateFlatHeights(100f);
        short[,] wdl = CreateFlatLattice(50);

        bool modified = PhaseEdgeBlender.BlendTileBoundary(
            heights, wdl, chunkX: 5, chunkY: 5, strength: 1f,
            neighborContributes: static (_, _) => false);

        Assert.False(modified);
        Assert.All(heights, h => Assert.Equal(100f, h));
    }

    private static float[] CreateFlatHeights(float value)
    {
        var heights = new float[145];
        Array.Fill(heights, value);
        return heights;
    }

    private static short[,] CreateFlatLattice(short value)
    {
        var lattice = new short[17, 17];
        for (int y = 0; y < 17; y++)
            for (int x = 0; x < 17; x++)
                lattice[y, x] = value;
        return lattice;
    }
}
