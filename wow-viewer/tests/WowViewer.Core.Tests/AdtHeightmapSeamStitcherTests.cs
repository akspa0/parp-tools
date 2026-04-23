using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtHeightmapSeamStitcherTests
{
    private const int TileHeightmapSize = AdtHeightmapSeamStitcher.TileHeightmapSize;

    [Fact]
    public void StitchSharedEdges_AveragesSharedVerticalBorder()
    {
        float[] left = CreateHeightmap(10f);
        float[] right = CreateHeightmap(30f);
        left[(100 * TileHeightmapSize) + (TileHeightmapSize - 1)] = 14f;
        right[100 * TileHeightmapSize] = 34f;
        left[(42 * TileHeightmapSize) + 42] = 123f;
        right[(42 * TileHeightmapSize) + 42] = 456f;

        AdtHeightmapSeamStitcher.StitchSharedEdges(new Dictionary<(int TileX, int TileY), float[]>
        {
            [(0, 0)] = left,
            [(1, 0)] = right,
        });

        Assert.Equal(24f, left[(100 * TileHeightmapSize) + (TileHeightmapSize - 1)]);
        Assert.Equal(24f, right[100 * TileHeightmapSize]);
        Assert.Equal(123f, left[(42 * TileHeightmapSize) + 42]);
        Assert.Equal(456f, right[(42 * TileHeightmapSize) + 42]);
    }

    [Fact]
    public void StitchSharedEdges_ReconcilesFourTileCorner()
    {
        float[] topLeft = CreateHeightmap(0f);
        float[] topRight = CreateHeightmap(0f);
        float[] bottomLeft = CreateHeightmap(0f);
        float[] bottomRight = CreateHeightmap(0f);

        topLeft[((TileHeightmapSize - 1) * TileHeightmapSize) + (TileHeightmapSize - 1)] = 4f;
        topRight[(TileHeightmapSize - 1) * TileHeightmapSize] = 8f;
        bottomLeft[TileHeightmapSize - 1] = 12f;
        bottomRight[0] = 16f;

        AdtHeightmapSeamStitcher.StitchSharedEdges(new Dictionary<(int TileX, int TileY), float[]>
        {
            [(0, 0)] = topLeft,
            [(1, 0)] = topRight,
            [(0, 1)] = bottomLeft,
            [(1, 1)] = bottomRight,
        });

        Assert.Equal(10f, topLeft[((TileHeightmapSize - 1) * TileHeightmapSize) + (TileHeightmapSize - 1)]);
        Assert.Equal(10f, topRight[(TileHeightmapSize - 1) * TileHeightmapSize]);
        Assert.Equal(10f, bottomLeft[TileHeightmapSize - 1]);
        Assert.Equal(10f, bottomRight[0]);
    }

    [Fact]
    public void AnchorPredictedEdgesToNeighbors_CopiesUnchangedNeighborBorder()
    {
        float[] predicted = CreateHeightmap(0f);
        float[] rightNeighbor = CreateHeightmap(0f);
        rightNeighbor[100 * TileHeightmapSize] = 77f;
        predicted[(42 * TileHeightmapSize) + 42] = 5f;

        AdtHeightmapSeamStitcher.AnchorPredictedEdgesToNeighbors(
            new Dictionary<(int TileX, int TileY), float[]>
            {
                [(0, 0)] = predicted,
            },
            new Dictionary<(int TileX, int TileY), float[]>
            {
                [(1, 0)] = rightNeighbor,
            });

        Assert.Equal(77f, predicted[(100 * TileHeightmapSize) + (TileHeightmapSize - 1)]);
        Assert.Equal(5f, predicted[(42 * TileHeightmapSize) + 42]);
    }

    [Fact]
    public void AnchorPredictedEdgesToNeighbors_ReconcilesPredictedCornerToAnchorAverage()
    {
        float[] predicted = CreateHeightmap(0f);
        float[] topNeighbor = CreateHeightmap(0f);
        float[] leftNeighbor = CreateHeightmap(0f);
        float[] diagonalNeighbor = CreateHeightmap(0f);

        topNeighbor[(TileHeightmapSize - 1) * TileHeightmapSize] = 6f;
        leftNeighbor[TileHeightmapSize - 1] = 12f;
        diagonalNeighbor[((TileHeightmapSize - 1) * TileHeightmapSize) + (TileHeightmapSize - 1)] = 18f;

        AdtHeightmapSeamStitcher.AnchorPredictedEdgesToNeighbors(
            new Dictionary<(int TileX, int TileY), float[]>
            {
                [(0, 0)] = predicted,
            },
            new Dictionary<(int TileX, int TileY), float[]>
            {
                [(0, -1)] = topNeighbor,
                [(-1, 0)] = leftNeighbor,
                [(-1, -1)] = diagonalNeighbor,
            });

        Assert.Equal(12f, predicted[0]);
    }

    private static float[] CreateHeightmap(float value)
    {
        float[] heightmap = new float[TileHeightmapSize * TileHeightmapSize];
        Array.Fill(heightmap, value);
        return heightmap;
    }
}