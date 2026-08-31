using WowViewer.Core.Editor.Operations;
using Xunit;

namespace WowViewer.Core.Editor.Tests.Operations;

public sealed class ChunkSelectionRegionTests
{
    [Fact]
    public void GlobalChunkCoordinate_Conversions_AreAccurate()
    {
        // Tile 32, 48, Chunk 7, 11
        var coord = GlobalChunkCoordinate.FromTileAndChunk(32, 48, 7, 11);
        Assert.Equal(32 * 16 + 7, coord.Gx);
        Assert.Equal(48 * 16 + 11, coord.Gy);
        Assert.Equal(32, coord.TileX);
        Assert.Equal(48, coord.TileY);
        Assert.Equal(7, coord.ChunkX);
        Assert.Equal(11, coord.ChunkY);
        Assert.True(coord.IsValid);
    }

    [Fact]
    public void GlobalChunkCoordinate_FromWorldPosition_MatchesTileMetric()
    {
        // World origin (0, 0) is at Tile 32, Chunk 0
        float worldOriginX = 0f;
        float worldOriginY = 0f;
        var coord = GlobalChunkCoordinate.FromWorldPosition(worldOriginX, worldOriginY);
        Assert.Equal(32, coord.TileX);
        Assert.Equal(32, coord.TileY);
        Assert.Equal(0, coord.ChunkX);
        Assert.Equal(0, coord.ChunkY);
    }

    [Fact]
    public void ChunkSelectionRegion_AddRectangle_SpansMultipleTiles()
    {
        var region = new ChunkSelectionRegion();
        // Spans from Tile 31, Chunk 14 -> Tile 32, Chunk 2 (a 5x5 chunk box across 4 tiles)
        var c1 = GlobalChunkCoordinate.FromTileAndChunk(31, 31, 14, 14);
        var c2 = GlobalChunkCoordinate.FromTileAndChunk(32, 32, 2, 2);

        region.AddRectangle(c1, c2);

        Assert.Equal(25, region.Count);
        Assert.True(region.TryGetBoundingBox(out var min, out var max));
        Assert.Equal(c1.Gx, min.Gx);
        Assert.Equal(c2.Gx, max.Gx);

        var affectedTiles = region.GetAffectedTiles();
        Assert.Equal(4, affectedTiles.Count);
        Assert.Contains((31, 31), affectedTiles);
        Assert.Contains((31, 32), affectedTiles);
        Assert.Contains((32, 31), affectedTiles);
        Assert.Contains((32, 32), affectedTiles);
    }

    [Fact]
    public void ChunkSelectionRegion_AddTile_AddsAll256Chunks()
    {
        var region = new ChunkSelectionRegion();
        region.AddTile(10, 20);

        Assert.Equal(256, region.Count);
        Assert.True(region.Contains(10, 20, 0, 0));
        Assert.True(region.Contains(10, 20, 15, 15));
        Assert.False(region.Contains(10, 21, 0, 0));

        var tiles = region.GetAffectedTiles();
        var single = Assert.Single(tiles);
        Assert.Equal((10, 20), single);

        region.RemoveTile(10, 20);
        Assert.Equal(0, region.Count);
        Assert.True(region.IsEmpty);
    }
}
