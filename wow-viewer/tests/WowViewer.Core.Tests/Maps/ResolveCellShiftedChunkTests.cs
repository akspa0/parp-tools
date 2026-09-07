using WowViewer.Core.Maps;
using Xunit;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Spec 232 FR-1: the cell fine-tune moves the layer as a RIGID map object — the composed chunk
/// of a target tile shows the layer's composed content from the supplying target tile shifted by
/// the cell offset, confined to the 64x64 grid.
/// </summary>
public sealed class ResolveCellShiftedChunkTests
{
    private static bool AllTilesExist(int _, int __) => true;

    [Fact]
    public void ZeroOffset_ReturnsTheSameTileAndChunk()
    {
        PhaseLayerSettings layer = new() { MapName = "Test" };

        (bool has, int tileX, int tileY, int chunkX, int chunkY) = Resolve(layer, 32, 32, 5, 7);

        Assert.True(has);
        Assert.Equal((32, 32, 5, 7), (tileX, tileY, chunkX, chunkY));
    }

    [Fact]
    public void CellShift_SlidesIntoTheNeighborTargetTile()
    {
        PhaseLayerSettings layer = new() { MapName = "Test", CellOffsetX = 1 };

        // Composed chunk (0,5) of target tile (32,32) shows global cell (512−1, ...) — the last
        // chunk of the neighboring target tile (31,32): a rigid slide, chunk content untouched.
        (bool has, int tileX, int tileY, int chunkX, int chunkY) = Resolve(layer, 32, 32, 0, 5);

        Assert.True(has);
        Assert.Equal((31, 32, 15, 5), (tileX, tileY, chunkX, chunkY));
    }

    [Fact]
    public void CellShift_ComposesWithTileOffset_AtBothLevels()
    {
        PhaseLayerSettings layer = new() { MapName = "Test", TileOffsetX = 1, CellOffsetX = 1 };

        // Rigid slide: composed chunk (0,5) of target 32 shows supplying target tile
        // (32 − 1 tile − 1 cell = 31), last chunk.
        (bool has, int tileX, int tileY, int chunkX, int chunkY) = Resolve(layer, 32, 32, 0, 5);

        Assert.True(has);
        Assert.Equal((31, 32, 15, 5), (tileX, tileY, chunkX, chunkY));

        // And the supplying target tile's own donor tile honors the tile offset: (31 − 1, 32).
        PhaseTileSource donor = PhaseCompositionPolicy.ResolveTileSource(layer, tileX, tileY, AllTilesExist);
        Assert.True(donor.HasSource);
        Assert.Equal((30, 32), (donor.SourceTileX, donor.SourceTileY));
    }

    [Fact]
    public void NegativeCellShift_AtTheFarEdge_GoesPastTheSupplyingTile()
    {
        PhaseLayerSettings layer = new() { MapName = "Test", CellOffsetX = -1 };

        // Composed chunk 15 with a −1 slide shows global cell (target·16 + 15 + 1) — first chunk
        // of the next target tile over.
        (bool has, int tileX, int tileY, int chunkX, int _) = Resolve(layer, 32, 32, 15, 5);

        Assert.True(has);
        Assert.Equal((33, 32), (tileX, tileY));
        Assert.Equal(0, chunkX);
    }

    [Fact]
    public void OutsideTheGrid_ContributesNothing()
    {
        PhaseLayerSettings layer = new() { MapName = "Test", CellOffsetX = 1 };

        // Composed chunk 0 of target tile 0 slides back to global −1: outside the 64x64 grid.
        Assert.False(Resolve(layer, 0, 0, 0, 5).HasSource);
    }

    private static (bool HasSource, int TileX, int TileY, int ChunkX, int ChunkY) Resolve(
        PhaseLayerSettings layer, int tileX, int tileY, int chunkX, int chunkY)
        => PhaseCompositionPolicy.ResolveCellShiftedChunk(layer, tileX, tileY, chunkX, chunkY);
}
