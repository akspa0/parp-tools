using WowViewer.Core.Maps;
using Xunit;

namespace WowViewer.Core.Tests.Maps;

/// <summary>
/// Spec 232 FR-1: chunk-granularity composition resolution. Locks the invariants that make the
/// cell-level fine-tune trustworthy: consistency with the discrete tile map at zero cell offset,
/// exact whole-cell re-indexing, tile-offset composition, and grid confinement.
/// </summary>
public sealed class ResolveChunkSourceTests
{
    private static bool AllTilesExist(int _, int __) => true;

    [Fact]
    public void ZeroTransform_ReturnsSameTileAndChunk()
    {
        PhaseLayerSettings layer = new() { MapName = "Test" };

        (bool has, int tileX, int tileY, int chunkX, int chunkY) = PhaseCompositionPolicy.ResolveChunkSource(
            layer, 32, 32, 5, 7, AllTilesExist);

        Assert.True(has);
        Assert.Equal((32, 32, 5, 7), (tileX, tileY, chunkX, chunkY));
    }

    [Fact]
    public void CellShift_ReindexesWithinTheTile()
    {
        PhaseLayerSettings layer = new() { MapName = "Test", CellOffsetX = 1 };

        (bool has, int tileX, int tileY, int chunkX, int chunkY) = PhaseCompositionPolicy.ResolveChunkSource(
            layer, 32, 32, 5, 5, AllTilesExist);

        Assert.True(has);
        // Composed chunk (5,5) shows donor chunk (4,5): a +1 cell shift moves content +1 in X.
        Assert.Equal((32, 32, 4, 5), (tileX, tileY, chunkX, chunkY));
    }

    [Fact]
    public void CellShift_PullsFromTheNeighboringTileAtTheBorder()
    {
        PhaseLayerSettings layer = new() { MapName = "Test", CellOffsetX = 1 };

        // Target chunk 0 with a +1 X cell shift must read the last chunk of the neighboring
        // donor tile — the cross-tile pull is the whole point of the cell fine-tune.
        (bool has, int tileX, int tileY, int chunkX, int chunkY) = PhaseCompositionPolicy.ResolveChunkSource(
            layer, 32, 32, 0, 5, AllTilesExist);

        Assert.True(has);
        Assert.Equal((31, 32, 15, 5), (tileX, tileY, chunkX, chunkY));
    }

    [Fact]
    public void CellShift_ComposesWithTileOffset()
    {
        PhaseLayerSettings layer = new() { MapName = "Test", TileOffsetX = 1, CellOffsetX = 1 };

        (bool has, int tileX, int _, int chunkX, int _) = PhaseCompositionPolicy.ResolveChunkSource(
            layer, 32, 32, 0, 5, AllTilesExist);

        Assert.True(has);
        // Total X shift = 1 tile + 1 cell: donor tile 32 − 2 = 30, last chunk.
        Assert.Equal((30, 32, 15, 5), (tileX, 32, chunkX, 5));
    }

    [Fact]
    public void RotationOnly_AgreesWithTheDiscreteTileMap()
    {
        // The pinned InverseTransformTile case (PhaseTileSourceTests): origin (10,20), 90° CW —
        // target tile (10,21) reads donor tile (9,20). The chunk-level map must land in that
        // same donor tile for every chunk of the target tile.
        PhaseLayerSettings layer = new()
        {
            MapName = "Test",
            RotationDegrees = 90f,
            RotationOriginTileX = 10f,
            RotationOriginTileY = 20f,
        };

        for (int cx = 0; cx < 16; cx++)
        {
            for (int cy = 0; cy < 16; cy++)
            {
                (bool has, int tileX, int tileY, _, _) = PhaseCompositionPolicy.ResolveChunkSource(
                    layer, 10, 21, cx, cy, AllTilesExist);
                Assert.True(has);
                Assert.Equal((9, 20), (tileX, tileY));
            }
        }
    }

    [Fact]
    public void OutsideTheGrid_ReturnsNoSource()
    {
        PhaseLayerSettings layer = new() { MapName = "Test" };

        Assert.False(PhaseCompositionPolicy.ResolveChunkSource(layer, 64, 0, 0, 0, AllTilesExist).HasSource);
        Assert.False(PhaseCompositionPolicy.ResolveChunkSource(layer, 0, -1, 0, 0, AllTilesExist).HasSource);
    }

    [Fact]
    public void ShiftedContentOutsideTheGrid_ReturnsNoSource()
    {
        // A +1 cell shift at donor tile edge 0 reads chunk −1 of tile 31 — out of grid: empty.
        PhaseLayerSettings layer = new() { MapName = "Test", CellOffsetX = 1 };

        (bool has, _, _, _, _) = PhaseCompositionPolicy.ResolveChunkSource(
            layer, 32, 32, 0, 5, static (sx, _) => sx != 31);

        Assert.False(has);
    }
}
