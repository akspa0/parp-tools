using System.Numerics;
using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests.World;

public sealed class CameraTileWindowSelectorTests
{
    private static readonly CameraTileWindowSelector Selector =
        new(mapOrigin: 100f, tileSize: 10f, mapEdge: 64);

    [Fact]
    public void RadiusTwoReturnsBoundedCameraCenteredWindow()
    {
        List<DirectionalTileCoord> tiles = Selector.GetTiles(CameraAt(32, 32), radius: 2);

        Assert.Equal(25, tiles.Count);
        Assert.Equal(new DirectionalTileCoord(32, 32), tiles[0]);
        Assert.All(tiles, tile =>
        {
            Assert.InRange(Math.Abs(tile.TileX - 32), 0, 2);
            Assert.InRange(Math.Abs(tile.TileY - 32), 0, 2);
        });
    }

    [Fact]
    public void MissingTilesAreExcludedWithoutExpandingWindow()
    {
        var selector = new CameraTileWindowSelector(
            mapOrigin: 100f,
            tileSize: 10f,
            mapEdge: 64,
            tileExists: (tileX, tileY) => tileX != 31 || tileY != 31);

        List<DirectionalTileCoord> tiles = selector.GetTiles(CameraAt(32, 32), radius: 1);

        Assert.Equal(8, tiles.Count);
        Assert.DoesNotContain(new DirectionalTileCoord(31, 31), tiles);
        Assert.All(tiles, tile =>
        {
            Assert.InRange(Math.Abs(tile.TileX - 32), 0, 1);
            Assert.InRange(Math.Abs(tile.TileY - 32), 0, 1);
        });
    }

    [Fact]
    public void RadiusZeroKeepsOnlyActiveTile()
    {
        List<DirectionalTileCoord> tiles = Selector.GetTiles(CameraAt(32, 32), radius: 0);

        Assert.Equal(new[] { new DirectionalTileCoord(32, 32) }, tiles);
    }

    private static Vector3 CameraAt(int tileX, int tileY)
        => new(
            100f - ((tileX + 0.5f) * 10f),
            100f - ((tileY + 0.5f) * 10f),
            0f);
}
