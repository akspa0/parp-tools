using System.Numerics;
using WowViewer.Core.Runtime.World;

namespace WowViewer.Core.Tests.World;

public sealed class DirectionalTileSelectorTests
{
    private static readonly DirectionalTileSelector Selector =
        new(mapOrigin: 100f, tileSize: 10f, mapEdge: 64);

    [Fact]
    public void CardinalHeadingReturnsOnlyActiveAndThreeForwardNeighbors()
    {
        List<DirectionalTileCoord> visible = Selector.GetVisibleTiles(CameraAt(32, 32), yaw: 0f, fovDegrees: 45f);

        Assert.Equal(4, visible.Count);
        Assert.Equal(new DirectionalTileCoord(32, 32), visible[0]);
        Assert.Contains(new DirectionalTileCoord(31, 32), visible);
        Assert.Contains(new DirectionalTileCoord(31, 31), visible);
        Assert.Contains(new DirectionalTileCoord(31, 33), visible);
        Assert.All(visible, tile => Assert.InRange(Math.Abs(tile.TileX - 32), 0, 1));
        Assert.All(visible, tile => Assert.InRange(Math.Abs(tile.TileY - 32), 0, 1));
    }

    [Fact]
    public void BackwardHeadingDoesNotSelectForwardOrRadialTiles()
    {
        List<DirectionalTileCoord> visible = Selector.GetVisibleTiles(CameraAt(32, 32), yaw: 180f, fovDegrees: 45f);

        Assert.Equal(4, visible.Count);
        Assert.Contains(new DirectionalTileCoord(33, 32), visible);
        Assert.Contains(new DirectionalTileCoord(33, 31), visible);
        Assert.Contains(new DirectionalTileCoord(33, 33), visible);
        Assert.DoesNotContain(new DirectionalTileCoord(31, 32), visible);
        Assert.All(visible, tile => Assert.InRange(Math.Abs(tile.TileX - 32), 0, 1));
        Assert.All(visible, tile => Assert.InRange(Math.Abs(tile.TileY - 32), 0, 1));
    }

    [Fact]
    public void MissingTilesAreExcludedWithoutExpandingTheCandidatePool()
    {
        var selector = new DirectionalTileSelector(
            mapOrigin: 100f,
            tileSize: 10f,
            mapEdge: 64,
            tileExists: (tileX, tileY) => tileX != 31 || tileY != 31);

        List<DirectionalTileCoord> visible = selector.GetVisibleTiles(CameraAt(32, 32), yaw: 0f, fovDegrees: 45f);

        Assert.Equal(3, visible.Count);
        Assert.DoesNotContain(new DirectionalTileCoord(31, 31), visible);
        Assert.DoesNotContain(visible, tile => Math.Abs(tile.TileX - 32) > 1 || Math.Abs(tile.TileY - 32) > 1);
    }

    [Fact]
    public void ZeroConeKeepsOnlyTheActiveTile()
    {
        List<DirectionalTileCoord> visible = Selector.GetVisibleTiles(CameraAt(32, 32), yaw: 90f, fovDegrees: 0f);

        Assert.Equal(new[] { new DirectionalTileCoord(32, 32) }, visible);
    }

    [Fact]
    public void RequestedDetailCountExpandsAcrossForwardRings()
    {
        List<DirectionalTileCoord> visible = Selector.GetVisibleTiles(
            CameraAt(32, 32),
            yaw: 0f,
            fovDegrees: 45f,
            maxTileCount: 9);

        Assert.Equal(9, visible.Count);
        Assert.Contains(new DirectionalTileCoord(30, 32), visible);
        Assert.Contains(new DirectionalTileCoord(30, 31), visible);
        Assert.Contains(new DirectionalTileCoord(30, 33), visible);
        Assert.DoesNotContain(visible, tile => tile.TileX > 32);
    }

    [Fact]
    public void RequestedDetailCountCanExceedTheLegacyFourTileBaseline()
    {
        List<DirectionalTileCoord> visible = Selector.GetVisibleTiles(
            CameraAt(32, 32),
            yaw: 0f,
            fovDegrees: 45f,
            maxTileCount: 25);

        Assert.Equal(25, visible.Count);
        Assert.Contains(new DirectionalTileCoord(28, 32), visible);
        Assert.All(visible, tile => Assert.InRange(tile.TileX, 28, 32));
    }

    [Fact]
    public void CameraTileUsesTheConfiguredAdtSpan()
    {
        var selector = new DirectionalTileSelector(
            mapOrigin: 17066.666f,
            tileSize: 533.333f,
            mapEdge: 64);

        List<DirectionalTileCoord> visible = selector.GetVisibleTiles(
            new Vector3(
                17066.666f - (32.5f * 533.333f),
                17066.666f - (31.5f * 533.333f),
                0f),
            yaw: 0f,
            fovDegrees: 45f,
            maxTileCount: 1);

        Assert.Equal(new DirectionalTileCoord(32, 31), Assert.Single(visible));
    }

    private static Vector3 CameraAt(int tileX, int tileY)
        => new(
            100f - ((tileX + 0.5f) * 10f),
            100f - ((tileY + 0.5f) * 10f),
            0f);
}
