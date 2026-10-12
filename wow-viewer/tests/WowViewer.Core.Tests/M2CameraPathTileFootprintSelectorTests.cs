using System.Numerics;
using WowViewer.Core.Runtime.M2;

namespace WowViewer.Core.Tests;

public sealed class M2CameraPathTileFootprintSelectorTests
{
    private const float MapOrigin = 17066.66666f;
    private const float WorldMapTileSize = 533.33333f;

    [Fact]
    public void SweptPathIncludesTilesBetweenSparseSamples()
    {
        M2CameraPathDocument path = new()
        {
            Interpolation = M2CameraPathInterpolation.Linear,
            Keyframes =
            [
                new() { TimeMs = 0, Position = WorldPosition(32, 32), Target = Vector3.UnitX },
                new() { TimeMs = 1000, Position = WorldPosition(30, 32), Target = Vector3.UnitX },
            ],
        };

        HashSet<(int tileX, int tileY)> tiles = CameraPathTileFootprintSelector.GetTiles(
            path, MapOrigin, WorldMapTileSize, 64, sampleSpacingMs: 1000, tileRadius: 0);

        Assert.Contains((32, 32), tiles);
        Assert.Contains((31, 32), tiles);
        Assert.Contains((30, 32), tiles);
    }

    [Fact]
    public void RadiusExpandsEachSweptTileAndExcludesMissingTiles()
    {
        M2CameraPathDocument path = new()
        {
            Interpolation = M2CameraPathInterpolation.Linear,
            Keyframes =
            [
                new() { TimeMs = 0, Position = WorldPosition(20, 20), Target = Vector3.UnitX },
                new() { TimeMs = 100, Position = WorldPosition(20, 20), Target = Vector3.UnitX },
            ],
        };

        HashSet<(int tileX, int tileY)> tiles = CameraPathTileFootprintSelector.GetTiles(
            path,
            MapOrigin,
            WorldMapTileSize,
            64,
            sampleSpacingMs: 1000,
            tileRadius: 1,
            tileExists: static (tileX, tileY) => (tileX, tileY) != (19, 19));

        Assert.Contains((20, 20), tiles);
        Assert.Contains((19, 20), tiles);
        Assert.DoesNotContain((19, 19), tiles);
        Assert.Equal(8, tiles.Count);
    }

    [Fact]
    public void FlybyUndeadOriginSelectsItsAvailableAzerothTile()
    {
        M2CameraPathDocument path = new()
        {
            Keyframes =
            [
                new()
                {
                    TimeMs = 0,
                    Position = new Vector3(1658.58f, 1662.91f, 141.234f),
                    Target = Vector3.UnitX,
                },
            ],
        };

        HashSet<(int tileX, int tileY)> tiles = CameraPathTileFootprintSelector.GetTiles(
            path,
            MapOrigin,
            WorldMapTileSize,
            64,
            sampleSpacingMs: 1000,
            tileRadius: 0,
            tileExists: static (tileX, tileY) => tileX == 28 && tileY == 28);

        Assert.Contains((28, 28), tiles);
        Assert.Single(tiles);
    }

    [Fact]
    public void TileLocalFlybyPathIsPlacedIntoItsMapTileBeforeWarmup()
    {
        M2CameraPathDocument path = new()
        {
            CoordinatesAreWorldSpace = false,
            Keyframes =
            [
                new() { TimeMs = 0, Position = Vector3.Zero, Target = Vector3.UnitX },
            ],
        };

        M2CameraPathPlacement.ApplyCinematicCameraOrigin(
            path,
            cameraId: 2,
            modelPath: "Cameras\\FlybyUndead.M2",
            origin: new Vector3(1658.58f, 1662.91f, 141.234f),
            facingRadians: 0f,
            tileX: 28,
            tileY: 28);

        HashSet<(int tileX, int tileY)> tiles = CameraPathTileFootprintSelector.GetTiles(
            path,
            MapOrigin,
            WorldMapTileSize,
            64,
            sampleSpacingMs: 1000,
            tileRadius: 0,
            tileExists: static (tileX, tileY) => tileX == 28 && tileY == 28);

        Assert.True(path.CoordinatesAreWorldSpace);
        Assert.Equal(new Vector3(1658.58f, 1662.91f, 141.234f), path.Keyframes[0].Position);
        Assert.Contains((28, 28), tiles);
        Assert.Single(tiles);
    }

    private static Vector3 WorldPosition(int tileX, int tileY)
        => new(
            MapOrigin - ((tileX + 0.5f) * WorldMapTileSize),
            MapOrigin - ((tileY + 0.5f) * WorldMapTileSize),
            0f);
}
