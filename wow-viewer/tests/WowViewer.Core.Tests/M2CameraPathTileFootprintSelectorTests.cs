using System.Numerics;
using WowViewer.Core.Runtime.M2;

namespace WowViewer.Core.Tests;

public sealed class M2CameraPathTileFootprintSelectorTests
{
    private const float MapOrigin = 17066.66666f;
    private const float TileSize = 8533.33333f;

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
            path, MapOrigin, TileSize, 64, sampleSpacingMs: 1000, tileRadius: 0);

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
            TileSize,
            64,
            sampleSpacingMs: 1000,
            tileRadius: 1,
            tileExists: static (tileX, tileY) => (tileX, tileY) != (19, 19));

        Assert.Contains((20, 20), tiles);
        Assert.Contains((19, 20), tiles);
        Assert.DoesNotContain((19, 19), tiles);
        Assert.Equal(8, tiles.Count);
    }

    private static Vector3 WorldPosition(int tileX, int tileY)
        => new(
            MapOrigin - ((tileX + 0.5f) * TileSize),
            MapOrigin - ((tileY + 0.5f) * TileSize),
            0f);
}
