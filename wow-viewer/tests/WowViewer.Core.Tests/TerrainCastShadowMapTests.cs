using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.Terrain;

namespace WowViewer.Core.Tests;

public sealed class TerrainCastShadowMapTests
{
    private const int Size = 257;

    /// <summary>
    /// Flat ground with a single tall wall running across it. Every sample is at height 0 except one
    /// row of the grid, which is raised. Under the authored solar bearing this must leave a shadow
    /// on exactly one side of the wall.
    /// </summary>
    private static float[,] BuildWallHeightfield(int wallRow, float wallHeight)
    {
        var height = new float[Size, Size];
        for (int column = 0; column < Size; column++)
            height[wallRow, column] = wallHeight;
        return height;
    }

    [Fact]
    public void Compute_FlatTerrainCastsNoShadow()
    {
        float[,]? occlusion = TerrainCastShadowMap.Compute(
            new float[Size, Size],
            TerrainSolarDirection.Evaluate(0.5f));

        Assert.NotNull(occlusion);
        for (int row = 0; row < Size; row++)
        {
            for (int column = 0; column < Size; column++)
                Assert.Equal(0f, occlusion![row, column]);
        }
    }

    [Fact]
    public void Compute_ReturnsNullWhenSunHasNoHorizontalBearing()
    {
        // Straight overhead: geometrically incapable of casting, so there is no map to build rather
        // than an all-zero map the compositor would still pay to sample.
        Assert.Null(TerrainCastShadowMap.Compute(BuildWallHeightfield(128, 200f), Vector3.UnitZ));
    }

    [Fact]
    public void Compute_ReturnsNullForMissingOrDegenerateHeightfield()
    {
        Assert.Null(TerrainCastShadowMap.Compute(null, TerrainSolarDirection.Evaluate(0.5f)));
        Assert.Null(TerrainCastShadowMap.Compute(new float[1, 1], TerrainSolarDirection.Evaluate(0.5f)));
    }

    /// <summary>
    /// The direction check. TerrainSolarDirection holds a fixed north-west bearing with positive
    /// world X and Y, and renderer axes map to the grid as world +X = grid -row. The sun is
    /// therefore always toward DECREASING row, so a wall shadows the rows BELOW it (higher index).
    /// Getting this backwards lights the wrong side of every ridge -- the same class of bug as the
    /// hillshade Y-axis inversion fixed in v0.5.2, and invisible without an asymmetric fixture.
    /// </summary>
    [Fact]
    public void Compute_ShadowFallsOnTheSideOfTheWallFacingAwayFromTheSun()
    {
        const int WallRow = 128;
        float[,] height = BuildWallHeightfield(WallRow, 200f);

        float[,]? occlusion = TerrainCastShadowMap.Compute(height, TerrainSolarDirection.Evaluate(0.5f));
        Assert.NotNull(occlusion);

        // Sample well clear of the wall on both sides, at a column far from the tile edges so the
        // ray does not simply run out of tile.
        const int Column = 200;
        float towardSun = occlusion![WallRow - 4, Column];
        float awayFromSun = occlusion[WallRow + 4, Column];

        Assert.Equal(0f, towardSun);
        Assert.True(
            awayFromSun > 0.9f,
            $"Expected the far side of the wall to be occluded, got {awayFromSun}.");
    }

    [Fact]
    public void Compute_LowerSunCastsLongerShadows()
    {
        const int WallRow = 100;
        float[,] height = BuildWallHeightfield(WallRow, 60f);

        // Noon is the highest elevation TerrainSolarDirection produces; 08:00 is materially lower.
        float[,]? noon = TerrainCastShadowMap.Compute(height, TerrainSolarDirection.Evaluate(0.5f));
        float[,]? morning = TerrainCastShadowMap.Compute(height, TerrainSolarDirection.Evaluate(8f / 24f));
        Assert.NotNull(noon);
        Assert.NotNull(morning);

        Assert.True(
            MeasureShadowLength(morning!, WallRow, 200) > MeasureShadowLength(noon!, WallRow, 200),
            "A lower sun must throw a longer shadow.");
    }

    [Fact]
    public void Compute_ShadowIsBoundedByOccluderHeight()
    {
        const int WallRow = 100;
        const int Column = 200;

        float[,]? shortWall = TerrainCastShadowMap.Compute(
            BuildWallHeightfield(WallRow, 20f),
            TerrainSolarDirection.Evaluate(0.5f));
        float[,]? tallWall = TerrainCastShadowMap.Compute(
            BuildWallHeightfield(WallRow, 200f),
            TerrainSolarDirection.Evaluate(0.5f));

        Assert.True(
            MeasureShadowLength(tallWall!, WallRow, Column) > MeasureShadowLength(shortWall!, WallRow, Column),
            "A taller occluder must throw a longer shadow.");
    }

    private static int MeasureShadowLength(float[,] occlusion, int wallRow, int column)
    {
        int length = 0;
        for (int row = wallRow + 1; row < Size; row++)
        {
            if (occlusion[row, column] <= 0f)
                break;
            length++;
        }

        return length;
    }
}
