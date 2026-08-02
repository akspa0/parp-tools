using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public sealed class MinimapLightOverlayTests
{
    private const int Resolution = 64;

    private static Vector3 TileCentreWorld(int tileX, int tileY) =>
        MinimapTileProjection.Unproject(Resolution / 2, Resolution / 2, Resolution, tileX, tileY);

    /// <summary>
    /// Round-trip guard on the axis crossover that makes this easy to get backwards: world X (North)
    /// varies with the tile ROW and world Y (West) with the tile COLUMN.
    /// </summary>
    [Theory]
    [InlineData(0, 0)]
    [InlineData(32, 32)]
    [InlineData(63, 12)]
    public void Projection_RoundTripsPixelToWorldAndBack(int tileX, int tileY)
    {
        const int PixelX = 17;
        const int PixelY = 43;

        Vector3 world = MinimapTileProjection.Unproject(PixelX, PixelY, Resolution, tileX, tileY);
        MinimapTileProjection.Project(world, tileX, tileY, out float u, out float v);

        Assert.True(MinimapTileProjection.IsWithinTile(u, v));
        Assert.Equal((PixelX + 0.5f) / Resolution, u, 4);
        Assert.Equal((PixelY + 0.5f) / Resolution, v, 4);
    }

    [Fact]
    public void Projection_IncreasingRowMovesSouthAndIncreasingColumnMovesEast()
    {
        Vector3 origin = MinimapTileProjection.Unproject(0, 0, Resolution, 30, 30);
        Vector3 downOneRow = MinimapTileProjection.Unproject(0, 10, Resolution, 30, 30);
        Vector3 rightOneColumn = MinimapTileProjection.Unproject(10, 0, Resolution, 30, 30);

        // +X is North, so advancing down the raster must decrease world X.
        Assert.True(downOneRow.X < origin.X);
        // +Y is West, so advancing right across the raster must decrease world Y.
        Assert.True(rightOneColumn.Y < origin.Y);
    }

    [Fact]
    public void Compose_TintsTheTileWithALightsColourAndMarksItsCentre()
    {
        using var terrain = new Image<Rgba32>(Resolution, Resolution, new Rgba32(40, 40, 40, 255));
        var light = new MinimapLightMarker(
            TileCentreWorld(30, 30),
            CoreRadius: 60f,
            OuterRadius: 200f,
            Color: new Vector3(1f, 0f, 0f),
            Name: "test");

        using Image<Rgba32> overlay = TerrainMinimapLightOverlayCompositor.Compose(
            terrain, 30, 30, [light], out int visible);

        Assert.Equal(1, visible);
        // Centre carries the solid swatch.
        Assert.True(overlay[Resolution / 2, Resolution / 2].R > 200);
        // Terrain outside every light's reach is untouched.
        Assert.Equal(new Rgba32(40, 40, 40, 255), overlay[0, 0]);
        // The original image is never mutated.
        Assert.Equal(new Rgba32(40, 40, 40, 255), terrain[Resolution / 2, Resolution / 2]);
    }

    /// <summary>
    /// The dome must follow LitSpatialSampler's falloff, so tint strength decreases with distance
    /// rather than being a flat disc.
    /// </summary>
    [Fact]
    public void Compose_DomeFadesWithDistanceFromTheLight()
    {
        using var terrain = new Image<Rgba32>(Resolution, Resolution, new Rgba32(0, 0, 0, 255));
        var light = new MinimapLightMarker(
            TileCentreWorld(30, 30),
            CoreRadius: 0f,
            OuterRadius: 400f,
            Color: Vector3.One,
            Name: "fade");

        using Image<Rgba32> overlay = TerrainMinimapLightOverlayCompositor.Compose(
            terrain,
            30,
            30,
            [light],
            out _,
            new MinimapLightOverlayOptions(DrawSwatch: false));

        int near = overlay[(Resolution / 2) + 4, Resolution / 2].R;
        int mid = overlay[(Resolution / 2) + 10, Resolution / 2].R;
        int far = overlay[(Resolution / 2) + 20, Resolution / 2].R;

        Assert.True(near > mid, $"Expected falloff, got near={near} mid={mid}.");
        Assert.True(mid > far, $"Expected falloff, got mid={mid} far={far}.");
    }

    /// <summary>
    /// A light centred on a neighbouring tile still spills across the seam, so reach is tested
    /// against the influence circle rather than against the centre being inside the tile.
    /// </summary>
    [Fact]
    public void Compose_DrawsALightWhoseCentreIsOnAnAdjacentTileButWhoseReachCrossesTheSeam()
    {
        using var terrain = new Image<Rgba32>(Resolution, Resolution, new Rgba32(0, 0, 0, 255));
        var neighbour = new MinimapLightMarker(
            TileCentreWorld(31, 30),
            CoreRadius: 300f,
            OuterRadius: 600f,
            Color: Vector3.One,
            Name: "neighbour");

        using Image<Rgba32> overlay = TerrainMinimapLightOverlayCompositor.Compose(
            terrain, 30, 30, [neighbour], out int visible);

        Assert.Equal(1, visible);
        Assert.True(overlay[Resolution - 1, Resolution / 2].R > 0, "Spill across the tile seam must be drawn.");
    }

    [Fact]
    public void Compose_IgnoresLightsThatCannotReachTheTile()
    {
        using var terrain = new Image<Rgba32>(Resolution, Resolution, new Rgba32(77, 77, 77, 255));
        var distant = new MinimapLightMarker(
            TileCentreWorld(5, 5),
            CoreRadius: 10f,
            OuterRadius: 20f,
            Color: Vector3.One,
            Name: "far away");

        using Image<Rgba32> overlay = TerrainMinimapLightOverlayCompositor.Compose(
            terrain, 30, 30, [distant], out int visible);

        Assert.Equal(0, visible);
        Assert.Equal(new Rgba32(77, 77, 77, 255), overlay[Resolution / 2, Resolution / 2]);
    }

    /// <summary>
    /// Overlapping domes blend toward their weighted mean rather than summing, so two adjacent
    /// lights stay individually readable instead of washing out to white.
    /// </summary>
    [Fact]
    public void Compose_OverlappingDomesDoNotSaturateToWhite()
    {
        using var terrain = new Image<Rgba32>(Resolution, Resolution, new Rgba32(0, 0, 0, 255));
        Vector3 centre = TileCentreWorld(30, 30);
        MinimapLightMarker Red(Vector3 at) => new(at, 100f, 400f, new Vector3(1f, 0f, 0f), "red");

        using Image<Rgba32> overlay = TerrainMinimapLightOverlayCompositor.Compose(
            terrain,
            30,
            30,
            [Red(centre), Red(centre)],
            out _,
            new MinimapLightOverlayOptions(DrawSwatch: false));

        Rgba32 pixel = overlay[Resolution / 2, Resolution / 2];
        Assert.True(pixel.G < 30, $"Two red lights must not bleach toward white, got G={pixel.G}.");
    }
}
