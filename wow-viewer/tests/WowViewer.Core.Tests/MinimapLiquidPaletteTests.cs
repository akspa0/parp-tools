using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class MinimapLiquidPaletteTests
{
    private static TerrainTileTensorPack BuildLiquidPack(AdtLiquidBasicType type) => new()
    {
        UnifiedLiquidMask = new float[3, 3]
        {
            { 1f, 1f, 0f },
            { 1f, 1f, 0f },
            { 0f, 0f, 0f }
        },
        LiquidBasicType257 = new byte[3, 3]
        {
            { (byte)type, (byte)type, LiquidBasicTypeConstants.NoLiquid },
            { (byte)type, (byte)type, LiquidBasicTypeConstants.NoLiquid },
            { LiquidBasicTypeConstants.NoLiquid, LiquidBasicTypeConstants.NoLiquid, LiquidBasicTypeConstants.NoLiquid }
        }
    };

    private static Rgba32 RenderWater(MinimapLiquidPalette palette, AdtLiquidBasicType type = AdtLiquidBasicType.Water)
    {
        using var terrain = new Image<Rgba32>(2, 2, new Rgba32(255, 255, 255, 255));
        using Image<Rgba32> liquid = TerrainMinimapLiquidCompositor.Compose(
            terrain, BuildLiquidPack(type), out _, palette);
        return liquid[0, 0];
    }

    /// <summary>
    /// The project restores the 0.5.3 pre-alpha era, whose minimap water is a cyan-teal rather than
    /// the later slate-blue. The default must follow the restoration target, not the live viewer.
    /// </summary>
    [Fact]
    public void Default_IsThePreAlpha053TealPalette()
    {
        Assert.Same(MinimapLiquidPalette.PreAlpha053, MinimapLiquidPalette.Default);
        Assert.Equal("viewer_flat_liquid_overlay_prealpha_0_5_3_teal_v4", MinimapLiquidPalette.Default.RenderProfile);
        Assert.Equal(MinimapLiquidPalette.Default.RenderProfile, TerrainMinimapLiquidCompositor.RenderProfile);
    }

    /// <summary>
    /// Authored 0.5.3 water is translucent -- seabed relief reads through it. Opacity is therefore a
    /// SIGNAL property, not a cosmetic one: an opaque slab erases the terrain information that water
    /// pixels still carry in a real minimap, in every synthesized training row.
    /// </summary>
    [Fact]
    public void PreAlpha053_WaterIsTranslucentEnoughToShowTerrainThrough()
    {
        Assert.InRange(MinimapLiquidPalette.PreAlpha053.Water.Opacity, 0.6f, 0.85f);

        // Same terrain under two different colours must still produce different water pixels.
        using var darkFloor = new Image<Rgba32>(2, 2, new Rgba32(20, 20, 20, 255));
        using var brightFloor = new Image<Rgba32>(2, 2, new Rgba32(230, 230, 230, 255));
        using Image<Rgba32> overDark = TerrainMinimapLiquidCompositor.Compose(
            darkFloor, BuildLiquidPack(AdtLiquidBasicType.Water), out _, MinimapLiquidPalette.PreAlpha053);
        using Image<Rgba32> overBright = TerrainMinimapLiquidCompositor.Compose(
            brightFloor, BuildLiquidPack(AdtLiquidBasicType.Water), out _, MinimapLiquidPalette.PreAlpha053);

        Assert.True(
            overBright[0, 0].R - overDark[0, 0].R > 30,
            "Seabed brightness must still read through the water overlay.");
    }

    /// <summary>
    /// The measured miss against the authored 0.5.3 comparison was overwhelmingly in GREEN -- the
    /// legacy palette renders water at 0.35 green where authored reads near 0.72. Blue moves much
    /// less. Pin the direction of the change so a future "tidy up the palette" cannot quietly walk
    /// the teal back to a blue.
    /// </summary>
    [Fact]
    public void PreAlpha053_IsMarkedlyGreenerThanTheLegacyViewerPalette()
    {
        MinimapLiquidStyle teal = MinimapLiquidPalette.PreAlpha053.Water;
        MinimapLiquidStyle legacy = MinimapLiquidPalette.ViewerFlatV1.Water;

        Assert.True(teal.Green > legacy.Green * 1.8f, $"Expected a markedly greener teal, got {teal.Green} vs {legacy.Green}.");
        Assert.True(teal.Green > teal.Red * 1.5f, "Teal must be green-dominant over red.");
        Assert.True(teal.Blue > teal.Green, "Teal still leans blue overall.");
        Assert.True(teal.Opacity > legacy.Opacity, "Authored minimap water shows little terrain through it.");
    }

    [Fact]
    public void PreAlpha053_RendersVisiblyDifferentWaterThanTheLegacyPalette()
    {
        Rgba32 teal = RenderWater(MinimapLiquidPalette.PreAlpha053);
        Rgba32 legacy = RenderWater(MinimapLiquidPalette.ViewerFlatV1);

        Assert.True(teal.G > legacy.G + 30, $"Teal water must render much greener: {teal.G} vs {legacy.G}.");
    }

    [Theory]
    [InlineData("prealpha053")]
    [InlineData("0.5.3")]
    [InlineData("teal")]
    [InlineData("default")]
    [InlineData(null)]
    public void TryResolve_MapsEraAliasesToThePreAlphaPalette(string? name)
    {
        Assert.Same(MinimapLiquidPalette.PreAlpha053, MinimapLiquidPalette.TryResolve(name));
    }

    [Theory]
    [InlineData("viewer")]
    [InlineData("v1")]
    [InlineData("legacy")]
    public void TryResolve_MapsViewerAliasesToTheLegacyPalette(string name)
    {
        Assert.Same(MinimapLiquidPalette.ViewerFlatV1, MinimapLiquidPalette.TryResolve(name));
    }

    [Fact]
    public void TryResolve_ReturnsNullForAnUnknownName()
    {
        Assert.Null(MinimapLiquidPalette.TryResolve("cataclysm"));
    }

    /// <summary>
    /// A single --water-color must not leave coastlines mismatched: ocean tracks water so the two
    /// keep their relative depth instead of one being overridden and the other left behind.
    /// </summary>
    [Fact]
    public void WithWaterColor_MovesOceanWithWaterAndRecordsTheOverrideInTheProfile()
    {
        MinimapLiquidPalette tuned = MinimapLiquidPalette.PreAlpha053.WithWaterColor(0.4f, 0.8f, 0.85f);

        Assert.Equal(0.8f, tuned.Water.Green, 5);
        Assert.True(tuned.Ocean.Green < tuned.Water.Green, "Ocean stays deeper than water.");
        Assert.Contains("custom_water", tuned.RenderProfile, StringComparison.Ordinal);

        // Magma and slime are untouched by a water override.
        Assert.Equal(MinimapLiquidPalette.PreAlpha053.Magma, tuned.Magma);
        Assert.Equal(MinimapLiquidPalette.PreAlpha053.Slime, tuned.Slime);
    }
}
