using System.Text.Json;
using WowViewer.Core.IO.Terrain;
using Xunit;

namespace WowViewer.Core.Tests;

public class TerrainBrushPasteTests
{
    [Fact]
    public void StockLibrary_LoadsAllExpectedPresets()
    {
        TerrainBrushLibrary lib = CuratedTerrainBrushLibrary.Instance;
        Assert.NotNull(lib);
        Assert.True(lib.Count >= 15, $"Expected at least 15 presets, found {lib.Count}");

        var categories = lib.GetCategories();
        Assert.Contains("Road", categories);
        Assert.Contains("Plaza", categories);
        Assert.Contains("Hill", categories);
        Assert.Contains("Garden", categories);
    }

    [Fact]
    public void StockLibrary_RoadSearch_ReturnsCobblestoneWalkways()
    {
        TerrainBrushLibrary lib = CuratedTerrainBrushLibrary.Instance;
        var roads = lib.Search("cobblestone");
        Assert.NotEmpty(roads);
        Assert.All(roads, r => Assert.Contains("cobblestone", r.Tags));
    }

    [Fact]
    public void StockLibrary_AllPresets_ObeyMaxSlopeWalkability()
    {
        TerrainBrushLibrary lib = CuratedTerrainBrushLibrary.Instance;
        foreach (TerrainBrushPaste paste in lib.AllPastes)
        {
            // Per NFR-001, walkability/roads/plazas must not exceed 25 degrees
            if (paste.Category is "Road" or "Plaza" or "Garden")
            {
                Assert.True(paste.MaxSlopeDegrees <= 25f,
                    $"Paste {paste.Id} has slope {paste.MaxSlopeDegrees} > 25 degrees.");
            }
        }
    }

    [Fact]
    public void StockLibrary_AllPresets_ObeyMaxFourLayers()
    {
        TerrainBrushLibrary lib = CuratedTerrainBrushLibrary.Instance;
        foreach (TerrainBrushPaste paste in lib.AllPastes)
        {
            Assert.True(paste.Layers.Count <= 4,
                $"Paste {paste.Id} has {paste.Layers.Count} layers, exceeding 4-layer engine limit.");
            Assert.NotEmpty(paste.Layers);
            Assert.False(string.IsNullOrWhiteSpace(paste.Layers[0].TexturePath));
        }
    }

    [Fact]
    public void JsonRoundTrip_PreservesAllProperties()
    {
        TerrainBrushLibrary originalLib = CuratedTerrainBrushLibrary.Instance;
        using var ms = new MemoryStream();
        originalLib.SaveToJson(ms);

        ms.Position = 0;
        TerrainBrushLibrary loadedLib = TerrainBrushLibrary.LoadFromJson(ms);

        Assert.Equal(originalLib.Count, loadedLib.Count);
        Assert.True(loadedLib.TryGetPaste("road_cobble_straight_01", out TerrainBrushPaste? road));
        Assert.NotNull(road);
        Assert.Equal("Cobblestone Straight Walkway", road.Name);
        Assert.Equal(2, road.Layers.Count);
        Assert.Equal(@"tileset\city\stormwindcobble.blp", road.Layers[1].TexturePath);
    }

    [Fact]
    public void SampleHeight_InterpolatesSmoothly()
    {
        TerrainBrushLibrary lib = CuratedTerrainBrushLibrary.Instance;
        Assert.True(lib.TryGetPaste("hill_gentle_knoll_01", out TerrainBrushPaste? hill));
        Assert.NotNull(hill);

        // Center of knoll (u=0.5, v=0.5) should be peak elevation ~4.5m
        float centerH = hill.SampleHeight(0.5f, 0.5f);
        Assert.InRange(centerH, 4.0f, 4.6f);

        // Edge of knoll (u=0.0, v=0.5) should be zero
        float edgeH = hill.SampleHeight(0.0f, 0.5f);
        Assert.InRange(edgeH, 0.0f, 0.1f);
    }
}
