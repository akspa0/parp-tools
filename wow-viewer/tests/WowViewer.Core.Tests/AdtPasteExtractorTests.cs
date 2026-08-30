using WowViewer.Core.IO.Maps;
using WowViewer.Core.IO.Terrain;
using WowViewer.Core.Maps;
using Xunit;

namespace WowViewer.Core.Tests;

public class AdtPasteExtractorTests
{
    [Fact]
    public void Resample145HeightsTo17x17_InterpolatesCorrectly()
    {
        // 145 heights with a known constant elevation of 10.0m
        float[] heights145 = new float[145];
        Array.Fill(heights145, 10.0f);

        float[] grid17x17 = AdtPasteExtractor.Resample145HeightsTo17x17(heights145);

        Assert.Equal(17 * 17, grid17x17.Length);
        Assert.All(grid17x17, h => Assert.Equal(10.0f, h));
    }

    [Fact]
    public void ExtractFromChunk_ProducesZeroedDeltasAndExtractedLayers()
    {
        // Construct mock 145 heights with slope
        float[] heights145 = new float[145];
        for (int i = 0; i < 145; i++)
            heights145[i] = 100.0f + (i % 9) * 0.5f;

        // Construct mock 64x64 uncompressed alpha
        byte[] rawAlpha = new byte[4096];
        Array.Fill(rawAlpha, (byte)210);

        var chunk = new LkMcnkData
        {
            IndexX = 0,
            IndexY = 0,
            BaseHeight = 100.0f,
            Heights = heights145,
            Layers =
            [
                new LkMclyEntry(TextureId: 0, Flags: 0, AlphaOffset: 0, EffectId: 0),
                new LkMclyEntry(TextureId: 1, Flags: 0, AlphaOffset: 0, EffectId: 0)
            ],
            AlphaMapData = rawAlpha,
            AlphaMapSize = rawAlpha.Length
        };

        string[] textures = [@"tileset\elwynn\elwynngrass.blp", @"tileset\city\stormwindcobble.blp"];

        TerrainBrushPaste paste = AdtPasteExtractor.ExtractFromChunk(
            chunk, textures, "test_extracted_chunk_01", "Extracted Test Chunk", "Road");

        Assert.NotNull(paste);
        Assert.Equal("test_extracted_chunk_01", paste.Id);
        Assert.Equal(17, paste.ResolutionX);
        Assert.Equal(17, paste.ResolutionY);
        Assert.Equal(2, paste.Layers.Count);

        // Layer 0 is base grass
        Assert.Equal(@"tileset\elwynn\elwynngrass.blp", paste.Layers[0].TexturePath);

        // Layer 1 is cobblestone with 210 alpha
        Assert.Equal(@"tileset\city\stormwindcobble.blp", paste.Layers[1].TexturePath);
        Assert.Equal(4096, paste.Layers[1].AlphaMask.Length);
        Assert.Equal(210, paste.Layers[1].AlphaMask[0]);

        // Baseline zeroing check
        Assert.Equal(0.0f, paste.HeightDeltas[0]);
    }
}
