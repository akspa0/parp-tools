using System.Numerics;
using WowViewer.Core.Editor.Procedural;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using Xunit;

namespace WowViewer.Core.Tests;

public class ProceduralTexturePainterTests
{
    private static RosettaAssetEntry TestAsset(string path, float size = 2f) => new(
        path,
        RosettaAssetKind.Model,
        new Vector3(-size / 2f, -size / 2f, -size / 2f),
        new Vector3(size / 2f, size / 2f, size / 2f));

    [Fact]
    public void GenerateChunkAlphaLayers_ProducesValid4LayerStackWithinLimits()
    {
        var assets = new List<RosettaAssetEntry>
        {
            TestAsset("item/objectcomponents/weapon/sword_1h_01.mdx", 1.5f)
        };

        var layoutOptions = new AdaptiveLayoutOptions(Density: DensityPreset.Compact);
        List<AdaptiveExhibitPlacement> placements = AdaptiveLayoutPacker.PackExhibits(assets, layoutOptions);

        ProceduralTexturePalette palette = ProceduralTexturePainter.ResolvePalette(ProceduralMapTheme.Garden);
        ChunkAlphaData alphaData = ProceduralTexturePainter.GenerateChunkAlphaLayers(0, 0, placements, palette);

        Assert.NotNull(alphaData);
        Assert.True(alphaData.TextureFilenames.Count >= 1 && alphaData.TextureFilenames.Count <= 4);
        Assert.Equal(alphaData.TextureFilenames.Count - 1, alphaData.AlphaLayers64x64.Count);

        Assert.Equal(palette.BaseTexture, alphaData.TextureFilenames[0]);

        foreach (byte[] layer in alphaData.AlphaLayers64x64)
        {
            Assert.Equal(4096, layer.Length);
        }
    }

    [Fact]
    public void GenerateChunkAlphaLayers_ProducesCleanPlazaInExhibitCenter()
    {
        var assets = new List<RosettaAssetEntry>
        {
            TestAsset("character/human/male/humanmale.mdx", 2.0f)
        };

        var layoutOptions = new AdaptiveLayoutOptions(Density: DensityPreset.Balanced);
        List<AdaptiveExhibitPlacement> placements = AdaptiveLayoutPacker.PackExhibits(assets, layoutOptions);
        var p = placements[0];

        ProceduralTexturePalette palette = ProceduralTexturePainter.ResolvePalette(ProceduralMapTheme.Garden);
        ChunkAlphaData alphaData = ProceduralTexturePainter.GenerateChunkAlphaLayers(0, 0, placements, palette);

        // Find clean plaza layer index
        int plazaIdx = -1;
        for (int i = 1; i < alphaData.TextureFilenames.Count; i++)
        {
            if (alphaData.TextureFilenames[i] == palette.CenterPlazaTexture)
            {
                plazaIdx = i - 1;
                break;
            }
        }

        Assert.True(plazaIdx >= 0, "Center plaza layer was not added to the chunk");

        byte[] plazaAlpha = alphaData.AlphaLayers64x64[plazaIdx];

        // Center pixel of the exhibit cell in 64x64 alpha grid
        float centerU = p.CellU + (p.CellSize * 0.5f);
        float centerV = p.CellV + (p.CellSize * 0.5f);

        int px = (int)(centerU / ProceduralTexturePainter.PixelWorldStep);
        int py = (int)(centerV / ProceduralTexturePainter.PixelWorldStep);

        if (px >= 0 && px < 64 && py >= 0 && py < 64)
        {
            int centerIdx = (py * 64) + px;
            Assert.True(plazaAlpha[centerIdx] > 200, $"Center plaza alpha was {plazaAlpha[centerIdx]}, expected > 200");
        }
    }
}
