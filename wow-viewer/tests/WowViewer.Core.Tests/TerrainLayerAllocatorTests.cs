using WowViewer.Core.IO.Terrain;
using Xunit;

namespace WowViewer.Core.Tests;

public class TerrainLayerAllocatorTests
{
    [Fact]
    public void MergeLayers_TwoLayersOntoOne_ProducesTwoLayers()
    {
        string[] existing = [@"tileset\elwynn\elwynngrass.blp"];
        List<TerrainPasteLayer> incoming =
        [
            new()
            {
                TexturePath = @"tileset\city\stormwindcobble.blp",
                Resolution = 64,
                AlphaMask = CreateTestAlpha(64, 180)
            }
        ];

        AllocatedChunkLayers result = TerrainLayerAllocator.MergeLayers(existing, null, incoming);

        Assert.Equal(2, result.TexturePaths.Length);
        Assert.Equal(@"tileset\elwynn\elwynngrass.blp", result.TexturePaths[0]);
        Assert.Equal(@"tileset\city\stormwindcobble.blp", result.TexturePaths[1]);
        Assert.Single(result.AlphaSplats);
        Assert.Equal(64 * 64, result.AlphaSplats[0].Length);
        Assert.Equal(180, result.AlphaSplats[0][0]);
    }

    [Fact]
    public void MergeLayers_ExceedingFourLayers_PrunesLowestEnergyToCapAtFour()
    {
        // 3 existing layers
        string[] existing =
        [
            @"tileset\elwynn\elwynngrass.blp",
            @"tileset\generic\dirt.blp",
            @"tileset\generic\rock.blp"
        ];
        List<byte[]> existingAlphas =
        [
            CreateTestAlpha(64, 200), // high energy
            CreateTestAlpha(64, 150)  // medium energy
        ];

        // 3 incoming layers -> Total 6 unique textures
        List<TerrainPasteLayer> incoming =
        [
            new() { TexturePath = @"tileset\city\stormwindcobble.blp", Resolution = 64, AlphaMask = CreateTestAlpha(64, 255) }, // Highest energy
            new() { TexturePath = @"tileset\city\whitemarble.blp", Resolution = 64, AlphaMask = CreateTestAlpha(64, 100) },      // Low energy
            new() { TexturePath = @"tileset\wailingcaverns\wcsand.blp", Resolution = 64, AlphaMask = CreateTestAlpha(64, 10) }  // Lowest energy -> should be pruned
        ];

        AllocatedChunkLayers result = TerrainLayerAllocator.MergeLayers(existing, existingAlphas, incoming);

        Assert.True(result.TexturePaths.Length <= 4, $"Expected <= 4 layers, got {result.TexturePaths.Length}");
        Assert.Equal(4, result.TexturePaths.Length);

        // Base texture preserved
        Assert.Equal(@"tileset\elwynn\elwynngrass.blp", result.TexturePaths[0]);

        // Highest energy textures preserved
        Assert.Contains(@"tileset\city\stormwindcobble.blp", result.TexturePaths);
        Assert.Contains(@"tileset\generic\dirt.blp", result.TexturePaths);

        // Lowest energy texture pruned
        Assert.DoesNotContain(@"tileset\wailingcaverns\wcsand.blp", result.TexturePaths);
    }

    private static byte[] CreateTestAlpha(int res, byte val)
    {
        byte[] b = new byte[res * res];
        Array.Fill(b, val);
        return b;
    }
}
