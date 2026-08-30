using WowViewer.Core.IO.Terrain;
using WowViewer.Core.Maps;
using Xunit;

namespace WowViewer.Core.Tests;

public class TemplatedTerrainGeneratorTests
{
    [Fact]
    public void GenerateMap_2x2GardenMuseum_SynthesizesFourTilesWithValidChunks()
    {
        var template = new TerrainMapTemplate
        {
            MapName = "GardenMuseumTest",
            Theme = BiomeTheme.GardenMuseum,
            TileRows = 2,
            TileCols = 2,
            BaseTileX = 30,
            BaseTileY = 30,
            PlazaSpacingChunks = 2
        };

        TemplatedMapResult result = TemplatedTerrainGenerator.GenerateMap(template);

        Assert.NotNull(result);
        Assert.Equal("GardenMuseumTest", result.MapName);
        Assert.Equal(4, result.Tiles.Count);

        // Verify each tile has 256 chunks
        foreach (var kvp in result.Tiles)
        {
            LkAdtData tile = kvp.Value;
            Assert.Equal(256, tile.Chunks.Count);
            Assert.True(tile.TextureNames.Count >= 3);

            // Verify all chunks strictly adhere to <= 4 layers
            foreach (LkMcnkData chunk in tile.Chunks)
            {
                Assert.True(chunk.Layers.Count <= 4,
                    $"Chunk ({chunk.IndexX}, {chunk.IndexY}) has {chunk.Layers.Count} layers > 4.");
                Assert.Equal(145, chunk.Heights.Length);
                Assert.Equal(145 * 3, chunk.Normals.Length);
            }
        }
    }

    [Fact]
    public void GenerateMap_PlazaNodes_ReceiveMarbleFloorAndCobblestoneBorder()
    {
        var template = new TerrainMapTemplate
        {
            MapName = "PlazaCheck",
            Theme = BiomeTheme.GardenMuseum,
            TileRows = 1,
            TileCols = 1,
            BaseTileX = 30,
            BaseTileY = 30,
            PlazaSpacingChunks = 2
        };

        TemplatedMapResult result = TemplatedTerrainGenerator.GenerateMap(template);
        LkAdtData tile = result.Tiles[(30, 30)];

        // Chunk (0, 0) is a plaza node
        LkMcnkData plazaChunk = tile.Chunks[0];
        Assert.True(plazaChunk.Layers.Count >= 2);

        // Verify presence of white marble and cobblestone
        var chunkTextureNames = plazaChunk.Layers
            .Select(l => tile.TextureNames[(int)l.TextureId])
            .ToList();

        Assert.Contains(@"tileset\elwynn\elwynngrass.blp", chunkTextureNames);
        Assert.Contains(@"tileset\city\stormwindcobble.blp", chunkTextureNames);
        Assert.Contains(@"tileset\city\whitemarble.blp", chunkTextureNames);
    }
}
