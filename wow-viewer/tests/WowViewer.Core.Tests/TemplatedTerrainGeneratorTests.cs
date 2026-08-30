using System.Linq;
using WowViewer.Core.IO.Maps;
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
            }
        }
    }

    [Fact]
    public void GenerateMap_PlazasHaveFlatGroundAndGardenDecorations()
    {
        var template = new TerrainMapTemplate
        {
            MapName = "GardenMuseumTest",
            Theme = BiomeTheme.GardenMuseum,
            TileRows = 1,
            TileCols = 1,
            BaseTileX = 32,
            BaseTileY = 32,
            PlazaSpacingChunks = 4
        };

        TemplatedMapResult result = TemplatedTerrainGenerator.GenerateMap(template);
        LkAdtData tile = result.Tiles[(32, 32)];

        // Verify objects/doodads are placed
        Assert.NotEmpty(tile.ModelNames);
        Assert.NotEmpty(tile.ModelPlacements);

        // Verify plaza chunk has low height variance (flat)
        // Plaza is placed at chunk (2, 2) since (2%4==2 && 2%4==2)
        LkMcnkData plazaChunk = tile.Chunks.First(c => c.IndexX == 2 && c.IndexY == 2);
        float minHeight = plazaChunk.Heights.Min();
        float maxHeight = plazaChunk.Heights.Max();
        float heightSpread = maxHeight - minHeight;

        // Spread in plaza chunk should be modest (not a wild mountain peak)
        Assert.True(heightSpread < 15.0f, $"Plaza chunk height spread was {heightSpread}, expected flat ground.");
    }

    [Fact]
    public void GenerateMap_AlphaCompatibleConversion_PreservesDimensionsAndTextures()
    {
        var template = new TerrainMapTemplate
        {
            MapName = "AlphaCompatTest",
            Theme = BiomeTheme.GardenMuseum,
            TileRows = 1,
            TileCols = 1,
            BaseTileX = 30,
            BaseTileY = 30
        };

        TemplatedMapResult result = TemplatedTerrainGenerator.GenerateMap(template);
        LkAdtData tile = result.Tiles[(30, 30)];

        AlphaTileData alphaTile = LkToAlphaConverter.ConvertTile(tile, 30, 30);
        Assert.NotNull(alphaTile);
        Assert.Equal(257, alphaTile.Heightmap.GetLength(0));
        Assert.Equal(257, alphaTile.Heightmap.GetLength(1));
        Assert.True(alphaTile.TextureNames.Count >= 3);
    }
}
