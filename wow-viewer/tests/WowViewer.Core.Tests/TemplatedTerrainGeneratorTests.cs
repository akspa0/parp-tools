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

    [Fact]
    public void BiomePalette_AllThemes_UseAuthenticBlpPaths()
    {
        foreach (BiomeTheme theme in Enum.GetValues<BiomeTheme>())
        {
            BiomePalette palette = BiomePalette.ForTheme(theme);
            Assert.False(string.IsNullOrWhiteSpace(palette.BaseGroundTexture));
            Assert.False(string.IsNullOrWhiteSpace(palette.PathTexture));
            Assert.False(string.IsNullOrWhiteSpace(palette.PlazaFloorTexture));
            Assert.False(string.IsNullOrWhiteSpace(palette.AccentTexture));

            string[] paths = [palette.BaseGroundTexture, palette.PathTexture, palette.PlazaFloorTexture, palette.AccentTexture];
            foreach (string path in paths)
            {
                Assert.EndsWith(".blp", path, StringComparison.OrdinalIgnoreCase);
                Assert.StartsWith("TILESET\\", path, StringComparison.OrdinalIgnoreCase);
                Assert.DoesNotContain("whitemarble", path, StringComparison.OrdinalIgnoreCase);
                Assert.DoesNotContain("elwynngrass.blp", path, StringComparison.OrdinalIgnoreCase);
            }
        }
    }

    [Fact]
    public void GenerateMap_UsesAuthenticModelPaths()
    {
        var template = new TerrainMapTemplate
        {
            MapName = "AuthenticModelTest",
            Theme = BiomeTheme.GardenMuseum,
            TileRows = 1,
            TileCols = 1,
            BaseTileX = 32,
            BaseTileY = 32,
            PlazaSpacingChunks = 2
        };

        TemplatedMapResult result = TemplatedTerrainGenerator.GenerateMap(template);
        LkAdtData tile = result.Tiles[(32, 32)];

        Assert.NotEmpty(tile.ModelNames);
        foreach (string model in tile.ModelNames)
        {
            Assert.True(model.EndsWith(".m2", StringComparison.OrdinalIgnoreCase) ||
                        model.EndsWith(".mdx", StringComparison.OrdinalIgnoreCase));
            Assert.StartsWith("World\\", model, StringComparison.OrdinalIgnoreCase);
        }
    }

    [Fact]
    public void GenerateMap_SynthesizesContinuousFractalReliefInNatureChunks()
    {
        var template = new TerrainMapTemplate
        {
            MapName = "FractalReliefTest",
            Theme = BiomeTheme.GardenMuseum,
            TileRows = 1,
            TileCols = 1,
            BaseTileX = 30,
            BaseTileY = 30,
            PlazaSpacingChunks = 4
        };

        TemplatedMapResult result = TemplatedTerrainGenerator.GenerateMap(template);
        LkAdtData tile = result.Tiles[(30, 30)];

        // Find a pure nature chunk far from plazas and pathways (e.g. index 1, 1 when spacing is 4)
        LkMcnkData natureChunk = tile.Chunks.First(c => c.IndexX == 1 && c.IndexY == 1);

        // Verify height variation exists (fractal harmonic noise)
        float min = natureChunk.Heights.Min();
        float max = natureChunk.Heights.Max();
        float spread = max - min;

        Assert.True(spread > 0.05f, $"Nature chunk had zero/flat relief (spread: {spread}), expected fractal noise variation.");
    }
}
