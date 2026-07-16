using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class TerrainMinimapCompositorTests
{
    [Fact]
    public void Compose_UsesWeightedMcalLayersAndNeutralLighting()
    {
        TerrainTileTensorPack pack = BuildPack(layerOneAlpha: 0.25f);
        var textures = new Dictionary<int, byte[,,]>
        {
            [0] = SolidTexture(255, 0, 0),
            [1] = SolidTexture(0, 255, 0)
        };
        var options = new TerrainMinimapCompositionOptions(2, TerrainMinimapLighting.Neutral);

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(pack, textures, options);

        Assert.Equal(new Rgba32(191, 64, 0, 255), image[0, 0]);
        Assert.Equal(new Rgba32(191, 64, 0, 255), image[1, 1]);
    }

    [Fact]
    public void Compose_CompositesOverlappingMcalLayersInTerrainRendererOrder()
    {
        TerrainTileTensorPack pack = BuildPack(layerOneAlpha: 0.5f, layerTwoAlpha: 0.5f);
        var textures = new Dictionary<int, byte[,,]>
        {
            [0] = SolidTexture(255, 0, 0),
            [1] = SolidTexture(0, 255, 0),
            [2] = SolidTexture(0, 0, 255)
        };

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            pack,
            textures,
            new TerrainMinimapCompositionOptions(2, TerrainMinimapLighting.Neutral));

        // Base red -> 50% green -> 50% blue, matching the terrain fragment shader's mix order.
        Assert.Equal(new Rgba32(64, 64, 128, 255), image[0, 0]);
    }

    [Fact]
    public void Compose_UsesRowMajorMcalAndMclyChunkCoordinates()
    {
        var alpha = new float[2, 2, 4];
        var textureIds = new int[2, 2, 4];
        for (int chunkY = 0; chunkY < 2; chunkY++)
        {
            for (int chunkX = 0; chunkX < 2; chunkX++)
                for (int layer = 0; layer < 4; layer++)
                    textureIds[chunkY, chunkX, layer] = -1;
        }

        textureIds[0, 0, 0] = 0;
        textureIds[0, 1, 0] = 1;
        textureIds[1, 0, 0] = 2;
        textureIds[1, 1, 0] = 3;
        var pack = new TerrainTileTensorPack
        {
            McalAlphaPack256 = alpha,
            MclyTextureIds = textureIds
        };
        var textures = new Dictionary<int, byte[,,]>
        {
            [0] = SolidTexture(255, 0, 0),
            [1] = SolidTexture(0, 255, 0),
            [2] = SolidTexture(0, 0, 255),
            [3] = SolidTexture(255, 255, 0)
        };

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            pack,
            textures,
            new TerrainMinimapCompositionOptions(2, TerrainMinimapLighting.Neutral));

        Assert.Equal(new Rgba32(255, 0, 0, 255), image[0, 0]);
        Assert.Equal(new Rgba32(0, 255, 0, 255), image[1, 0]);
        Assert.Equal(new Rgba32(0, 0, 255, 255), image[0, 1]);
        Assert.Equal(new Rgba32(255, 255, 0, 255), image[1, 1]);
    }

    [Fact]
    public void Compose_DoesNotBlendAnAbsentMclyLayer()
    {
        TerrainTileTensorPack pack = BuildPack(
            layerOneAlpha: 1f,
            layerMask: new bool[1, 1, 4]
            {
                { { true, false, false, false } }
            });
        var textures = new Dictionary<int, byte[,,]>
        {
            [0] = SolidTexture(255, 0, 0),
            [1] = SolidTexture(0, 255, 0)
        };

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            pack,
            textures,
            new TerrainMinimapCompositionOptions(1, TerrainMinimapLighting.Neutral));

        Assert.Equal(new Rgba32(255, 0, 0, 255), image[0, 0]);
    }

    [Fact]
    public void Compose_UsesTheAverageMaterialColourAtMinimapScale()
    {
        var alpha = new float[4, 4, 4];
        var textureIds = new int[1, 1, 4];
        textureIds[0, 0, 0] = 0;
        textureIds[0, 0, 1] = -1;
        textureIds[0, 0, 2] = -1;
        textureIds[0, 0, 3] = -1;
        var texture = new byte[4, 4, 3];
        for (int y = 0; y < 4; y++)
        {
            for (int x = 0; x < 4; x++)
            {
                texture[y, x, 0] = (byte)(x * 85);
                texture[y, x, 1] = (byte)(y * 85);
            }
        }

        var pack = new TerrainTileTensorPack
        {
            TileX = 0,
            TileY = 0,
            McalAlphaPack256 = alpha,
            MclyTextureIds = textureIds
        };

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            pack,
            new Dictionary<int, byte[,,]> { [0] = texture },
            new TerrainMinimapCompositionOptions(4, TerrainMinimapLighting.Neutral));

        Assert.Equal(new Rgba32(128, 128, 0, 255), image[0, 0]);
        Assert.Equal(new Rgba32(128, 128, 0, 255), image[1, 0]);
        Assert.Equal(new Rgba32(128, 128, 0, 255), image[0, 1]);
        Assert.Equal(new Rgba32(128, 128, 0, 255), image[3, 3]);
    }

    [Fact]
    public void Compose_FiltersHighFrequencyDiffuseTexturesInsteadOfAliasingThem()
    {
        TerrainTileTensorPack pack = BuildPack(layerOneAlpha: 0f);
        var checker = new byte[256, 256, 3];
        for (int y = 0; y < 256; y++)
        {
            for (int x = 0; x < 256; x++)
            {
                byte value = (byte)(((x + y) & 1) == 0 ? 0 : 255);
                checker[y, x, 0] = value;
                checker[y, x, 1] = value;
                checker[y, x, 2] = value;
            }
        }

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            pack,
            new Dictionary<int, byte[,,]> { [0] = checker },
            new TerrainMinimapCompositionOptions(1, TerrainMinimapLighting.Neutral));

        Assert.Equal(new Rgba32(128, 128, 128, 255), image[0, 0]);
    }

    [Fact]
    public void Compose_AppliesDeclaredLightingAndMcshShadow()
    {
        TerrainTileTensorPack pack = BuildPack(layerOneAlpha: 0f, shadow: 1f);
        var textures = new Dictionary<int, byte[,,]> { [0] = SolidTexture(255, 255, 255) };
        var lighting = new TerrainMinimapLighting(
            Vector3.UnitZ,
            Vector3.One,
            Vector3.Zero,
            McshShadowStrength: 0.5f,
            ApplyMcshToMinimap: true);

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            pack,
            textures,
            new TerrainMinimapCompositionOptions(1, lighting));

        Assert.Equal(new Rgba32(128, 128, 128, 255), image[0, 0]);
    }

    [Fact]
    public void Compose_KeepsMcshOutOfNormalMinimapRgb()
    {
        TerrainTileTensorPack pack = BuildPack(layerOneAlpha: 0f, shadow: 1f);
        var lighting = new TerrainMinimapLighting(
            Vector3.UnitZ,
            Vector3.One,
            Vector3.Zero,
            McshShadowStrength: 1f);

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            pack,
            new Dictionary<int, byte[,,]> { [0] = SolidTexture(255, 255, 255) },
            new TerrainMinimapCompositionOptions(1, lighting));

        Assert.Equal(new Rgba32(255, 255, 255, 255), image[0, 0]);
    }

    [Fact]
    public void Compose_UsesBaseLayerWhenMcalIsAbsent()
    {
        var textureIds = new int[1, 1, 4];
        textureIds[0, 0, 0] = 0;
        textureIds[0, 0, 1] = 1;
        var pack = new TerrainTileTensorPack
        {
            MclyTextureIds = textureIds,
        };

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            pack,
            new Dictionary<int, byte[,,]>
            {
                [0] = SolidTexture(255, 0, 0),
                [1] = SolidTexture(0, 255, 0),
            },
            new TerrainMinimapCompositionOptions(1, TerrainMinimapLighting.Neutral));

        Assert.Equal(new Rgba32(255, 0, 0, 255), image[0, 0]);
    }

    [Fact]
    public void Compose_UsesTheDeclaredCatalogFallbackWhenMclyIsAbsent()
    {
        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            new TerrainTileTensorPack(),
            new Dictionary<int, byte[,,]> { [0] = SolidTexture(80, 120, 40) },
            new TerrainMinimapCompositionOptions(1, TerrainMinimapLighting.Neutral));

        Assert.Equal(new Rgba32(80, 120, 40, 255), image[0, 0]);
    }

    [Fact]
    public void Compose_ToleratesMcnrMaskThatDoesNotMatchNormalGrid()
    {
        var textureIds = new int[1, 1, 4];
        textureIds[0, 0, 0] = 0;
        var pack = new TerrainTileTensorPack
        {
            McalAlphaPack256 = new float[2, 2, 4],
            MclyTextureIds = textureIds,
            McnrNormalXyz = new float[3, 3, 3],
            McnrMask257 = new bool[1, 1],
        };

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            pack,
            new Dictionary<int, byte[,,]> { [0] = SolidTexture(255, 255, 255) },
            new TerrainMinimapCompositionOptions(2, TerrainMinimapLighting.Neutral));

        Assert.Equal(new Rgba32(255, 255, 255, 255), image[1, 1]);
    }

    [Fact]
    public void Compose_InterpolatesMcNrVertexLightingInsteadOfSamplingDenseMaskGaps()
    {
        var normals = new float[3, 3, 3];
        var normalMask = new bool[3, 3];
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                if ((x & 1) != (y & 1))
                    continue;

                normals[y, x, 2] = 1f;
                normalMask[y, x] = true;
            }
        }

        // The first output-pixel centre lies halfway between the top-left outer vertex
        // and the centre inner vertex. Its Lambert term must be 0.5, not UnitZ from a
        // dense-raster gap or an un-interpolated vertex sample.
        normals[1, 1, 2] = -1f;
        TerrainTileTensorPack pack = BuildPack(
            layerOneAlpha: 0f,
            normals: normals,
            normalMask: normalMask);
        var lighting = new TerrainMinimapLighting(
            Vector3.UnitZ,
            Vector3.One,
            Vector3.Zero,
            McshShadowStrength: 0f);

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            pack,
            new Dictionary<int, byte[,,]> { [0] = SolidTexture(255, 255, 255) },
            new TerrainMinimapCompositionOptions(2, lighting));

        Assert.Equal(new Rgba32(128, 128, 128, 255), image[0, 0]);
    }

    [Fact]
    public void Stitch_PreservesTransparentUnoccupiedTileSlots()
    {
        string root = Path.Combine(Path.GetTempPath(), $"wowviewer-minimap-test-{Guid.NewGuid():N}");
        Directory.CreateDirectory(root);
        try
        {
            string firstPath = Path.Combine(root, "first.png");
            string thirdPath = Path.Combine(root, "third.png");
            using (var first = new Image<Rgba32>(2, 2, new Rgba32(255, 0, 0, 255)))
                first.SaveAsPng(firstPath);
            using (var third = new Image<Rgba32>(2, 2, new Rgba32(0, 0, 255, 255)))
                third.SaveAsPng(thirdPath);

            string outputPath = Path.Combine(root, "stitched.png");
            TerrainMinimapStitchResult result = TerrainMinimapStitcher.Stitch(
                new Dictionary<(int TileX, int TileY), string>
                {
                    [(1, 2)] = firstPath,
                    [(3, 2)] = thirdPath
                },
                outputPath,
                tileResolution: 2);

            Assert.Equal((1, 2, 3, 2), (result.MinTileX, result.MinTileY, result.MaxTileX, result.MaxTileY));
            Assert.Equal((6, 2), (result.Width, result.Height));
            using Image<Rgba32> stitched = Image.Load<Rgba32>(outputPath);
            Assert.Equal(new Rgba32(255, 0, 0, 255), stitched[0, 0]);
            Assert.Equal(new Rgba32(0, 0, 0, 0), stitched[2, 0]);
            Assert.Equal(new Rgba32(0, 0, 255, 255), stitched[4, 0]);
        }
        finally
        {
            Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void ComposeLiquid_OverlaysResolvedLiquidTypeWithoutMutatingTerrainBaseline()
    {
        TerrainTileTensorPack pack = new()
        {
            UnifiedLiquidMask = new float[3, 3]
            {
                { 1f, 1f, 0f },
                { 1f, 1f, 0f },
                { 0f, 0f, 0f }
            },
            LiquidBasicType257 = new byte[3, 3]
            {
                { (byte)AdtLiquidBasicType.Ocean, (byte)AdtLiquidBasicType.Ocean, LiquidBasicTypeConstants.NoLiquid },
                { (byte)AdtLiquidBasicType.Ocean, (byte)AdtLiquidBasicType.Ocean, LiquidBasicTypeConstants.NoLiquid },
                { LiquidBasicTypeConstants.NoLiquid, LiquidBasicTypeConstants.NoLiquid, LiquidBasicTypeConstants.NoLiquid }
            }
        };
        using var terrain = new Image<Rgba32>(2, 2, new Rgba32(255, 255, 255, 255));

        using Image<Rgba32> liquid = TerrainMinimapLiquidCompositor.Compose(terrain, pack, out int liquidPixelCount);

        // Ocean follows the live viewer's (0.10, 0.25, 0.55, 0.60) flat-liquid pass.
        Assert.Equal(new Rgba32(117, 140, 186, 255), liquid[0, 0]);
        Assert.Equal(new Rgba32(255, 255, 255, 255), liquid[1, 1]);
        Assert.Equal(new Rgba32(255, 255, 255, 255), terrain[0, 0]);
        Assert.Equal(1, liquidPixelCount);
    }

    [Theory]
    [InlineData((byte)AdtLiquidBasicType.Water, 136, 164, 206)]
    [InlineData((byte)AdtLiquidBasicType.Magma, 226, 121, 73)]
    [InlineData((byte)AdtLiquidBasicType.Slime, 122, 205, 114)]
    public void ComposeLiquid_UsesDistinctDeterministicPaletteForEachBasicType(
        byte basicType,
        byte expectedRed,
        byte expectedGreen,
        byte expectedBlue)
    {
        TerrainTileTensorPack pack = new()
        {
            UnifiedLiquidMask = new float[3, 3]
            {
                { 1f, 1f, 0f },
                { 1f, 1f, 0f },
                { 0f, 0f, 0f }
            },
            LiquidBasicType257 = new byte[3, 3]
            {
                { basicType, basicType, LiquidBasicTypeConstants.NoLiquid },
                { basicType, basicType, LiquidBasicTypeConstants.NoLiquid },
                { LiquidBasicTypeConstants.NoLiquid, LiquidBasicTypeConstants.NoLiquid, LiquidBasicTypeConstants.NoLiquid }
            }
        };
        using var terrain = new Image<Rgba32>(2, 2, new Rgba32(255, 255, 255, 255));

        using Image<Rgba32> liquid = TerrainMinimapLiquidCompositor.Compose(terrain, pack, out int liquidPixelCount);

        Assert.Equal(new Rgba32(expectedRed, expectedGreen, expectedBlue, 255), liquid[0, 0]);
        Assert.Equal(1, liquidPixelCount);
    }

    private static TerrainTileTensorPack BuildPack(
        float layerOneAlpha,
        float layerTwoAlpha = 0f,
        float shadow = 0f,
        bool[,,]? layerMask = null,
        float[,,]? normals = null,
        bool[,]? normalMask = null)
    {
        var alpha = new float[2, 2, 4];
        for (int y = 0; y < 2; y++)
            for (int x = 0; x < 2; x++)
            {
                alpha[y, x, 1] = layerOneAlpha;
                alpha[y, x, 2] = layerTwoAlpha;
            }

        var textureIds = new int[1, 1, 4];
        textureIds[0, 0, 0] = 0;
        textureIds[0, 0, 1] = 1;
        textureIds[0, 0, 2] = 2;
        textureIds[0, 0, 3] = -1;

        return new TerrainTileTensorPack
        {
            TileX = 0,
            TileY = 0,
            McalAlphaPack256 = alpha,
            MclyTextureIds = textureIds,
            MclyLayerMask = layerMask,
            McshShadowMask256 = new[,] { { shadow } },
            McnrNormalXyz = normals,
            McnrMask257 = normalMask
        };
    }

    private static byte[,,] SolidTexture(byte red, byte green, byte blue)
    {
        return new byte[1, 1, 3] { { { red, green, blue } } };
    }
}
