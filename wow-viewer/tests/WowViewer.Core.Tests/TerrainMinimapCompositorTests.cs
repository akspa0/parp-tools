using System.Numerics;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;
using WowViewer.Core.Terrain;

namespace WowViewer.Core.Tests;

public sealed class TerrainMinimapCompositorTests
{
    [Theory]
    [InlineData(0.25f)]
    [InlineData(0.35f)]
    [InlineData(0.50f)]
    [InlineData(0.65f)]
    [InlineData(0.75f)]
    public void WhiteTopEdgeLighting_StaysAchromaticAndKeepsTheRasterSunAboveTerrain(float gameTime)
    {
        TerrainMinimapLighting lighting = TerrainMinimapLighting.CreateWhiteTopEdge(gameTime);

        // The lighting vector is in renderer/world axes (+X = North, +Y = West, +Z = Up). Raw MCNR
        // normals are ADT-grid vectors and the compositor converts them through
        // TerrainNormalGeometry before N.L. The bearing is fixed north-west at every
        // time of day (matching the traced constant-azimuth native ray) instead of sweeping through
        // a zero-horizontal, straight-overhead sun at solar noon.
        // CreateWhiteTopEdge itself stays unscaled -- MinimapShadingMatch sweeps this factory
        // across every hour and needs headroom against clipping; only CreateNoonWhiteGlobal (the
        // production fixed-noon entry point) applies TerrainMinimapLighting.ExposureFactor.
        Assert.Equal(Vector3.One, lighting.DirectionalColor);
        Assert.Equal(0.25f, lighting.AmbientColor.X, 6);
        Assert.Equal(lighting.AmbientColor.X, lighting.AmbientColor.Y, 6);
        Assert.Equal(lighting.AmbientColor.X, lighting.AmbientColor.Z, 6);
        Assert.True(lighting.LightDirection.Z > 0f);
        Assert.True(lighting.LightDirection.X > 0f);
        Assert.True(lighting.LightDirection.Y > 0f);
    }

    [Fact]
    public void NoonWhiteGlobal_IsFixedAchromaticAndDoesNotCrushFlatTerrain()
    {
        TerrainMinimapLighting lighting = TerrainMinimapLighting.CreateNoonWhiteGlobal();

        Assert.Equal(TerrainSolarDirection.Evaluate(0.5f), lighting.LightDirection);
        Assert.Equal(Vector3.One, lighting.DirectionalColor);
        Assert.Equal(new Vector3(0.25f), lighting.AmbientColor);
        Assert.True(lighting.ToneMapped);

        Vector3 flatTerrainLight = TerrainLightingMath.Evaluate(
            Vector3.UnitZ,
            lighting.LightDirection,
            lighting.DirectionalColor,
            lighting.AmbientColor,
            shadowMask: 0f,
            toneMapped: lighting.ToneMapped);
        // Tone-mapped output stays within [0,1) by construction (Reinhard asymptotes to 1, never
        // reaches or exceeds it) -- assert it's meaningfully lifted above the raw 0.25 ambient
        // floor instead of the old raw-linear ">1" overexposure assertion.
        Assert.True(flatTerrainLight.X is > 0.25f and < 1f);
        Assert.Equal(flatTerrainLight.X, flatTerrainLight.Y, 6);
        Assert.Equal(flatTerrainLight.X, flatTerrainLight.Z, 6);
    }

    [Fact]
    public void Compose_TransformsAdtMcnrNormalsToWorldBeforeApplyingSunlight()
    {
        var normals = new float[3, 3, 3];
        var normalMask = new bool[3, 3];
        float component = 1f / MathF.Sqrt(2f);
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                if ((x & 1) != (y & 1))
                    continue;

                // ADT (0,-Y,+Z) transforms to renderer/world (+X,0,+Z).
                normals[y, x, 1] = -component;
                normals[y, x, 2] = component;
                normalMask[y, x] = true;
            }
        }

        TerrainTileTensorPack pack = BuildPack(
            layerOneAlpha: 0f,
            normals: normals,
            normalMask: normalMask);
        var lighting = new TerrainMinimapLighting(
            Vector3.Normalize(new Vector3(1f, 0f, 1f)),
            Vector3.One,
            Vector3.Zero,
            McshShadowStrength: 0f);

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            pack,
            new Dictionary<int, byte[,,]> { [0] = SolidTexture(255, 255, 255) },
            new TerrainMinimapCompositionOptions(1, lighting));

        // Correct world-space N.L is 1.0. The old raw-ADT/world dot was 0.5 and rendered gray.
        Assert.Equal(new Rgba32(255, 255, 255, 255), image[0, 0]);
    }

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
            MclyTextureIds = textureIds,
            MclyTextureNames = ["test_0.blp", "test_1.blp", "test_2.blp", "test_3.blp"]
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
            MclyTextureIds = textureIds,
            MclyTextureNames = ["test_0.blp"]
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
            MclyTextureNames = ["test_0.blp", "test_1.blp"],
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

    /// <summary>
    /// A tile with no MTEX names is UNTEXTURED, not empty: it still carries real MCVT heights and
    /// MCNR normals. It renders on a neutral white base so its terrain shape survives, rather than
    /// short-circuiting to a flat white image that discards it. The base must not be coloured from
    /// an unrelated catalog BLP, which is what the supplied texture here would be.
    /// </summary>
    [Fact]
    public void Compose_RendersAnUntexturedTileOnANeutralWhiteBase()
    {
        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            new TerrainTileTensorPack(),
            new Dictionary<int, byte[,,]> { [0] = SolidTexture(80, 120, 40) },
            new TerrainMinimapCompositionOptions(1, TerrainMinimapLighting.Neutral));

        Assert.Equal(new Rgba32(255, 255, 255, 255), image[0, 0]);
    }

    [Fact]
    public void Compose_ShadesAnUntexturedTileInsteadOfDiscardingItsShape()
    {
        // Flat normals with a light 60 degrees off vertical: lambert 0.5, so a shaded untextured
        // tile lands strictly between black and white. The pre-v0.5.2 path returned solid white
        // here regardless of lighting, erasing the tile's terrain shape.
        var lighting = new TerrainMinimapLighting(
            new Vector3(MathF.Sqrt(3f) / 2f, 0f, 0.5f),
            Vector3.One,
            Vector3.Zero,
            McshShadowStrength: 0f);

        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            new TerrainTileTensorPack(),
            new Dictionary<int, byte[,,]> { [0] = SolidTexture(80, 120, 40) },
            new TerrainMinimapCompositionOptions(1, lighting));

        Assert.InRange(image[0, 0].R, 120, 140);
        Assert.Equal(image[0, 0].R, image[0, 0].G);
        Assert.Equal(image[0, 0].R, image[0, 0].B);
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
            MclyTextureNames = ["test_0.blp"],
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
            // Fully wet: bilinear sampling of a per-vertex mask puts the waterline half a cell in
            // from the last wet vertex, so a shoreline fixture no longer has any 100%-covered pixel.
            // Liquid-type and palette behaviour is what this test is about, so keep it interior.
            UnifiedLiquidMask = new float[3, 3]
            {
                { 1f, 1f, 1f },
                { 1f, 1f, 1f },
                { 1f, 1f, 1f }
            },
            LiquidBasicType257 = new byte[3, 3]
            {
                { (byte)AdtLiquidBasicType.Ocean, (byte)AdtLiquidBasicType.Ocean, (byte)AdtLiquidBasicType.Ocean },
                { (byte)AdtLiquidBasicType.Ocean, (byte)AdtLiquidBasicType.Ocean, (byte)AdtLiquidBasicType.Ocean },
                { (byte)AdtLiquidBasicType.Ocean, (byte)AdtLiquidBasicType.Ocean, (byte)AdtLiquidBasicType.Ocean }
            }
        };
        using var terrain = new Image<Rgba32>(2, 2, new Rgba32(255, 255, 255, 255));

        using Image<Rgba32> liquid = TerrainMinimapLiquidCompositor.Compose(
            terrain, pack, out int liquidPixelCount, MinimapLiquidPalette.ViewerFlatV1);

        // Ocean follows the live viewer's (0.10, 0.25, 0.55, 0.60) flat-liquid pass. Pinned to the
        // ViewerFlatV1 palette explicitly: the default is now the 0.5.3-era teal, and this test is
        // about the viewer-matching palette specifically.
        Assert.Equal(new Rgba32(117, 140, 186, 255), liquid[0, 0]);
        Assert.Equal(new Rgba32(117, 140, 186, 255), liquid[1, 1]);
        Assert.Equal(new Rgba32(255, 255, 255, 255), terrain[0, 0]);
        Assert.Equal(4, liquidPixelCount);
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
                { 1f, 1f, 1f },
                { 1f, 1f, 1f },
                { 1f, 1f, 1f }
            },
            LiquidBasicType257 = new byte[3, 3]
            {
                { basicType, basicType, basicType },
                { basicType, basicType, basicType },
                { basicType, basicType, basicType }
            }
        };
        using var terrain = new Image<Rgba32>(2, 2, new Rgba32(255, 255, 255, 255));

        using Image<Rgba32> liquid = TerrainMinimapLiquidCompositor.Compose(
            terrain, pack, out int liquidPixelCount, MinimapLiquidPalette.ViewerFlatV1);

        Assert.Equal(new Rgba32(expectedRed, expectedGreen, expectedBlue, 255), liquid[0, 0]);
        Assert.Equal(4, liquidPixelCount);
    }

    /// <summary>
    /// Renders one flat mid-grey pixel under a given lighting profile and light direction, and
    /// returns its red channel. Flat (null) normals mean N is UnitZ everywhere, so the light
    /// direction alone sets the Lambert term: UnitZ gives 1.0, a horizontal direction gives 0.0.
    /// </summary>
    private static byte RenderFlatGrey(TerrainMinimapLighting profile, Vector3 lightDirection)
    {
        using Image<Rgba32> image = TerrainMinimapCompositor.Compose(
            BuildPack(layerOneAlpha: 0f),
            new Dictionary<int, byte[,,]> { [0] = SolidTexture(128, 128, 128) },
            new TerrainMinimapCompositionOptions(1, profile with { LightDirection = lightDirection }));
        return image[0, 0].R;
    }

    /// <summary>
    /// The regression guard for the flat-render bug: synthesized minimaps read as shadowless next
    /// to authored ones because the exposure-20 Reinhard curve squeezed the entire Lambert range
    /// into ~13% of albedo. Exposure 20 was fitted against MEAN authored brightness only, and a
    /// saturating curve fixes the mean by destroying the range.
    ///
    /// This asserts both halves of the fix at once: linear-space shading must recover several times
    /// the shading range WITHOUT drifting the brightness the old calibration matched.
    /// </summary>
    [Fact]
    public void Compose_LinearSpaceShadingRestoresShadingRangeAtTheSameFlatGroundBrightness()
    {
        Vector3 fullyLit = Vector3.UnitZ;                                  // lambert 1
        Vector3 grazing = Vector3.UnitX;                                   // lambert 0
        Vector3 flatGroundAtNoon = TerrainSolarDirection.Evaluate(0.5f);   // lambert 0.894

        TerrainMinimapLighting toneMapped = TerrainMinimapLighting.CreateNoonWhiteGlobal();
        TerrainMinimapLighting linear = TerrainMinimapLighting.CreateShadedTerrain(0.5f);

        int toneMappedRange = RenderFlatGrey(toneMapped, fullyLit) - RenderFlatGrey(toneMapped, grazing);
        int linearRange = RenderFlatGrey(linear, fullyLit) - RenderFlatGrey(linear, grazing);

        Assert.True(
            linearRange > toneMappedRange * 4,
            $"Linear-space shading range {linearRange} must far exceed the tone-mapped {toneMappedRange}.");

        // The brightness anchor is FLAT GROUND UNDER THE NOON SUN, not an arbitrary mid-lambert.
        // The first cut of this fix anchored at lambert 0.5 while real terrain sits at 0.894, which
        // rendered ordinary ground 19% too bright and read as washed out.
        Assert.InRange(
            RenderFlatGrey(linear, flatGroundAtNoon) - RenderFlatGrey(toneMapped, flatGroundAtNoon),
            -4,
            4);
    }

    /// <summary>
    /// Ambient is the floor every shadowed pixel lands on, so it is the real contrast control. The
    /// derived gain must keep it from doubling as a brightness control: changing ambient has to move
    /// shadow depth while leaving lit flat ground where it was, or every contrast tweak turns into a
    /// brightness regression.
    /// </summary>
    [Fact]
    public void ShadedTerrain_AmbientChangesShadowDepthWithoutChangingOverallBrightness()
    {
        Vector3 flatGroundAtNoon = TerrainSolarDirection.Evaluate(0.5f);
        Vector3 grazing = Vector3.UnitX;

        TerrainMinimapLighting high = TerrainMinimapLighting.CreateShadedTerrain(0.5f, 0.25f, 0.70f);
        TerrainMinimapLighting low = TerrainMinimapLighting.CreateShadedTerrain(0.5f, 0.08f, 0.70f);

        Assert.InRange(
            RenderFlatGrey(low, flatGroundAtNoon) - RenderFlatGrey(high, flatGroundAtNoon),
            -4,
            4);
        Assert.True(
            RenderFlatGrey(low, grazing) < RenderFlatGrey(high, grazing) - 20,
            "Lower ambient must produce materially darker unlit terrain.");
    }

    [Fact]
    public void ShadedTerrain_TintedAmbientColorsShadowsWithoutTintingLitGround()
    {
        // Shadowed ground is lit by ambient alone, so a cool ambient is what makes shadows a
        // different hue rather than just a darker copy of lit ground.
        var coolAmbient = new Vector3(0.09f, 0.11f, 0.16f);
        TerrainMinimapLighting tinted = TerrainMinimapLighting.CreateShadedTerrain(0.5f, coolAmbient, 0.70f);

        using Image<Rgba32> unlit = TerrainMinimapCompositor.Compose(
            BuildPack(layerOneAlpha: 0f),
            new Dictionary<int, byte[,,]> { [0] = SolidTexture(128, 128, 128) },
            new TerrainMinimapCompositionOptions(1, tinted with { LightDirection = Vector3.UnitX }));
        using Image<Rgba32> lit = TerrainMinimapCompositor.Compose(
            BuildPack(layerOneAlpha: 0f),
            new Dictionary<int, byte[,,]> { [0] = SolidTexture(128, 128, 128) },
            new TerrainMinimapCompositionOptions(1, tinted with { LightDirection = Vector3.UnitZ }));

        Assert.True(unlit[0, 0].B > unlit[0, 0].R + 10, "Shadowed ground must take the ambient's tint.");
        Assert.InRange(lit[0, 0].B - lit[0, 0].R, -6, 6);
    }

    [Fact]
    public void ShadedTerrain_EnablesCastShadowsAndLinearShadingWithoutTheToneMap()
    {
        TerrainMinimapLighting lighting = TerrainMinimapLighting.CreateShadedTerrain(0.5f);

        Assert.True(lighting.ApplyCastShadows);
        Assert.True(lighting.LinearSpaceShading);
        Assert.False(lighting.ToneMapped);
        Assert.False(lighting.ApplyMcshToMinimap);
        Assert.Equal(TerrainSolarDirection.Evaluate(0.5f), lighting.LightDirection);
    }

    /// <summary>
    /// Cast shadows and linear shading must stay opt-in on the base factory. MinimapShadingMatch
    /// sweeps CreateWhiteTopEdge across all 24 hours and its Spec 111 calibration is defined
    /// against the legacy response; silently changing it would invalidate every bucketed tile.
    /// </summary>
    [Fact]
    public void WhiteTopEdge_LeavesCastShadowsAndLinearShadingOff()
    {
        TerrainMinimapLighting lighting = TerrainMinimapLighting.CreateWhiteTopEdge(0.5f);

        Assert.False(lighting.ApplyCastShadows);
        Assert.False(lighting.LinearSpaceShading);
        Assert.False(lighting.ToneMapped);
    }

    [Fact]
    public void Compose_CastShadowsDarkenGroundBehindARidgeThatLambertAloneLeavesLit()
    {
        // Flat ground (all normals UnitZ, so Lambert is uniform across the whole tile) with one
        // tall wall in the heightfield. Any darkening here is cast shadow by construction: Lambert
        // cannot vary on a surface whose normal never changes.
        const int Size = 257;
        const int WallRow = 100;
        var height = new float[Size, Size];
        for (int column = 0; column < Size; column++)
            height[WallRow, column] = 300f;

        TerrainTileTensorPack pack = BuildPack(layerOneAlpha: 0f, height257: height);
        var textures = new Dictionary<int, byte[,,]> { [0] = SolidTexture(200, 200, 200) };
        // Pin the strength rather than inheriting the default: this test is about cast shadows
        // existing and landing on the correct side, not about whatever the tuned default happens
        // to be (which has moved as the solar model was corrected).
        TerrainMinimapLighting lighting = TerrainMinimapLighting.CreateShadedTerrain(0.5f, 0.12f, 0.70f);

        using Image<Rgba32> withShadows = TerrainMinimapCompositor.Compose(
            pack, textures, new TerrainMinimapCompositionOptions(Size, lighting));
        using Image<Rgba32> withoutShadows = TerrainMinimapCompositor.Compose(
            pack, textures, new TerrainMinimapCompositionOptions(Size, lighting with { ApplyCastShadows = false }));

        // The sun bears north-west (world +X/+Y = grid -row/-column), so the wall shadows the rows
        // below it and leaves the rows above it lit.
        Assert.True(
            withShadows[200, WallRow + 4].R < withoutShadows[200, WallRow + 4].R - 20,
            "Ground behind the ridge must be materially darker once cast shadows are enabled.");
        Assert.Equal(withoutShadows[200, WallRow - 4].R, withShadows[200, WallRow - 4].R);
    }

    private static TerrainTileTensorPack BuildPack(
        float layerOneAlpha,
        float layerTwoAlpha = 0f,
        float shadow = 0f,
        bool[,,]? layerMask = null,
        float[,,]? normals = null,
        bool[,]? normalMask = null,
        float[,]? height257 = null)
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
            MclyTextureNames = ["test_0.blp", "test_1.blp", "test_2.blp"],
            MclyLayerMask = layerMask,
            McshShadowMask256 = new[,] { { shadow } },
            McnrNormalXyz = normals,
            McnrMask257 = normalMask,
            Height257 = height257
        };
    }

    private static byte[,,] SolidTexture(byte red, byte green, byte blue)
    {
        return new byte[1, 1, 3] { { { red, green, blue } } };
    }
}
