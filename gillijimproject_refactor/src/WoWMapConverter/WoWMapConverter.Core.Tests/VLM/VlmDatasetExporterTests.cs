using WoWMapConverter.Core.VLM;
using Xunit;
using SixLabors.ImageSharp;
using SixLabors.ImageSharp.PixelFormats;
using System.Text;
using System.Numerics;
using WowViewer.Core.IO.Files;

namespace WoWMapConverter.Core.Tests.VLM;

public sealed class VlmDatasetExporterTests
{
    [Fact]
    public void TryResolveArchiveMapDirectoryAlias_ExactNormalizedMatch_ReturnsDirectory()
    {
        string[] knownFiles =
        [
            "World/Maps/LostIsles/LostIsles.wdt"
        ];

        string? resolved = VlmDatasetExporter.TryResolveArchiveMapDirectoryAlias("Lost Isles", knownFiles);

        Assert.Equal("LostIsles", resolved);
    }

    [Fact]
    public void TryResolveArchiveMapDirectoryAlias_NearMatch_ReturnsBestDirectory()
    {
        string[] knownFiles =
        [
            "World/Maps/Deephome/Deephome.wdt",
            "World/Maps/EmeraldDream/EmeraldDream.wdt"
        ];

        string? resolved = VlmDatasetExporter.TryResolveArchiveMapDirectoryAlias("Deepholm", knownFiles);

        Assert.Equal("Deephome", resolved);
    }

    [Fact]
    public void MergeLegacyLiquids_KeepsLegacyChunksNotCoveredByMh2o()
    {
        VlmLiquidData[] mh2oLiquids =
        [
            new(ChunkIndex: 5, LiquidType: 0, MinHeight: 1f, MaxHeight: 2f, MaskPath: null, Heights: null),
            new(ChunkIndex: 9, LiquidType: 0, MinHeight: 1f, MaxHeight: 2f, MaskPath: null, Heights: null)
        ];

        VlmLiquidData[] legacyLiquids =
        [
            new(ChunkIndex: 5, LiquidType: 0, MinHeight: 1f, MaxHeight: 2f, MaskPath: null, Heights: null),
            new(ChunkIndex: 7, LiquidType: 0, MinHeight: 1f, MaxHeight: 2f, MaskPath: null, Heights: null)
        ];

        VlmLiquidData[] merged = VlmDatasetExporter.MergeLegacyLiquids(mh2oLiquids, legacyLiquids).ToArray();

        Assert.Single(merged);
        Assert.Equal(7, merged[0].ChunkIndex);
    }

    [Fact]
    public void RenderMccvImage_PreservesRawStoredChannelOrder()
    {
        byte[] chunkColors = new byte[145 * 4];
        for (int index = 0; index < 145; index++)
        {
            int offset = index * 4;
            chunkColors[offset + 0] = 30;
            chunkColors[offset + 1] = 20;
            chunkColors[offset + 2] = 10;
            chunkColors[offset + 3] = 255;
        }

        Dictionary<int, byte[]> mccv = new()
        {
            [0] = chunkColors
        };

        byte[] imageBytes = VlmDatasetExporter.RenderMccvImage(mccv, 145);

        using Image<Rgba32> image = Image.Load<Rgba32>(imageBytes);
        Assert.Equal(new Rgba32(30, 20, 10, 255), image[0, 0]);
    }

    [Fact]
    public void RenderMccvImage_MissingChunksUseNeutralRawValue()
    {
        byte[] imageBytes = VlmDatasetExporter.RenderMccvImage(new Dictionary<int, byte[]>(), 145);

        using Image<Rgba32> image = Image.Load<Rgba32>(imageBytes);
        Assert.Equal(new Rgba32(127, 127, 127, 127), image[72, 72]);
    }

    [Fact]
    public void ReadVirtualAssetBytes_LooseFileOverridesArchive()
    {
        string root = Path.Combine(Path.GetTempPath(), $"vlm-loose-override-{Guid.NewGuid():N}");
        try
        {
            string assetDirectory = Path.Combine(root, "World", "Maps", "Azeroth");
            Directory.CreateDirectory(assetDirectory);

            string loosePath = Path.Combine(assetDirectory, "Azeroth_0_0.adt");
            File.WriteAllBytes(loosePath, Encoding.UTF8.GetBytes("loose"));

            FakeArchiveReader archiveReader = new(new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase)
            {
                ["World\\Maps\\Azeroth\\Azeroth_0_0.adt"] = Encoding.UTF8.GetBytes("archive")
            });

            byte[]? bytes = VlmDatasetExporter.ReadVirtualAssetBytes(
                [root],
                "World/Maps/Azeroth/Azeroth_0_0.adt",
                archiveReader);

            Assert.NotNull(bytes);
            Assert.Equal("loose", Encoding.UTF8.GetString(bytes!));
        }
        finally
        {
            if (Directory.Exists(root))
                Directory.Delete(root, recursive: true);
        }
    }

    [Fact]
    public void TryParseTileFilter_ValidCoordinate_ReturnsExpectedTile()
    {
        bool parsed = VlmDatasetExporter.TryParseTileFilter("23_24", out int tileX, out int tileY);

        Assert.True(parsed);
        Assert.Equal(23, tileX);
        Assert.Equal(24, tileY);
    }

    [Fact]
    public void GetTerrainOnlyMaskPaths_ExcludesShadowArtifacts()
    {
        string[] alphaPaths =
        [
            "stitched/tile_alpha_0.png",
            "stitched/tile_alpha_1.png"
        ];

        string[] filtered = VlmDatasetExporter.GetTerrainOnlyMaskPaths(alphaPaths, "stitched/tile_shadow.png").ToArray();

        Assert.Equal(
            ["stitched/tile_alpha_0.png", "stitched/tile_alpha_1.png"],
            filtered);
    }

    [Fact]
    public void ResolveChunkTextureFallbackPaths_UsesNearestChunkBaseTexture()
    {
        VlmChunkLayers[] chunks =
        [
            new(
                ChunkIndex: 0,
                Layers:
                [
                    new(TextureId: 1, TexturePath: "Textures/terrain/grass.blp", Flags: 0, AlphaOffset: 0, EffectId: 0)
                ]),
            new(
                ChunkIndex: 17,
                Layers: [])
        ];

        string?[] resolved = VlmDatasetExporter.ResolveChunkTextureFallbackPaths(chunks);

        Assert.Equal("Textures/terrain/grass.blp", resolved[0]);
        Assert.Equal("Textures/terrain/grass.blp", resolved[17]);
    }

    [Fact]
    public async Task ReplaceMaskedPixelsWithBakedChunksAsync_UsesNearestChunkBaseTextureFallback()
    {
        using Image<Rgba32> result = new(32, 32);
        using Image<L8> mask = new(32, 32);

        for (int y = 0; y < result.Height; y++)
        {
            for (int x = 0; x < result.Width; x++)
                result[x, y] = new Rgba32(10, 20, 30, 255);
        }

        for (int y = 2; y < 4; y++)
        {
            for (int x = 2; x < 4; x++)
                mask[x, y] = new L8(255);
        }

        VlmChunkLayers[] chunks =
        [
            new(
                ChunkIndex: 0,
                Layers:
                [
                    new(TextureId: 1, TexturePath: "Textures/terrain/grass.blp", Flags: 0, AlphaOffset: 0, EffectId: 0)
                ]),
            new(
                ChunkIndex: 17,
                Layers: [])
        ];

        List<(int ChunkIndex, string? FallbackTexturePath)> calls = [];

        async Task<Image<Rgba32>?> BakeChunkAsync(VlmChunkLayers chunk, string? fallbackTexturePath)
        {
            calls.Add((chunk.ChunkIndex, fallbackTexturePath));

            if (chunk.ChunkIndex != 17)
                return null;

            Image<Rgba32> baked = new(2, 2);
            for (int y = 0; y < baked.Height; y++)
            {
                for (int x = 0; x < baked.Width; x++)
                    baked[x, y] = new Rgba32(200, 150, 100, 255);
            }

            return await Task.FromResult<Image<Rgba32>?>(baked);
        }

        bool replaced = await VlmDatasetExporter.ReplaceMaskedPixelsWithBakedChunksAsync(result, mask, chunks, BakeChunkAsync);

        Assert.True(replaced);
        Assert.Single(calls);
        Assert.Equal((17, "Textures/terrain/grass.blp"), calls[0]);
        Assert.Equal(new Rgba32(200, 150, 100, 255), result[2, 2]);
        Assert.Equal(new Rgba32(10, 20, 30, 255), result[0, 0]);
    }

    [Fact]
    public void RenderChunkValueMap_FillsChunkBlocksWithUInt16Values()
    {
        ushort[] values = new ushort[256];
        values[0] = 1024;
        values[17] = 2048;

        byte[] imageBytes = VlmDatasetExporter.RenderChunkValueMap(values, 32, 32);

        using Image<L16> image = Image.Load<L16>(imageBytes);
        Assert.Equal((ushort)1024, image[0, 0].PackedValue);
        Assert.Equal((ushort)2048, image[2, 2].PackedValue);
    }

    [Fact]
    public void RenderChunkFlagMap_PacksUInt32FlagsIntoRgbaChannels()
    {
        uint[] flags = new uint[256];
        flags[0] = 0x12345678;

        byte[] imageBytes = VlmDatasetExporter.RenderChunkFlagMap(flags, 16, 16);

        using Image<Rgba32> image = Image.Load<Rgba32>(imageBytes);
        Assert.Equal(new Rgba32(0x78, 0x56, 0x34, 0x12), image[0, 0]);
    }

    [Fact]
    public void RenderHolesMask_MapsChunkHoleBitsToTilePixels()
    {
        int[] holes = new int[256];
        holes[0] = 1 << 5;

        byte[] imageBytes = VlmDatasetExporter.RenderHolesMask(holes, 64, 64);

        using Image<L8> image = Image.Load<L8>(imageBytes);
        Assert.Equal((byte)255, image[1, 1].PackedValue);
        Assert.Equal((byte)0, image[0, 0].PackedValue);
    }

    [Fact]
    public void TransformFootprintPolygonToWorldForTesting_UsesRotYForXzFootprints()
    {
        Vector2[] localPolygon =
        [
            new Vector2(1f, 0f),
            new Vector2(0f, 1f),
            new Vector2(-1f, 0f),
        ];
        VlmObjectPlacement obj = new(
            Name: "test",
            NameId: 1,
            UniqueId: 1,
            X: 100f,
            Y: 200f,
            Z: 300f,
            RotX: 0f,
            RotY: 90f,
            RotZ: 0f,
            Scale: 1f,
            Category: "m2");

        Vector2[] transformed = VlmDatasetExporter.TransformFootprintPolygonToWorldForTesting(localPolygon, obj, secondaryAxisIsZ: true);

        Assert.Equal(3, transformed.Length);
        Assert.Equal(100f, transformed[0].X, 3);
        Assert.Equal(301f, transformed[0].Y, 3);
        Assert.Equal(99f, transformed[1].X, 3);
        Assert.Equal(300f, transformed[1].Y, 3);
        Assert.Equal(100f, transformed[2].X, 3);
        Assert.Equal(299f, transformed[2].Y, 3);
    }

    [Fact]
    public void TransformFootprintPolygonToWorldForTesting_UsesRotZForXyFallbackFootprints()
    {
        Vector2[] localPolygon =
        [
            new Vector2(1f, 0f),
            new Vector2(0f, 1f),
        ];
        VlmObjectPlacement obj = new(
            Name: "test",
            NameId: 1,
            UniqueId: 1,
            X: 100f,
            Y: 200f,
            Z: 300f,
            RotX: 0f,
            RotY: 0f,
            RotZ: 90f,
            Scale: 1f,
            Category: "m2");

        Vector2[] transformed = VlmDatasetExporter.TransformFootprintPolygonToWorldForTesting(localPolygon, obj, secondaryAxisIsZ: false);

        Assert.Equal(2, transformed.Length);
        Assert.Equal(100f, transformed[0].X, 3);
        Assert.Equal(201f, transformed[0].Y, 3);
        Assert.Equal(99f, transformed[1].X, 3);
        Assert.Equal(200f, transformed[1].Y, 3);
    }

    [Fact]
    public void ScoreTileContentForTesting_UsesTerrainAndObjectSignals()
    {
        FakeArchiveReader archiveReader = new(new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["World\\Maps\\Azeroth\\Azeroth_1_2.adt"] = BuildChunkedFile("MH2O", "MDDF", "MCLQ", "MCNK"),
            ["World\\Maps\\Azeroth\\Azeroth_1_2_tex0.adt"] = BuildChunkedFile("MCAL", "MCLY"),
            ["World\\Maps\\Azeroth\\Azeroth_1_2_obj0.adt"] = BuildChunkedFile("MODF")
        });

        int score = VlmDatasetExporter.ScoreTileContentForTesting("Azeroth", 1, 2, false, archiveReader, Array.Empty<string>());

        Assert.Equal(218, score);
    }

    [Fact]
    public void SelectTilesForProcessingForTesting_InterestingOnlyPrefersScoredTiles()
    {
        FakeArchiveReader archiveReader = new(new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase)
        {
            ["World\\Maps\\Azeroth\\Azeroth_0_0.adt"] = BuildChunkedFile("MCNK"),
            ["World\\Maps\\Azeroth\\Azeroth_1_0.adt"] = BuildChunkedFile("MH2O")
        });

        List<int> selected = VlmDatasetExporter.SelectTilesForProcessingForTesting(
            [0, 1],
            1,
            false,
            "Azeroth",
            archiveReader,
            Array.Empty<string>(),
            true,
            1);

        Assert.Equal([1], selected);
    }

    [Fact]
    public void SelectTilesForProcessingForTesting_InterestingOnlyFallsBackToSingleRepresentativeTile()
    {
        FakeArchiveReader archiveReader = new(new Dictionary<string, byte[]>(StringComparer.OrdinalIgnoreCase));

        List<int> selected = VlmDatasetExporter.SelectTilesForProcessingForTesting(
            [0, 1, 2],
            32,
            false,
            "Azeroth",
            archiveReader,
            Array.Empty<string>(),
            true,
            1);

        Assert.Single(selected);
        Assert.Equal(1, selected[0]);
    }

    [Theory]
    [InlineData("")]
    [InlineData("23")]
    [InlineData("23_64")]
    [InlineData("abc_def")]
    public void TryParseTileFilter_InvalidCoordinate_ReturnsFalse(string value)
    {
        bool parsed = VlmDatasetExporter.TryParseTileFilter(value, out _, out _);

        Assert.False(parsed);
    }

    private sealed class FakeArchiveReader : IArchiveReader
    {
        private readonly Dictionary<string, byte[]> _files;

        public FakeArchiveReader(Dictionary<string, byte[]> files)
        {
            _files = files;
        }

        public bool FileExists(string virtualPath)
        {
            return _files.ContainsKey(virtualPath.Replace('/', '\\'));
        }

        public byte[]? ReadFile(string virtualPath)
        {
            return _files.TryGetValue(virtualPath.Replace('/', '\\'), out byte[]? bytes) ? bytes : null;
        }
    }

    private static byte[] BuildChunkedFile(params string[] fourCcs)
    {
        using MemoryStream stream = new();
        foreach (string fourCc in fourCcs)
        {
            stream.Write(Encoding.ASCII.GetBytes(fourCc));
            stream.Write(BitConverter.GetBytes(0));
        }

        return stream.ToArray();
    }
}