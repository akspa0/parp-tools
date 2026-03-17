using MdxViewer.Export;
using MdxViewer.DataSources;
using MdxViewer.Terrain;
using System.Numerics;
using WoWMapConverter.Core.Formats.Liquids;
using WoWMapConverter.Core.Formats.LichKing;
using Xunit;

namespace MdxViewer.Tests.Terrain;

public sealed class TerrainAlphaParityTests
{
    [Fact]
    public void BuildAlphaShadowArray_PreservesExplicitEdgeTexels()
    {
        var chunk = CreateExplicitChunk(chunkX: 4, chunkY: 2);

        var packed = TerrainTileMeshBuilder.BuildAlphaShadowArray(new[] { chunk });
        var slice = ExtractSlice(packed, chunk.ChunkX, chunk.ChunkY);

        Assert.Equal((byte)17, slice[GetChannelOffset(38, 63, 0)]);
        Assert.Equal((byte)41, slice[GetChannelOffset(63, 12, 1)]);
        Assert.Equal((byte)99, slice[GetChannelOffset(63, 63, 2)]);
        Assert.Equal((byte)255, slice[GetChannelOffset(63, 31, 3)]);
    }

    [Fact]
    public void AlphaAtlas_RoundTrip_MatchesPackedSlice_ForExplicitLayers()
    {
        var chunk = CreateExplicitChunk(chunkX: 6, chunkY: 2);

        var packed = TerrainTileMeshBuilder.BuildAlphaShadowArray(new[] { chunk });
        var packedSlice = ExtractSlice(packed, chunk.ChunkX, chunk.ChunkY);

        using var atlas = TerrainImageIo.BuildAlphaAtlasFromChunks(new[] { chunk });
        var roundTripped = TerrainImageIo.DecodeAlphaShadowArrayFromAtlas(atlas);
        var atlasSlice = ExtractSlice(roundTripped, chunk.ChunkX, chunk.ChunkY);

        Assert.Equal(packedSlice, atlasSlice);
        Assert.Equal((byte)17, atlasSlice[GetChannelOffset(38, 63, 0)]);
        Assert.Equal((byte)41, atlasSlice[GetChannelOffset(63, 12, 1)]);
        Assert.Equal((byte)99, atlasSlice[GetChannelOffset(63, 63, 2)]);
        Assert.Equal((byte)255, atlasSlice[GetChannelOffset(63, 31, 3)]);
    }

    [Fact]
    public void AlphaTerrainExtractAlphaMaps_AppliesLegacyEdgeFixToFourBitAlpha()
    {
        var mcal = new byte[2048];
        mcal[31] = 0x73;
        mcal[31 + (62 * 32)] = 0xB5;

        var mcly = new byte[32];
        var maps = AlphaTerrainAdapter.ExtractAlphaMaps(mcal, mcly, 2);

        Assert.True(maps.TryGetValue(1, out var alpha));
        Assert.NotNull(alpha);
        Assert.Equal((byte)(3 * 17), alpha![62]);
        Assert.Equal((byte)(3 * 17), alpha[63]);
        Assert.Equal((byte)(5 * 17), alpha[62 * 64 + 62]);
        Assert.Equal((byte)(5 * 17), alpha[63 * 64 + 62]);
        Assert.Equal((byte)(5 * 17), alpha[63 * 64 + 63]);
    }

    [Fact]
    public void AlphaTerrainExtractAlphaMaps_FlaggedWidePayload_IsCopiedRaw()
    {
        var mcal = new byte[4096];
        mcal[0] = 12;
        mcal[63] = 34;
        mcal[4095] = 56;

        var mcly = new byte[32];
        BitConverter.GetBytes(0x200u).CopyTo(mcly, 20);

        var maps = AlphaTerrainAdapter.ExtractAlphaMaps(mcal, mcly, 2);

        Assert.True(maps.TryGetValue(1, out var alpha));
        Assert.NotNull(alpha);
        Assert.Equal(4096, alpha!.Length);
        Assert.Equal((byte)12, alpha[0]);
        Assert.Equal((byte)34, alpha[63]);
        Assert.Equal((byte)56, alpha[4095]);
    }

    [Fact]
    public void TerrainTileMeshBuilder_BuildIndices_CanIgnoreHoleMask()
    {
        var withHoles = TerrainTileMeshBuilder.BuildIndices(0x1, ignoreHoleMask: false);
        var withoutHoles = TerrainTileMeshBuilder.BuildIndices(0x1, ignoreHoleMask: true);

        Assert.Equal(720, withHoles.Length);
        Assert.Equal(768, withoutHoles.Length);
    }

    [Fact]
    public void StandardTerrainAdapter_LegacyProfiles_IgnoreLkStyleLayerFlags()
    {
        var textureLayers = new List<MclyEntry>
        {
            new() { TextureId = 0, Flags = 0, AlphaMapOffset = 0, EffectId = 0 },
            new() { TextureId = 1, Flags = MclyFlags.UseAlpha | MclyFlags.CompressedAlpha, AlphaMapOffset = 0, EffectId = 0 },
        };

        var mcalRawData = new byte[2048];
        mcalRawData[0] = 0x21;

        var maps = StandardTerrainAdapter.ExtractAlphaMaps(
            textureLayers,
            new Mcal(mcalRawData),
            mcalRawData,
            TerrainAlphaDecodeMode.LegacySequential,
            useBigAlpha: false,
            doNotFixAlphaMap: false);

        Assert.True(maps.TryGetValue(1, out var alpha));
        Assert.NotNull(alpha);
        Assert.Equal((byte)(1 * 17), alpha![0]);
        Assert.Equal((byte)(2 * 17), alpha[1]);
    }

    [Fact]
    public void StandardTerrainAdapter_LegacyProfiles_PreserveFinalHighNibble()
    {
        var textureLayers = new List<MclyEntry>
        {
            new() { TextureId = 0, Flags = 0, AlphaMapOffset = 0, EffectId = 0 },
            new() { TextureId = 1, Flags = 0, AlphaMapOffset = 0, EffectId = 0 },
        };

        var mcalRawData = new byte[2048];
        mcalRawData[31] = 0x73;

        var maps = StandardTerrainAdapter.ExtractAlphaMaps(
            textureLayers,
            alphaDecoder: null,
            mcalRawData,
            TerrainAlphaDecodeMode.LegacySequential,
            useBigAlpha: false,
            doNotFixAlphaMap: false);

        var alpha = Assert.Contains(1, maps);
        Assert.Equal((byte)(3 * 17), alpha[62]);
        Assert.Equal((byte)(7 * 17), alpha[63]);
    }

    [Fact]
    public void StandardTerrainAdapter_BuildMh2oLiquid_UsesExistsBitmapAndLayerPrecedence()
    {
        var water = new Mh2oInstance
        {
            LiquidTypeId = 1,
            VertexFormat = Mh2oVertexFormat.HeightDepth,
            MinHeightLevel = 10f,
            MaxHeightLevel = 10f,
            XOffset = 0,
            YOffset = 0,
            Width = 2,
            Height = 2,
            HeightMap = CreateFilledHeights(3, 3, 10f),
            DepthMap = CreateFilledDepth(9, 1),
        };

        var magma = new Mh2oInstance
        {
            LiquidTypeId = 5,
            VertexFormat = Mh2oVertexFormat.HeightDepth,
            MinHeightLevel = 30f,
            MaxHeightLevel = 30f,
            XOffset = 1,
            YOffset = 1,
            Width = 2,
            Height = 1,
            ExistsBitmap = new byte[] { 0b0000_0001 },
            HeightMap = CreateFilledHeights(3, 2, 30f),
            DepthMap = CreateFilledDepth(6, 2),
        };

        var liquid = StandardTerrainAdapter.BuildMh2oLiquid(
            new[] { water, magma },
            tileX: 3,
            tileY: 7,
            chunkX: 1,
            chunkY: 2,
            new Vector3(100f, 200f, 0f));

        Assert.NotNull(liquid);
        Assert.Equal(LiquidType.Magma, liquid!.Type);
        Assert.Equal((byte)0, liquid.TileFlags![0]);
        Assert.Equal((byte)0, liquid.TileFlags[9]);
        Assert.Equal((byte)0x0F, liquid.TileFlags[10]);
        Assert.Equal(30f, liquid.Heights[10]);
        Assert.Equal(10f, liquid.Heights[0]);
    }

    [Fact]
    public void StandardTerrainAdapter_LichKingProfiles_UseStrictMcalDecode()
    {
        var textureLayers = new List<MclyEntry>
        {
            new() { TextureId = 0, Flags = 0, AlphaMapOffset = 0, EffectId = 0 },
            new() { TextureId = 1, Flags = MclyFlags.UseAlpha, AlphaMapOffset = 0, EffectId = 0 },
        };

        var mcalRawData = new byte[4096];
        mcalRawData[0] = 12;
        mcalRawData[63] = 34;
        mcalRawData[4095] = 56;

        var maps = StandardTerrainAdapter.ExtractAlphaMaps(
            textureLayers,
            new Mcal(mcalRawData),
            mcalRawData,
            TerrainAlphaDecodeMode.LichKingStrict,
            useBigAlpha: true,
            doNotFixAlphaMap: false);

        Assert.True(maps.TryGetValue(1, out var alpha));
        Assert.NotNull(alpha);
        Assert.Equal((byte)12, alpha![0]);
        Assert.Equal((byte)34, alpha[63]);
        Assert.Equal((byte)56, alpha[4095]);
    }

    [Fact]
    public void StandardTerrainAdapter_LichKingProfiles_DoNotInferAlpha_WhenUseAlphaFlagIsMissing()
    {
        var textureLayers = new List<MclyEntry>
        {
            new() { TextureId = 0, Flags = 0, AlphaMapOffset = 0, EffectId = 0 },
            new() { TextureId = 1, Flags = 0, AlphaMapOffset = 0, EffectId = 0 },
        };

        var mcalRawData = new byte[4096];
        mcalRawData[0] = 12;
        mcalRawData[63] = 34;
        mcalRawData[4095] = 56;

        var maps = StandardTerrainAdapter.ExtractAlphaMaps(
            textureLayers,
            new Mcal(mcalRawData),
            mcalRawData,
            TerrainAlphaDecodeMode.LichKingStrict,
            useBigAlpha: false,
            doNotFixAlphaMap: false);

        Assert.DoesNotContain(1, maps.Keys);
    }

    [Fact]
    public void StandardTerrainAdapter_LichKingProfiles_RelaxedFallbackOnlyRuns_WhenOffsetExists()
    {
        var textureLayers = new List<MclyEntry>
        {
            new() { TextureId = 0, Flags = 0, AlphaMapOffset = 0, EffectId = 0 },
            new() { TextureId = 1, Flags = 0, AlphaMapOffset = 32, EffectId = 0 },
            new() { TextureId = 2, Flags = 0, AlphaMapOffset = 98, EffectId = 0 },
        };

        var mcalRawData = new byte[256];
        Array.Copy(BuildCompressedAlpha(fillValue: 173), 0, mcalRawData, 32, 66);

        var maps = StandardTerrainAdapter.ExtractAlphaMaps(
            textureLayers,
            new Mcal(mcalRawData),
            mcalRawData,
            TerrainAlphaDecodeMode.LichKingStrict,
            useBigAlpha: false,
            doNotFixAlphaMap: false);

        var alpha = Assert.Contains(1, maps);
        Assert.Equal(4096, alpha.Length);
        Assert.All(alpha, static value => Assert.Equal((byte)173, value));
    }

    [Fact]
    public void Mcal_FourBitDecode_PreservesFinalHighNibble_WhenDoNotFixAlphaMap()
    {
        var mcalRawData = new byte[2048];
        mcalRawData[31] = 0x73;

        var alpha = new Mcal(mcalRawData).GetAlphaMapForLayer(
            new MclyEntry { TextureId = 1, Flags = MclyFlags.UseAlpha, AlphaMapOffset = 0, EffectId = 0 },
            bigAlpha: false,
            doNotFixAlphaMap: true);

        Assert.NotNull(alpha);
        Assert.Equal((byte)(3 * 17), alpha![62]);
        Assert.Equal((byte)(7 * 17), alpha[63]);
    }

    [Fact]
    public void StandardTerrainAdapter_ResolveChunkCoordinates_PrefersValidHeaderIndices()
    {
        var coords = StandardTerrainAdapter.ResolveChunkCoordinates(mcinIndex: 5, headerChunkX: 14, headerChunkY: 3);

        Assert.Equal((14, 3), coords);
    }

    [Fact]
    public void StandardTerrainAdapter_ResolveChunkCoordinates_FallsBackToMcinSlot_WhenHeaderIndicesAreInvalid()
    {
        var coords = StandardTerrainAdapter.ResolveChunkCoordinates(mcinIndex: 34, headerChunkX: 99, headerChunkY: -1);

        Assert.Equal((2, 2), coords);
    }

    [Fact]
    public void FormatProfiles_LichKingBuilds_DoNotUseSplitTextureAdt()
    {
        Assert.False(FormatProfileRegistry.ResolveAdtProfile("3.3.5.12340").UseSplitTextureAdt);
        Assert.False(FormatProfileRegistry.ResolveAdtProfile("3.0.1.8303").UseSplitTextureAdt);
    }

    [Fact]
    public void FormatProfiles_EarlySplitBuilds_UseSplitTextureAdt()
    {
        Assert.True(FormatProfileRegistry.ResolveAdtProfile("0.7.0.3694").UseSplitTextureAdt);
        Assert.True(FormatProfileRegistry.ResolveAdtProfile("0.9.1.3810").UseSplitTextureAdt);
    }

    [Fact]
    public void TerrainRenderer_RemapMissingDiffuseTextureIndices_InvalidatesMissingSlices()
    {
        var texIndices = new ushort[]
        {
            0, 1, 2, 3,
            3, 2, 1, 0,
            0xFFFF, 4, 1, 2,
        };

        bool changed = TerrainRenderer.RemapMissingDiffuseTextureIndices(
            texIndices,
            new[] { true, false, true, false });

        Assert.True(changed);
        Assert.Equal(
            new ushort[]
            {
                0, 0xFFFF, 2, 0xFFFF,
                0xFFFF, 2, 0xFFFF, 0,
                0xFFFF, 0xFFFF, 0xFFFF, 2,
            },
            texIndices);
    }

    private static TerrainChunkData CreateExplicitChunk(int chunkX, int chunkY)
    {
        var alpha1 = new byte[64 * 64];
        var alpha2 = new byte[64 * 64];
        var alpha3 = new byte[64 * 64];
        var shadow = new byte[64 * 64];

        alpha1[(63 * 64) + 38] = 17;
        alpha2[(12 * 64) + 63] = 41;
        alpha3[(63 * 64) + 63] = 99;
        shadow[(31 * 64) + 63] = 255;

        return new TerrainChunkData
        {
            ChunkX = chunkX,
            ChunkY = chunkY,
            Layers =
            [
                new TerrainLayer { TextureIndex = 0, Flags = 0 },
                new TerrainLayer { TextureIndex = 1, Flags = 0x100 },
                new TerrainLayer { TextureIndex = 2, Flags = 0x100 },
                new TerrainLayer { TextureIndex = 3, Flags = 0x100 },
            ],
            AlphaMaps = new Dictionary<int, byte[]>
            {
                [1] = alpha1,
                [2] = alpha2,
                [3] = alpha3,
            },
            ShadowMap = shadow,
        };
    }

    private static byte[] ExtractSlice(byte[] alphaShadow, int chunkX, int chunkY)
    {
        var slice = (chunkY * 16) + chunkX;
        var sliceSize = TerrainTileMeshBuilder.AlphaShadowSliceSize * TerrainTileMeshBuilder.AlphaShadowSliceSize * 4;
        var buffer = new byte[sliceSize];
        Array.Copy(alphaShadow, slice * sliceSize, buffer, 0, sliceSize);
        return buffer;
    }

    private static int GetChannelOffset(int x, int y, int channel)
    {
        return ((y * TerrainTileMeshBuilder.AlphaShadowSliceSize) + x) * 4 + channel;
    }

    private static byte[] BuildCompressedAlpha(byte fillValue)
    {
        var data = new List<byte>();
        int remaining = 64 * 64;
        while (remaining > 0)
        {
            int count = Math.Min(127, remaining);
            data.Add((byte)(0x80 | count));
            data.Add(fillValue);
            remaining -= count;
        }

        return data.ToArray();
    }

    private static float[] CreateFilledHeights(int width, int height, float value)
    {
        var result = new float[width * height];
        Array.Fill(result, value);
        return result;
    }

    private static byte[] CreateFilledDepth(int count, byte value)
    {
        var result = new byte[count];
        Array.Fill(result, value);
        return result;
    }
}