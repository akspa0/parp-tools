using System.Buffers.Binary;
using System.Numerics;
using System.Reflection;
using System.Text;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtTensorPackLiquidTests
{
    [Fact]
    public void Build_Mh2oAtSeaLevelZero_PreservesUnifiedLiquidMask()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_liquid_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string rootPath = Path.Combine(tempDir, "tile_0_0.adt");

            File.WriteAllBytes(rootPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MHDR", new byte[64]),
                .. CreateChunk("MH2O", CreateMh2oPayloadWithZeroHeights()),
                .. CreateChunk("MCNK", CreateRootMcnkPayload(0, 0)),
            ]);

            TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(rootPath, buildVersion: "3.3.5.12340");

            Assert.NotNull(pack.UnifiedLiquidMask);
            Assert.NotNull(pack.UnifiedLiquidHeight);
            Assert.Equal(1.0f, pack.UnifiedLiquidMask![4, 82]);
            Assert.Equal(0.0f, pack.UnifiedLiquidHeight![4, 82]);
            Assert.Equal(16, pack.UnifiedLiquidMask.Cast<float>().Count(static value => value > 0.5f));
            Assert.Contains("unified_liquid_mask", pack.AvailableSignals);
            Assert.Contains("unified_liquid_height", pack.AvailableSignals);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void Build_Pre310RootMcnkHeaderOffsetMclq_PreservesLegacyLiquid()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_liquid_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string rootPath = Path.Combine(tempDir, "tile_0_0.adt");

            File.WriteAllBytes(rootPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MHDR", new byte[64]),
                .. CreateChunk("MCNK", CreateRootMcnkPayloadWithHeaderOffsetMclq(flags: 0x08u, indexX: 0, indexY: 0, surfaceHeight: 27f, layerStride: 0x324)),
            ]);

            TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(rootPath, buildVersion: "3.0.1.8303");

            Assert.NotNull(pack.MclqSurfaceHeight);
            Assert.NotNull(pack.MclqPresenceMask);
            Assert.NotNull(pack.MclqTypeMask);
            Assert.True(pack.MclqPresenceMask![0, 0]);
            Assert.Equal(27f, pack.MclqSurfaceHeight![0, 0]);
            Assert.Equal(2, pack.MclqTypeMask![0, 0]);
            Assert.Contains("mclq_surface_height", pack.AvailableSignals);
            Assert.Contains("mclq_type_mask", pack.AvailableSignals);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void BuildUnifiedLiquid_UsesPerPixelMh2oMclqWlPrecedence()
    {
        float[,] mh2oHeight = new float[257, 257];
        bool[,] mh2oPresence = new bool[257, 257];
        mh2oHeight[0, 0] = 30f;
        mh2oPresence[0, 0] = true;

        float[,] mclqHeight = new float[129, 129];
        bool[,] mclqPresence = new bool[129, 129];
        for (int y = 0; y < 129; y++)
            for (int x = 0; x < 129; x++)
                mclqHeight[y, x] = 20f;
        mclqPresence[0, 0] = true;

        float[,] wlMask = new float[257, 257];
        float[,] wlHeight = new float[257, 257];
        wlMask[0, 0] = 1f;
        wlHeight[0, 0] = 10f;
        wlMask[200, 200] = 1f;
        wlHeight[200, 200] = 10f;

        HashSet<string> signals = new(StringComparer.OrdinalIgnoreCase);
        (float[,]? mask, float[,]? height) = InvokeBuildUnifiedLiquid(
            mh2oHeight,
            mh2oPresence,
            mclqHeight,
            mclqPresence,
            wlMask,
            wlHeight,
            signals);

        Assert.NotNull(mask);
        Assert.NotNull(height);
        Assert.Equal(30f, height![0, 0]); // MH2O wins over MCLQ and WL*.
        Assert.Equal(20f, height[1, 1]); // MCLQ remains where MH2O is absent.
        Assert.Equal(10f, height[200, 200]); // WL* remains where neither higher source covers.
        Assert.Equal(1f, mask![0, 0]);
        Assert.Equal(1f, mask[1, 1]);
        Assert.Equal(1f, mask[200, 200]);
        Assert.Contains("unified_liquid_mask", signals);
        Assert.Contains("unified_liquid_height", signals);
    }

    [Fact]
    public void Build_WlReadFailureMakesStrictLiquidEvidenceUnknown()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_liquid_wl_failure_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string rootPath = Path.Combine(tempDir, "Test_0_0.adt");
            File.WriteAllBytes(rootPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MHDR", new byte[64]),
                .. CreateChunk("MCNK", CreateRootMcnkPayload(0, 0)),
            ]);
            File.WriteAllBytes(Path.Combine(tempDir, "broken.wlw"), [0, 0, 0, 0]);

            TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(rootPath, buildVersion: "3.3.5.12340");

            ObjectGeometryTargetProvenance provenance = Assert.IsType<ObjectGeometryTargetProvenance>(pack.ObjectGeometryTargetProvenance);
            Assert.Equal(ObjectGeometryTargetStatus.LiquidVisibilityUnknown, provenance.Status);
            Assert.Equal(ObjectGeometryLiquidEvidenceStatus.Unknown, provenance.LiquidEvidenceStatus);
            Assert.False(provenance.IsMaterialized);
            Assert.Null(pack.ObjectGeometryVisibleMask257);
            Assert.Null(pack.ObjectGeometryVisibleTopElevation257);
            Assert.Null(pack.ObjectGeometryVisibleTerrainElevation257);
            Assert.Null(pack.ObjectGeometryVisibleSource257);
            Assert.Null(pack.WlLiquidMask);
            Assert.Null(pack.WlLiquidHeight);
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
    }

    [Fact]
    public void WlLiquidRasterizer_FillsContiguousBlockQuadsInsteadOfSparseMarkers()
    {
        Vector3[] vertices = new Vector3[16];
        const float vertexSpacing = 32f;
        for (int row = 0; row < 4; row++)
        {
            for (int column = 0; column < 4; column++)
            {
                // The file stores this 4x4 grid in reverse (lower-right to upper-left).
                vertices[15 - (row * 4 + column)] = new Vector3(
                    WlLiquidRasterizer.MapOrigin - (row * vertexSpacing),
                    WlLiquidRasterizer.MapOrigin - (column * vertexSpacing),
                    100f + (row * 10f) + column);
            }
        }

        var file = new WlFile
        {
            Blocks = [new WlBlock { Vertices = vertices }]
        };

        bool success = WlLiquidRasterizer.TryRasterize(
            [file],
            0,
            0,
            out float[,]? mask,
            out float[,]? heights,
            out byte[,]? basicTypes);

        Assert.True(success);
        Assert.NotNull(mask);
        Assert.NotNull(heights);
        Assert.NotNull(basicTypes);
        Assert.True(mask!.Cast<float>().Count(static value => value > 0.5f) > 2_000);
        Assert.Equal(1f, mask[0, 0]);
        Assert.Equal(1f, mask[45, 45]);
        Assert.Equal(0f, mask[46, 46]);
        Assert.True(heights![40, 40] > heights[0, 0]);
        Assert.Equal((byte)AdtLiquidBasicType.Water, basicTypes![0, 0]);
    }

    [Fact]
    public void WlLiquidRasterizer_RemovesSurfacesBelowAlignedTerrain()
    {
        float[,] mask = new float[3, 3];
        float[,] heights = new float[3, 3];
        float[,] terrain = new float[3, 3];
        byte[,] types = new byte[3, 3];
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                mask[y, x] = 1f;
                heights[y, x] = 100f;
                terrain[y, x] = 99f;
                types[y, x] = (byte)AdtLiquidBasicType.Water;
            }
        }
        terrain[1, 1] = 101f;

        int retained = WlLiquidRasterizer.KeepOnlyAboveTerrain(mask, heights, terrain, types);

        Assert.Equal(8, retained);
        Assert.Equal(0f, mask[1, 1]);
        Assert.Equal(0f, heights[1, 1]);
        Assert.Equal(LiquidBasicTypeConstants.NoLiquid, types[1, 1]);
        Assert.Equal(1f, mask[0, 0]);
        Assert.Equal(100f, heights[0, 0]);
    }

    [Fact]
    public void WlLiquidRasterizer_UsesTheSameNonSquareRasterAxesAsTerrainForHeightGating()
    {
        Vector3[] vertices = new Vector3[16];
        const float terrainOffsetY = 100f;
        const float terrainOffsetX = 200f;
        const float spacing = 20f;
        for (int row = 0; row < 4; row++)
        {
            for (int column = 0; column < 4; column++)
            {
                vertices[15 - (row * 4 + column)] = new Vector3(
                    WlLiquidRasterizer.MapOrigin - terrainOffsetY - (row * spacing),
                    WlLiquidRasterizer.MapOrigin - terrainOffsetX - (column * spacing),
                    100f);
            }
        }

        var file = new WlFile
        {
            Header = new WlHeader { FileType = WlFileType.WLW, LiquidType = WlLiquidType.StillWater },
            Blocks = [new WlBlock { Vertices = vertices }]
        };

        Assert.True(WlLiquidRasterizer.TryRasterize(
            [file], 0, 0, out float[,]? mask, out float[,]? heights, out byte[,]? types));
        Assert.NotNull(mask);
        Assert.NotNull(heights);
        Assert.NotNull(types);

        // World Y maps to raster X and world X maps to raster Y, matching the ADT terrain lattice.
        Assert.Equal(1f, mask![48, 96]);
        Assert.Equal(0f, mask[96, 48]);

        float[,] terrain = new float[257, 257];
        for (int y = 0; y < 257; y++)
            for (int x = 0; x < 257; x++)
                terrain[y, x] = 99f;
        terrain[48, 96] = 101f;

        WlLiquidRasterizer.KeepOnlyAboveTerrain(mask, heights!, terrain, types);

        Assert.Equal(0f, mask[48, 96]);
        Assert.Equal(LiquidBasicTypeConstants.NoLiquid, types![48, 96]);
    }

    [Fact]
    public void WlLiquidRasterizer_UsesHeaderTypeAndLavaFamilyFallback()
    {
        Vector3[] vertices = CreateWlVertices();
        var slime = new WlFile
        {
            Header = new WlHeader { FileType = WlFileType.WLW, LiquidType = WlLiquidType.Slime },
            Blocks = [new WlBlock { Vertices = vertices }]
        };

        Assert.True(WlLiquidRasterizer.TryRasterize(
            [slime], 0, 0, out _, out _, out byte[,]? slimeTypes));
        Assert.NotNull(slimeTypes);
        Assert.Equal((byte)AdtLiquidBasicType.Slime, slimeTypes![0, 0]);

        var lava = new WlFile
        {
            // WLL is lava even if the shared header's raw class is water.
            Header = new WlHeader { FileType = WlFileType.WLL, LiquidType = WlLiquidType.StillWater },
            Blocks = [new WlBlock { Vertices = vertices }]
        };

        Assert.True(WlLiquidRasterizer.TryRasterize(
            [lava], 0, 0, out _, out _, out byte[,]? lavaTypes));
        Assert.NotNull(lavaTypes);
        Assert.Equal((byte)AdtLiquidBasicType.Magma, lavaTypes![0, 0]);
    }

    [Fact]
    public void WlFileReader_UsesWlwHeaderAndWllFamilyForLiquidType()
    {
        WlFile slime = ReadWlHeader(3, "liquid.wlw");
        WlFile lava = ReadWlHeader(0, "liquid.wll");

        Assert.Equal(WlLiquidType.Slime, slime.Header.LiquidType);
        Assert.Equal((ushort)3, slime.Header.RawLiquidType);
        Assert.Equal(WlLiquidType.Magma, lava.Header.LiquidType);
        Assert.Equal((ushort)0, lava.Header.RawLiquidType);
    }

    [Fact]
    public void LiquidBasicTypePackBuilder_OverlaysWlTypesOnlyWhereNoExplicitSurfaceOwnsThePixel()
    {
        float[,] wlMask = new float[3, 3];
        byte[,] wlTypes = new byte[3, 3];
        for (int y = 0; y < 3; y++)
        {
            for (int x = 0; x < 3; x++)
            {
                wlMask[y, x] = 1f;
                wlTypes[y, x] = (byte)AdtLiquidBasicType.Magma;
            }
        }

        float[,] mh2oHeights = new float[3, 3];
        bool[,] mh2oPresence = new bool[3, 3];
        mh2oPresence[1, 1] = true;
        byte[,] resolved = new byte[3, 3];
        for (int y = 0; y < 3; y++)
            for (int x = 0; x < 3; x++)
                resolved[y, x] = LiquidBasicTypeConstants.NoLiquid;
        resolved[1, 1] = (byte)AdtLiquidBasicType.Ocean;

        byte[,]? result = LiquidBasicTypePackBuilder.OverlayWlFallbackTypes(
            resolved,
            wlMask,
            wlTypes,
            mh2oHeights,
            mh2oPresence,
            null,
            null);

        Assert.NotNull(result);
        Assert.Equal((byte)AdtLiquidBasicType.Magma, result![0, 0]);
        Assert.Equal((byte)AdtLiquidBasicType.Ocean, result[1, 1]);
    }

    private static Vector3[] CreateWlVertices()
    {
        Vector3[] vertices = new Vector3[16];
        const float vertexSpacing = 32f;
        for (int row = 0; row < 4; row++)
        {
            for (int column = 0; column < 4; column++)
            {
                vertices[15 - (row * 4 + column)] = new Vector3(
                    WlLiquidRasterizer.MapOrigin - (row * vertexSpacing),
                    WlLiquidRasterizer.MapOrigin - (column * vertexSpacing),
                    100f);
            }
        }

        return vertices;
    }

    private static WlFile ReadWlHeader(ushort rawLiquidType, string fileName)
    {
        using var stream = new MemoryStream();
        using (var writer = new BinaryWriter(stream, Encoding.ASCII, leaveOpen: true))
        {
            writer.Write(Encoding.ASCII.GetBytes("*QIL"));
            writer.Write((ushort)1);
            writer.Write((ushort)0);
            writer.Write(rawLiquidType);
            writer.Write((ushort)0);
            writer.Write(0u);
        }

        stream.Position = 0;
        return WlFileReader.Read(stream, fileName);
    }

    private static (float[,]? mask, float[,]? height) InvokeBuildUnifiedLiquid(
        float[,]? mh2oHeight,
        bool[,]? mh2oPresence,
        float[,]? mclqHeight,
        bool[,]? mclqPresence,
        float[,]? wlMask,
        float[,]? wlHeight,
        HashSet<string> signals)
    {
        MethodInfo? method = typeof(AdtTensorPackBuilder).GetMethod(
            "BuildUnifiedLiquid",
            BindingFlags.Static | BindingFlags.Public | BindingFlags.NonPublic);
        Assert.NotNull(method);
        object result = Assert.IsType<ValueTuple<float[,]?, float[,]?>>(method!.Invoke(
            null,
            [mh2oHeight, mh2oPresence, mclqHeight, mclqPresence, wlMask, wlHeight, signals]));
        return (ValueTuple<float[,]?, float[,]?>)result;
    }

    private static byte[] CreateMh2oPayloadWithZeroHeights()
    {
        const int chunkCount = 256;
        const int headerSize = 12;
        const int attributesSize = 16;
        const int layerSize = 24;
        const int width = 2;
        const int height = 2;
        const int vertexCount = (width + 1) * (height + 1);

        int headersSize = chunkCount * headerSize;
        int attributesOffset = headersSize;
        int layerOffset = attributesOffset + attributesSize;
        int vertexOffset = layerOffset + layerSize;
        int depthOffset = vertexOffset + (vertexCount * sizeof(float));

        byte[] payload = new byte[depthOffset + vertexCount];

        int headerOffset = 5 * headerSize;
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(headerOffset, 4), (uint)layerOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(headerOffset + 4, 4), 1u);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(headerOffset + 8, 4), (uint)attributesOffset);

        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(layerOffset, 2), 17);
        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(layerOffset + 2, 2), 0);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(layerOffset + 4, 4), 0f);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(layerOffset + 8, 4), 0f);
        payload[layerOffset + 12] = 1;
        payload[layerOffset + 13] = 2;
        payload[layerOffset + 14] = width;
        payload[layerOffset + 15] = height;
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(layerOffset + 16, 4), 0u);
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(layerOffset + 20, 4), (uint)vertexOffset);

        for (int index = 0; index < vertexCount; index++)
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(vertexOffset + (index * sizeof(float)), sizeof(float)), 0f);

        for (int index = 0; index < vertexCount; index++)
            payload[depthOffset + index] = 1;

        return payload;
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        byte[] bytes = new byte[8 + payload.Length];
        Array.Copy(FourCC.FromString(id).ToFileBytes(), 0, bytes, 0, 4);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4), (uint)payload.Length);
        Array.Copy(payload, 0, bytes, 8, payload.Length);
        return bytes;
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        byte[] bytes = new byte[4];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes, value);
        return bytes;
    }

    private static byte[] CreateRootMcnkPayload(uint indexX, uint indexY)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        stream.Write(CreateChunk("MCVT", new byte[145 * sizeof(float)]));
        stream.Write(CreateChunk("MCNR", new byte[145 * 3]));
        return stream.ToArray();
    }

    private static byte[] CreateRootMcnkPayloadWithHeaderOffsetMclq(uint flags, uint indexX, uint indexY, float surfaceHeight, int layerStride)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00, 4), flags);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        stream.Write(CreateChunk("MCVT", new byte[145 * sizeof(float)]));
        stream.Write(CreateChunk("MCNR", new byte[145 * 3]));

        int mclqChunkHeaderOffsetInPayload = checked((int)stream.Length);
        byte[] mclqPayload = CreateLegacyMclqPayload(surfaceHeight, layerStride);
        stream.Write(CreateChunk("MCLQ", mclqPayload));

        byte[] payload = stream.ToArray();
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0x60, 4), (uint)(mclqChunkHeaderOffsetInPayload + 8));
        BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(0x64, 4), (uint)mclqPayload.Length);
        return payload;
    }

    private static byte[] CreateLegacyMclqPayload(float surfaceHeight, int layerStride)
    {
        byte[] payload = new byte[layerStride];
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(0, 4), surfaceHeight);
        BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(4, 4), surfaceHeight);

        for (int index = 0; index < 81; index++)
            BinaryPrimitives.WriteSingleLittleEndian(payload.AsSpan(8 + (index * 8) + 4, 4), surfaceHeight);

        for (int index = 0; index < 64; index++)
            payload[0x290 + index] = 0;

        return payload;
    }
}
