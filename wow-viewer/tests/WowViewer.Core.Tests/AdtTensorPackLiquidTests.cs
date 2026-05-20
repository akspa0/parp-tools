using System.Buffers.Binary;
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
            Assert.Equal(1.0f, pack.UnifiedLiquidMask![2, 81]);
            Assert.Equal(0.0f, pack.UnifiedLiquidHeight![2, 81]);
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
