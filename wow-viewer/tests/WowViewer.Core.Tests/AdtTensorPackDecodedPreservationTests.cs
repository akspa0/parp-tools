using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtTensorPackDecodedPreservationTests
{
    [Fact]
    public void Build_DecodesSpecBackedPreservationSignals()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_tensor_pack_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string rootPath = Path.Combine(tempDir, "tile_0_0.adt");
            string texPath = Path.Combine(tempDir, "tile_0_0_tex0.adt");

            File.WriteAllBytes(rootPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MHDR", new byte[64]),
                .. CreateChunk("MFBO", CreateMfboPayload()),
                .. CreateChunk("MCNK", CreateRootMcnkPayload(indexX: 0, indexY: 0, mclvPayload: CreateMclvPayload())),
            ]);

            File.WriteAllBytes(texPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MAMP", [5]),
                .. CreateChunk("MTEX", CreateStringBlock("base.blp")),
                .. CreateChunk("MCNK", CreateTexChunkPayload(CreateChunk("MCLY", CreateMclyPayload([0u], [0u])), CreateChunk("MCMT", [7, 8, 9, 10]))),
            ]);

            TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(rootPath, texPath, buildVersion: "4.0.0.11927");

            Assert.Equal(5, Assert.Single(pack.MampValue!));
            Assert.NotNull(pack.MfboFlightBounds);
            Assert.Equal(101, pack.MfboFlightBounds![0, 0, 0]);
            Assert.Equal(-202, pack.MfboFlightBounds[1, 2, 2]);
            Assert.NotNull(pack.MclvLightingBytes);
            Assert.Equal((byte)0x11, pack.MclvLightingBytes![0, 0, 0]);
            Assert.Equal((byte)0x22, pack.MclvLightingBytes[0, 0, 1]);
            Assert.Equal((byte)0x33, pack.MclvLightingBytes[0, 0, 2]);
            Assert.Equal((byte)0x44, pack.MclvLightingBytes[0, 0, 3]);
            Assert.NotNull(pack.McmtMaterialIds);
            Assert.Equal((byte)7, pack.McmtMaterialIds![0, 0, 0]);
            Assert.Equal((byte)8, pack.McmtMaterialIds[0, 0, 1]);
            Assert.Contains("mamp_value", pack.AvailableSignals);
            Assert.Contains("mfbo_flight_bounds", pack.AvailableSignals);
            Assert.Contains("mclv_lighting_bytes", pack.AvailableSignals);
            Assert.Contains("mcmt_material_ids", pack.AvailableSignals);
            Assert.DoesNotContain(pack.RawChunks, static chunk => chunk.ChunkId is "MAMP" or "MFBO" or "MCLV" or "MCMT");
        }
        finally
        {
            if (Directory.Exists(tempDir))
                Directory.Delete(tempDir, recursive: true);
        }
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

    private static byte[] CreateMfboPayload()
    {
        byte[] payload = new byte[36];
        short[] values = [101, 102, 103, 104, 105, 106, 107, 108, 109, -201, -202, -203, -204, -205, -206, -207, -208, -202];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteInt16LittleEndian(payload.AsSpan(index * sizeof(short), sizeof(short)), values[index]);
        return payload;
    }

    private static byte[] CreateMclvPayload()
    {
        byte[] payload = new byte[145 * 4];
        payload[0] = 0x11;
        payload[1] = 0x22;
        payload[2] = 0x33;
        payload[3] = 0x44;
        return payload;
    }

    private static byte[] CreateRootMcnkPayload(uint indexX, uint indexY, byte[] mclvPayload)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        stream.Write(CreateChunk("MCVT", new byte[145 * sizeof(float)]));
        stream.Write(CreateChunk("MCNR", new byte[435]));
        stream.Write(CreateChunk("MCLV", mclvPayload));
        return stream.ToArray();
    }

    private static byte[] CreateTexChunkPayload(byte[] mclyChunk, byte[] mcmtChunk)
    {
        using MemoryStream stream = new();
        stream.Write(mclyChunk);
        stream.Write(mcmtChunk);
        return stream.ToArray();
    }

    private static byte[] CreateMclyPayload(uint[] layerFlags, uint[] layerOffsets)
    {
        byte[] payload = new byte[layerFlags.Length * 16];
        for (int index = 0; index < layerFlags.Length; index++)
        {
            int offset = index * 16;
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset, 4), (uint)index);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 4, 4), layerFlags[index]);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 8, 4), layerOffsets[index]);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 12, 4), 0u);
        }

        return payload;
    }

    private static byte[] CreateStringBlock(params string[] entries)
    {
        using MemoryStream stream = new();
        foreach (string entry in entries)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(entry);
            stream.Write(bytes);
            stream.WriteByte(0);
        }

        return stream.ToArray();
    }
}