using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtMcrfPreservationTests
{
    [Fact]
    public void Build_DecodesPreCataMcrfReferencesAndRemovesRawFallback()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_mcrf_refs_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string rootPath = Path.Combine(tempDir, "tile_0_0.adt");

            File.WriteAllBytes(rootPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MHDR", new byte[64]),
                .. CreateChunk("MCNK", CreateRootMcnkPayload(indexX: 0, indexY: 0, doodadRefs: [11, 12], wmoRefs: [21], trailingChunkId: "MCSE", trailingPayload: [0x44, 0x55, 0x66, 0x77])),
            ]);

            TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(rootPath, buildVersion: "3.3.5.12340");

            Assert.NotNull(pack.McrfDoodadRefCounts16);
            Assert.NotNull(pack.McrfDoodadRefIndices);
            Assert.NotNull(pack.McrfWmoRefCounts16);
            Assert.NotNull(pack.McrfWmoRefIndices);
            Assert.Equal(2, pack.McrfDoodadRefCounts16![0, 0]);
            Assert.Equal([11, 12], pack.McrfDoodadRefIndices!);
            Assert.Equal(1, pack.McrfWmoRefCounts16![0, 0]);
            Assert.Equal([21], pack.McrfWmoRefIndices!);
            Assert.Contains("mcrf_doodad_ref_indices", pack.AvailableSignals);
            Assert.Contains("mcrf_wmo_ref_indices", pack.AvailableSignals);
            Assert.DoesNotContain(pack.RawChunks, static chunk => chunk.ChunkId == "MCRF");
            Assert.Contains(pack.RawChunks, static chunk => chunk.ChunkId == "MCSE");
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

    private static byte[] CreateIntPayload(params int[] values)
    {
        byte[] payload = new byte[values.Length * sizeof(int)];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteInt32LittleEndian(payload.AsSpan(index * sizeof(int), sizeof(int)), values[index]);
        return payload;
    }

    private static byte[] CreateRootMcnkPayload(uint indexX, uint indexY, int[] doodadRefs, int[] wmoRefs, string trailingChunkId, byte[] trailingPayload)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x14, 4), doodadRefs.Length);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x3C, 4), wmoRefs.Length);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        stream.Write(CreateChunk("MCVT", new byte[145 * sizeof(float)]));
        stream.Write(CreateChunk("MCNR", new byte[435]));
        stream.Write(CreateChunk("MCRF", CreateIntPayload([.. doodadRefs, .. wmoRefs])));
        stream.Write(CreateChunk(trailingChunkId, trailingPayload));
        return stream.ToArray();
    }
}