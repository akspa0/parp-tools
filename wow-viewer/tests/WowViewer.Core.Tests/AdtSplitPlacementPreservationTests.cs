using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtSplitPlacementPreservationTests
{
    [Fact]
    public void Build_DecodesSplitPlacementReferencesAndRemovesRawFallback()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_split_refs_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string rootPath = Path.Combine(tempDir, "tile_0_0.adt");
            string objPath = Path.Combine(tempDir, "tile_0_0_obj0.adt");

            File.WriteAllBytes(rootPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MHDR", new byte[64]),
                .. CreateChunk("MCNK", CreateRootMcnkPayload(0, 0)),
            ]);

            File.WriteAllBytes(objPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MCNK", CreateSplitPlacementPayload(("MCRD", CreateIntPayload(11, 12)), ("MCRW", CreateIntPayload(21)))),
            ]);

            TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(rootPath, buildVersion: "4.0.0.11927");

            Assert.NotNull(pack.McrdRefCounts16);
            Assert.NotNull(pack.McrdRefIndices);
            Assert.NotNull(pack.McrwRefCounts16);
            Assert.NotNull(pack.McrwRefIndices);
            Assert.Equal(2, pack.McrdRefCounts16![0, 0]);
            Assert.Equal([11, 12], pack.McrdRefIndices!);
            Assert.Equal(1, pack.McrwRefCounts16![0, 0]);
            Assert.Equal([21], pack.McrwRefIndices!);
            Assert.Contains("mcrd_ref_indices", pack.AvailableSignals);
            Assert.Contains("mcrw_ref_indices", pack.AvailableSignals);
            Assert.DoesNotContain(pack.RawChunks, static chunk => chunk.ChunkId is "MCRD" or "MCRW");
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

    private static byte[] CreateRootMcnkPayload(uint indexX, uint indexY)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        stream.Write(CreateChunk("MCVT", new byte[145 * sizeof(float)]));
        stream.Write(CreateChunk("MCNR", new byte[435]));
        return stream.ToArray();
    }

    private static byte[] CreateSplitPlacementPayload(params (string Id, byte[] Payload)[] subchunks)
    {
        using MemoryStream stream = new();
        foreach ((string id, byte[] payload) in subchunks)
            stream.Write(CreateChunk(id, payload));
        return stream.ToArray();
    }
}