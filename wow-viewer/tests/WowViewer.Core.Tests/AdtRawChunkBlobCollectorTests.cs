using System.Buffers.Binary;
using WowViewer.Core.IO.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtRawChunkBlobCollectorTests
{
    [Fact]
    public void Collect_PreservesUnconsumedRootTexAndObjChunks()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_raw_chunks_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string rootPath = Path.Combine(tempDir, "tile_0_0.adt");
            string texPath = Path.Combine(tempDir, "tile_0_0_tex0.adt");
            string objPath = Path.Combine(tempDir, "tile_0_0_obj0.adt");

            File.WriteAllBytes(rootPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MHDR", new byte[64]),
                .. CreateChunk("MFBO", [1, 2, 3, 4]),
                .. CreateChunk("MCNK", CreateRootMcnkPayload(indexX: 0, indexY: 0, includeMcvt: true, includeMcse: true, includeMcrf: true, includeUnknownMclv: true)),
            ]);

            File.WriteAllBytes(texPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MAMP", [9]),
                .. CreateChunk("MTXF", [7, 7, 7, 7]),
                .. CreateChunk("MCNK", CreateSplitMcnkPayload(("MCLY", new byte[16]), ("MCMT", [5, 6, 7, 8]), ("MCSH", new byte[32]))),
            ]);

            File.WriteAllBytes(objPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MDDF", new byte[36]),
                .. CreateChunk("MCNK", CreateSplitMcnkPayload(("MCRD", [0xAA, 0xBB, 0xCC, 0xDD]))),
            ]);

            IReadOnlyList<WowViewer.Core.Maps.TerrainRawChunkBlob> rawChunks = AdtRawChunkBlobCollector.Collect(rootPath);

            Assert.Contains(rawChunks, static chunk => chunk.SourceKind == "root" && chunk.Scope == "mcnk-subchunk" && chunk.ChunkId == "MCSE" && HasBytes(chunk.Data, 0x10, 0x11, 0x12, 0x13));
            Assert.DoesNotContain(rawChunks, static chunk => chunk.ChunkId == "MFBO");
            Assert.DoesNotContain(rawChunks, static chunk => chunk.ChunkId == "MCVT");
            Assert.DoesNotContain(rawChunks, static chunk => chunk.ChunkId == "MCRF");
            Assert.DoesNotContain(rawChunks, static chunk => chunk.ChunkId == "MCLV");
            Assert.DoesNotContain(rawChunks, static chunk => chunk.ChunkId == "MCLY");
            Assert.DoesNotContain(rawChunks, static chunk => chunk.ChunkId == "MCMT");
            Assert.DoesNotContain(rawChunks, static chunk => chunk.ChunkId == "MCSH");
            Assert.DoesNotContain(rawChunks, static chunk => chunk.ChunkId == "MAMP");
            Assert.DoesNotContain(rawChunks, static chunk => chunk.ChunkId == "MCRD");
            Assert.DoesNotContain(rawChunks, static chunk => chunk.ChunkId == "MTXF");
            Assert.DoesNotContain(rawChunks, static chunk => chunk.ChunkId == "MDDF");
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
        Array.Copy(WowViewer.Core.Chunks.FourCC.FromString(id).ToFileBytes(), 0, bytes, 0, 4);
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

    private static byte[] CreateSplitMcnkPayload(params (string Id, byte[] Payload)[] subchunks)
    {
        using MemoryStream stream = new();
        foreach ((string id, byte[] payload) in subchunks)
            stream.Write(CreateChunk(id, payload));
        return stream.ToArray();
    }

    private static byte[] CreateRootMcnkPayload(uint indexX, uint indexY, bool includeMcvt, bool includeMcse, bool includeMcrf, bool includeUnknownMclv)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        if (includeMcvt)
            stream.Write(CreateChunk("MCVT", new byte[16]));

        if (includeMcse)
            stream.Write(CreateChunk("MCSE", [0x10, 0x11, 0x12, 0x13]));

        if (includeMcrf)
            stream.Write(CreateChunk("MCRF", [0x21, 0x22, 0x23, 0x24]));

        if (includeUnknownMclv)
            stream.Write(CreateChunk("MCLV", [0x31, 0x32, 0x33, 0x34]));

        return stream.ToArray();
    }

    private static bool HasBytes(byte[] actual, params byte[] expected)
    {
        if (actual.Length != expected.Length)
            return false;

        for (int index = 0; index < actual.Length; index++)
        {
            if (actual[index] != expected[index])
                return false;
        }

        return true;
    }
}