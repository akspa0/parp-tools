using System.Buffers.Binary;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class AdtMcsePreservationTests
{
    [Fact]
    public void Build_DecodesMcseSignalsWhileKeepingRawFallback()
    {
        string tempDir = Path.Combine(Path.GetTempPath(), $"wowviewer_mcse_{Guid.NewGuid():N}");
        Directory.CreateDirectory(tempDir);

        try
        {
            string rootPath = Path.Combine(tempDir, "tile_0_0.adt");

            File.WriteAllBytes(rootPath,
            [
                .. CreateChunk("MVER", CreateUInt32Payload(18)),
                .. CreateChunk("MHDR", new byte[64]),
                .. CreateChunk("MCNK", CreateRootMcnkPayload(0, 0, CreateMcseEntry(1001, 1.5f, 2.5f, 3.5f, 9f, 8f, 7f), CreateMcseEntry(1002, 4.5f, 5.5f, 6.5f, 6f, 5f, 4f))),
            ]);

            TerrainTileTensorPack pack = AdtTensorPackBuilder.Build(rootPath, buildVersion: "3.3.5.12340");

            Assert.NotNull(pack.McseEmitterCounts16);
            Assert.NotNull(pack.McseEntryIds);
            Assert.NotNull(pack.McsePositionXyz);
            Assert.NotNull(pack.McseEntryBytes);
            Assert.Equal(2, pack.McseEmitterCounts16![0, 0]);
            Assert.Equal([1001, 1002], pack.McseEntryIds!);
            Assert.Equal(1.5f, pack.McsePositionXyz![0, 0]);
            Assert.Equal(2.5f, pack.McsePositionXyz[0, 1]);
            Assert.Equal(3.5f, pack.McsePositionXyz[0, 2]);
            Assert.Equal(2, pack.McseEntryBytes!.GetLength(0));
            Assert.Equal(AdtMcseReader.StandardEntrySize, pack.McseEntryBytes.GetLength(1));
            Assert.Equal(0xE9, pack.McseEntryBytes[0, 0]);
            Assert.Equal(0x03, pack.McseEntryBytes[0, 1]);
            Assert.Contains("mcse_entry_bytes", pack.AvailableSignals);
            Assert.Contains("mcse_entry_ids", pack.AvailableSignals);
            Assert.Contains("mcse_position_xyz", pack.AvailableSignals);
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

    private static byte[] CreateRootMcnkPayload(uint indexX, uint indexY, params byte[][] mcseEntries)
    {
        byte[] header = new byte[128];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), indexX);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), indexY);
        BinaryPrimitives.WriteInt32LittleEndian(header.AsSpan(0x5C, 4), mcseEntries.Length);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        stream.Write(CreateChunk("MCVT", new byte[145 * sizeof(float)]));
        stream.Write(CreateChunk("MCNR", new byte[435]));
        stream.Write(CreateChunk("MCSE", [.. mcseEntries.SelectMany(static entry => entry)]));
        stream.Write(CreateChunk("MCLV", [0xAA, 0xBB, 0xCC, 0xDD]));
        return stream.ToArray();
    }

    private static byte[] CreateMcseEntry(int entryId, float x, float y, float z, float sizeX, float sizeY, float sizeZ)
    {
        byte[] entry = new byte[AdtMcseReader.StandardEntrySize];
        BinaryPrimitives.WriteInt32LittleEndian(entry.AsSpan(0x00, 4), entryId);
        BinaryPrimitives.WriteSingleLittleEndian(entry.AsSpan(0x04, 4), x);
        BinaryPrimitives.WriteSingleLittleEndian(entry.AsSpan(0x08, 4), y);
        BinaryPrimitives.WriteSingleLittleEndian(entry.AsSpan(0x0C, 4), z);
        BinaryPrimitives.WriteSingleLittleEndian(entry.AsSpan(0x10, 4), sizeX);
        BinaryPrimitives.WriteSingleLittleEndian(entry.AsSpan(0x14, 4), sizeY);
        BinaryPrimitives.WriteSingleLittleEndian(entry.AsSpan(0x18, 4), sizeZ);
        return entry;
    }
}