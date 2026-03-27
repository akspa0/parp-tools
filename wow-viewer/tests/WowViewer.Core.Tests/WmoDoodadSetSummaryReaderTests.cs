using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoDoodadSetSummaryReaderTests
{
    [Fact]
    public void Read_ModsBuffer_ProducesDoodadSetSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODS", CreateModsPayload(
                CreateModsEntry("Default", 0, 4),
                CreateModsEntry("Empty", 10, 0),
                CreateModsEntry("RaidSet", 12, 6))),
        ];

        using MemoryStream stream = new(bytes);
        WmoDoodadSetSummary summary = WmoDoodadSetSummaryReader.Read(stream, "synthetic_sets_root.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(96, summary.PayloadSizeBytes);
        Assert.Equal(3, summary.EntryCount);
        Assert.Equal(2, summary.NonEmptySetCount);
        Assert.Equal(7, summary.LongestNameLength);
        Assert.Equal(10, summary.TotalDoodadRefs);
        Assert.Equal(12, summary.MaxStartIndex);
        Assert.Equal(18, summary.MaxRangeEnd);
    }

    private static byte[] CreateModsPayload(params byte[][] entries)
    {
        using MemoryStream stream = new();
        foreach (byte[] entry in entries)
            stream.Write(entry, 0, entry.Length);

        return stream.ToArray();
    }

    private static byte[] CreateModsEntry(string name, uint startIndex, uint count)
    {
        byte[] bytes = new byte[32];
        byte[] nameBytes = System.Text.Encoding.ASCII.GetBytes(name);
        Array.Copy(nameBytes, 0, bytes, 0, Math.Min(nameBytes.Length, 20));
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(20, 4), startIndex);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24, 4), count);
        return bytes;
    }
}
