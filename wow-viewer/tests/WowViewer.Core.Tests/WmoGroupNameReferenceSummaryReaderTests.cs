using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupNameReferenceSummaryReaderTests
{
    [Fact]
    public void Read_MogiAndMognBuffers_ProducesReferenceSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(groupCount: 3)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGN", CreateStringBlock("group_one", "raid_hall")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGI", CreateMogiPayload(32, 0, 10, 999)),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupNameReferenceSummary summary = WmoGroupNameReferenceSummaryReader.Read(stream, "synthetic_mogi_mogn_root.wmo");

        Assert.Equal(3, summary.EntryCount);
        Assert.Equal(2, summary.ResolvedNameCount);
        Assert.Equal(1, summary.UnresolvedNameCount);
        Assert.Equal(2, summary.DistinctResolvedNameCount);
        Assert.Equal(9, summary.MaxResolvedNameLength);
    }

    private static byte[] CreateMohd(uint groupCount)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), groupCount);
        return bytes;
    }

    private static byte[] CreateStringBlock(params string[] entries)
    {
        using MemoryStream stream = new();
        foreach (string entry in entries)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(entry);
            stream.Write(bytes, 0, bytes.Length);
            stream.WriteByte(0);
        }

        return stream.ToArray();
    }

    private static byte[] CreateMogiPayload(int entrySize, params int[] nameOffsets)
    {
        byte[] bytes = new byte[nameOffsets.Length * entrySize];
        for (int i = 0; i < nameOffsets.Length; i++)
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(i * entrySize + 28, 4), nameOffsets[i]);

        return bytes;
    }
}
