using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupNameTableSummaryReaderTests
{
    [Fact]
    public void Read_MognBuffer_ProducesGroupNameSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGN", CreateStringBlock("group_one", "lower-cave", "raid_hall")),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupNameTableSummary summary = WmoGroupNameTableSummaryReader.Read(stream, "synthetic_group_names_root.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(31, summary.PayloadSizeBytes);
        Assert.Equal(3, summary.NameCount);
        Assert.Equal(10, summary.LongestEntryLength);
        Assert.Equal(21, summary.MaxOffset);
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
}
