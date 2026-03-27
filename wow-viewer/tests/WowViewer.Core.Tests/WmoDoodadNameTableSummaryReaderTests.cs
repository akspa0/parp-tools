using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoDoodadNameTableSummaryReaderTests
{
    [Fact]
    public void Read_ModnBuffer_ProducesDoodadNameTableSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODN", CreateStringBlock("foo.mdx", "bar.m2", "doodads/baz.mdx")),
        ];

        using MemoryStream stream = new(bytes);
        WmoDoodadNameTableSummary summary = WmoDoodadNameTableSummaryReader.Read(stream, "synthetic_doodads_root.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(31, summary.PayloadSizeBytes);
        Assert.Equal(3, summary.NameCount);
        Assert.Equal(15, summary.LongestEntryLength);
        Assert.Equal(15, summary.MaxOffset);
        Assert.Equal(2, summary.DistinctExtensionCount);
        Assert.Equal(2, summary.MdxEntryCount);
        Assert.Equal(1, summary.M2EntryCount);
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
