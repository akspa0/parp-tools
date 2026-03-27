using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoDoodadNameReferenceSummaryReaderTests
{
    [Fact]
    public void Read_ModnAndModdBuffers_ProducesReferenceSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODN", CreateStringBlock("foo.mdx", "bar.m2")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODD", CreateModdPayload(0, 8, 999)),
        ];

        using MemoryStream stream = new(bytes);
        WmoDoodadNameReferenceSummary summary = WmoDoodadNameReferenceSummaryReader.Read(stream, "synthetic_modd_modn_root.wmo");

        Assert.Equal(3, summary.EntryCount);
        Assert.Equal(2, summary.ResolvedNameCount);
        Assert.Equal(1, summary.UnresolvedNameCount);
        Assert.Equal(2, summary.DistinctResolvedNameCount);
        Assert.Equal(7, summary.MaxResolvedNameLength);
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

    private static byte[] CreateModdPayload(params uint[] nameOffsets)
    {
        byte[] bytes = new byte[nameOffsets.Length * 40];
        for (int i = 0; i < nameOffsets.Length; i++)
            BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(i * 40, 4), nameOffsets[i]);

        return bytes;
    }
}
