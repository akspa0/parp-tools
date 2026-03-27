using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoTextureTableSummaryReaderTests
{
    [Fact]
    public void Read_MotxBuffer_ProducesTextureTableSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("foo.blp", "bar.png", "textures/baz.blp")),
        ];

        using MemoryStream stream = new(bytes);
        WmoTextureTableSummary summary = WmoTextureTableSummaryReader.Read(stream, "synthetic_textures_root.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(33, summary.PayloadSizeBytes);
        Assert.Equal(3, summary.TextureCount);
        Assert.Equal(16, summary.LongestEntryLength);
        Assert.Equal(16, summary.MaxOffset);
        Assert.Equal(2, summary.DistinctExtensionCount);
        Assert.Equal(2, summary.BlpEntryCount);
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
