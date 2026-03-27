using System.Buffers.Binary;
using WowViewer.Core.IO.Maps;
using WowViewer.Core.Maps;

namespace WowViewer.Core.Tests;

public sealed class WdtSummaryReaderTests
{
    [Fact]
    public void Read_StandardWdtBuffer_ProducesSemanticSummary()
    {
        byte[] mainData = new byte[64 * 64 * 8];
        WriteUInt32(mainData, 0, 1);
        WriteUInt32(mainData, 8, 1);
        WriteUInt32(mainData, 16, 0);

        byte[] mphdData = new byte[32];
        WriteUInt32(mphdData, 0, 1);

        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MPHD", mphdData),
            .. CreateChunk("MAIN", mainData),
            .. CreateChunk("MMDX", CreateStringBlock("foo.mdx", "bar.mdx")),
            .. CreateChunk("MWMO", CreateStringBlock("baz.wmo")),
            .. CreateChunk("MDDF", new byte[36 * 2]),
            .. CreateChunk("MODF", new byte[64]),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "standard.wdt");
        WdtSummary summary = WdtSummaryReader.Read(stream, fileSummary);

        Assert.True(summary.IsWmoBased);
        Assert.Equal(2, summary.TilesWithData);
        Assert.Equal(4096, summary.TotalTiles);
        Assert.Equal(8, summary.MainCellSizeBytes);
        Assert.Equal(2, summary.DoodadNameCount);
        Assert.Equal(1, summary.WorldModelNameCount);
        Assert.Equal(2, summary.DoodadPlacementCount);
        Assert.Equal(1, summary.WorldModelPlacementCount);
    }

    [Fact]
    public void Read_AlphaWdtBuffer_ProducesSemanticSummary()
    {
        byte[] mainData = new byte[64 * 64 * 16];
        WriteUInt32(mainData, 0, 128);
        WriteUInt32(mainData, 16, 256);

        byte[] mphdData = new byte[32];
        WriteUInt32(mphdData, 8, 2);

        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(18)),
            .. CreateChunk("MPHD", mphdData),
            .. CreateChunk("MAIN", mainData),
            .. CreateChunk("MDNM", CreateStringBlock("alpha_a.mdx", "alpha_b.mdx", "alpha_c.mdx")),
            .. CreateChunk("MONM", CreateStringBlock("alpha_world.wmo")),
            .. CreateChunk("MODF", new byte[64 * 2]),
        ];

        using MemoryStream stream = new(bytes);
        MapFileSummary fileSummary = MapFileSummaryReader.Read(stream, "alpha.wdt");
        WdtSummary summary = WdtSummaryReader.Read(stream, fileSummary);

        Assert.True(summary.IsWmoBased);
        Assert.Equal(2, summary.TilesWithData);
        Assert.Equal(16, summary.MainCellSizeBytes);
        Assert.Equal(3, summary.DoodadNameCount);
        Assert.Equal(1, summary.WorldModelNameCount);
        Assert.Equal(0, summary.DoodadPlacementCount);
        Assert.Equal(2, summary.WorldModelPlacementCount);
    }

    [Fact]
    public void Read_DevelopmentWdt_ProducesExpectedSemanticSummary()
    {
        WdtSummary summary = WdtSummaryReader.Read(MapTestPaths.DevelopmentWdtPath);

        Assert.False(summary.IsWmoBased);
        Assert.Equal(1496, summary.TilesWithData);
        Assert.Equal(4096, summary.TotalTiles);
        Assert.Equal(8, summary.MainCellSizeBytes);
        Assert.Equal(0, summary.DoodadNameCount);
        Assert.Equal(0, summary.WorldModelNameCount);
        Assert.Equal(0, summary.DoodadPlacementCount);
        Assert.Equal(0, summary.WorldModelPlacementCount);
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

    private static void WriteUInt32(byte[] bytes, int offset, uint value)
    {
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(offset, 4), value);
    }
}