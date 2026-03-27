using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoVisibleBlockSummaryReaderTests
{
    [Fact]
    public void Read_MovbBuffer_ProducesVisibleBlockSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOVB", CreateBlocks((4, 0), (3, 4), (2, 8))),
        ];

        using MemoryStream stream = new(bytes);
        WmoVisibleBlockSummary summary = WmoVisibleBlockSummaryReader.Read(stream, "synthetic_visible_blocks_root.wmo");

        Assert.Equal(12, summary.PayloadSizeBytes);
        Assert.Equal(3, summary.BlockCount);
        Assert.Equal(9, summary.TotalVertexRefs);
        Assert.Equal(2, summary.MinVerticesPerBlock);
        Assert.Equal(4, summary.MaxVerticesPerBlock);
        Assert.Equal(0, summary.MinFirstVertex);
        Assert.Equal(8, summary.MaxFirstVertex);
        Assert.Equal(10, summary.MaxVertexEnd);
    }

    private static byte[] CreateBlocks(params (ushort VertexCount, ushort FirstVertex)[] blocks)
    {
        byte[] bytes = new byte[blocks.Length * 4];
        for (int i = 0; i < blocks.Length; i++)
        {
            int offset = i * 4;
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset, 2), blocks[i].VertexCount);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 2, 2), blocks[i].FirstVertex);
        }

        return bytes;
    }
}
