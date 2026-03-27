using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoVisibleBlockReferenceSummaryReaderTests
{
    [Fact]
    public void Read_MovbAndMovvBuffers_ProducesReferenceSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOVV", CreateVertices(new Vector3(1f, 0f, 0f), new Vector3(2f, 0f, 0f), new Vector3(3f, 0f, 0f), new Vector3(4f, 0f, 0f), new Vector3(5f, 0f, 0f), new Vector3(6f, 0f, 0f))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOVB", CreateBlocks((4, 0), (0, 4), (3, 5))),
        ];

        using MemoryStream stream = new(bytes);
        WmoVisibleBlockReferenceSummary summary = WmoVisibleBlockReferenceSummaryReader.Read(stream, "synthetic_visible_block_refs_root.wmo");

        Assert.Equal(3, summary.BlockCount);
        Assert.Equal(6, summary.VisibleVertexCount);
        Assert.Equal(1, summary.ZeroVertexBlockCount);
        Assert.Equal(1, summary.CoveredBlockCount);
        Assert.Equal(1, summary.OutOfRangeBlockCount);
        Assert.Equal(8, summary.MaxVertexEnd);
    }

    private static byte[] CreateVertices(params Vector3[] vertices)
    {
        byte[] bytes = new byte[vertices.Length * 12];
        for (int i = 0; i < vertices.Length; i++)
        {
            int offset = i * 12;
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(vertices[i].X));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 4, 4), BitConverter.SingleToInt32Bits(vertices[i].Y));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 8, 4), BitConverter.SingleToInt32Bits(vertices[i].Z));
        }

        return bytes;
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
