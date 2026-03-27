using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoVisibleVertexSummaryReaderTests
{
    [Fact]
    public void Read_MovvBuffer_ProducesVisibleVertexSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOVV", CreateVertices(new Vector3(1f, 2f, 3f), new Vector3(-4f, 5f, -6f), new Vector3(7f, -8f, 9f))),
        ];

        using MemoryStream stream = new(bytes);
        WmoVisibleVertexSummary summary = WmoVisibleVertexSummaryReader.Read(stream, "synthetic_visible_vertices_root.wmo");

        Assert.Equal(36, summary.PayloadSizeBytes);
        Assert.Equal(3, summary.VertexCount);
        Assert.Equal(new Vector3(-4f, -8f, -6f), summary.BoundsMin);
        Assert.Equal(new Vector3(7f, 5f, 9f), summary.BoundsMax);
    }

    [Fact]
    public void Read_WithoutMovvChunk_ThrowsInvalidDataException()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPT", new byte[20]),
        ];

        using MemoryStream stream = new(bytes);
        Assert.Throws<InvalidDataException>(() => WmoVisibleVertexSummaryReader.Read(stream, "synthetic_missing_movv_root.wmo"));
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
}
