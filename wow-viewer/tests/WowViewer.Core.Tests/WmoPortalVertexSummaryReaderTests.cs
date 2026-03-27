using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoPortalVertexSummaryReaderTests
{
    [Fact]
    public void Read_MopvBuffer_ProducesPortalVertexSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPV", CreateVertices(new Vector3(1f, 2f, 3f), new Vector3(-4f, 5f, -6f), new Vector3(7f, -8f, 9f))),
        ];

        using MemoryStream stream = new(bytes);
        WmoPortalVertexSummary summary = WmoPortalVertexSummaryReader.Read(stream, "synthetic_portal_vertices_root.wmo");

        Assert.Equal(36, summary.PayloadSizeBytes);
        Assert.Equal(3, summary.VertexCount);
        Assert.Equal(new Vector3(-4f, -8f, -6f), summary.BoundsMin);
        Assert.Equal(new Vector3(7f, 5f, 9f), summary.BoundsMax);
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
