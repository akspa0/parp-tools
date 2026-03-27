using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupVertexSummaryReaderTests
{
    [Fact]
    public void Read_MovtBuffer_ProducesVertexSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x80,
                ("MOVT", CreateVertexPayload(new Vector3(1f, 2f, 3f), new Vector3(-4f, 5f, -6f), new Vector3(7f, -8f, 9f))))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupVertexSummary summary = WmoGroupVertexSummaryReader.Read(stream, "synthetic_movt_000.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(36, summary.PayloadSizeBytes);
        Assert.Equal(3, summary.VertexCount);
        Assert.Equal(new Vector3(-4f, -8f, -6f), summary.BoundsMin);
        Assert.Equal(new Vector3(7f, 5f, 9f), summary.BoundsMax);
    }

    private static byte[] CreateMogpPayload(int headerSize, params (string Id, byte[] Payload)[] subchunks)
    {
        byte[] header = new byte[headerSize];
        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        foreach ((string id, byte[] payload) in subchunks)
            stream.Write(MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload));

        return stream.ToArray();
    }

    private static byte[] CreateVertexPayload(params Vector3[] vertices)
    {
        byte[] bytes = new byte[vertices.Length * 12];
        for (int index = 0; index < vertices.Length; index++)
        {
            int offset = index * 12;
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(vertices[index].X));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 4, 4), BitConverter.SingleToInt32Bits(vertices[index].Y));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 8, 4), BitConverter.SingleToInt32Bits(vertices[index].Z));
        }

        return bytes;
    }
}
