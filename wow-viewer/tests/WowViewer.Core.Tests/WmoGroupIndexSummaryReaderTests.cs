using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupIndexSummaryReaderTests
{
    [Fact]
    public void Read_MoviBuffer_ProducesIndexSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x80,
                ("MOVI", CreateIndexPayload(0, 1, 2, 2, 2, 3)))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupIndexSummary summary = WmoGroupIndexSummaryReader.Read(stream, "synthetic_movi_000.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal("MOVI", summary.ChunkId);
        Assert.Equal(12, summary.PayloadSizeBytes);
        Assert.Equal(6, summary.IndexCount);
        Assert.Equal(2, summary.TriangleCount);
        Assert.Equal(4, summary.DistinctIndexCount);
        Assert.Equal(0, summary.MinIndex);
        Assert.Equal(3, summary.MaxIndex);
        Assert.Equal(1, summary.DegenerateTriangleCount);
    }

    [Fact]
    public void Read_MoinBuffer_ProducesIndexSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x44,
                ("MOIN", CreateIndexPayload(4, 5, 6)))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupIndexSummary summary = WmoGroupIndexSummaryReader.Read(stream, "synthetic_moin_000.wmo");

        Assert.Null(summary.Version);
        Assert.Equal("MOIN", summary.ChunkId);
        Assert.Equal(3, summary.IndexCount);
        Assert.Equal(1, summary.TriangleCount);
        Assert.Equal(3, summary.DistinctIndexCount);
        Assert.Equal(4, summary.MinIndex);
        Assert.Equal(6, summary.MaxIndex);
        Assert.Equal(0, summary.DegenerateTriangleCount);
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

    private static byte[] CreateIndexPayload(params ushort[] values)
    {
        byte[] bytes = new byte[values.Length * 2];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(index * 2, 2), values[index]);

        return bytes;
    }
}
