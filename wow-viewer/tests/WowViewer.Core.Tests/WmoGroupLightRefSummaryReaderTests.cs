using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupLightRefSummaryReaderTests
{
    [Fact]
    public void Read_WmoGroupLightRefBuffer_ProducesRefSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x80,
                ("MOLR", CreateRefPayload(2, 5, 2, 7)))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupLightRefSummary summary = WmoGroupLightRefSummaryReader.Read(stream, "synthetic_molr_000.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(8, summary.PayloadSizeBytes);
        Assert.Equal(4, summary.RefCount);
        Assert.Equal(3, summary.DistinctRefCount);
        Assert.Equal(2, summary.MinRef);
        Assert.Equal(7, summary.MaxRef);
        Assert.Equal(1, summary.DuplicateRefCount);
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

    private static byte[] CreateRefPayload(params ushort[] values)
    {
        byte[] bytes = new byte[values.Length * 2];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(index * 2, 2), values[index]);

        return bytes;
    }
}