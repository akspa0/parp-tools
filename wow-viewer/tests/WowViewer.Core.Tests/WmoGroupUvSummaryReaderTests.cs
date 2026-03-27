using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupUvSummaryReaderTests
{
    [Fact]
    public void Read_WmoGroupUvBuffer_ProducesUvSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x80,
                ("MOTV", CreateUvPayload((0.1f, 0.2f), (0.8f, 0.9f), (-0.2f, 0.4f))),
                ("MOTV", CreateUvPayload((0.3f, 0.7f), (0.6f, 0.5f))))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupUvSummary summary = WmoGroupUvSummaryReader.Read(stream, "synthetic_uv_000.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(24, summary.PrimaryPayloadSizeBytes);
        Assert.Equal(3, summary.PrimaryUvCount);
        Assert.Equal(-0.2f, summary.MinU);
        Assert.Equal(0.8f, summary.MaxU);
        Assert.Equal(0.2f, summary.MinV);
        Assert.Equal(0.9f, summary.MaxV);
        Assert.Equal(1, summary.AdditionalUvSetCount);
        Assert.Equal(2, summary.TotalAdditionalUvCount);
        Assert.Equal(2, summary.MaxAdditionalUvCount);
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

    private static byte[] CreateUvPayload(params (float U, float V)[] values)
    {
        using MemoryStream stream = new();
        foreach ((float u, float v) in values)
        {
            byte[] buffer = new byte[8];
            BinaryPrimitives.WriteInt32LittleEndian(buffer.AsSpan(0, 4), BitConverter.SingleToInt32Bits(u));
            BinaryPrimitives.WriteInt32LittleEndian(buffer.AsSpan(4, 4), BitConverter.SingleToInt32Bits(v));
            stream.Write(buffer, 0, buffer.Length);
        }

        return stream.ToArray();
    }
}
