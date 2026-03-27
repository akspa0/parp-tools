using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupVertexColorSummaryReaderTests
{
    [Fact]
    public void Read_WmoGroupVertexColorBuffer_ProducesColorSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x80,
                ("MOCV", new byte[] { 10, 20, 30, 40, 50, 60, 70, 80 }),
                ("MOCV", new byte[] { 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12 }))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupVertexColorSummary summary = WmoGroupVertexColorSummaryReader.Read(stream, "synthetic_mocv_000.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(8, summary.PrimaryPayloadSizeBytes);
        Assert.Equal(2, summary.PrimaryColorCount);
        Assert.Equal(30, summary.MinRed);
        Assert.Equal(70, summary.MaxRed);
        Assert.Equal(20, summary.MinGreen);
        Assert.Equal(60, summary.MaxGreen);
        Assert.Equal(10, summary.MinBlue);
        Assert.Equal(50, summary.MaxBlue);
        Assert.Equal(40, summary.MinAlpha);
        Assert.Equal(80, summary.MaxAlpha);
        Assert.Equal(60, summary.AverageAlpha);
        Assert.Equal(1, summary.AdditionalColorSetCount);
        Assert.Equal(3, summary.TotalAdditionalColorCount);
        Assert.Equal(3, summary.MaxAdditionalColorCount);
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
}
