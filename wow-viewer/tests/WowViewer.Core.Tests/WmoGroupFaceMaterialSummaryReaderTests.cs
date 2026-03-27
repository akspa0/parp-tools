using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupFaceMaterialSummaryReaderTests
{
    [Fact]
    public void Read_V17StyleMopyBuffer_ProducesFaceMaterialSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x80,
                ("MOPY", new byte[] { 1, 3, 0, 7, 2, 255, 0, 3 }))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupFaceMaterialSummary summary = WmoGroupFaceMaterialSummaryReader.Read(stream, "synthetic_mopy_v17_000.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(8, summary.PayloadSizeBytes);
        Assert.Equal(2, summary.EntrySizeBytes);
        Assert.Equal(4, summary.FaceCount);
        Assert.Equal(2, summary.DistinctMaterialIdCount);
        Assert.Equal(7, summary.HighestMaterialId);
        Assert.Equal(1, summary.HiddenFaceCount);
        Assert.Equal(2, summary.FlaggedFaceCount);
    }

    [Fact]
    public void Read_V16StyleMopyBuffer_ProducesFaceMaterialSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(16)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x44,
                ("MOPY", new byte[] { 0, 4, 0, 0, 1, 255, 0, 0 }))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupFaceMaterialSummary summary = WmoGroupFaceMaterialSummaryReader.Read(stream, "synthetic_mopy_v16_000.wmo");

        Assert.Equal((uint)16, summary.Version);
        Assert.Equal(4, summary.EntrySizeBytes);
        Assert.Equal(2, summary.FaceCount);
        Assert.Equal(1, summary.DistinctMaterialIdCount);
        Assert.Equal(4, summary.HighestMaterialId);
        Assert.Equal(1, summary.HiddenFaceCount);
        Assert.Equal(1, summary.FlaggedFaceCount);
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
