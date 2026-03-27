using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoSkyboxSummaryReaderTests
{
    [Fact]
    public void Read_MosbBuffer_ProducesSkyboxSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOSB", System.Text.Encoding.UTF8.GetBytes("Sky\0")),
        ];

        using MemoryStream stream = new(bytes);
        WmoSkyboxSummary summary = WmoSkyboxSummaryReader.Read(stream, "synthetic_skybox_root.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(4, summary.PayloadSizeBytes);
        Assert.Equal("Sky", summary.SkyboxName);
    }
}
