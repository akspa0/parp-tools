using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoOpaqueChunkSummaryReaderTests
{
    [Fact]
    public void Read_McvpBuffer_ProducesOpaqueChunkSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MCVP", new byte[12]),
        ];

        using MemoryStream stream = new(bytes);
        WmoOpaqueChunkSummary summary = WmoOpaqueChunkSummaryReader.Read(stream, "synthetic_mcvp_root.wmo", WmoChunkIds.Mcvp);

        Assert.Equal("MCVP", summary.ChunkId);
        Assert.Equal(12, summary.PayloadSizeBytes);
    }
}
