using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoPortalVertexRangeSummaryReaderTests
{
    [Fact]
    public void Read_MoptAndMopvBuffers_ProducesPortalVertexRangeSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPV", new byte[6 * 12]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPT", CreatePortalInfos((0, 4), (4, 0), (5, 3))),
        ];

        using MemoryStream stream = new(bytes);
        WmoPortalVertexRangeSummary summary = WmoPortalVertexRangeSummaryReader.Read(stream, "synthetic_mopt_mopv_root.wmo");

        Assert.Equal(3, summary.EntryCount);
        Assert.Equal(6, summary.VertexCount);
        Assert.Equal(1, summary.ZeroVertexPortalCount);
        Assert.Equal(1, summary.CoveredPortalCount);
        Assert.Equal(1, summary.OutOfRangePortalCount);
        Assert.Equal(8, summary.MaxVertexEnd);
    }

    private static byte[] CreatePortalInfos(params (ushort Start, ushort Count)[] infos)
    {
        byte[] bytes = new byte[infos.Length * 20];
        for (int i = 0; i < infos.Length; i++)
        {
            int offset = i * 20;
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset, 2), infos[i].Start);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 2, 2), infos[i].Count);
        }

        return bytes;
    }
}
