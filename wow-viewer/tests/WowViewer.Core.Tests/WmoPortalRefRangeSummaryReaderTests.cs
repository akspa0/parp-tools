using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoPortalRefRangeSummaryReaderTests
{
    [Fact]
    public void Read_MoprAndMoptBuffers_ProducesPortalRefRangeSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPT", new byte[3 * 20]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPR", CreatePortalRefs((0, 1), (1, 2), (2, 0), (4, 1))),
        ];

        using MemoryStream stream = new(bytes);
        WmoPortalRefRangeSummary summary = WmoPortalRefRangeSummaryReader.Read(stream, "synthetic_mopr_mopt_root.wmo");

        Assert.Equal(4, summary.RefCount);
        Assert.Equal(3, summary.PortalCount);
        Assert.Equal(3, summary.CoveredRefCount);
        Assert.Equal(1, summary.OutOfRangeRefCount);
        Assert.Equal(4, summary.MaxPortalIndex);
        Assert.Equal(4, summary.DistinctPortalRefCount);
    }

    private static byte[] CreatePortalRefs(params (ushort PortalIndex, ushort GroupIndex)[] refs)
    {
        byte[] bytes = new byte[refs.Length * 8];
        for (int i = 0; i < refs.Length; i++)
        {
            int offset = i * 8;
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset, 2), refs[i].PortalIndex);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 2, 2), refs[i].GroupIndex);
        }

        return bytes;
    }
}
