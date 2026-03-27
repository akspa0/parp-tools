using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoPortalGroupRangeSummaryReaderTests
{
    [Fact]
    public void Read_MoprAndMogiBuffers_ProducesPortalGroupRangeSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(groupCount: 3)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGI", new byte[3 * 32]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPR", CreatePortalRefs((0, 0), (1, 2), (2, 1), (0, 5))),
        ];

        using MemoryStream stream = new(bytes);
        WmoPortalGroupRangeSummary summary = WmoPortalGroupRangeSummaryReader.Read(stream, "synthetic_mopr_mogi_root.wmo");

        Assert.Equal(4, summary.RefCount);
        Assert.Equal(3, summary.GroupCount);
        Assert.Equal(3, summary.CoveredRefCount);
        Assert.Equal(1, summary.OutOfRangeRefCount);
        Assert.Equal(4, summary.DistinctGroupRefCount);
        Assert.Equal(5, summary.MaxGroupIndex);
    }

    private static byte[] CreateMohd(uint groupCount)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), groupCount);
        return bytes;
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
