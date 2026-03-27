using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoPortalRefSummaryReaderTests
{
    [Fact]
    public void Read_MoprBuffer_ProducesPortalRefSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPR", CreatePortalRefs((1, 5, 1), (2, 7, -1), (1, 3, 0))),
        ];

        using MemoryStream stream = new(bytes);
        WmoPortalRefSummary summary = WmoPortalRefSummaryReader.Read(stream, "synthetic_portal_refs_root.wmo");

        Assert.Equal(24, summary.PayloadSizeBytes);
        Assert.Equal(3, summary.EntryCount);
        Assert.Equal(2, summary.DistinctPortalIndexCount);
        Assert.Equal(7, summary.MaxGroupIndex);
        Assert.Equal(1, summary.PositiveSideCount);
        Assert.Equal(1, summary.NegativeSideCount);
        Assert.Equal(1, summary.NeutralSideCount);
    }

    private static byte[] CreatePortalRefs(params (ushort PortalIndex, ushort GroupIndex, short Side)[] refs)
    {
        byte[] bytes = new byte[refs.Length * 8];
        for (int i = 0; i < refs.Length; i++)
        {
            int offset = i * 8;
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset, 2), refs[i].PortalIndex);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 2, 2), refs[i].GroupIndex);
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(offset + 4, 2), refs[i].Side);
        }

        return bytes;
    }
}
