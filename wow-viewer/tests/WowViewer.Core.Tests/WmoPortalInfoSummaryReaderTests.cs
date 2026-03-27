using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoPortalInfoSummaryReaderTests
{
    [Fact]
    public void Read_MoptBuffer_ProducesPortalInfoSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", new byte[64]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPT", CreatePortalInfos((2, 4, 1f), (10, 3, -2f))),
        ];

        using MemoryStream stream = new(bytes);
        WmoPortalInfoSummary summary = WmoPortalInfoSummaryReader.Read(stream, "synthetic_portal_info_root.wmo");

        Assert.Equal(40, summary.PayloadSizeBytes);
        Assert.Equal(2, summary.EntryCount);
        Assert.Equal(10, summary.MaxStartVertex);
        Assert.Equal(4, summary.MaxVertexCount);
        Assert.Equal(-2f, summary.MinPlaneD);
        Assert.Equal(1f, summary.MaxPlaneD);
    }

    private static byte[] CreatePortalInfos(params (ushort Start, ushort Count, float PlaneD)[] infos)
    {
        byte[] bytes = new byte[infos.Length * 20];
        for (int i = 0; i < infos.Length; i++)
        {
            int offset = i * 20;
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset, 2), infos[i].Start);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 2, 2), infos[i].Count);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 4, 4), BitConverter.SingleToInt32Bits(0f));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 8, 4), BitConverter.SingleToInt32Bits(0f));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 12, 4), BitConverter.SingleToInt32Bits(0f));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 16, 4), BitConverter.SingleToInt32Bits(infos[i].PlaneD));
        }

        return bytes;
    }
}
