using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupBspNodeSummaryReaderTests
{
    [Fact]
    public void Read_WmoGroupBspNodeBuffer_ProducesNodeSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x80,
                ("MOBN", CreateNodes(
                    (0x0000, 1, -1, 2, 0, -4.0f),
                    (0x0004, -1, 3, 1, 2, 2.5f))))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupBspNodeSummary summary = WmoGroupBspNodeSummaryReader.Read(stream, "synthetic_mobn_000.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(32, summary.PayloadSizeBytes);
        Assert.Equal(2, summary.NodeCount);
        Assert.Equal(1, summary.LeafNodeCount);
        Assert.Equal(1, summary.BranchNodeCount);
        Assert.Equal(2, summary.ChildReferenceCount);
        Assert.Equal(2, summary.NoChildReferenceCount);
        Assert.Equal(1, summary.OutOfRangeChildReferenceCount);
        Assert.Equal(1, summary.MinFaceCount);
        Assert.Equal(2, summary.MaxFaceCount);
        Assert.Equal(0, summary.MinFaceStart);
        Assert.Equal(2, summary.MaxFaceStart);
        Assert.Equal(3, summary.MaxFaceEnd);
        Assert.Equal(-4.0f, summary.MinPlaneDistance);
        Assert.Equal(2.5f, summary.MaxPlaneDistance);
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

    private static byte[] CreateNodes(params (ushort Flags, short NegativeChild, short PositiveChild, ushort FaceCount, int FaceStart, float PlaneDistance)[] nodes)
    {
        byte[] bytes = new byte[nodes.Length * 16];
        for (int index = 0; index < nodes.Length; index++)
        {
            int offset = index * 16;
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset, 2), nodes[index].Flags);
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(offset + 2, 2), nodes[index].NegativeChild);
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(offset + 4, 2), nodes[index].PositiveChild);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 6, 2), nodes[index].FaceCount);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 8, 4), nodes[index].FaceStart);
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset + 12, 4), BitConverter.SingleToInt32Bits(nodes[index].PlaneDistance));
        }

        return bytes;
    }
}