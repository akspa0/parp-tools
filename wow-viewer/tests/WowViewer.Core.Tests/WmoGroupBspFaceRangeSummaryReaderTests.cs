using System.Buffers.Binary;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupBspFaceRangeSummaryReaderTests
{
    [Fact]
    public void Read_MobnAndMobrBuffers_ProducesRangeSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(0x80,
                ("MOBN", CreateNodes(
                    (0x0004, -1, -1, 0, 0, 0.0f),
                    (0x0004, -1, -1, 2, 0, 1.0f),
                    (0x0004, -1, -1, 3, 3, 2.0f))),
                ("MOBR", CreateRefPayload(0, 1, 2, 3, 4)))),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupBspFaceRangeSummary summary = WmoGroupBspFaceRangeSummaryReader.Read(stream, "synthetic_mobn_mobr_000.wmo");

        Assert.Equal(3, summary.NodeCount);
        Assert.Equal(5, summary.FaceRefCount);
        Assert.Equal(1, summary.ZeroFaceNodeCount);
        Assert.Equal(1, summary.CoveredNodeCount);
        Assert.Equal(1, summary.OutOfRangeNodeCount);
        Assert.Equal(6, summary.MaxFaceEnd);
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

    private static byte[] CreateRefPayload(params ushort[] values)
    {
        byte[] bytes = new byte[values.Length * 2];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(index * 2, 2), values[index]);

        return bytes;
    }
}