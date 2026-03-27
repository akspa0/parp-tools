using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoEmbeddedGroupSummaryReaderTests
{
    [Fact]
    public void Read_AlphaRootWithEmbeddedGroups_ProducesAggregateSummary()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(14)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMO", MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(groupCount: 2))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(
                headerSize: 0x44,
                flags: 0x8,
                boundsMin: new Vector3(-2f, -3f, -4f),
                boundsMax: new Vector3(5f, 6f, 7f),
                portalStart: 10,
                portalCount: 1,
                transBatchCount: 0,
                intBatchCount: 1,
                extBatchCount: 0,
                groupLiquid: 0,
                nameOffset: 1,
                descriptiveNameOffset: 2,
                subchunks:
                [
                    ("MOPY", new byte[8]),
                    ("MOVI", new byte[6]),
                    ("MOVT", new byte[24]),
                    ("MONR", new byte[24]),
                    ("MOBA", new byte[24]),
                    ("MODR", new byte[4]),
                ])),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(
                headerSize: 0x80,
                flags: 0x2009,
                boundsMin: new Vector3(-7f, 1f, 2f),
                boundsMax: new Vector3(3f, 8f, 9f),
                portalStart: 20,
                portalCount: 2,
                transBatchCount: 1,
                intBatchCount: 2,
                extBatchCount: 3,
                groupLiquid: 7,
                nameOffset: 3,
                descriptiveNameOffset: 4,
                subchunks:
                [
                    ("MOPY", new byte[12]),
                    ("MOIN", new byte[12]),
                    ("MOVT", new byte[36]),
                    ("MONR", new byte[36]),
                    ("MOBA", new byte[48]),
                    ("MODR", new byte[2]),
                    ("MLIQ", new byte[32]),
                ])),
        ];

        using MemoryStream stream = new(bytes);
        WmoEmbeddedGroupSummary summary = WmoEmbeddedGroupSummaryReader.Read(stream, "synthetic_alpha_root.wmo");

        Assert.Equal((uint)14, summary.Version);
        Assert.Equal(2, summary.GroupCount);
        Assert.Equal(0x44, summary.MinHeaderSizeBytes);
        Assert.Equal(0x80, summary.MaxHeaderSizeBytes);
        Assert.Equal(2, summary.GroupsWithPortals);
        Assert.Equal(1, summary.GroupsWithLiquid);
        Assert.Equal(5, summary.TotalFaceMaterialCount);
        Assert.Equal(5, summary.TotalVertexCount);
        Assert.Equal(9, summary.TotalIndexCount);
        Assert.Equal(5, summary.TotalNormalCount);
        Assert.Equal(3, summary.TotalBatchCount);
        Assert.Equal(3, summary.TotalDoodadRefCount);
        Assert.Equal(new Vector3(-7f, -3f, -4f), summary.BoundsMin);
        Assert.Equal(new Vector3(5f, 8f, 9f), summary.BoundsMax);
    }

    private static byte[] CreateMohd(uint groupCount)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), groupCount);
        return bytes;
    }

    private static byte[] CreateMogpPayload(
        int headerSize,
        uint flags,
        Vector3 boundsMin,
        Vector3 boundsMax,
        ushort portalStart,
        ushort portalCount,
        ushort transBatchCount,
        ushort intBatchCount,
        ushort extBatchCount,
        uint groupLiquid,
        uint nameOffset,
        uint descriptiveNameOffset,
        params (string Id, byte[] Payload)[] subchunks)
    {
        byte[] header = new byte[headerSize];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00, 4), nameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), descriptiveNameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), flags);
        WriteSingle(header, 0x0C, boundsMin.X);
        WriteSingle(header, 0x10, boundsMin.Y);
        WriteSingle(header, 0x14, boundsMin.Z);
        WriteSingle(header, 0x18, boundsMax.X);
        WriteSingle(header, 0x1C, boundsMax.Y);
        WriteSingle(header, 0x20, boundsMax.Z);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x24, 2), portalStart);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x26, 2), portalCount);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x28, 2), transBatchCount);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x2A, 2), intBatchCount);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x2C, 2), extBatchCount);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x34, 4), groupLiquid);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        foreach ((string id, byte[] payload) in subchunks)
            stream.Write(MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload));

        return stream.ToArray();
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}
