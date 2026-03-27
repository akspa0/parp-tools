using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoGroupSummaryReaderTests
{
    [Fact]
    public void Read_WmoGroupBuffer_ProducesSemanticSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MVER", CreateUInt32Payload(17)),
            .. CreateChunk("MOGP", CreateMogpPayload(
                headerSize: 0x80,
                flags: 0x2009,
                boundsMin: new Vector3(-2f, -3f, -4f),
                boundsMax: new Vector3(5f, 6f, 7f),
                portalStart: 11,
                portalCount: 2,
                transBatchCount: 1,
                intBatchCount: 2,
                extBatchCount: 3,
                groupLiquid: 7,
                nameOffset: 123,
                descriptiveNameOffset: 456,
                subchunks:
                [
                    ("MOPY", new byte[8]),
                    ("MOVI", new byte[12]),
                    ("MOVT", new byte[36]),
                    ("MONR", new byte[36]),
                    ("MOTV", new byte[24]),
                    ("MOTV", new byte[24]),
                    ("MOBA", new byte[48]),
                    ("MOCV", new byte[12]),
                    ("MODR", new byte[6]),
                    ("MLIQ", new byte[32]),
                ])),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupSummary summary = WmoGroupSummaryReader.Read(stream, "synthetic_000.wmo");

        Assert.Equal((uint)17, summary.Version);
        Assert.Equal(0x80, summary.HeaderSizeBytes);
        Assert.Equal((uint)123, summary.NameOffset);
        Assert.Equal((uint)456, summary.DescriptiveNameOffset);
        Assert.Equal(0x2009u, summary.Flags);
        Assert.Equal(new Vector3(-2f, -3f, -4f), summary.BoundsMin);
        Assert.Equal(new Vector3(5f, 6f, 7f), summary.BoundsMax);
        Assert.Equal(11, summary.PortalStart);
        Assert.Equal(2, summary.PortalCount);
        Assert.Equal(1, summary.TransparentBatchCount);
        Assert.Equal(2, summary.InteriorBatchCount);
        Assert.Equal(3, summary.ExteriorBatchCount);
        Assert.Equal(6, summary.DeclaredBatchCount);
        Assert.Equal(7u, summary.GroupLiquid);
        Assert.Equal(4, summary.FaceMaterialCount);
        Assert.Equal(3, summary.VertexCount);
        Assert.Equal(6, summary.IndexCount);
        Assert.Equal(3, summary.NormalCount);
        Assert.Equal(3, summary.PrimaryUvCount);
        Assert.Equal(1, summary.AdditionalUvSetCount);
        Assert.Equal(2, summary.BatchCount);
        Assert.Equal(3, summary.VertexColorCount);
        Assert.Equal(3, summary.DoodadRefCount);
        Assert.True(summary.HasLiquid);
    }

    [Fact]
    public void Read_WmoGroupBufferWithoutMver_ProducesSemanticSummary()
    {
        byte[] bytes =
        [
            .. CreateChunk("MOGP", CreateMogpPayload(
                headerSize: 0x44,
                flags: 0x8,
                boundsMin: Vector3.Zero,
                boundsMax: Vector3.One,
                portalStart: 0,
                portalCount: 0,
                transBatchCount: 0,
                intBatchCount: 1,
                extBatchCount: 0,
                groupLiquid: 0,
                nameOffset: 5,
                descriptiveNameOffset: 6,
                subchunks:
                [
                    ("MOPY", new byte[12]),
                    ("MOIN", new byte[6]),
                    ("MOVT", new byte[24]),
                ])),
        ];

        using MemoryStream stream = new(bytes);
        WmoGroupSummary summary = WmoGroupSummaryReader.Read(stream, "synthetic_nomver_000.wmo");

        Assert.Null(summary.Version);
        Assert.Equal(0x44, summary.HeaderSizeBytes);
        Assert.Equal(3, summary.FaceMaterialCount);
        Assert.Equal(3, summary.IndexCount);
        Assert.Equal(2, summary.VertexCount);
        Assert.False(summary.HasLiquid);
    }

    private static byte[] CreateChunk(string id, byte[] payload)
    {
        return MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload);
    }

    private static byte[] CreateUInt32Payload(uint value)
    {
        return MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(value);
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
            stream.Write(CreateChunk(id, payload));

        return stream.ToArray();
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}
