using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoEmbeddedGroupDetailReaderTests
{
    [Fact]
    public void Read_AlphaRootWithEmbeddedGroups_ProducesPerGroupDetails()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(14)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMO", MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(groupCount: 2))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(
                headerSize: 0x80,
                flags: 0x2009,
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
                    ("MOTV", CreateUvs((0.1f, 0.2f), (0.3f, 0.4f))),
                    ("MOCV", CreateColors((1, 2, 3, 4), (5, 6, 7, 8))),
                    ("MOBA", new byte[24]),
                    ("MODR", CreateRefs(9, 9, 10)),
                    ("MOLR", CreateRefs(4, 7)),
                    ("MOBN", CreateNodes((0x0004, -1, -1, 2, 0, 1.0f))),
                    ("MOBR", CreateRefs(0, 1)),
                ])),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(
                headerSize: 0x80,
                flags: 0x8,
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
                    ("MOBN", CreateNodes((0x0000, 0, -1, 1, 0, -2.0f), (0x0004, -1, -1, 1, 1, 3.0f))),
                    ("MOBR", CreateRefs(3, 5)),
                    ("MLIQ", CreateLiquidPayload(1, 1, 1, 1, 2.5f, 0x00)),
                ])),
        ];

        using MemoryStream stream = new(bytes);
        IReadOnlyList<WmoEmbeddedGroupDetail> details = WmoEmbeddedGroupDetailReader.Read(stream, "synthetic_alpha_root.wmo");

        Assert.Equal(2, details.Count);

        Assert.Equal(0, details[0].GroupIndex);
        Assert.NotNull(details[0].FaceMaterialSummary);
        Assert.NotNull(details[0].IndexSummary);
        Assert.NotNull(details[0].VertexSummary);
        Assert.NotNull(details[0].NormalSummary);
        Assert.NotNull(details[0].BatchSummary);
        Assert.NotNull(details[0].UvSummary);
        Assert.NotNull(details[0].VertexColorSummary);
        Assert.NotNull(details[0].DoodadRefSummary);
        Assert.NotNull(details[0].LightRefSummary);
        Assert.NotNull(details[0].BspNodeSummary);
        Assert.NotNull(details[0].BspFaceSummary);
        Assert.NotNull(details[0].BspFaceRangeSummary);
        Assert.Equal(2, details[0].FaceMaterialSummary!.FaceCount);
        Assert.Equal(3, details[0].IndexSummary!.IndexCount);
        Assert.Equal(2, details[0].VertexSummary!.VertexCount);
        Assert.Equal(2, details[0].NormalSummary!.NormalCount);
        Assert.Equal(1, details[0].BatchSummary!.EntryCount);
        Assert.Equal(2, details[0].UvSummary!.PrimaryUvCount);
        Assert.Equal(2, details[0].VertexColorSummary!.PrimaryColorCount);
        Assert.Equal(3, details[0].DoodadRefSummary!.RefCount);
        Assert.Equal(2, details[0].LightRefSummary!.RefCount);
        Assert.Equal(1, details[0].BspNodeSummary!.NodeCount);
        Assert.Equal(2, details[0].BspFaceSummary!.RefCount);
        Assert.Equal(1, details[0].BspFaceRangeSummary!.CoveredNodeCount);
        Assert.Null(details[0].LiquidSummary);

        Assert.Equal(1, details[1].GroupIndex);
        Assert.NotNull(details[1].FaceMaterialSummary);
        Assert.NotNull(details[1].IndexSummary);
        Assert.NotNull(details[1].VertexSummary);
        Assert.NotNull(details[1].NormalSummary);
        Assert.NotNull(details[1].BatchSummary);
        Assert.NotNull(details[1].LiquidSummary);
        Assert.Null(details[1].LightRefSummary);
        Assert.NotNull(details[1].BspNodeSummary);
        Assert.NotNull(details[1].BspFaceSummary);
        Assert.NotNull(details[1].BspFaceRangeSummary);
        Assert.Equal(3, details[1].FaceMaterialSummary!.FaceCount);
        Assert.Equal(6, details[1].IndexSummary!.IndexCount);
        Assert.Equal(3, details[1].VertexSummary!.VertexCount);
        Assert.Equal(3, details[1].NormalSummary!.NormalCount);
        Assert.Equal(2, details[1].BatchSummary!.EntryCount);
        Assert.Equal(2, details[1].BspNodeSummary!.NodeCount);
        Assert.Equal(2, details[1].BspFaceSummary!.RefCount);
        Assert.Equal(2, details[1].BspFaceRangeSummary!.CoveredNodeCount);
        Assert.Equal(0, details[1].BspFaceRangeSummary!.OutOfRangeNodeCount);
        Assert.Equal(1, details[1].LiquidSummary!.HeightCount);
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

    private static byte[] CreateRefs(params ushort[] values)
    {
        byte[] bytes = new byte[values.Length * 2];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(index * 2, 2), values[index]);

        return bytes;
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

    private static byte[] CreateUvs(params (float U, float V)[] values)
    {
        byte[] bytes = new byte[values.Length * 8];
        for (int index = 0; index < values.Length; index++)
        {
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(index * 8, 4), BitConverter.SingleToInt32Bits(values[index].U));
            BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(index * 8 + 4, 4), BitConverter.SingleToInt32Bits(values[index].V));
        }

        return bytes;
    }

    private static byte[] CreateColors(params (byte Blue, byte Green, byte Red, byte Alpha)[] values)
    {
        byte[] bytes = new byte[values.Length * 4];
        for (int index = 0; index < values.Length; index++)
        {
            int offset = index * 4;
            bytes[offset] = values[index].Blue;
            bytes[offset + 1] = values[index].Green;
            bytes[offset + 2] = values[index].Red;
            bytes[offset + 3] = values[index].Alpha;
        }

        return bytes;
    }

    private static byte[] CreateLiquidPayload(int xVertexCount, int yVertexCount, int xTileCount, int yTileCount, float height, byte firstTileNibble)
    {
        byte[] bytes = new byte[30 + (xVertexCount * yVertexCount * 8) + (xTileCount * yTileCount)];
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(0, 4), xVertexCount);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(4, 4), yVertexCount);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(8, 4), xTileCount);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(12, 4), yTileCount);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(20, 4), BitConverter.SingleToInt32Bits(height));
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(28, 2), 3);
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(30 + 4, 4), BitConverter.SingleToInt32Bits(height));
        bytes[38] = firstTileNibble;
        return bytes;
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}