using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoRenderDocumentReaderTests
{
    [Fact]
    public void Read_AlphaRootWithEmbeddedGroups_ProducesRenderableDocument()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(14)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMO", MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 1, groupCount: 1))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("stone.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMT", CreateMomtEntry(48, texture1Offset: 0)),
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
                    ("MOPY", CreateMopyEntryV14(flags: 0x01, materialId: 0x02, legacyExtraValue: 0x0304)),
                    ("MOVI", CreateIndices(0, 1, 2)),
                    ("MOVT", CreateVertices(new Vector3(1f, 2f, 3f), new Vector3(4f, 5f, 6f), new Vector3(7f, 8f, 9f))),
                    ("MONR", CreateVertices(Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ)),
                    ("MOTV", CreateUvs((0.1f, 0.2f), (0.3f, 0.4f), (0.5f, 0.6f))),
                    ("MOTV", CreateUvs((0.7f, 0.8f), (0.9f, 1.0f), (1.1f, 1.2f))),
                    ("MOCV", CreateColors((1, 2, 3, 4), (5, 6, 7, 8), (9, 10, 11, 12))),
                    ("MOCV", CreateColors((13, 14, 15, 16), (17, 18, 19, 20), (21, 22, 23, 24))),
                    ("MOBA", CreateMobaEntry(materialIdRaw: 0x02, firstIndex: 0, indexCount: 3, flags: 0x80)),
                    ("MODR", CreateRefs(9, 10)),
                    ("MOLR", CreateRefs(4)),
                ])),
        ];

        using MemoryStream stream = new(bytes);
        WmoRenderDocument document = WmoRenderDocumentReader.Read(stream, "synthetic_alpha_document.wmo");

        Assert.Equal((uint)14, document.Version);
        Assert.Single(document.Materials);
        Assert.Equal("stone.blp", document.Materials[0].Texture1Name);
        WmoEmbeddedGroupMeshDetail group = Assert.Single(document.Groups);
        Assert.Equal(3, group.Mesh.Vertices.Count);
        Assert.Equal(3, group.Mesh.Normals.Count);
        Assert.Equal("MOVI", group.Mesh.IndexChunkId);
        Assert.Equal(new ushort[] { 0, 1, 2 }, group.Mesh.Indices);
        Assert.Equal(3, group.Mesh.PrimaryUvs.Count);
        Assert.Single(group.Mesh.AdditionalUvSets);
        Assert.Equal(3, group.Mesh.PrimaryVertexColorsBgra.Count);
        Assert.Single(group.Mesh.AdditionalVertexColorSetsBgra);
        WmoGroupFaceMaterialDetail face = Assert.Single(group.Mesh.FaceMaterials);
        Assert.Equal((byte)0x01, face.Flags);
        Assert.Equal((byte)0x02, face.MaterialId);
        Assert.Equal((ushort)0x0304, face.LegacyExtraValue);
        WmoGroupBatchDetail batch = Assert.Single(group.Mesh.Batches);
        Assert.True(batch.HasMaterialId);
        Assert.Equal(0, batch.FirstIndex);
        Assert.Equal(3, batch.IndexCount);
        Assert.Equal((byte)0x80, batch.Flags);
        Assert.Equal(new ushort[] { 9, 10 }, group.DoodadRefs);
        Assert.Equal(new ushort[] { 4 }, group.LightRefs);
    }

    [Fact]
    public void Read_V17RootWithEmbeddedGroups_UsesStandardLayouts()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 1, groupCount: 1)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("modern.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMT", CreateMomtEntry(64, texture1Offset: 0)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", CreateMogpPayload(
                headerSize: 0x44,
                flags: 0x8,
                boundsMin: new Vector3(-1f, -1f, -1f),
                boundsMax: new Vector3(1f, 1f, 1f),
                portalStart: 0,
                portalCount: 0,
                transBatchCount: 0,
                intBatchCount: 1,
                extBatchCount: 0,
                groupLiquid: 0,
                nameOffset: 0,
                descriptiveNameOffset: 0,
                subchunks:
                [
                    ("MOPY", CreateMopyEntryV17(flags: 0x05, materialId: 0x06)),
                    ("MOIN", CreateIndices(0, 2, 1)),
                    ("MOVT", CreateVertices(new Vector3(0f, 0f, 0f), new Vector3(1f, 0f, 0f), new Vector3(0f, 1f, 0f))),
                    ("MONR", CreateVertices(Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ)),
                    ("MOBA", CreateMobaEntry(materialIdRaw: 0x06, firstIndex: 0, indexCount: 3, flags: 0x00)),
                ])),
        ];

        using MemoryStream stream = new(bytes);
        WmoRenderDocument document = WmoRenderDocumentReader.Read(stream, "synthetic_v17_document.wmo");

        WmoEmbeddedGroupMeshDetail group = Assert.Single(document.Groups);
        Assert.Equal((uint)17, group.Mesh.Version);
        Assert.Equal(0x44, group.Mesh.HeaderSizeBytes);
        Assert.Equal("MOIN", group.Mesh.IndexChunkId);
        WmoGroupFaceMaterialDetail face = Assert.Single(group.Mesh.FaceMaterials);
        Assert.Null(face.LegacyExtraValue);
        Assert.Equal((byte)0x05, face.Flags);
        Assert.Equal((byte)0x06, face.MaterialId);
    }

    [Fact]
    public void Read_Castle01AlphaPerAssetMpq_DocumentCountsMatchSummary()
    {
        string mpqPath = WmoTestPaths.Castle01AlphaMpqPath;
        if (!File.Exists(mpqPath))
            return;

        byte[] bytes = AlphaArchiveReader.ReadWithMpqFallback(mpqPath)!;

        using MemoryStream stream = new(bytes);
        WmoRenderDocument document = WmoRenderDocumentReader.Read(stream, mpqPath);

        Assert.Equal(document.Summary.ReportedMaterialCount, document.Materials.Count);
        Assert.Equal(2, document.Groups.Count);
        Assert.Equal(document.Groups.Sum(static group => group.GroupSummary.VertexCount), document.Groups.Sum(static group => group.Mesh.Vertices.Count));
        Assert.Equal(document.Groups.Sum(static group => group.GroupSummary.IndexCount), document.Groups.Sum(static group => group.Mesh.Indices.Count));
        Assert.Equal(document.Groups.Sum(static group => group.GroupSummary.NormalCount), document.Groups.Sum(static group => group.Mesh.Normals.Count));
    }

    private static byte[] CreateMohd(uint materialCount, uint groupCount)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), materialCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), groupCount);
        return bytes;
    }

    private static byte[] CreateMomtEntry(int entrySize, uint texture1Offset)
    {
        byte[] bytes = new byte[entrySize];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12, 4), texture1Offset);
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
        if (headerSize >= 0x38)
            BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x34, 4), groupLiquid);

        using MemoryStream stream = new();
        stream.Write(header, 0, header.Length);
        foreach ((string id, byte[] payload) in subchunks)
            stream.Write(MapFileSummaryReaderTestsAccessor.CreateChunk(id, payload));

        return stream.ToArray();
    }

    private static byte[] CreateVertices(params Vector3[] values)
    {
        byte[] bytes = new byte[values.Length * 12];
        for (int index = 0; index < values.Length; index++)
        {
            WriteSingle(bytes, index * 12 + 0, values[index].X);
            WriteSingle(bytes, index * 12 + 4, values[index].Y);
            WriteSingle(bytes, index * 12 + 8, values[index].Z);
        }

        return bytes;
    }

    private static byte[] CreateIndices(params ushort[] values)
    {
        byte[] bytes = new byte[values.Length * 2];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(index * 2, 2), values[index]);

        return bytes;
    }

    private static byte[] CreateRefs(params ushort[] values)
    {
        return CreateIndices(values);
    }

    private static byte[] CreateUvs(params (float U, float V)[] values)
    {
        byte[] bytes = new byte[values.Length * 8];
        for (int index = 0; index < values.Length; index++)
        {
            WriteSingle(bytes, index * 8 + 0, values[index].U);
            WriteSingle(bytes, index * 8 + 4, values[index].V);
        }

        return bytes;
    }

    private static byte[] CreateColors(params (byte Blue, byte Green, byte Red, byte Alpha)[] values)
    {
        byte[] bytes = new byte[values.Length * 4];
        for (int index = 0; index < values.Length; index++)
        {
            int offset = index * 4;
            bytes[offset + 0] = values[index].Blue;
            bytes[offset + 1] = values[index].Green;
            bytes[offset + 2] = values[index].Red;
            bytes[offset + 3] = values[index].Alpha;
        }

        return bytes;
    }

    private static byte[] CreateMopyEntryV14(byte flags, byte materialId, ushort legacyExtraValue)
    {
        byte[] bytes = new byte[4];
        bytes[0] = flags;
        bytes[1] = materialId;
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2, 2), legacyExtraValue);
        return bytes;
    }

    private static byte[] CreateMopyEntryV17(byte flags, byte materialId)
    {
        return [flags, materialId];
    }

    private static byte[] CreateMobaEntry(byte materialIdRaw, ushort firstIndex, ushort indexCount, byte flags)
    {
        byte[] bytes = new byte[24];
        bytes[1] = materialIdRaw;
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(14, 2), firstIndex);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(16, 2), indexCount);
        bytes[22] = flags;
        return bytes;
    }

    private static byte[] CreateStringBlock(params string[] entries)
    {
        using MemoryStream stream = new();
        foreach (string entry in entries)
        {
            byte[] bytes = System.Text.Encoding.UTF8.GetBytes(entry);
            stream.Write(bytes, 0, bytes.Length);
            stream.WriteByte(0);
        }

        if ((stream.Length & 1) != 0)
            stream.WriteByte(0);

        return stream.ToArray();
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}