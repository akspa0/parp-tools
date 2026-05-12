using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoV17ToV14ConverterTests
{
    [Fact]
    public void Convert_SyntheticV17RootAndGroup_ProducesReadableV14Document()
    {
        byte[] rootBytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 1, groupCount: 1)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("modern.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMT", CreateMomtEntry(64, texture1Offset: 0)),
        ];

        byte[] groupBytes = CreateGroupFile(17, CreateMogpPayload(
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
                ("MOTV", CreateUvs((0.1f, 0.2f), (0.3f, 0.4f), (0.5f, 0.6f))),
                ("MOBA", CreateMobaEntryV17(materialIdRaw: 0x06, firstIndex: 0, indexCount: 3, firstVertex: 0, lastVertex: 2, flags: 0x80)),
            ]));

        byte[] converted = WmoV17ToV14Converter.Convert(rootBytes, [groupBytes], "synthetic_v17_root.wmo");

        using MemoryStream renderStream = new(converted);
        WmoRenderDocument document = WmoRenderDocumentReader.Read(renderStream, "converted_v14.wmo");

        Assert.Equal((uint)14, document.Version);
        WmoMaterialDetail material = Assert.Single(document.Materials);
        Assert.Equal(48, material.EntrySizeBytes);

        WmoEmbeddedGroupMeshDetail group = Assert.Single(document.Groups);
        Assert.Equal("MOVI", group.Mesh.IndexChunkId);

        WmoGroupFaceMaterialDetail face = Assert.Single(group.Mesh.FaceMaterials);
        Assert.Equal((byte)0x05, face.Flags);
        Assert.Equal((byte)0x06, face.MaterialId);
        Assert.Equal((ushort)0, face.LegacyExtraValue);

        WmoGroupBatchDetail batch = Assert.Single(group.Mesh.Batches);
        Assert.True(batch.HasMaterialId);
        Assert.Equal(6, batch.MaterialId);
        Assert.Equal((ushort)0, batch.FirstIndex);
        Assert.Equal((ushort)3, batch.IndexCount);
        Assert.Equal((byte)0x80, batch.Flags);

        using MemoryStream topLevelStream = new(converted);
        IReadOnlyList<ChunkSpan> topLevelChunks = ChunkedFileReader.ReadTopLevelChunks(topLevelStream, padOddChunkSizes: false);
        ChunkSpan momoChunk = Assert.Single(topLevelChunks, static chunk => chunk.Header.Id == WmoChunkIds.Momo);
        byte[] momoPayload = ReadChunkPayload(converted, momoChunk);

        using MemoryStream momoStream = new(momoPayload);
        IReadOnlyList<ChunkSpan> momoChunks = ChunkedFileReader.ReadTopLevelChunks(momoStream, padOddChunkSizes: false);
        ChunkSpan mogpChunk = Assert.Single(momoChunks, static chunk => chunk.Header.Id == WmoChunkIds.Mogp);
        byte[] legacyGroupPayload = ReadChunkPayload(momoPayload, mogpChunk);
        byte[] legacyGroupFile = CreateGroupFile(14, legacyGroupPayload);

        using MemoryStream groupStream = new(legacyGroupFile);
        WmoGroupFaceMaterialSummary faceSummary = WmoGroupFaceMaterialSummaryReader.Read(groupStream, "converted_group_000.wmo");
        Assert.Equal((uint)14, faceSummary.Version);
        Assert.Equal(4, faceSummary.EntrySizeBytes);
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

    private static byte[] CreateGroupFile(uint version, byte[] mogpPayload)
    {
        return
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(version)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", mogpPayload),
        ];
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

    private static byte[] CreateMopyEntryV17(byte flags, byte materialId)
    {
        return [flags, materialId];
    }

    private static byte[] CreateMobaEntryV17(byte materialIdRaw, uint firstIndex, ushort indexCount, ushort firstVertex, ushort lastVertex, byte flags)
    {
        byte[] bytes = new byte[24];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12, 4), firstIndex);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(16, 2), indexCount);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(18, 2), firstVertex);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(20, 2), lastVertex);
        bytes[22] = flags;
        bytes[23] = materialIdRaw;
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

    private static byte[] ReadChunkPayload(byte[] bytes, ChunkSpan chunk)
    {
        return bytes.AsSpan(checked((int)chunk.DataOffset), checked((int)chunk.Header.Size)).ToArray();
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }
}