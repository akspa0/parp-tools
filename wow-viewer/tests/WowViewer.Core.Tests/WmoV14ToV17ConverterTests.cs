using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoV14ToV17ConverterTests
{
    [Fact]
    public void Convert_SyntheticV14Root_ProducesReadableSplitV17Document()
    {
        byte[] embeddedGroup = CreateMogpPayload(
            headerSize: 0x80,
            flags: 0x2009,
            boundsMin: new Vector3(-2f, -3f, -4f),
            boundsMax: new Vector3(5f, 6f, 7f),
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
                ("MOPY", CreateMopyEntryV14(flags: 0x01, materialId: 0x02, legacyExtraValue: 0x0304)),
                ("MOIN", CreateIndices(0, 1, 2)),
                ("MOVT", CreateVertices(new Vector3(1f, 2f, 3f), new Vector3(4f, 5f, 6f), new Vector3(7f, 8f, 9f))),
                ("MONR", CreateVertices(Vector3.UnitX, Vector3.UnitY, Vector3.UnitZ)),
                ("MOTV", CreateUvs((0.1f, 0.2f), (0.3f, 0.4f), (0.5f, 0.6f))),
                ("MOBA", CreateMobaEntryLegacy(materialIdRaw: 0x02, firstIndex: 0, indexCount: 3, flags: 0x80)),
            ]);

        byte[] rootBytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(14)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMO",
            [
                .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 1, groupCount: 1)),
                .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("legacy.blp")),
                .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMT", CreateMomtEntry(entrySize: 48, texture1Offset: 0)),
                .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGN", CreateStringBlock("group_000")),
                .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGI", CreateMogiEntryV14(flags: 0x2009, boundsMin: new Vector3(-2f, -3f, -4f), boundsMax: new Vector3(5f, 6f, 7f), nameOffset: 0)),
                .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGP", embeddedGroup),
            ]),
        ];

        WmoV14ToV17Converter.SplitWmoResult converted = WmoV14ToV17Converter.Convert(rootBytes, "synthetic_v14_root.wmo");

        Assert.Single(converted.GroupBytes);

        using MemoryStream rootSummaryStream = new(converted.RootBytes);
        WmoSummary rootSummary = WmoSummaryReader.Read(rootSummaryStream, "converted_v17_root.wmo");
        Assert.Equal((uint)17, rootSummary.Version);
        Assert.Equal(1, rootSummary.MaterialEntryCount);
        Assert.Equal(1, rootSummary.GroupInfoCount);

        using MemoryStream materialStream = new(converted.RootBytes);
        WmoMaterialDetail material = Assert.Single(WmoMaterialDetailReader.Read(materialStream, "converted_v17_root.wmo"));
        Assert.Equal(64, material.EntrySizeBytes);
        Assert.Equal("legacy.blp", material.Texture1Name);

        using MemoryStream groupNameStream = new(converted.RootBytes);
        WmoGroupNameReferenceSummary groupNameSummary = WmoGroupNameReferenceSummaryReader.Read(groupNameStream, "converted_v17_root.wmo");
        Assert.Equal(1, groupNameSummary.ResolvedNameCount);

        using MemoryStream groupSummaryStream = new(converted.GroupBytes[0]);
        WmoGroupSummary groupSummary = WmoGroupSummaryReader.Read(groupSummaryStream, "converted_v17_group_000.wmo");
        Assert.Equal((uint)17, groupSummary.Version);
        Assert.Equal(0x44, groupSummary.HeaderSizeBytes);

        using MemoryStream faceSummaryStream = new(converted.GroupBytes[0]);
        WmoGroupFaceMaterialSummary faceSummary = WmoGroupFaceMaterialSummaryReader.Read(faceSummaryStream, "converted_v17_group_000.wmo");
        Assert.Equal(2, faceSummary.EntrySizeBytes);
        Assert.Equal(1, faceSummary.FaceCount);

        using MemoryStream meshStream = new(converted.GroupBytes[0]);
        WmoGroupMeshDetail mesh = WmoGroupMeshDetailReader.Read(meshStream, "converted_v17_group_000.wmo");
        Assert.Equal("MOVI", mesh.IndexChunkId);

        WmoGroupFaceMaterialDetail face = Assert.Single(mesh.FaceMaterials);
        Assert.Equal((byte)0x01, face.Flags);
        Assert.Equal((byte)0x02, face.MaterialId);
        Assert.Null(face.LegacyExtraValue);

        WmoGroupBatchDetail batch = Assert.Single(mesh.Batches);
        Assert.Equal(0, batch.FirstIndex);
        Assert.Equal(3, batch.IndexCount);
        Assert.Equal((byte)0x80, batch.Flags);
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

    private static byte[] CreateMogiEntryV14(uint flags, Vector3 boundsMin, Vector3 boundsMax, uint nameOffset)
    {
        byte[] bytes = new byte[40];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8, 4), flags);
        WriteSingle(bytes, 12, boundsMin.X);
        WriteSingle(bytes, 16, boundsMin.Y);
        WriteSingle(bytes, 20, boundsMin.Z);
        WriteSingle(bytes, 24, boundsMax.X);
        WriteSingle(bytes, 28, boundsMax.Y);
        WriteSingle(bytes, 32, boundsMax.Z);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(36, 4), nameOffset);
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

    private static byte[] CreateMopyEntryV14(byte flags, byte materialId, ushort legacyExtraValue)
    {
        byte[] bytes = new byte[4];
        bytes[0] = flags;
        bytes[1] = materialId;
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2, 2), legacyExtraValue);
        return bytes;
    }

    private static byte[] CreateMobaEntryLegacy(byte materialIdRaw, ushort firstIndex, ushort indexCount, byte flags)
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