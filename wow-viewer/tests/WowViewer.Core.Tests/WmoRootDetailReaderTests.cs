using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.IO.Files;
using WowViewer.Core.IO.Wmo;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Tests;

public sealed class WmoRootDetailReaderTests
{
    [Fact]
    public void Read_SyntheticRoot_ProducesPortalAndDoodadDetails()
    {
        byte[] bytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 1, groupCount: 1, portalCount: 1, doodadPlacementCount: 2, doodadSetCount: 1, doodadNameCount: 2)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("stone.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMT", CreateMomtEntry(64, texture1Offset: 0)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPV", CreateVertices(new Vector3(0f, 0f, 0f), new Vector3(1f, 0f, 0f), new Vector3(1f, 1f, 0f), new Vector3(0f, 1f, 0f))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPT", CreatePortalInfo(0, 4, Vector3.UnitZ, 2.0f)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPR", CreatePortalRef(0, 0, 1)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODN", CreateStringBlock("doodad_a.mdx", "doodad_b.m2")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODS", CreateDoodadSet("Default", 0, 2, 0)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODD", CreateDoodadPlacements(
                CreateDoodadPlacement(0, new Vector3(2f, 3f, 4f), Quaternion.Identity, 1.0f, 0xFF112233),
                CreateDoodadPlacement(13, new Vector3(5f, 6f, 7f), Quaternion.Identity, 0.5f, 0xAA445566))),
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
                ("MOPY", CreateMopyEntryV17(flags: 0x05, materialId: 0x06)),
                ("MOIN", CreateIndices(0, 2, 1)),
                ("MOVT", CreateVertices(new Vector3(0f, 0f, 0f), new Vector3(1f, 0f, 0f), new Vector3(0f, 1f, 0f))),
                ("MONR", CreateVertices(Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ)),
                ("MOBA", CreateMobaEntry(materialIdRaw: 0x06, firstIndex: 0, indexCount: 3, flags: 0x00))))
        ];

        using MemoryStream stream = new(bytes);
        WmoRenderDocument document = WmoRenderDocumentReader.Read(stream, "synthetic_root_details.wmo");

        Assert.Single(document.Portals);
        Assert.Equal(4, document.PortalVertices.Count);
        Assert.Single(document.PortalReferences);
        Assert.Single(document.DoodadSets);
        Assert.Equal(2, document.DoodadPlacements.Count);
        Assert.Equal("Default", document.DoodadSets[0].Name);
        Assert.Equal("doodad_a.mdx", document.DoodadPlacements[0].ModelPath);
        Assert.Equal("doodad_b.m2", document.DoodadPlacements[1].ModelPath);
        Assert.Equal(WmoDoodadModelKind.Mdx, document.DoodadPlacements[0].ModelKind);
        Assert.Equal(WmoDoodadModelKind.M2, document.DoodadPlacements[1].ModelKind);
        Assert.Equal(Vector3.UnitZ, document.Portals[0].Normal);
        Assert.Equal(2.0f, document.Portals[0].PlaneDistance);
    }

    [Fact]
    public void Read_Castle01AlphaPerAssetMpq_ProducesRootPortalAndDoodadOwnership()
    {
        string mpqPath = WmoTestPaths.Castle01AlphaMpqPath;
        if (!File.Exists(mpqPath))
            return;

        byte[] bytes = AlphaArchiveReader.ReadWithMpqFallback(mpqPath)!;
        using MemoryStream stream = new(bytes);
        WmoRenderDocument document = WmoRenderDocumentReader.Read(stream, mpqPath);

        Assert.Equal(document.Summary.ReportedPortalCount, document.Portals.Count);
        Assert.Equal(document.Summary.ReportedDoodadSetCount, document.DoodadSets.Count);
        Assert.Equal(document.Summary.ReportedDoodadPlacementCount, document.DoodadPlacements.Count);
    }

    private static byte[] CreateMohd(uint materialCount, uint groupCount, uint portalCount, uint doodadPlacementCount, uint doodadSetCount, uint doodadNameCount)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), materialCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), groupCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8, 4), portalCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(16, 4), doodadNameCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(20, 4), doodadPlacementCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24, 4), doodadSetCount);
        return bytes;
    }

    private static byte[] CreateMomtEntry(int entrySize, uint texture1Offset)
    {
        byte[] bytes = new byte[entrySize];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(12, 4), texture1Offset);
        return bytes;
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

    private static byte[] CreatePortalInfo(ushort startVertex, ushort vertexCount, Vector3 normal, float planeDistance)
    {
        byte[] bytes = new byte[20];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(0, 2), startVertex);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2, 2), vertexCount);
        WriteSingle(bytes, 4, normal.X);
        WriteSingle(bytes, 8, normal.Y);
        WriteSingle(bytes, 12, normal.Z);
        WriteSingle(bytes, 16, planeDistance);
        return bytes;
    }

    private static byte[] CreatePortalRef(ushort portalIndex, ushort groupIndex, short side)
    {
        byte[] bytes = new byte[8];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(0, 2), portalIndex);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(2, 2), groupIndex);
        BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(4, 2), side);
        return bytes;
    }

    private static byte[] CreateDoodadSet(string name, uint startIndex, uint count, uint flags)
    {
        byte[] bytes = new byte[32];
        byte[] nameBytes = System.Text.Encoding.UTF8.GetBytes(name);
        Array.Copy(nameBytes, bytes, Math.Min(nameBytes.Length, 20));
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(20, 4), startIndex);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(24, 4), count);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(28, 4), flags);
        return bytes;
    }

    private static byte[] CreateDoodadPlacements(params byte[][] entries)
    {
        using MemoryStream stream = new();
        foreach (byte[] entry in entries)
            stream.Write(entry, 0, entry.Length);

        return stream.ToArray();
    }

    private static byte[] CreateDoodadPlacement(uint nameIndex, Vector3 position, Quaternion rotation, float scale, uint color)
    {
        byte[] bytes = new byte[40];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), nameIndex);
        WriteSingle(bytes, 4, position.X);
        WriteSingle(bytes, 8, position.Y);
        WriteSingle(bytes, 12, position.Z);
        WriteSingle(bytes, 16, rotation.X);
        WriteSingle(bytes, 20, rotation.Y);
        WriteSingle(bytes, 24, rotation.Z);
        WriteSingle(bytes, 28, rotation.W);
        WriteSingle(bytes, 32, scale);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(36, 4), color);
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

    private static byte[] CreateIndices(params ushort[] values)
    {
        byte[] bytes = new byte[values.Length * 2];
        for (int index = 0; index < values.Length; index++)
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(index * 2, 2), values[index]);

        return bytes;
    }

    private static byte[] CreateMopyEntryV17(byte flags, byte materialId)
    {
        return [flags, materialId];
    }

    private static byte[] CreateMobaEntry(ushort materialIdRaw, uint firstIndex, ushort indexCount, byte flags)
    {
        byte[] bytes = new byte[24];
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(12, 2), materialIdRaw);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(16, 4), firstIndex);
        BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(20, 2), indexCount);
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