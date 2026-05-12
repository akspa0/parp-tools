using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
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

    [Fact]
    public void Convert_WhenSourceExceedsLegacyGroupLimit_MergesOverflowIntoFinalLegacyGroup()
    {
        const int sourceGroupCount = 385;

        byte[] rootBytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 1, groupCount: sourceGroupCount)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("compat.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMT", CreateMomtEntry(64, texture1Offset: 0)),
        ];

        List<byte[]> groupBytes = new(sourceGroupCount);
        for (int groupIndex = 0; groupIndex < sourceGroupCount; groupIndex++)
        {
            float offset = groupIndex * 10f;
            groupBytes.Add(CreateGroupFile(17, CreateMogpPayload(
                headerSize: 0x44,
                flags: 0x8,
                boundsMin: new Vector3(offset, 0f, 0f),
                boundsMax: new Vector3(offset + 1f, 1f, 1f),
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
                    ("MOPY", CreateMopyEntryV17(flags: 0x01, materialId: 0x02)),
                    ("MOIN", CreateIndices(0, 1, 2)),
                    ("MOVT", CreateVertices(new Vector3(offset, 0f, 0f), new Vector3(offset + 1f, 0f, 0f), new Vector3(offset, 1f, 0f))),
                    ("MONR", CreateVertices(Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ)),
                    ("MOBA", CreateMobaEntryV17(materialIdRaw: 0x02, firstIndex: 0, indexCount: 3, firstVertex: 0, lastVertex: 2, flags: 0x00)),
                ])));
        }

        byte[] converted = WmoV17ToV14Converter.Convert(rootBytes, groupBytes, "synthetic_v17_large_root.wmo");

        using MemoryStream summaryStream = new(converted);
        WmoSummary summary = WmoSummaryReader.Read(summaryStream, "converted_v14_large.wmo");
        Assert.Equal((uint)14, summary.Version);
        Assert.Equal(384, summary.ReportedGroupCount);
        Assert.Equal(384, summary.GroupInfoCount);

        using MemoryStream renderStream = new(converted);
        WmoRenderDocument document = WmoRenderDocumentReader.Read(renderStream, "converted_v14_large.wmo");
        Assert.Equal(384, document.Groups.Count);

        WmoEmbeddedGroupMeshDetail mergedGroup = Assert.Single(document.Groups, static group => group.Mesh.Vertices.Count == 6);
        Assert.Equal(6, mergedGroup.Mesh.Vertices.Count);
        Assert.Equal(6, mergedGroup.Mesh.Indices.Count);
        Assert.Equal(2, mergedGroup.Mesh.FaceMaterials.Count);
        Assert.Equal(2, mergedGroup.Mesh.Batches.Count);
        Assert.Equal(new Vector3(3830f, 0f, 0f), mergedGroup.GroupSummary.BoundsMin);
        Assert.Equal(new Vector3(3841f, 1f, 1f), mergedGroup.GroupSummary.BoundsMax);
    }

    [Fact]
    public void Convert_WhenSourceExceedsLegacyGroupLimit_RemapsPortalRefsIntoMergedGroupRange()
    {
        const int sourceGroupCount = 385;

        byte[] rootBytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 1, groupCount: sourceGroupCount, portalCount: 1)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("compat.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMT", CreateMomtEntry(64, texture1Offset: 0)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPV", CreatePortalVertices(
                new Vector3(3830f, 0f, 0f),
                new Vector3(3831f, 0f, 0f),
                new Vector3(3831f, 1f, 0f),
                new Vector3(3830f, 1f, 0f))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPT", CreatePortalInfo(startVertex: 0, vertexCount: 4, normal: Vector3.UnitX, planeDistance: 0f)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPR", CreatePortalRefs(
                (portalIndex: 0, groupIndex: 383, side: (short)-1),
                (portalIndex: 0, groupIndex: 384, side: (short)1))),
        ];

        List<byte[]> groupBytes = new(sourceGroupCount);
        for (int groupIndex = 0; groupIndex < sourceGroupCount; groupIndex++)
        {
            float offset = groupIndex * 10f;
            bool hasPortal = groupIndex >= 383;
            ushort portalStart = groupIndex switch
            {
                383 => 0,
                384 => 1,
                _ => 0,
            };
            ushort portalCount = hasPortal ? (ushort)1 : (ushort)0;

            groupBytes.Add(CreateGroupFile(17, CreateMogpPayload(
                headerSize: 0x44,
                flags: 0x8,
                boundsMin: new Vector3(offset, 0f, 0f),
                boundsMax: new Vector3(offset + 1f, 1f, 1f),
                portalStart: portalStart,
                portalCount: portalCount,
                transBatchCount: 0,
                intBatchCount: 1,
                extBatchCount: 0,
                groupLiquid: 0,
                nameOffset: 0,
                descriptiveNameOffset: 0,
                subchunks:
                [
                    ("MOPY", CreateMopyEntryV17(flags: 0x01, materialId: 0x02)),
                    ("MOIN", CreateIndices(0, 1, 2)),
                    ("MOVT", CreateVertices(new Vector3(offset, 0f, 0f), new Vector3(offset + 1f, 0f, 0f), new Vector3(offset, 1f, 0f))),
                    ("MONR", CreateVertices(Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ)),
                    ("MOBA", CreateMobaEntryV17(materialIdRaw: 0x02, firstIndex: 0, indexCount: 3, firstVertex: 0, lastVertex: 2, flags: 0x00)),
                ])));
        }

        byte[] converted = WmoV17ToV14Converter.Convert(rootBytes, groupBytes, "synthetic_v17_large_portal_root.wmo");

        using MemoryStream summaryStream = new(converted);
        WmoSummary summary = WmoSummaryReader.Read(summaryStream, "converted_v14_large_portal.wmo");
        Assert.Equal(384, summary.ReportedGroupCount);
        Assert.Equal(1, summary.ReportedPortalCount);

        using MemoryStream portalRefRangeStream = new(converted);
        WmoPortalRefRangeSummary portalRefRange = WmoPortalRefRangeSummaryReader.Read(portalRefRangeStream, "converted_v14_large_portal.wmo");
        Assert.Equal(0, portalRefRange.OutOfRangeRefCount);
        Assert.Equal(portalRefRange.RefCount, portalRefRange.CoveredRefCount);

        using MemoryStream portalGroupRangeStream = new(converted);
        WmoPortalGroupRangeSummary portalGroupRange = WmoPortalGroupRangeSummaryReader.Read(portalGroupRangeStream, "converted_v14_large_portal.wmo");
        Assert.Equal(0, portalGroupRange.OutOfRangeRefCount);
        Assert.Equal(portalGroupRange.RefCount, portalGroupRange.CoveredRefCount);

        using MemoryStream renderStream = new(converted);
        WmoRenderDocument document = WmoRenderDocumentReader.Read(renderStream, "converted_v14_large_portal.wmo");
        Assert.Contains(document.Groups, static group => group.GroupSummary.PortalCount > 0);
    }

    [Fact]
    public void Convert_WhenGroupBatchStartIndicesExceedLegacyRange_SplitsGroupIntoMultipleLegacyGroups()
    {
        byte[] rootBytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 1, groupCount: 1)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("compat.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMT", CreateMomtEntry(64, texture1Offset: 0)),
        ];

        const ushort batchIndexCount = 32769;
        byte[] groupBytes = CreateGroupFile(17, CreateMogpPayload(
            headerSize: 0x44,
            flags: 0x8,
            boundsMin: new Vector3(-1f, -1f, -1f),
            boundsMax: new Vector3(1f, 1f, 1f),
            portalStart: 0,
            portalCount: 0,
            transBatchCount: 0,
            intBatchCount: 3,
            extBatchCount: 0,
            groupLiquid: 0,
            nameOffset: 0,
            descriptiveNameOffset: 0,
            subchunks:
            [
                ("MOPY", CreateRepeatedMopyEntriesV17(faceCount: 21847, flags: 0x01, materialId: 0x02)),
                ("MOIN", CreateRepeatedTriangleIndices(21847, 0, 1, 2)),
                ("MOVT", CreateVertices(new Vector3(0f, 0f, 0f), new Vector3(1f, 0f, 0f), new Vector3(0f, 1f, 0f))),
                ("MONR", CreateVertices(Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ)),
                ("MOBA", CreateMobaPayloadV17(
                    CreateMobaEntryV17(materialIdRaw: 0x02, firstIndex: 0, indexCount: batchIndexCount, firstVertex: 0, lastVertex: 2, flags: 0x00),
                    CreateMobaEntryV17(materialIdRaw: 0x02, firstIndex: batchIndexCount, indexCount: batchIndexCount, firstVertex: 0, lastVertex: 2, flags: 0x00),
                    CreateMobaEntryV17(materialIdRaw: 0x02, firstIndex: 65538, indexCount: 3, firstVertex: 0, lastVertex: 2, flags: 0x00))),
            ]));

        byte[] converted = WmoV17ToV14Converter.Convert(rootBytes, [groupBytes], "synthetic_v17_oversized_batch_root.wmo");

        using MemoryStream summaryStream = new(converted);
        WmoSummary summary = WmoSummaryReader.Read(summaryStream, "converted_v14_oversized_batch.wmo");
        Assert.Equal((uint)14, summary.Version);
        Assert.Equal(2, summary.ReportedGroupCount);

        using MemoryStream renderStream = new(converted);
        WmoRenderDocument document = WmoRenderDocumentReader.Read(renderStream, "converted_v14_oversized_batch.wmo");
        Assert.Equal(2, document.Groups.Count);
        Assert.Equal(65541, document.Groups.Sum(static group => group.Mesh.Indices.Count));
        Assert.All(document.Groups.SelectMany(static group => group.Mesh.Batches), static batch => Assert.InRange(batch.FirstIndex, 0, ushort.MaxValue));
        Assert.All(document.Groups, static group => Assert.True(group.Mesh.Batches.Count > 0));
    }

    [Fact]
    public void Convert_WhenSourceCarriesSkyboxAndVisibilityChunks_EmitsStrictAlphaRootChunkOrder()
    {
        byte[] rootBytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 1, groupCount: 1, portalCount: 1)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOTX", CreateStringBlock("modern.blp")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOMT", CreateMomtEntry(64, texture1Offset: 0)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGN", CreateStringBlock("group")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOGI", CreateMogiEntry(flags: 0x8, boundsMin: Vector3.Zero, boundsMax: Vector3.One, nameOffset: 0)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOSB", CreateStringBlock("sky")),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPV", CreatePortalVertices(
                new Vector3(0f, 0f, 0f),
                new Vector3(1f, 0f, 0f),
                new Vector3(1f, 1f, 0f),
                new Vector3(0f, 1f, 0f))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPT", CreatePortalInfo(startVertex: 0, vertexCount: 4, normal: Vector3.UnitZ, planeDistance: 0f)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOPR", CreatePortalRefs((portalIndex: 0, groupIndex: 0, side: (short)1))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOVV", CreateVertices(new Vector3(2f, 2f, 2f))),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOVB", CreateIndices(0, 1)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOLT", [0x01]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODS", [0x02]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODN", [0x03]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MODD", [0x04]),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MFOG", [0x05]),
        ];

        byte[] groupBytes = CreateGroupFile(17, CreateMogpPayload(
            headerSize: 0x44,
            flags: 0x8,
            boundsMin: Vector3.Zero,
            boundsMax: Vector3.One,
            portalStart: 0,
            portalCount: 1,
            transBatchCount: 0,
            intBatchCount: 1,
            extBatchCount: 0,
            groupLiquid: 0,
            nameOffset: 0,
            descriptiveNameOffset: 0,
            subchunks:
            [
                ("MOPY", CreateMopyEntryV17(flags: 0x01, materialId: 0x00)),
                ("MOIN", CreateIndices(0, 1, 2)),
                ("MOVT", CreateVertices(new Vector3(0f, 0f, 0f), new Vector3(1f, 0f, 0f), new Vector3(0f, 1f, 0f))),
                ("MONR", CreateVertices(Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ)),
                ("MOBA", CreateMobaEntryV17(materialIdRaw: 0x00, firstIndex: 0, indexCount: 3, firstVertex: 0, lastVertex: 2, flags: 0x00)),
            ]));

        byte[] converted = WmoV17ToV14Converter.Convert(rootBytes, [groupBytes], "synthetic_v17_root_with_optional_chunks.wmo");

        using MemoryStream topLevelStream = new(converted);
        IReadOnlyList<ChunkSpan> topLevelChunks = ChunkedFileReader.ReadTopLevelChunks(topLevelStream, padOddChunkSizes: false);
        ChunkSpan momoChunk = Assert.Single(topLevelChunks, static chunk => chunk.Header.Id == WmoChunkIds.Momo);
        byte[] momoPayload = ReadChunkPayload(converted, momoChunk);

        using MemoryStream momoStream = new(momoPayload);
        IReadOnlyList<ChunkSpan> momoChunks = ChunkedFileReader.ReadTopLevelChunks(momoStream, padOddChunkSizes: false);
        FourCC[] rootChunkIds = momoChunks
            .TakeWhile(static chunk => chunk.Header.Id != WmoChunkIds.Mogp)
            .Select(static chunk => chunk.Header.Id)
            .ToArray();

        FourCC[] expectedRootChunkIds =
        [
            WmoChunkIds.Mohd,
            WmoChunkIds.Motx,
            WmoChunkIds.Momt,
            WmoChunkIds.Mogn,
            WmoChunkIds.Mogi,
            WmoChunkIds.Mopv,
            WmoChunkIds.Mopt,
            WmoChunkIds.Mopr,
            WmoChunkIds.Molt,
            WmoChunkIds.Mods,
            WmoChunkIds.Modn,
            WmoChunkIds.Modd,
            WmoChunkIds.Mfog,
        ];

        Assert.True(rootChunkIds.SequenceEqual(expectedRootChunkIds));
    }

    [Fact]
    public void Convert_WhenGroupVertexBudgetExceedsAlphaLimit_SplitsAndCompactsLegacyGroups()
    {
        const int verticesPerBatch = 25002;
        const int totalVertices = verticesPerBatch * 2;
        const int trianglesPerBatch = verticesPerBatch / 3;

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
            boundsMin: Vector3.Zero,
            boundsMax: new Vector3(totalVertices, 8f, 1f),
            portalStart: 0,
            portalCount: 0,
            transBatchCount: 0,
            intBatchCount: 2,
            extBatchCount: 0,
            groupLiquid: 0,
            nameOffset: 0,
            descriptiveNameOffset: 0,
            subchunks:
            [
                ("MOPY", CreateRepeatedMopyEntriesV17(faceCount: trianglesPerBatch * 2, flags: 0x01, materialId: 0x02)),
                ("MOIN", CreateSequentialTriangleIndices(0, trianglesPerBatch, verticesPerBatch, trianglesPerBatch)),
                ("MOVT", CreateSequentialVertices(totalVertices)),
                ("MONR", CreateRepeatedVertices(totalVertices, Vector3.UnitZ)),
                ("MOBA", CreateMobaPayloadV17(
                    CreateMobaEntryV17(materialIdRaw: 0x02, firstIndex: 0, indexCount: (ushort)verticesPerBatch, firstVertex: 0, lastVertex: (ushort)(verticesPerBatch - 1), flags: 0x00),
                    CreateMobaEntryV17(materialIdRaw: 0x02, firstIndex: (uint)verticesPerBatch, indexCount: (ushort)verticesPerBatch, firstVertex: (ushort)verticesPerBatch, lastVertex: (ushort)(totalVertices - 1), flags: 0x00))),
            ]));

        byte[] converted = WmoV17ToV14Converter.Convert(rootBytes, [groupBytes], "synthetic_v17_vertex_budget_root.wmo");

        using MemoryStream summaryStream = new(converted);
        WmoSummary summary = WmoSummaryReader.Read(summaryStream, "converted_v14_vertex_budget.wmo");
        Assert.Equal((uint)14, summary.Version);
        Assert.Equal(2, summary.ReportedGroupCount);

        using MemoryStream renderStream = new(converted);
        WmoRenderDocument document = WmoRenderDocumentReader.Read(renderStream, "converted_v14_vertex_budget.wmo");
        Assert.Equal(2, document.Groups.Count);
        Assert.All(document.Groups, static group => Assert.InRange(group.Mesh.Vertices.Count, 1, 0xBFFF));
        Assert.All(document.Groups, static group => Assert.Equal(group.Mesh.Vertices.Count, group.Mesh.Normals.Count));
        Assert.Equal(verticesPerBatch, document.Groups[0].Mesh.Vertices.Count);
        Assert.Equal(verticesPerBatch, document.Groups[1].Mesh.Vertices.Count);
    }

    [Fact]
    public void Convert_WhenSourceHasNoPortalOrOptionalRootChunks_StillEmitsEmptyAlphaRootChain()
    {
        byte[] rootBytes =
        [
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MVER", MapFileSummaryReaderTestsAccessor.CreateUInt32Payload(17)),
            .. MapFileSummaryReaderTestsAccessor.CreateChunk("MOHD", CreateMohd(materialCount: 1, groupCount: 1, portalCount: 0)),
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
                ("MOPY", CreateMopyEntryV17(flags: 0x01, materialId: 0x00)),
                ("MOIN", CreateIndices(0, 1, 2)),
                ("MOVT", CreateVertices(new Vector3(0f, 0f, 0f), new Vector3(1f, 0f, 0f), new Vector3(0f, 1f, 0f))),
                ("MONR", CreateVertices(Vector3.UnitZ, Vector3.UnitZ, Vector3.UnitZ)),
                ("MOBA", CreateMobaEntryV17(materialIdRaw: 0x00, firstIndex: 0, indexCount: 3, firstVertex: 0, lastVertex: 2, flags: 0x00)),
            ]));

        byte[] converted = WmoV17ToV14Converter.Convert(rootBytes, [groupBytes], "synthetic_v17_root_without_optional_chunks.wmo");

        using MemoryStream topLevelStream = new(converted);
        IReadOnlyList<ChunkSpan> topLevelChunks = ChunkedFileReader.ReadTopLevelChunks(topLevelStream, padOddChunkSizes: false);
        ChunkSpan momoChunk = Assert.Single(topLevelChunks, static chunk => chunk.Header.Id == WmoChunkIds.Momo);
        byte[] momoPayload = ReadChunkPayload(converted, momoChunk);

        using MemoryStream momoStream = new(momoPayload);
        IReadOnlyList<ChunkSpan> momoChunks = ChunkedFileReader.ReadTopLevelChunks(momoStream, padOddChunkSizes: false);
        FourCC[] rootChunkIds = momoChunks
            .TakeWhile(static chunk => chunk.Header.Id != WmoChunkIds.Mogp)
            .Select(static chunk => chunk.Header.Id)
            .ToArray();

        FourCC[] expectedRootChunkIds =
        [
            WmoChunkIds.Mohd,
            WmoChunkIds.Motx,
            WmoChunkIds.Momt,
            WmoChunkIds.Mogn,
            WmoChunkIds.Mogi,
            WmoChunkIds.Mopv,
            WmoChunkIds.Mopt,
            WmoChunkIds.Mopr,
            WmoChunkIds.Molt,
            WmoChunkIds.Mods,
            WmoChunkIds.Modn,
            WmoChunkIds.Modd,
            WmoChunkIds.Mfog,
        ];

        Assert.True(rootChunkIds.SequenceEqual(expectedRootChunkIds));

        ChunkSpan mopvChunk = Assert.Single(momoChunks, static chunk => chunk.Header.Id == WmoChunkIds.Mopv);
        ChunkSpan moptChunk = Assert.Single(momoChunks, static chunk => chunk.Header.Id == WmoChunkIds.Mopt);
        ChunkSpan moprChunk = Assert.Single(momoChunks, static chunk => chunk.Header.Id == WmoChunkIds.Mopr);
        ChunkSpan moltChunk = Assert.Single(momoChunks, static chunk => chunk.Header.Id == WmoChunkIds.Molt);
        ChunkSpan modsChunk = Assert.Single(momoChunks, static chunk => chunk.Header.Id == WmoChunkIds.Mods);
        ChunkSpan modnChunk = Assert.Single(momoChunks, static chunk => chunk.Header.Id == WmoChunkIds.Modn);
        ChunkSpan moddChunk = Assert.Single(momoChunks, static chunk => chunk.Header.Id == WmoChunkIds.Modd);
        ChunkSpan mfogChunk = Assert.Single(momoChunks, static chunk => chunk.Header.Id == WmoChunkIds.Mfog);

        Assert.Equal(0u, mopvChunk.Header.Size);
        Assert.Equal(0u, moptChunk.Header.Size);
        Assert.Equal(0u, moprChunk.Header.Size);
        Assert.Equal(0u, moltChunk.Header.Size);
        Assert.Equal(0u, modsChunk.Header.Size);
        Assert.Equal(0u, modnChunk.Header.Size);
        Assert.Equal(0u, moddChunk.Header.Size);
        Assert.Equal(0u, mfogChunk.Header.Size);
    }

    private static byte[] CreateMohd(uint materialCount, uint groupCount, uint portalCount = 0)
    {
        byte[] bytes = new byte[64];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), materialCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(4, 4), groupCount);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(8, 4), portalCount);
        return bytes;
    }

    private static byte[] CreateMogiEntry(uint flags, Vector3 boundsMin, Vector3 boundsMax, uint nameOffset)
    {
        byte[] bytes = new byte[32];
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(0, 4), flags);
        WriteSingle(bytes, 4, boundsMin.X);
        WriteSingle(bytes, 8, boundsMin.Y);
        WriteSingle(bytes, 12, boundsMin.Z);
        WriteSingle(bytes, 16, boundsMax.X);
        WriteSingle(bytes, 20, boundsMax.Y);
        WriteSingle(bytes, 24, boundsMax.Z);
        BinaryPrimitives.WriteUInt32LittleEndian(bytes.AsSpan(28, 4), nameOffset);
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

    private static byte[] CreateRepeatedVertices(int count, Vector3 value)
    {
        Vector3[] values = Enumerable.Repeat(value, count).ToArray();
        return CreateVertices(values);
    }

    private static byte[] CreateSequentialVertices(int count)
    {
        Vector3[] values = new Vector3[count];
        for (int index = 0; index < count; index++)
            values[index] = new Vector3(index, index % 7, 0f);

        return CreateVertices(values);
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

    private static byte[] CreateMobaPayloadV17(params byte[][] entries)
    {
        using MemoryStream stream = new();
        foreach (byte[] entry in entries)
            stream.Write(entry, 0, entry.Length);

        return stream.ToArray();
    }

    private static byte[] CreateRepeatedMopyEntriesV17(int faceCount, byte flags, byte materialId)
    {
        byte[] bytes = new byte[faceCount * 2];
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            int offset = faceIndex * 2;
            bytes[offset] = flags;
            bytes[offset + 1] = materialId;
        }

        return bytes;
    }

    private static byte[] CreateRepeatedTriangleIndices(int triangleCount, ushort a, ushort b, ushort c)
    {
        byte[] bytes = new byte[triangleCount * 3 * 2];
        for (int triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++)
        {
            int offset = triangleIndex * 6;
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 0, 2), a);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 2, 2), b);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 4, 2), c);
        }

        return bytes;
    }

    private static byte[] CreateSequentialTriangleIndices(int firstVertexA, int triangleCountA, int firstVertexB, int triangleCountB)
    {
        ushort[] values = new ushort[(triangleCountA + triangleCountB) * 3];
        int cursor = 0;
        cursor = WriteSequentialTriangleIndices(values, cursor, firstVertexA, triangleCountA);
        _ = WriteSequentialTriangleIndices(values, cursor, firstVertexB, triangleCountB);
        return CreateIndices(values);
    }

    private static int WriteSequentialTriangleIndices(ushort[] values, int cursor, int firstVertex, int triangleCount)
    {
        for (int triangleIndex = 0; triangleIndex < triangleCount; triangleIndex++)
        {
            int vertex = firstVertex + triangleIndex * 3;
            values[cursor++] = checked((ushort)vertex);
            values[cursor++] = checked((ushort)(vertex + 1));
            values[cursor++] = checked((ushort)(vertex + 2));
        }

        return cursor;
    }

    private static byte[] CreatePortalVertices(params Vector3[] values)
    {
        return CreateVertices(values);
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

    private static byte[] CreatePortalRefs(params (ushort portalIndex, ushort groupIndex, short side)[] values)
    {
        byte[] bytes = new byte[values.Length * 8];
        for (int index = 0; index < values.Length; index++)
        {
            int offset = index * 8;
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset, 2), values[index].portalIndex);
            BinaryPrimitives.WriteUInt16LittleEndian(bytes.AsSpan(offset + 2, 2), values[index].groupIndex);
            BinaryPrimitives.WriteInt16LittleEndian(bytes.AsSpan(offset + 4, 2), values[index].side);
        }

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