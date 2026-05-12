using System.Buffers.Binary;
using System.Numerics;
using WowViewer.Core.Chunks;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoV17ToV14Converter
{
    private const int LegacyMaxGroupCount = 384;

    private static readonly FourCC[] RootChunkOrder =
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
        WmoChunkIds.Mcvp,
    ];

    public static void Convert(string v17RootPath, string outputPath, string? groupsDirectory = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(v17RootPath);
        ArgumentException.ThrowIfNullOrWhiteSpace(outputPath);

        byte[] rootBytes = File.ReadAllBytes(v17RootPath);
        RootChunkPayloads rootPayloads = ParseRootPayloads(rootBytes, v17RootPath);

        string groupDirectory = groupsDirectory ?? Path.GetDirectoryName(v17RootPath) ?? ".";
        string baseName = Path.GetFileNameWithoutExtension(v17RootPath);
        List<byte[]> groupBytes = new(rootPayloads.ReportedGroupCount);
        for (int groupIndex = 0; groupIndex < rootPayloads.ReportedGroupCount; groupIndex++)
        {
            string groupPath = Path.Combine(groupDirectory, $"{baseName}_{groupIndex:D3}.wmo");
            if (!File.Exists(groupPath))
                throw new FileNotFoundException($"Expected WMO group file '{groupPath}' was not found.", groupPath);

            groupBytes.Add(File.ReadAllBytes(groupPath));
        }

        byte[] converted = Convert(rootBytes, groupBytes, v17RootPath);
        Directory.CreateDirectory(Path.GetDirectoryName(outputPath) ?? ".");
        File.WriteAllBytes(outputPath, converted);
    }

    public static byte[] Convert(byte[] v17RootBytes, IReadOnlyList<byte[]> groupBytes, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(v17RootBytes);
        ArgumentNullException.ThrowIfNull(groupBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        RootChunkPayloads rootPayloads = ParseRootPayloads(v17RootBytes, sourcePath);
        if (rootPayloads.ReportedGroupCount != groupBytes.Count)
        {
            throw new InvalidDataException(
                $"WMO root reports {rootPayloads.ReportedGroupCount} groups, but {groupBytes.Count} group payloads were supplied.");
        }

        List<LegacyGroupDocument> sourceLegacyGroups = new(groupBytes.Count);
        for (int groupIndex = 0; groupIndex < groupBytes.Count; groupIndex++)
        {
            string groupSourcePath = $"{sourcePath}#{groupIndex:D3}";
            if (TryCreateLegacySeedFromOversizedV17Group(groupBytes[groupIndex], groupSourcePath, out LegacyGroupDocument? oversizedSeed))
            {
                sourceLegacyGroups.Add(oversizedSeed);
                continue;
            }

            byte[] legacyGroupPayload = ConvertGroupPayload(groupBytes[groupIndex], groupSourcePath);
            sourceLegacyGroups.Add(ParseLegacyGroup(legacyGroupPayload, $"{sourcePath}#legacy[{groupIndex:D3}]"));
        }

        List<int> sourceToExpandedFirstIndex = new(sourceLegacyGroups.Count);
        List<LegacyGroupDocument> legacyGroups = new();
        bool splitAnyGroups = false;
        foreach (LegacyGroupDocument group in sourceLegacyGroups)
        {
            sourceToExpandedFirstIndex.Add(legacyGroups.Count);
            IReadOnlyList<LegacyGroupDocument> splitGroups = SplitLegacyGroupForLegacyBatchIndexLimit(group);
            splitAnyGroups |= splitGroups.Count > 1;
            legacyGroups.AddRange(splitGroups);
        }

        LegacyPortalLayout portalLayout = splitAnyGroups
            ? ExpandPortalLayout(sourceLegacyGroups, legacyGroups, sourceToExpandedFirstIndex, rootPayloads.PayloadsById)
            : BuildDefaultPortalLayout(rootPayloads.PayloadsById);

        if (portalLayout.UpdatedGroups is not null)
            legacyGroups = portalLayout.UpdatedGroups.ToList();

        IReadOnlyDictionary<FourCC, byte[]> effectiveRootPayloads = ApplyPortalLayout(rootPayloads.PayloadsById, portalLayout);
        if (legacyGroups.Count > LegacyMaxGroupCount)
        {
            MergeOverflowResult mergeResult = MergeOverflowGroups(legacyGroups, effectiveRootPayloads);
            portalLayout = mergeResult.PortalLayout;
            legacyGroups = (portalLayout.UpdatedGroups ?? mergeResult.Groups).ToList();
        }

        return BuildV14Root(rootPayloads.PayloadsById, legacyGroups, portalLayout);
    }

    private static RootChunkPayloads ParseRootPayloads(byte[] rootBytes, string sourcePath)
    {
        using MemoryStream stream = new(rootBytes, writable: false);
        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        if (version != 17)
            throw new InvalidDataException($"WMO root version '{version?.ToString() ?? "unknown"}' is not supported. Expected 17.");

        byte[] mohdPayload = WmoRootReaderCommon.ReadRequiredChunkPayload(stream, chunks, WmoChunkIds.Mohd);
        if (mohdPayload.Length < 64)
            throw new InvalidDataException($"MOHD payload is too short ({mohdPayload.Length} bytes). Expected at least 64 bytes.");

        Dictionary<FourCC, byte[]> payloadsById = [];
        payloadsById[WmoChunkIds.Mohd] = mohdPayload.ToArray();

        foreach (FourCC chunkId in RootChunkOrder.Where(static id => id != WmoChunkIds.Mohd))
        {
            byte[]? payload = WmoRootReaderCommon.TryReadChunkPayload(stream, chunks, chunkId);
            if (payload is null || payload.Length == 0)
                continue;

            payloadsById[chunkId] = chunkId == WmoChunkIds.Momt
                ? DownconvertMomtPayload(payload)
                : payload;
        }

        int reportedGroupCount = checked((int)BinaryPrimitives.ReadUInt32LittleEndian(mohdPayload.AsSpan(4, 4)));
        return new RootChunkPayloads(reportedGroupCount, payloadsById);
    }

    private static byte[] ConvertGroupPayload(byte[] groupBytes, string sourcePath)
    {
        using MemoryStream stream = new(groupBytes, writable: false);
        (uint? version, byte[] mogpPayload) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        if (version is null || version < 16 || version > 17)
            throw new InvalidDataException($"WMO group version '{version?.ToString() ?? "unknown"}' is not supported. Expected 16 or 17.");

        if (mogpPayload.Length < 0x3C)
            throw new InvalidDataException($"MOGP payload is too short ({mogpPayload.Length} bytes). Expected at least 60 bytes.");

        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogpPayload);

        byte[]? mopyPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Mopy);
        byte[]? indexPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Movi)
            ?? WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Moin);
        byte[]? movtPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Movt);
        byte[]? monrPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Monr);
        byte[]? motvPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Motv);
        byte[]? mobaPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Moba);

        if (mopyPayload is null || indexPayload is null || movtPayload is null || monrPayload is null || mobaPayload is null)
            throw new InvalidDataException("WMO group is missing one or more required subchunks for v17->v14 conversion.");

        byte[] legacyHeader = new byte[0x44];
        mogpPayload.AsSpan(0, Math.Min(0x3C, mogpPayload.Length)).CopyTo(legacyHeader);

        using MemoryStream payloadStream = new();
        using BinaryWriter writer = new(payloadStream);
        writer.Write(legacyHeader);
        WriteChunk(writer, WmoChunkIds.Mopy, DownconvertMopyPayload(mopyPayload, version.Value));
        WriteChunk(writer, WmoChunkIds.Movi, indexPayload);
        WriteChunk(writer, WmoChunkIds.Movt, movtPayload);
        WriteChunk(writer, WmoChunkIds.Monr, monrPayload);
        if (motvPayload is not null)
            WriteChunk(writer, WmoChunkIds.Motv, motvPayload);
        WriteChunk(writer, WmoChunkIds.Moba, DownconvertMobaPayload(mobaPayload, version.Value));

        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Molr);
        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Modr);
        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Mobn);
        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Mobr);
        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Mocv);
        WriteOptionalGroupChunk(writer, mogpPayload, headerSizeBytes, WmoChunkIds.Mliq);

        return payloadStream.ToArray();
    }

    private static void WriteOptionalGroupChunk(BinaryWriter writer, byte[] mogpPayload, int headerSizeBytes, FourCC chunkId)
    {
        byte[]? payload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, chunkId);
        if (payload is { Length: > 0 })
            WriteChunk(writer, chunkId, payload);
    }

    private static byte[] DownconvertMomtPayload(byte[] payload)
    {
        const int sourceEntrySize = 64;
        const int targetEntrySize = 48;

        if (payload.Length == 0 || payload.Length % sourceEntrySize != 0)
            return payload;

        int entryCount = payload.Length / sourceEntrySize;
        byte[] converted = new byte[entryCount * targetEntrySize];
        for (int entryIndex = 0; entryIndex < entryCount; entryIndex++)
        {
            payload.AsSpan(entryIndex * sourceEntrySize, targetEntrySize)
                .CopyTo(converted.AsSpan(entryIndex * targetEntrySize, targetEntrySize));
        }

        return converted;
    }

    private static byte[] DownconvertMopyPayload(byte[] payload, uint version)
    {
        if (payload.Length == 0 || version <= 16)
            return payload;

        if (payload.Length % 2 != 0)
            throw new InvalidDataException($"MOPY payload size {payload.Length} is not divisible by 2.");

        int faceCount = payload.Length / 2;
        byte[] converted = new byte[faceCount * 4];
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
        {
            int sourceOffset = faceIndex * 2;
            int targetOffset = faceIndex * 4;
            converted[targetOffset] = payload[sourceOffset];
            converted[targetOffset + 1] = payload[sourceOffset + 1];
        }

        return converted;
    }

    private static byte[] DownconvertMobaPayload(byte[] payload, uint version)
    {
        const int batchEntrySize = 24;

        if (payload.Length == 0 || version <= 16)
            return payload;

        if (payload.Length % batchEntrySize != 0)
            throw new InvalidDataException($"MOBA payload size {payload.Length} is not divisible by {batchEntrySize}.");

        int batchCount = payload.Length / batchEntrySize;
        byte[] converted = new byte[payload.Length];
        for (int batchIndex = 0; batchIndex < batchCount; batchIndex++)
        {
            int sourceOffset = batchIndex * batchEntrySize;
            int targetOffset = batchIndex * batchEntrySize;

            uint startIndex = BinaryPrimitives.ReadUInt32LittleEndian(payload.AsSpan(sourceOffset + 12, 4));
            if (startIndex > ushort.MaxValue)
            {
                throw new InvalidDataException(
                    $"MOBA batch {batchIndex} firstIndex {startIndex} exceeds the legacy ushort range and requires group splitting.");
            }

            ushort indexCount = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(sourceOffset + 16, 2));
            ushort startVertex = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(sourceOffset + 18, 2));
            ushort endVertex = BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(sourceOffset + 20, 2));
            byte flags = payload[sourceOffset + 22];
            byte materialId = payload[sourceOffset + 23];

            converted[targetOffset] = 0;
            converted[targetOffset + 1] = materialId;
            payload.AsSpan(sourceOffset, 12).CopyTo(converted.AsSpan(targetOffset + 2, 12));
            BinaryPrimitives.WriteUInt16LittleEndian(converted.AsSpan(targetOffset + 14, 2), checked((ushort)startIndex));
            BinaryPrimitives.WriteUInt16LittleEndian(converted.AsSpan(targetOffset + 16, 2), indexCount);
            BinaryPrimitives.WriteUInt16LittleEndian(converted.AsSpan(targetOffset + 18, 2), startVertex);
            BinaryPrimitives.WriteUInt16LittleEndian(converted.AsSpan(targetOffset + 20, 2), endVertex);
            converted[targetOffset + 22] = flags;
            converted[targetOffset + 23] = 0;
        }

        return converted;
    }

    private static bool TryCreateLegacySeedFromOversizedV17Group(byte[] groupBytes, string sourcePath, out LegacyGroupDocument legacyGroup)
    {
        ArgumentNullException.ThrowIfNull(groupBytes);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        using MemoryStream stream = new(groupBytes, writable: false);
        (uint? version, byte[] mogpPayload) = WmoGroupReaderCommon.ReadGroupPayload(stream, sourcePath);
        if (version is null || version <= 16)
        {
            legacyGroup = null!;
            return false;
        }

        int headerSizeBytes = WmoGroupReaderCommon.FindHeaderSize(mogpPayload);
        byte[]? mobaPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, WmoChunkIds.Moba);
        if (!RequiresLegacyGroupSplit(mobaPayload))
        {
            legacyGroup = null!;
            return false;
        }

        WmoGroupSummary sourceSummary = WmoGroupSummaryReader.ReadMogpPayload(mogpPayload, sourcePath, version);
        WmoGroupMeshDetail sourceMesh = WmoGroupMeshDetailReader.ReadMogpPayload(mogpPayload, sourcePath, version);
        List<WmoGroupFaceMaterialDetail> legacyFaceMaterials = new(sourceMesh.FaceMaterials.Count);
        foreach (WmoGroupFaceMaterialDetail face in sourceMesh.FaceMaterials)
            legacyFaceMaterials.Add(new WmoGroupFaceMaterialDetail(legacyFaceMaterials.Count, face.Flags, face.MaterialId, face.LegacyExtraValue ?? 0));

        List<ushort> doodadRefs = ReadRefs(mogpPayload, sourceMesh.HeaderSizeBytes, WmoChunkIds.Modr);
        List<ushort> lightRefs = ReadRefs(mogpPayload, sourceMesh.HeaderSizeBytes, WmoChunkIds.Molr);
        byte[]? liquidPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, sourceMesh.HeaderSizeBytes, WmoChunkIds.Mliq);
        bool hasLiquid = liquidPayload is { Length: > 0 };

        uint flags = NormalizeFlags(
            sourceSummary.Flags,
            legacyFaceMaterials.Count,
            sourceMesh.PrimaryUvs.Count,
            sourceMesh.AdditionalUvSets.Count,
            sourceMesh.PrimaryVertexColorsBgra.Count,
            sourceMesh.AdditionalVertexColorSetsBgra.Count,
            doodadRefs.Count,
            lightRefs.Count,
            hasLiquid);

        WmoGroupSummary legacySummary = new(
            sourceSummary.SourcePath,
            14,
            0x44,
            sourceSummary.NameOffset,
            sourceSummary.DescriptiveNameOffset,
            flags,
            sourceSummary.BoundsMin,
            sourceSummary.BoundsMax,
            sourceSummary.PortalStart,
            sourceSummary.PortalCount,
            sourceSummary.TransparentBatchCount,
            sourceSummary.InteriorBatchCount,
            sourceSummary.ExteriorBatchCount,
            hasLiquid ? sourceSummary.GroupLiquid : 0,
            legacyFaceMaterials.Count,
            sourceMesh.Vertices.Count,
            sourceMesh.Indices.Count,
            sourceMesh.Normals.Count,
            sourceMesh.PrimaryUvs.Count,
            sourceMesh.AdditionalUvSets.Count,
            sourceMesh.Batches.Count,
            sourceMesh.PrimaryVertexColorsBgra.Count,
            doodadRefs.Count,
            lightRefs.Count,
            sourceMesh.Indices.Count > 0 ? 1 : 0,
            sourceMesh.Indices.Count / 3,
            hasLiquid);

        WmoGroupMeshDetail legacyMesh = new(
            sourceMesh.SourcePath,
            14,
            0x44,
            WmoChunkIds.Movi.ToString(),
            sourceMesh.Vertices,
            sourceMesh.Normals,
            sourceMesh.Indices,
            sourceMesh.PrimaryUvs,
            sourceMesh.AdditionalUvSets,
            sourceMesh.PrimaryVertexColorsBgra,
            sourceMesh.AdditionalVertexColorSetsBgra,
            legacyFaceMaterials,
            sourceMesh.Batches);

        legacyGroup = new LegacyGroupDocument(
            legacySummary,
            legacyMesh,
            doodadRefs,
            lightRefs,
            liquidPayload,
            sourceSummary.TransparentBatchCount,
            sourceSummary.InteriorBatchCount,
            sourceSummary.ExteriorBatchCount);
        return true;
    }

    private static bool RequiresLegacyGroupSplit(byte[]? mobaPayload)
    {
        if (mobaPayload is null || mobaPayload.Length == 0)
            return false;

        const int batchEntrySize = 24;
        if (mobaPayload.Length % batchEntrySize != 0)
            throw new InvalidDataException($"MOBA payload size {mobaPayload.Length} is not divisible by {batchEntrySize}.");

        for (int batchOffset = 0; batchOffset < mobaPayload.Length; batchOffset += batchEntrySize)
        {
            uint firstIndex = BinaryPrimitives.ReadUInt32LittleEndian(mobaPayload.AsSpan(batchOffset + 12, 4));
            if (firstIndex > ushort.MaxValue)
                return true;
        }

        return false;
    }

    private static IReadOnlyList<LegacyGroupDocument> SplitLegacyGroupForLegacyBatchIndexLimit(LegacyGroupDocument group)
    {
        if (group.Mesh.Batches.Count == 0)
            return [group];

        int maxBatchEnd = group.Mesh.Batches.Max(static batch => batch.FirstIndex + batch.IndexCount);
        if (maxBatchEnd <= ushort.MaxValue)
            return [group];

        List<List<int>> partitions = PartitionLegacyGroupBatches(group.Mesh.Batches);
        if (partitions.Count == 1)
            return [group];

        List<LegacyGroupDocument> splitGroups = new(partitions.Count);
        for (int partitionIndex = 0; partitionIndex < partitions.Count; partitionIndex++)
            splitGroups.Add(CreateSplitLegacyGroup(group, partitions[partitionIndex], partitionIndex == 0));

        return splitGroups;
    }

    private static List<List<int>> PartitionLegacyGroupBatches(IReadOnlyList<WmoGroupBatchDetail> batches)
    {
        List<List<int>> partitions = [];
        List<int> currentPartition = [];
        int currentIndexCount = 0;

        for (int batchIndex = 0; batchIndex < batches.Count; batchIndex++)
        {
            WmoGroupBatchDetail batch = batches[batchIndex];
            if (currentPartition.Count > 0 && currentIndexCount + batch.IndexCount > ushort.MaxValue)
            {
                partitions.Add(currentPartition);
                currentPartition = [];
                currentIndexCount = 0;
            }

            currentPartition.Add(batchIndex);
            currentIndexCount += batch.IndexCount;
        }

        if (currentPartition.Count > 0)
            partitions.Add(currentPartition);

        return partitions;
    }

    private static LegacyGroupDocument CreateSplitLegacyGroup(LegacyGroupDocument group, IReadOnlyList<int> batchOrdinals, bool isPrimarySplit)
    {
        List<ushort> indices = [];
        List<WmoGroupFaceMaterialDetail> faceMaterials = [];
        List<WmoGroupBatchDetail> batches = [];
        int transparentBatchCount = 0;
        int interiorBatchCount = 0;
        int exteriorBatchCount = 0;
        int firstIndex = 0;

        foreach (int batchOrdinal in batchOrdinals)
        {
            WmoGroupBatchDetail batch = group.Mesh.Batches[batchOrdinal];
            if (batch.IndexCount % 3 != 0)
                throw new InvalidDataException($"Legacy batch {batchOrdinal} indexCount {batch.IndexCount} is not divisible by 3.");

            if (batch.FirstIndex % 3 != 0)
                throw new InvalidDataException($"Legacy batch {batchOrdinal} firstIndex {batch.FirstIndex} is not aligned to triangle boundaries.");

            if (batch.FirstIndex + batch.IndexCount > group.Mesh.Indices.Count)
                throw new InvalidDataException($"Legacy batch {batchOrdinal} overruns the group's index buffer.");

            int faceStart = batch.FirstIndex / 3;
            int faceCount = batch.IndexCount / 3;
            if (faceStart + faceCount > group.Mesh.FaceMaterials.Count)
                throw new InvalidDataException($"Legacy batch {batchOrdinal} overruns the group's face-material buffer.");

            for (int index = 0; index < batch.IndexCount; index++)
                indices.Add(group.Mesh.Indices[batch.FirstIndex + index]);

            for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
            {
                WmoGroupFaceMaterialDetail face = group.Mesh.FaceMaterials[faceStart + faceIndex];
                faceMaterials.Add(new WmoGroupFaceMaterialDetail(faceMaterials.Count, face.Flags, face.MaterialId, face.LegacyExtraValue));
            }

            byte[] rawEntryBytes = CreateLegacyBatchEntry(batch, firstIndex, group.Mesh.Version);
            batches.Add(new WmoGroupBatchDetail(
                batches.Count,
                batches.Count * 24,
                batch.MaterialIdRaw,
                batch.HasMaterialId,
                firstIndex,
                batch.IndexCount,
                batch.Flags,
                rawEntryBytes));

            CountBatchRegion(group, batchOrdinal, ref transparentBatchCount, ref interiorBatchCount, ref exteriorBatchCount);
            firstIndex += batch.IndexCount;
        }

        (Vector3 boundsMin, Vector3 boundsMax) = ComputeBoundsForIndices(group.Mesh.Vertices, indices, group.Summary.BoundsMin, group.Summary.BoundsMax);
        IReadOnlyList<ushort> doodadRefs = isPrimarySplit ? group.DoodadRefs : Array.Empty<ushort>();
        IReadOnlyList<ushort> lightRefs = isPrimarySplit ? group.LightRefs : Array.Empty<ushort>();
        byte[]? liquidPayload = isPrimarySplit ? group.LiquidPayload : null;
        bool hasLiquid = liquidPayload is { Length: > 0 };

        uint flags = NormalizeFlags(
            group.Summary.Flags,
            faceMaterials.Count,
            group.Mesh.PrimaryUvs.Count,
            group.Mesh.AdditionalUvSets.Count,
            group.Mesh.PrimaryVertexColorsBgra.Count,
            group.Mesh.AdditionalVertexColorSetsBgra.Count,
            doodadRefs.Count,
            lightRefs.Count,
            hasLiquid);

        WmoGroupSummary summary = new(
            group.Summary.SourcePath,
            group.Summary.Version,
            group.Summary.HeaderSizeBytes,
            group.Summary.NameOffset,
            group.Summary.DescriptiveNameOffset,
            flags,
            boundsMin,
            boundsMax,
            isPrimarySplit ? group.Summary.PortalStart : 0,
            isPrimarySplit ? group.Summary.PortalCount : 0,
            transparentBatchCount,
            interiorBatchCount,
            exteriorBatchCount,
            hasLiquid ? group.Summary.GroupLiquid : 0,
            faceMaterials.Count,
            group.Mesh.Vertices.Count,
            indices.Count,
            group.Mesh.Normals.Count,
            group.Mesh.PrimaryUvs.Count,
            group.Mesh.AdditionalUvSets.Count,
            batches.Count,
            group.Mesh.PrimaryVertexColorsBgra.Count,
            doodadRefs.Count,
            lightRefs.Count,
            indices.Count > 0 ? 1 : 0,
            indices.Count / 3,
            hasLiquid);

        WmoGroupMeshDetail mesh = new(
            group.Mesh.SourcePath,
            group.Mesh.Version,
            group.Mesh.HeaderSizeBytes,
            group.Mesh.IndexChunkId,
            group.Mesh.Vertices,
            group.Mesh.Normals,
            indices,
            group.Mesh.PrimaryUvs,
            group.Mesh.AdditionalUvSets,
            group.Mesh.PrimaryVertexColorsBgra,
            group.Mesh.AdditionalVertexColorSetsBgra,
            faceMaterials,
            batches);

        return new LegacyGroupDocument(
            summary,
            mesh,
            doodadRefs.ToList(),
            lightRefs.ToList(),
            liquidPayload,
            transparentBatchCount,
            interiorBatchCount,
            exteriorBatchCount);
    }

    private static void CountBatchRegion(LegacyGroupDocument group, int batchOrdinal, ref int transparentBatchCount, ref int interiorBatchCount, ref int exteriorBatchCount)
    {
        if (batchOrdinal < group.TransparentBatchCount)
        {
            transparentBatchCount++;
            return;
        }

        if (batchOrdinal < group.TransparentBatchCount + group.InteriorBatchCount)
        {
            interiorBatchCount++;
            return;
        }

        exteriorBatchCount++;
    }

    private static (Vector3 BoundsMin, Vector3 BoundsMax) ComputeBoundsForIndices(
        IReadOnlyList<Vector3> vertices,
        IReadOnlyList<ushort> indices,
        Vector3 fallbackMin,
        Vector3 fallbackMax)
    {
        if (indices.Count == 0)
            return (fallbackMin, fallbackMax);

        Vector3 boundsMin = vertices[indices[0]];
        Vector3 boundsMax = boundsMin;
        foreach (ushort vertexIndex in indices)
        {
            if (vertexIndex >= vertices.Count)
                throw new InvalidDataException($"Group index {vertexIndex} exceeds the available vertex count {vertices.Count}.");

            Vector3 vertex = vertices[vertexIndex];
            boundsMin = Vector3.Min(boundsMin, vertex);
            boundsMax = Vector3.Max(boundsMax, vertex);
        }

        return (boundsMin, boundsMax);
    }

    private static byte[] CreateLegacyBatchEntry(WmoGroupBatchDetail batch, int firstIndex, uint? sourceVersion)
    {
        byte[] entry = new byte[24];
        int boundsOffset = sourceVersion > 16 ? 0 : 2;
        batch.RawEntryBytes.AsSpan(boundsOffset, 12).CopyTo(entry.AsSpan(2, 12));
        entry[0] = 0;
        entry[1] = batch.MaterialIdRaw;
        BinaryPrimitives.WriteUInt16LittleEndian(entry.AsSpan(14, 2), checked((ushort)firstIndex));
        BinaryPrimitives.WriteUInt16LittleEndian(entry.AsSpan(16, 2), batch.IndexCount);
        BinaryPrimitives.WriteUInt16LittleEndian(entry.AsSpan(18, 2), BinaryPrimitives.ReadUInt16LittleEndian(batch.RawEntryBytes.AsSpan(18, 2)));
        BinaryPrimitives.WriteUInt16LittleEndian(entry.AsSpan(20, 2), BinaryPrimitives.ReadUInt16LittleEndian(batch.RawEntryBytes.AsSpan(20, 2)));
        entry[22] = batch.Flags;
        entry[23] = 0;
        return entry;
    }

    private static LegacyPortalLayout ExpandPortalLayout(
        IReadOnlyList<LegacyGroupDocument> sourceGroups,
        IReadOnlyList<LegacyGroupDocument> expandedGroups,
        IReadOnlyList<int> sourceToExpandedFirstIndex,
        IReadOnlyDictionary<FourCC, byte[]> rootPayloads)
    {
        if (!TryGetPortalLayout(rootPayloads, out int portalCount, out byte[]? moprPayload) || moprPayload is null)
            return new LegacyPortalLayout(false, 0, null, expandedGroups);

        List<RootPortalReference> sourceRefs = ReadRootPortalReferences(moprPayload);
        List<List<RootPortalReference>> expandedRefs = Enumerable.Range(0, expandedGroups.Count).Select(static _ => new List<RootPortalReference>()).ToList();
        for (int sourceGroupIndex = 0; sourceGroupIndex < sourceGroups.Count; sourceGroupIndex++)
        {
            LegacyGroupDocument sourceGroup = sourceGroups[sourceGroupIndex];
            if (sourceGroup.Summary.PortalCount == 0)
                continue;

            int start = sourceGroup.Summary.PortalStart;
            int count = sourceGroup.Summary.PortalCount;
            if (start < 0 || count < 0 || start + count > sourceRefs.Count)
                return new LegacyPortalLayout(false, 0, null, expandedGroups);

            int expandedGroupIndex = sourceToExpandedFirstIndex[sourceGroupIndex];
            for (int portalRefIndex = 0; portalRefIndex < count; portalRefIndex++)
            {
                RootPortalReference sourceRef = sourceRefs[start + portalRefIndex];
                expandedRefs[expandedGroupIndex].Add(sourceRef with { GroupIndex = checked((ushort)expandedGroupIndex) });
            }
        }

        int cursor = 0;
        List<LegacyGroupDocument> updatedGroups = new(expandedGroups.Count);
        List<RootPortalReference> remappedRefs = new();
        for (int sourceGroupIndex = 0; sourceGroupIndex < sourceGroups.Count; sourceGroupIndex++)
        {
            int expandedStart = sourceToExpandedFirstIndex[sourceGroupIndex];
            int expandedEnd = sourceGroupIndex + 1 < sourceToExpandedFirstIndex.Count
                ? sourceToExpandedFirstIndex[sourceGroupIndex + 1]
                : expandedGroups.Count;

            for (int expandedIndex = expandedStart; expandedIndex < expandedEnd; expandedIndex++)
            {
                List<RootPortalReference> refs = expandedIndex == expandedStart ? expandedRefs[expandedIndex] : [];
                updatedGroups.Add(UpdatePortalRange(expandedGroups[expandedIndex], cursor, refs.Count));
                if (expandedIndex == expandedStart)
                {
                    remappedRefs.AddRange(refs);
                    cursor += refs.Count;
                }
            }
        }

        byte[] remappedPayload = WriteRootPortalReferences(remappedRefs);
        return new LegacyPortalLayout(true, portalCount, remappedPayload, updatedGroups);
    }

    private static IReadOnlyDictionary<FourCC, byte[]> ApplyPortalLayout(
        IReadOnlyDictionary<FourCC, byte[]> rootPayloads,
        LegacyPortalLayout portalLayout)
    {
        if (!portalLayout.KeepPortalChunks || portalLayout.MoprPayload is null)
            return rootPayloads;

        Dictionary<FourCC, byte[]> updated = rootPayloads.ToDictionary(static pair => pair.Key, static pair => pair.Value.ToArray());
        updated[WmoChunkIds.Mopr] = portalLayout.MoprPayload;
        return updated;
    }

    private static MergeOverflowResult MergeOverflowGroups(IReadOnlyList<LegacyGroupDocument> sourceGroups, IReadOnlyDictionary<FourCC, byte[]> rootPayloads)
    {
        List<List<int>> buckets = CreateSpatialBuckets(sourceGroups, LegacyMaxGroupCount);
        List<LegacyGroupDocument> mergedGroups = new(buckets.Count);
        Dictionary<int, int> sourceToBucket = new(sourceGroups.Count);
        for (int bucketIndex = 0; bucketIndex < buckets.Count; bucketIndex++)
        {
            List<int> bucket = buckets[bucketIndex];
            foreach (int sourceGroupIndex in bucket)
                sourceToBucket[sourceGroupIndex] = bucketIndex;

            mergedGroups.Add(bucket.Count == 1
                ? sourceGroups[bucket[0]]
                : MergeLegacyGroups(bucket.Select(index => sourceGroups[index]).ToList()));
        }

        LegacyPortalLayout portalLayout = RebuildPortalLayout(sourceGroups, mergedGroups, sourceToBucket, rootPayloads);
        return new MergeOverflowResult(mergedGroups, portalLayout);
    }

    private static List<List<int>> CreateSpatialBuckets(IReadOnlyList<LegacyGroupDocument> groups, int targetBucketCount)
    {
        List<List<int>> buckets = [Enumerable.Range(0, groups.Count).ToList()];
        while (buckets.Count < targetBucketCount)
        {
            int bucketIndex = SelectBucketToSplit(buckets, groups);
            if (bucketIndex < 0)
                break;

            SplitBucket(buckets, bucketIndex, groups);
        }

        return buckets;
    }

    private static int SelectBucketToSplit(IReadOnlyList<List<int>> buckets, IReadOnlyList<LegacyGroupDocument> groups)
    {
        int selectedIndex = -1;
        float selectedScore = float.MinValue;
        for (int bucketIndex = 0; bucketIndex < buckets.Count; bucketIndex++)
        {
            List<int> bucket = buckets[bucketIndex];
            if (bucket.Count <= 1)
                continue;

            float score = ComputeBucketSplitScore(bucket, groups);
            if (score <= selectedScore)
                continue;

            selectedScore = score;
            selectedIndex = bucketIndex;
        }

        return selectedIndex;
    }

    private static float ComputeBucketSplitScore(IReadOnlyList<int> bucket, IReadOnlyList<LegacyGroupDocument> groups)
    {
        Vector3 min = GetGroupCenter(groups[bucket[0]]);
        Vector3 max = min;
        for (int index = 1; index < bucket.Count; index++)
        {
            Vector3 center = GetGroupCenter(groups[bucket[index]]);
            min = Vector3.Min(min, center);
            max = Vector3.Max(max, center);
        }

        Vector3 extents = max - min;
        return Math.Max(extents.X, Math.Max(extents.Y, extents.Z)) * bucket.Count;
    }

    private static void SplitBucket(List<List<int>> buckets, int bucketIndex, IReadOnlyList<LegacyGroupDocument> groups)
    {
        List<int> bucket = buckets[bucketIndex];
        int axis = GetLargestExtentAxis(bucket, groups);
        bucket.Sort((left, right) => GetAxisValue(GetGroupCenter(groups[left]), axis).CompareTo(GetAxisValue(GetGroupCenter(groups[right]), axis)));

        int midpoint = bucket.Count / 2;
        if (midpoint <= 0 || midpoint >= bucket.Count)
            return;

        List<int> lower = bucket.GetRange(0, midpoint);
        List<int> upper = bucket.GetRange(midpoint, bucket.Count - midpoint);
        buckets[bucketIndex] = lower;
        buckets.Insert(bucketIndex + 1, upper);
    }

    private static int GetLargestExtentAxis(IReadOnlyList<int> bucket, IReadOnlyList<LegacyGroupDocument> groups)
    {
        Vector3 min = GetGroupCenter(groups[bucket[0]]);
        Vector3 max = min;
        for (int index = 1; index < bucket.Count; index++)
        {
            Vector3 center = GetGroupCenter(groups[bucket[index]]);
            min = Vector3.Min(min, center);
            max = Vector3.Max(max, center);
        }

        Vector3 extents = max - min;
        if (extents.Y >= extents.X && extents.Y >= extents.Z)
            return 1;

        if (extents.Z >= extents.X && extents.Z >= extents.Y)
            return 2;

        return 0;
    }

    private static Vector3 GetGroupCenter(LegacyGroupDocument group)
    {
        return (group.Summary.BoundsMin + group.Summary.BoundsMax) * 0.5f;
    }

    private static float GetAxisValue(Vector3 value, int axis)
    {
        return axis switch
        {
            1 => value.Y,
            2 => value.Z,
            _ => value.X,
        };
    }

    private static LegacyGroupDocument ParseLegacyGroup(byte[] mogpPayload, string sourcePath)
    {
        WmoGroupSummary summary = WmoGroupSummaryReader.ReadMogpPayload(mogpPayload, sourcePath, 14);
        WmoGroupMeshDetail mesh = WmoGroupMeshDetailReader.ReadMogpPayload(mogpPayload, sourcePath, 14);
        List<ushort> doodadRefs = ReadRefs(mogpPayload, mesh.HeaderSizeBytes, WmoChunkIds.Modr);
        List<ushort> lightRefs = ReadRefs(mogpPayload, mesh.HeaderSizeBytes, WmoChunkIds.Molr);
        byte[]? liquidPayload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, mesh.HeaderSizeBytes, WmoChunkIds.Mliq);
        return new LegacyGroupDocument(summary, mesh, doodadRefs, lightRefs, liquidPayload, summary.TransparentBatchCount, summary.InteriorBatchCount, summary.ExteriorBatchCount);
    }

    private static LegacyPortalLayout BuildDefaultPortalLayout(IReadOnlyDictionary<FourCC, byte[]> rootPayloads)
    {
        if (!TryGetPortalLayout(rootPayloads, out int portalCount, out byte[]? moprPayload))
            return new LegacyPortalLayout(false, 0, null);

        return new LegacyPortalLayout(true, portalCount, moprPayload);
    }

    private static LegacyPortalLayout RebuildPortalLayout(
        IReadOnlyList<LegacyGroupDocument> sourceGroups,
        IReadOnlyList<LegacyGroupDocument> mergedGroups,
        IReadOnlyDictionary<int, int> sourceToBucket,
        IReadOnlyDictionary<FourCC, byte[]> rootPayloads)
    {
        if (!TryGetPortalLayout(rootPayloads, out int portalCount, out byte[]? moprPayload) || moprPayload is null)
        {
            return new LegacyPortalLayout(false, 0, null);
        }

        List<RootPortalReference> sourceRefs = ReadRootPortalReferences(moprPayload);
        List<List<RootPortalReference>> bucketRefs = Enumerable.Range(0, mergedGroups.Count).Select(static _ => new List<RootPortalReference>()).ToList();
        for (int sourceGroupIndex = 0; sourceGroupIndex < sourceGroups.Count; sourceGroupIndex++)
        {
            WmoGroupSummary summary = sourceGroups[sourceGroupIndex].Summary;
            if (summary.PortalCount == 0)
                continue;

            int start = summary.PortalStart;
            int count = summary.PortalCount;
            if (start < 0 || count < 0 || start + count > sourceRefs.Count)
                return new LegacyPortalLayout(false, 0, null);

            int bucketIndex = sourceToBucket[sourceGroupIndex];
            for (int portalRefIndex = 0; portalRefIndex < count; portalRefIndex++)
            {
                RootPortalReference sourceRef = sourceRefs[start + portalRefIndex];
                bucketRefs[bucketIndex].Add(sourceRef with { GroupIndex = checked((ushort)bucketIndex) });
            }
        }

        int cursor = 0;
        List<LegacyGroupDocument> updatedGroups = new(mergedGroups.Count);
        List<RootPortalReference> remappedRefs = new();
        for (int bucketIndex = 0; bucketIndex < mergedGroups.Count; bucketIndex++)
        {
            List<RootPortalReference> refs = bucketRefs[bucketIndex];
            updatedGroups.Add(UpdatePortalRange(mergedGroups[bucketIndex], cursor, refs.Count));
            remappedRefs.AddRange(refs);
            cursor += refs.Count;
        }

        byte[] remappedPayload = WriteRootPortalReferences(remappedRefs);
        return new LegacyPortalLayout(true, portalCount, remappedPayload, updatedGroups);
    }

    private static bool TryGetPortalLayout(IReadOnlyDictionary<FourCC, byte[]> rootPayloads, out int portalCount, out byte[]? moprPayload)
    {
        portalCount = 0;
        moprPayload = null;
        if (!rootPayloads.TryGetValue(WmoChunkIds.Mopv, out byte[]? mopvPayload)
            || !rootPayloads.TryGetValue(WmoChunkIds.Mopt, out byte[]? moptPayload)
            || !rootPayloads.TryGetValue(WmoChunkIds.Mopr, out byte[]? moprRawPayload))
        {
            return false;
        }

        if (moptPayload.Length % 20 != 0 || moprRawPayload.Length % 8 != 0 || mopvPayload.Length % 12 != 0)
            return false;

        portalCount = moptPayload.Length / 20;
        moprPayload = moprRawPayload.ToArray();
        return true;
    }

    private static List<RootPortalReference> ReadRootPortalReferences(byte[] payload)
    {
        List<RootPortalReference> refs = new(payload.Length / 8);
        for (int offset = 0; offset < payload.Length; offset += 8)
        {
            refs.Add(new RootPortalReference(
                BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset, 2)),
                BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 2, 2)),
                BinaryPrimitives.ReadInt16LittleEndian(payload.AsSpan(offset + 4, 2)),
                BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset + 6, 2))));
        }

        return refs;
    }

    private static byte[] WriteRootPortalReferences(IReadOnlyList<RootPortalReference> refs)
    {
        byte[] payload = new byte[refs.Count * 8];
        for (int index = 0; index < refs.Count; index++)
        {
            int offset = index * 8;
            BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(offset, 2), refs[index].PortalIndex);
            BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(offset + 2, 2), refs[index].GroupIndex);
            BinaryPrimitives.WriteInt16LittleEndian(payload.AsSpan(offset + 4, 2), refs[index].Side);
            BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(offset + 6, 2), refs[index].Padding);
        }

        return payload;
    }

    private static LegacyGroupDocument UpdatePortalRange(LegacyGroupDocument group, int portalStart, int portalCount)
    {
        WmoGroupSummary summary = new(
            group.Summary.SourcePath,
            group.Summary.Version,
            group.Summary.HeaderSizeBytes,
            group.Summary.NameOffset,
            group.Summary.DescriptiveNameOffset,
            group.Summary.Flags,
            group.Summary.BoundsMin,
            group.Summary.BoundsMax,
            portalStart,
            portalCount,
            group.TransparentBatchCount,
            group.InteriorBatchCount,
            group.ExteriorBatchCount,
            group.Summary.GroupLiquid,
            group.Mesh.FaceMaterials.Count,
            group.Mesh.Vertices.Count,
            group.Mesh.Indices.Count,
            group.Mesh.Normals.Count,
            group.Mesh.PrimaryUvs.Count,
            group.Mesh.AdditionalUvSets.Count,
            group.Mesh.Batches.Count,
            group.Mesh.PrimaryVertexColorsBgra.Count,
            group.DoodadRefs.Count,
            group.LightRefs.Count,
            group.Summary.BspNodeCount,
            group.Summary.BspFaceRefCount,
            group.LiquidPayload is { Length: > 0 });

        return new LegacyGroupDocument(summary, group.Mesh, group.DoodadRefs, group.LightRefs, group.LiquidPayload, group.TransparentBatchCount, group.InteriorBatchCount, group.ExteriorBatchCount);
    }

    private static List<ushort> ReadRefs(byte[] mogpPayload, int headerSizeBytes, FourCC chunkId)
    {
        byte[]? payload = WmoGroupReaderCommon.TryReadFirstSubchunkPayload(mogpPayload, headerSizeBytes, chunkId);
        if (payload is null)
            return [];

        if ((payload.Length & 1) != 0)
            throw new InvalidDataException($"{chunkId} payload size {payload.Length} is not divisible by 2.");

        List<ushort> refs = new(payload.Length / 2);
        for (int offset = 0; offset < payload.Length; offset += 2)
            refs.Add(BinaryPrimitives.ReadUInt16LittleEndian(payload.AsSpan(offset, 2)));

        return refs;
    }

    private static LegacyGroupDocument MergeLegacyGroups(IReadOnlyList<LegacyGroupDocument> groups)
    {
        if (groups.Count == 0)
            throw new InvalidOperationException("At least one legacy group is required for merge.");

        if (groups.Count == 1)
            return groups[0];

        LegacyGroupDocument seed = groups[0];
        Vector3 boundsMin = seed.Summary.BoundsMin;
        Vector3 boundsMax = seed.Summary.BoundsMax;
        uint flags = seed.Summary.Flags;
        int transparentBatchCount = 0;
        int interiorBatchCount = 0;
        int exteriorBatchCount = 0;

        List<Vector3> vertices = [];
        List<Vector3> normals = [];
        List<ushort> indices = [];
        List<Vector2> primaryUvs = [];
        int additionalUvSetCount = groups.Max(static group => group.Mesh.AdditionalUvSets.Count);
        List<List<Vector2>> additionalUvSets = Enumerable.Range(0, additionalUvSetCount).Select(static _ => new List<Vector2>()).ToList();
        List<uint> primaryVertexColors = [];
        int additionalVertexColorSetCount = groups.Max(static group => group.Mesh.AdditionalVertexColorSetsBgra.Count);
        List<List<uint>> additionalVertexColorSets = Enumerable.Range(0, additionalVertexColorSetCount).Select(static _ => new List<uint>()).ToList();
        List<WmoGroupFaceMaterialDetail> faceMaterials = [];
        List<WmoGroupBatchDetail> batches = [];
        SortedSet<ushort> doodadRefs = [];
        SortedSet<ushort> lightRefs = [];

        int vertexOffset = 0;
        int indexOffset = 0;
        foreach (LegacyGroupDocument group in groups)
        {
            boundsMin = Vector3.Min(boundsMin, group.Summary.BoundsMin);
            boundsMax = Vector3.Max(boundsMax, group.Summary.BoundsMax);
            flags |= group.Summary.Flags;
            transparentBatchCount += group.TransparentBatchCount;
            interiorBatchCount += group.InteriorBatchCount;
            exteriorBatchCount += group.ExteriorBatchCount;

            int groupVertexCount = group.Mesh.Vertices.Count;
            AppendRange(vertices, group.Mesh.Vertices);
            AppendRange(normals, group.Mesh.Normals, groupVertexCount, Vector3.UnitZ);
            AppendRange(primaryUvs, group.Mesh.PrimaryUvs, groupVertexCount, Vector2.Zero);
            AppendRange(primaryVertexColors, group.Mesh.PrimaryVertexColorsBgra, groupVertexCount, 0u);

            for (int setIndex = 0; setIndex < additionalUvSetCount; setIndex++)
            {
                IReadOnlyList<Vector2> set = setIndex < group.Mesh.AdditionalUvSets.Count
                    ? group.Mesh.AdditionalUvSets[setIndex]
                    : Array.Empty<Vector2>();
                AppendRange(additionalUvSets[setIndex], set, groupVertexCount, Vector2.Zero);
            }

            for (int setIndex = 0; setIndex < additionalVertexColorSetCount; setIndex++)
            {
                IReadOnlyList<uint> set = setIndex < group.Mesh.AdditionalVertexColorSetsBgra.Count
                    ? group.Mesh.AdditionalVertexColorSetsBgra[setIndex]
                    : Array.Empty<uint>();
                AppendRange(additionalVertexColorSets[setIndex], set, groupVertexCount, 0u);
            }

            foreach (ushort index in group.Mesh.Indices)
                indices.Add(checked((ushort)(index + vertexOffset)));

            foreach (WmoGroupFaceMaterialDetail face in group.Mesh.FaceMaterials)
                faceMaterials.Add(new WmoGroupFaceMaterialDetail(faceMaterials.Count, face.Flags, face.MaterialId, face.LegacyExtraValue));

            foreach (WmoGroupBatchDetail batch in group.Mesh.Batches)
                batches.Add(RebaseBatch(batch, indexOffset, vertexOffset, batches.Count));

            foreach (ushort doodadRef in group.DoodadRefs)
                doodadRefs.Add(doodadRef);

            foreach (ushort lightRef in group.LightRefs)
                lightRefs.Add(lightRef);

            vertexOffset += groupVertexCount;
            indexOffset += group.Mesh.Indices.Count;
        }

        bool hasLiquid = groups.Count == 1 && groups[0].LiquidPayload is { Length: > 0 };
        if (!hasLiquid)
            flags &= ~(uint)WmoGroupFlags.HasLiquidChunk;

        flags = NormalizeFlags(flags, faceMaterials.Count, primaryUvs.Count, additionalUvSets.Count, primaryVertexColors.Count, additionalVertexColorSets.Count, doodadRefs.Count, lightRefs.Count, hasLiquid);
        WmoGroupSummary mergedSummary = new(
            seed.Summary.SourcePath,
            seed.Summary.Version,
            0x44,
            seed.Summary.NameOffset,
            seed.Summary.DescriptiveNameOffset,
            flags,
            boundsMin,
            boundsMax,
            0,
            0,
            transparentBatchCount,
            interiorBatchCount,
            exteriorBatchCount,
            hasLiquid ? seed.Summary.GroupLiquid : 0,
            faceMaterials.Count,
            vertices.Count,
            indices.Count,
            normals.Count,
            primaryUvs.Count,
            additionalUvSets.Count,
            batches.Count,
            primaryVertexColors.Count,
            doodadRefs.Count,
            lightRefs.Count,
            indices.Count > 0 ? 1 : 0,
            indices.Count / 3,
            hasLiquid);

        WmoGroupMeshDetail mergedMesh = new(
            seed.Mesh.SourcePath,
            seed.Mesh.Version,
            0x44,
            WmoChunkIds.Movi.ToString(),
            vertices,
            normals,
            indices,
            primaryUvs,
            additionalUvSets,
            primaryVertexColors,
            additionalVertexColorSets,
            faceMaterials,
            batches);

        return new LegacyGroupDocument(
            mergedSummary,
            mergedMesh,
            doodadRefs.ToList(),
            lightRefs.ToList(),
            hasLiquid ? groups[0].LiquidPayload : null,
            transparentBatchCount,
            interiorBatchCount,
            exteriorBatchCount);
    }

    private static WmoGroupBatchDetail RebaseBatch(WmoGroupBatchDetail batch, int indexOffset, int vertexOffset, int batchIndex)
    {
        byte[] rawEntryBytes = batch.RawEntryBytes.ToArray();
        ushort firstIndex = checked((ushort)(batch.FirstIndex + indexOffset));
        BinaryPrimitives.WriteUInt16LittleEndian(rawEntryBytes.AsSpan(14, 2), firstIndex);

        ushort firstVertex = BinaryPrimitives.ReadUInt16LittleEndian(rawEntryBytes.AsSpan(18, 2));
        ushort lastVertex = BinaryPrimitives.ReadUInt16LittleEndian(rawEntryBytes.AsSpan(20, 2));
        BinaryPrimitives.WriteUInt16LittleEndian(rawEntryBytes.AsSpan(18, 2), checked((ushort)(firstVertex + vertexOffset)));
        BinaryPrimitives.WriteUInt16LittleEndian(rawEntryBytes.AsSpan(20, 2), checked((ushort)(lastVertex + vertexOffset)));

        return new WmoGroupBatchDetail(batchIndex, batch.PayloadOffset, batch.MaterialIdRaw, batch.HasMaterialId, firstIndex, batch.IndexCount, batch.Flags, rawEntryBytes);
    }

    private static void AppendRange<T>(List<T> target, IReadOnlyList<T> source)
    {
        foreach (T value in source)
            target.Add(value);
    }

    private static void AppendRange<T>(List<T> target, IReadOnlyList<T> source, int expectedCount, T defaultValue)
    {
        int count = Math.Min(source.Count, expectedCount);
        for (int index = 0; index < count; index++)
            target.Add(source[index]);

        for (int index = count; index < expectedCount; index++)
            target.Add(defaultValue);
    }

    private static uint NormalizeFlags(uint flags, int faceCount, int primaryUvCount, int additionalUvSetCount, int primaryVertexColorCount, int additionalVertexColorSetCount, int doodadRefCount, int lightRefCount, bool hasLiquid)
    {
        if (faceCount > 0)
            flags |= (uint)WmoGroupFlags.HasBspChunks;
        else
            flags &= ~(uint)WmoGroupFlags.HasBspChunks;

        if (primaryVertexColorCount > 0)
            flags |= (uint)WmoGroupFlags.HasVertexColorChunk;
        else
            flags &= ~(uint)WmoGroupFlags.HasVertexColorChunk;

        if (additionalVertexColorSetCount > 0)
            flags |= (uint)WmoGroupFlags.HasSecondaryVertexColorChunk;
        else
            flags &= ~(uint)WmoGroupFlags.HasSecondaryVertexColorChunk;

        if (additionalUvSetCount > 0)
            flags |= (uint)WmoGroupFlags.HasSecondaryUvSet;
        else
            flags &= ~(uint)WmoGroupFlags.HasSecondaryUvSet;

        if (additionalUvSetCount > 1)
            flags |= (uint)WmoGroupFlags.HasTertiaryUvSet;
        else
            flags &= ~(uint)WmoGroupFlags.HasTertiaryUvSet;

        if (doodadRefCount > 0)
            flags |= (uint)WmoGroupFlags.HasDoodadRefChunk;
        else
            flags &= ~(uint)WmoGroupFlags.HasDoodadRefChunk;

        if (lightRefCount > 0)
            flags |= (uint)WmoGroupFlags.HasLightRefChunk;
        else
            flags &= ~(uint)WmoGroupFlags.HasLightRefChunk;

        if (hasLiquid)
            flags |= (uint)WmoGroupFlags.HasLiquidChunk;
        else
            flags &= ~(uint)WmoGroupFlags.HasLiquidChunk;

        _ = primaryUvCount;
        return flags;
    }

    private static byte[] BuildLegacyGroupPayload(LegacyGroupDocument group)
    {
        uint flags = NormalizeFlags(
            group.Summary.Flags,
            group.Mesh.FaceMaterials.Count,
            group.Mesh.PrimaryUvs.Count,
            group.Mesh.AdditionalUvSets.Count,
            group.Mesh.PrimaryVertexColorsBgra.Count,
            group.Mesh.AdditionalVertexColorSetsBgra.Count,
            group.DoodadRefs.Count,
            group.LightRefs.Count,
            group.LiquidPayload is { Length: > 0 });

        byte[] header = new byte[0x44];
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x00, 4), group.Summary.NameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x04, 4), group.Summary.DescriptiveNameOffset);
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x08, 4), flags);
        WriteVector3(header, 0x0C, group.Summary.BoundsMin);
        WriteVector3(header, 0x18, group.Summary.BoundsMax);
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x24, 2), checked((ushort)group.Summary.PortalStart));
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x26, 2), checked((ushort)group.Summary.PortalCount));
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x28, 2), checked((ushort)group.TransparentBatchCount));
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x2A, 2), checked((ushort)group.InteriorBatchCount));
        BinaryPrimitives.WriteUInt16LittleEndian(header.AsSpan(0x2C, 2), checked((ushort)group.ExteriorBatchCount));
        BinaryPrimitives.WriteUInt32LittleEndian(header.AsSpan(0x34, 4), group.LiquidPayload is { Length: > 0 } ? group.Summary.GroupLiquid : 0u);

        using MemoryStream payloadStream = new();
        using BinaryWriter writer = new(payloadStream);
        writer.Write(header);
        WriteMopyChunk(writer, group.Mesh.FaceMaterials);
        WriteChunk(writer, WmoChunkIds.Movi, ToUInt16Bytes(group.Mesh.Indices));
        WriteChunk(writer, WmoChunkIds.Movt, ToVector3Bytes(group.Mesh.Vertices));
        WriteChunk(writer, WmoChunkIds.Monr, ToVector3Bytes(group.Mesh.Normals));

        if (group.Mesh.PrimaryUvs.Count > 0)
            WriteChunk(writer, WmoChunkIds.Motv, ToVector2Bytes(group.Mesh.PrimaryUvs));

        foreach (IReadOnlyList<Vector2> uvSet in group.Mesh.AdditionalUvSets)
            WriteChunk(writer, WmoChunkIds.Motv, ToVector2Bytes(uvSet));

        WriteChunk(writer, WmoChunkIds.Moba, ToBatchBytes(group.Mesh.Batches));

        if (group.LightRefs.Count > 0)
            WriteChunk(writer, WmoChunkIds.Molr, ToUInt16Bytes(group.LightRefs));

        if (group.DoodadRefs.Count > 0)
            WriteChunk(writer, WmoChunkIds.Modr, ToUInt16Bytes(group.DoodadRefs));

        if (group.Mesh.Indices.Count > 0)
        {
            WriteChunk(writer, WmoChunkIds.Mobn, CreateLeafBspNode());
            WriteChunk(writer, WmoChunkIds.Mobr, CreateFaceRefBytes(group.Mesh.Indices.Count / 3));
        }

        if (group.Mesh.PrimaryVertexColorsBgra.Count > 0)
            WriteChunk(writer, WmoChunkIds.Mocv, ToUInt32Bytes(group.Mesh.PrimaryVertexColorsBgra));

        foreach (IReadOnlyList<uint> colorSet in group.Mesh.AdditionalVertexColorSetsBgra)
            WriteChunk(writer, WmoChunkIds.Mocv, ToUInt32Bytes(colorSet));

        if (group.LiquidPayload is { Length: > 0 })
            WriteChunk(writer, WmoChunkIds.Mliq, group.LiquidPayload);

        return payloadStream.ToArray();
    }

    private static void WriteMopyChunk(BinaryWriter writer, IReadOnlyList<WmoGroupFaceMaterialDetail> faceMaterials)
    {
        byte[] payload = new byte[faceMaterials.Count * 4];
        for (int faceIndex = 0; faceIndex < faceMaterials.Count; faceIndex++)
        {
            int offset = faceIndex * 4;
            payload[offset] = faceMaterials[faceIndex].Flags;
            payload[offset + 1] = faceMaterials[faceIndex].MaterialId;
            BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(offset + 2, 2), faceMaterials[faceIndex].LegacyExtraValue ?? 0);
        }

        WriteChunk(writer, WmoChunkIds.Mopy, payload);
    }

    private static byte[] ToUInt16Bytes(IReadOnlyList<ushort> values)
    {
        byte[] payload = new byte[values.Count * 2];
        for (int index = 0; index < values.Count; index++)
            BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(index * 2, 2), values[index]);

        return payload;
    }

    private static byte[] ToUInt32Bytes(IReadOnlyList<uint> values)
    {
        byte[] payload = new byte[values.Count * 4];
        for (int index = 0; index < values.Count; index++)
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(index * 4, 4), values[index]);

        return payload;
    }

    private static byte[] ToVector3Bytes(IReadOnlyList<Vector3> values)
    {
        byte[] payload = new byte[values.Count * 12];
        for (int index = 0; index < values.Count; index++)
            WriteVector3(payload, index * 12, values[index]);

        return payload;
    }

    private static byte[] ToVector2Bytes(IReadOnlyList<Vector2> values)
    {
        byte[] payload = new byte[values.Count * 8];
        for (int index = 0; index < values.Count; index++)
        {
            WriteSingle(payload, index * 8, values[index].X);
            WriteSingle(payload, index * 8 + 4, values[index].Y);
        }

        return payload;
    }

    private static byte[] ToBatchBytes(IReadOnlyList<WmoGroupBatchDetail> batches)
    {
        byte[] payload = new byte[batches.Count * 24];
        for (int batchIndex = 0; batchIndex < batches.Count; batchIndex++)
            batches[batchIndex].RawEntryBytes.AsSpan(0, 24).CopyTo(payload.AsSpan(batchIndex * 24, 24));

        return payload;
    }

    private static byte[] CreateLeafBspNode()
    {
        byte[] payload = new byte[16];
        BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(0, 2), 0x4);
        BinaryPrimitives.WriteInt16LittleEndian(payload.AsSpan(2, 2), -1);
        BinaryPrimitives.WriteInt16LittleEndian(payload.AsSpan(4, 2), -1);
        return payload;
    }

    private static byte[] CreateFaceRefBytes(int faceCount)
    {
        byte[] payload = new byte[faceCount * 2];
        for (int faceIndex = 0; faceIndex < faceCount; faceIndex++)
            BinaryPrimitives.WriteUInt16LittleEndian(payload.AsSpan(faceIndex * 2, 2), checked((ushort)faceIndex));

        return payload;
    }

    private static void WriteVector3(byte[] bytes, int offset, Vector3 value)
    {
        WriteSingle(bytes, offset, value.X);
        WriteSingle(bytes, offset + 4, value.Y);
        WriteSingle(bytes, offset + 8, value.Z);
    }

    private static void WriteSingle(byte[] bytes, int offset, float value)
    {
        BinaryPrimitives.WriteInt32LittleEndian(bytes.AsSpan(offset, 4), BitConverter.SingleToInt32Bits(value));
    }

    private static byte[] BuildV14Root(IReadOnlyDictionary<FourCC, byte[]> rootChunkPayloads, IReadOnlyList<LegacyGroupDocument> groups, LegacyPortalLayout portalLayout)
    {
        Dictionary<FourCC, byte[]> payloadsById = rootChunkPayloads.ToDictionary(static pair => pair.Key, static pair => pair.Value.ToArray());
        payloadsById[WmoChunkIds.Mohd] = BuildLegacyMohdPayload(payloadsById[WmoChunkIds.Mohd], groups.Count, portalLayout.KeepPortalChunks ? portalLayout.PortalCount : 0);
        payloadsById[WmoChunkIds.Mogi] = BuildLegacyMogiPayload(groups);
        if (portalLayout.KeepPortalChunks)
        {
            if (portalLayout.MoprPayload is not null)
                payloadsById[WmoChunkIds.Mopr] = portalLayout.MoprPayload;
        }
        else
        {
            payloadsById.Remove(WmoChunkIds.Mopv);
            payloadsById.Remove(WmoChunkIds.Mopt);
            payloadsById.Remove(WmoChunkIds.Mopr);
        }

        using MemoryStream rootStream = new();
        using BinaryWriter writer = new(rootStream);

        WriteChunk(writer, WmoChunkIds.Mver, BitConverter.GetBytes((uint)14));

        using MemoryStream momoStream = new();
        using BinaryWriter momoWriter = new(momoStream);
        foreach (FourCC chunkId in RootChunkOrder)
        {
            if (payloadsById.TryGetValue(chunkId, out byte[]? payload) && payload.Length > 0)
                WriteChunk(momoWriter, chunkId, payload);
        }

        foreach (LegacyGroupDocument group in groups)
            WriteChunk(momoWriter, WmoChunkIds.Mogp, BuildLegacyGroupPayload(group));

        WriteChunk(writer, WmoChunkIds.Momo, momoStream.ToArray());
        writer.Flush();
        return rootStream.ToArray();
    }

    private static byte[] BuildLegacyMohdPayload(byte[] originalMohdPayload, int groupCount, int portalCount)
    {
        byte[] mohdPayload = originalMohdPayload.ToArray();
        BinaryPrimitives.WriteUInt32LittleEndian(mohdPayload.AsSpan(4, 4), checked((uint)groupCount));
        BinaryPrimitives.WriteUInt32LittleEndian(mohdPayload.AsSpan(8, 4), checked((uint)portalCount));

        return mohdPayload;
    }

    private static byte[] BuildLegacyMogiPayload(IReadOnlyList<LegacyGroupDocument> groups)
    {
        byte[] payload = new byte[groups.Count * 32];
        for (int groupIndex = 0; groupIndex < groups.Count; groupIndex++)
        {
            WmoGroupSummary summary = groups[groupIndex].Summary;
            int offset = groupIndex * 32;
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset, 4), summary.Flags);
            WriteVector3(payload, offset + 4, summary.BoundsMin);
            WriteVector3(payload, offset + 16, summary.BoundsMax);
            BinaryPrimitives.WriteUInt32LittleEndian(payload.AsSpan(offset + 28, 4), summary.NameOffset);
        }

        return payload;
    }

    private static void WriteChunk(BinaryWriter writer, FourCC chunkId, byte[] payload)
    {
        writer.Write(chunkId.ToFileBytes());
        writer.Write(payload.Length);
        writer.Write(payload);
    }

    private sealed record RootChunkPayloads(int ReportedGroupCount, IReadOnlyDictionary<FourCC, byte[]> PayloadsById);

    private sealed record MergeOverflowResult(IReadOnlyList<LegacyGroupDocument> Groups, LegacyPortalLayout PortalLayout);

    private sealed record LegacyPortalLayout(bool KeepPortalChunks, int PortalCount, byte[]? MoprPayload, IReadOnlyList<LegacyGroupDocument>? UpdatedGroups = null);

    private sealed record RootPortalReference(ushort PortalIndex, ushort GroupIndex, short Side, ushort Padding);

    private sealed record LegacyGroupDocument(
        WmoGroupSummary Summary,
        WmoGroupMeshDetail Mesh,
        IReadOnlyList<ushort> DoodadRefs,
        IReadOnlyList<ushort> LightRefs,
        byte[]? LiquidPayload,
        int TransparentBatchCount,
        int InteriorBatchCount,
        int ExteriorBatchCount);
}