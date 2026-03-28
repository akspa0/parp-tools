using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoEmbeddedGroupDetailReader
{
    public static IReadOnlyList<WmoEmbeddedGroupDetail> Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static IReadOnlyList<WmoEmbeddedGroupDetail> Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        (uint? version, IReadOnlyList<ChunkSpan> chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        List<ChunkSpan> groupChunks = chunks.Where(static chunk => chunk.Header.Id == WmoChunkIds.Mogp).ToList();
        if (groupChunks.Count == 0)
            throw new InvalidDataException("WMO embedded-group detail read requires one or more MOGP chunks in the root file.");

        List<WmoEmbeddedGroupDetail> details = new(groupChunks.Count);
        for (int groupIndex = 0; groupIndex < groupChunks.Count; groupIndex++)
        {
            ChunkSpan groupChunk = groupChunks[groupIndex];
            string detailSourcePath = $"{sourcePath}#MOGP[{groupIndex}]@{groupChunk.HeaderOffset}";
            byte[] mogp = WmoGroupReaderCommon.ReadChunkPayload(stream, groupChunk);

            WmoGroupSummary groupSummary = WmoGroupSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version);
            WmoGroupLiquidSummary? liquidSummary = groupSummary.HasLiquid
                ? WmoGroupLiquidSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupBatchSummary? batchSummary = groupSummary.BatchCount > 0
                ? WmoGroupBatchSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupFaceMaterialSummary? faceMaterialSummary = groupSummary.FaceMaterialCount > 0
                ? WmoGroupFaceMaterialSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupUvSummary? uvSummary = groupSummary.PrimaryUvCount > 0
                ? WmoGroupUvSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupVertexColorSummary? vertexColorSummary = groupSummary.VertexColorCount > 0
                ? WmoGroupVertexColorSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupDoodadRefSummary? doodadRefSummary = groupSummary.DoodadRefCount > 0
                ? WmoGroupDoodadRefSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupLightRefSummary? lightRefSummary = groupSummary.LightRefCount > 0
                ? WmoGroupLightRefSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupIndexSummary? indexSummary = groupSummary.IndexCount > 0
                ? WmoGroupIndexSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupVertexSummary? vertexSummary = groupSummary.VertexCount > 0
                ? WmoGroupVertexSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupNormalSummary? normalSummary = groupSummary.NormalCount > 0
                ? WmoGroupNormalSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupBspNodeSummary? bspNodeSummary = groupSummary.BspNodeCount > 0
                ? WmoGroupBspNodeSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupBspFaceSummary? bspFaceSummary = groupSummary.BspFaceRefCount > 0
                ? WmoGroupBspFaceSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;
            WmoGroupBspFaceRangeSummary? bspFaceRangeSummary = groupSummary.BspNodeCount > 0 && groupSummary.BspFaceRefCount > 0
                ? WmoGroupBspFaceRangeSummaryReader.ReadMogpPayload(mogp, detailSourcePath, version)
                : null;

            details.Add(new WmoEmbeddedGroupDetail(
                groupIndex,
                groupChunk.HeaderOffset,
                groupSummary,
                liquidSummary,
                batchSummary,
                faceMaterialSummary,
                uvSummary,
                vertexColorSummary,
                doodadRefSummary,
                lightRefSummary,
                indexSummary,
                vertexSummary,
                normalSummary,
                bspNodeSummary,
                bspFaceSummary,
                bspFaceRangeSummary));
        }

        return details;
    }
}