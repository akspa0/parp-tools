using System.Numerics;
using WowViewer.Core.IO.Chunked;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.IO.Wmo;

public static class WmoEmbeddedGroupSummaryReader
{
    public static WmoEmbeddedGroupSummary Read(string path)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(path);
        using FileStream stream = File.OpenRead(path);
        return Read(stream, Path.GetFullPath(path));
    }

    public static WmoEmbeddedGroupSummary Read(Stream stream, string sourcePath = "<memory>")
    {
        ArgumentNullException.ThrowIfNull(stream);
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);

        var (version, chunks) = WmoRootReaderCommon.ReadRootChunks(stream, sourcePath);
        List<ChunkSpan> groupChunks = chunks.Where(static chunk => chunk.Header.Id == WmoChunkIds.Mogp).ToList();
        if (groupChunks.Count == 0)
            throw new InvalidDataException("WMO embedded-group summary requires one or more MOGP chunks in the root file.");

        int minHeaderSizeBytes = int.MaxValue;
        int maxHeaderSizeBytes = 0;
        int groupsWithPortals = 0;
        int groupsWithLiquid = 0;
        int totalFaceMaterialCount = 0;
        int totalVertexCount = 0;
        int totalIndexCount = 0;
        int totalNormalCount = 0;
        int totalBatchCount = 0;
        int totalDoodadRefCount = 0;
        int totalLightRefCount = 0;
        int totalBspNodeCount = 0;
        int totalBspFaceRefCount = 0;
        Vector3 boundsMin = new(float.MaxValue, float.MaxValue, float.MaxValue);
        Vector3 boundsMax = new(float.MinValue, float.MinValue, float.MinValue);

        foreach (ChunkSpan groupChunk in groupChunks)
        {
            byte[] mogp = WmoGroupReaderCommon.ReadChunkPayload(stream, groupChunk);
            WmoGroupSummary groupSummary = WmoGroupSummaryReader.ReadMogpPayload(mogp, $"{sourcePath}#MOGP@{groupChunk.HeaderOffset}", version);

            minHeaderSizeBytes = Math.Min(minHeaderSizeBytes, groupSummary.HeaderSizeBytes);
            maxHeaderSizeBytes = Math.Max(maxHeaderSizeBytes, groupSummary.HeaderSizeBytes);
            if (groupSummary.PortalCount > 0)
                groupsWithPortals++;
            if (groupSummary.HasLiquid)
                groupsWithLiquid++;

            totalFaceMaterialCount += groupSummary.FaceMaterialCount;
            totalVertexCount += groupSummary.VertexCount;
            totalIndexCount += groupSummary.IndexCount;
            totalNormalCount += groupSummary.NormalCount;
            totalBatchCount += groupSummary.BatchCount;
            totalDoodadRefCount += groupSummary.DoodadRefCount;
            totalLightRefCount += groupSummary.LightRefCount;
            totalBspNodeCount += groupSummary.BspNodeCount;
            totalBspFaceRefCount += groupSummary.BspFaceRefCount;
            boundsMin = Vector3.Min(boundsMin, groupSummary.BoundsMin);
            boundsMax = Vector3.Max(boundsMax, groupSummary.BoundsMax);
        }

        return new WmoEmbeddedGroupSummary(
            sourcePath,
            version,
            groupChunks.Count,
            minHeaderSizeBytes,
            maxHeaderSizeBytes,
            groupsWithPortals,
            groupsWithLiquid,
            totalFaceMaterialCount,
            totalVertexCount,
            totalIndexCount,
            totalNormalCount,
            totalBatchCount,
            totalDoodadRefCount,
            totalLightRefCount,
            totalBspNodeCount,
            totalBspFaceRefCount,
            boundsMin,
            boundsMax);
    }
}
