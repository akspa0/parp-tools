using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoEmbeddedGroupSummary
{
    public WmoEmbeddedGroupSummary(
        string sourcePath,
        uint? version,
        int groupCount,
        int minHeaderSizeBytes,
        int maxHeaderSizeBytes,
        int groupsWithPortals,
        int groupsWithLiquid,
        int totalFaceMaterialCount,
        int totalVertexCount,
        int totalIndexCount,
        int totalNormalCount,
        int totalBatchCount,
        int totalDoodadRefCount,
        Vector3 boundsMin,
        Vector3 boundsMax)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(groupCount);
        ArgumentOutOfRangeException.ThrowIfNegative(minHeaderSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(maxHeaderSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(groupsWithPortals);
        ArgumentOutOfRangeException.ThrowIfNegative(groupsWithLiquid);
        ArgumentOutOfRangeException.ThrowIfNegative(totalFaceMaterialCount);
        ArgumentOutOfRangeException.ThrowIfNegative(totalVertexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(totalIndexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(totalNormalCount);
        ArgumentOutOfRangeException.ThrowIfNegative(totalBatchCount);
        ArgumentOutOfRangeException.ThrowIfNegative(totalDoodadRefCount);

        SourcePath = sourcePath;
        Version = version;
        GroupCount = groupCount;
        MinHeaderSizeBytes = minHeaderSizeBytes;
        MaxHeaderSizeBytes = maxHeaderSizeBytes;
        GroupsWithPortals = groupsWithPortals;
        GroupsWithLiquid = groupsWithLiquid;
        TotalFaceMaterialCount = totalFaceMaterialCount;
        TotalVertexCount = totalVertexCount;
        TotalIndexCount = totalIndexCount;
        TotalNormalCount = totalNormalCount;
        TotalBatchCount = totalBatchCount;
        TotalDoodadRefCount = totalDoodadRefCount;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int GroupCount { get; }
    public int MinHeaderSizeBytes { get; }
    public int MaxHeaderSizeBytes { get; }
    public int GroupsWithPortals { get; }
    public int GroupsWithLiquid { get; }
    public int TotalFaceMaterialCount { get; }
    public int TotalVertexCount { get; }
    public int TotalIndexCount { get; }
    public int TotalNormalCount { get; }
    public int TotalBatchCount { get; }
    public int TotalDoodadRefCount { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
}
