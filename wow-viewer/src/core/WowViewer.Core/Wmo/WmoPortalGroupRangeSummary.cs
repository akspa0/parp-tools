namespace WowViewer.Core.Wmo;

public sealed class WmoPortalGroupRangeSummary
{
    public WmoPortalGroupRangeSummary(
        string sourcePath,
        uint? version,
        int refCount,
        int groupCount,
        int coveredRefCount,
        int outOfRangeRefCount,
        int distinctGroupRefCount,
        int maxGroupIndex)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(refCount);
        ArgumentOutOfRangeException.ThrowIfNegative(groupCount);
        ArgumentOutOfRangeException.ThrowIfNegative(coveredRefCount);
        ArgumentOutOfRangeException.ThrowIfNegative(outOfRangeRefCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctGroupRefCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxGroupIndex);

        SourcePath = sourcePath;
        Version = version;
        RefCount = refCount;
        GroupCount = groupCount;
        CoveredRefCount = coveredRefCount;
        OutOfRangeRefCount = outOfRangeRefCount;
        DistinctGroupRefCount = distinctGroupRefCount;
        MaxGroupIndex = maxGroupIndex;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int RefCount { get; }
    public int GroupCount { get; }
    public int CoveredRefCount { get; }
    public int OutOfRangeRefCount { get; }
    public int DistinctGroupRefCount { get; }
    public int MaxGroupIndex { get; }
}
