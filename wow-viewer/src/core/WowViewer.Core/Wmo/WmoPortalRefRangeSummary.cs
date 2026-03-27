namespace WowViewer.Core.Wmo;

public sealed class WmoPortalRefRangeSummary
{
    public WmoPortalRefRangeSummary(
        string sourcePath,
        uint? version,
        int refCount,
        int portalCount,
        int coveredRefCount,
        int outOfRangeRefCount,
        int distinctPortalRefCount,
        int maxPortalIndex)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(refCount);
        ArgumentOutOfRangeException.ThrowIfNegative(portalCount);
        ArgumentOutOfRangeException.ThrowIfNegative(coveredRefCount);
        ArgumentOutOfRangeException.ThrowIfNegative(outOfRangeRefCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctPortalRefCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxPortalIndex);

        SourcePath = sourcePath;
        Version = version;
        RefCount = refCount;
        PortalCount = portalCount;
        CoveredRefCount = coveredRefCount;
        OutOfRangeRefCount = outOfRangeRefCount;
        DistinctPortalRefCount = distinctPortalRefCount;
        MaxPortalIndex = maxPortalIndex;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int RefCount { get; }
    public int PortalCount { get; }
    public int CoveredRefCount { get; }
    public int OutOfRangeRefCount { get; }
    public int DistinctPortalRefCount { get; }
    public int MaxPortalIndex { get; }
}
