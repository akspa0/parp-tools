namespace WowViewer.Core.Wmo;

public sealed class WmoPortalRefSummary
{
    public WmoPortalRefSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int entryCount,
        int distinctPortalIndexCount,
        int maxGroupIndex,
        int positiveSideCount,
        int negativeSideCount,
        int neutralSideCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctPortalIndexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxGroupIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(positiveSideCount);
        ArgumentOutOfRangeException.ThrowIfNegative(negativeSideCount);
        ArgumentOutOfRangeException.ThrowIfNegative(neutralSideCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        EntryCount = entryCount;
        DistinctPortalIndexCount = distinctPortalIndexCount;
        MaxGroupIndex = maxGroupIndex;
        PositiveSideCount = positiveSideCount;
        NegativeSideCount = negativeSideCount;
        NeutralSideCount = neutralSideCount;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int PayloadSizeBytes { get; }
    public int EntryCount { get; }
    public int DistinctPortalIndexCount { get; }
    public int MaxGroupIndex { get; }
    public int PositiveSideCount { get; }
    public int NegativeSideCount { get; }
    public int NeutralSideCount { get; }
}
