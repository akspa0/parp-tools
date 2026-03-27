namespace WowViewer.Core.Wmo;

public sealed class WmoDoodadSetRangeSummary
{
    public WmoDoodadSetRangeSummary(
        string sourcePath,
        uint? version,
        int entryCount,
        int placementCount,
        int emptySetCount,
        int fullyCoveredSetCount,
        int outOfRangeSetCount,
        int maxRangeEnd)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(placementCount);
        ArgumentOutOfRangeException.ThrowIfNegative(emptySetCount);
        ArgumentOutOfRangeException.ThrowIfNegative(fullyCoveredSetCount);
        ArgumentOutOfRangeException.ThrowIfNegative(outOfRangeSetCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxRangeEnd);

        SourcePath = sourcePath;
        Version = version;
        EntryCount = entryCount;
        PlacementCount = placementCount;
        EmptySetCount = emptySetCount;
        FullyCoveredSetCount = fullyCoveredSetCount;
        OutOfRangeSetCount = outOfRangeSetCount;
        MaxRangeEnd = maxRangeEnd;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int EntryCount { get; }
    public int PlacementCount { get; }
    public int EmptySetCount { get; }
    public int FullyCoveredSetCount { get; }
    public int OutOfRangeSetCount { get; }
    public int MaxRangeEnd { get; }
}
