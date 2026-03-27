namespace WowViewer.Core.Wmo;

public sealed class WmoDoodadSetSummary
{
    public WmoDoodadSetSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int entryCount,
        int nonEmptySetCount,
        int longestNameLength,
        int totalDoodadRefs,
        int maxStartIndex,
        int maxRangeEnd)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(nonEmptySetCount);
        ArgumentOutOfRangeException.ThrowIfNegative(longestNameLength);
        ArgumentOutOfRangeException.ThrowIfNegative(totalDoodadRefs);
        ArgumentOutOfRangeException.ThrowIfNegative(maxStartIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(maxRangeEnd);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        EntryCount = entryCount;
        NonEmptySetCount = nonEmptySetCount;
        LongestNameLength = longestNameLength;
        TotalDoodadRefs = totalDoodadRefs;
        MaxStartIndex = maxStartIndex;
        MaxRangeEnd = maxRangeEnd;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int EntryCount { get; }

    public int NonEmptySetCount { get; }

    public int LongestNameLength { get; }

    public int TotalDoodadRefs { get; }

    public int MaxStartIndex { get; }

    public int MaxRangeEnd { get; }
}
