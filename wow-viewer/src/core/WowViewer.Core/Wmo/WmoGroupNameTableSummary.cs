namespace WowViewer.Core.Wmo;

public sealed class WmoGroupNameTableSummary
{
    public WmoGroupNameTableSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int nameCount,
        int longestEntryLength,
        int maxOffset)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(nameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(longestEntryLength);
        ArgumentOutOfRangeException.ThrowIfNegative(maxOffset);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        NameCount = nameCount;
        LongestEntryLength = longestEntryLength;
        MaxOffset = maxOffset;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int NameCount { get; }

    public int LongestEntryLength { get; }

    public int MaxOffset { get; }
}
