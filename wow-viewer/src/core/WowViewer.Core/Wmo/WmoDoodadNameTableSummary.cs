namespace WowViewer.Core.Wmo;

public sealed class WmoDoodadNameTableSummary
{
    public WmoDoodadNameTableSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int nameCount,
        int longestEntryLength,
        int maxOffset,
        int distinctExtensionCount,
        int mdxEntryCount,
        int m2EntryCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(nameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(longestEntryLength);
        ArgumentOutOfRangeException.ThrowIfNegative(maxOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctExtensionCount);
        ArgumentOutOfRangeException.ThrowIfNegative(mdxEntryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(m2EntryCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        NameCount = nameCount;
        LongestEntryLength = longestEntryLength;
        MaxOffset = maxOffset;
        DistinctExtensionCount = distinctExtensionCount;
        MdxEntryCount = mdxEntryCount;
        M2EntryCount = m2EntryCount;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int NameCount { get; }

    public int LongestEntryLength { get; }

    public int MaxOffset { get; }

    public int DistinctExtensionCount { get; }

    public int MdxEntryCount { get; }

    public int M2EntryCount { get; }
}
