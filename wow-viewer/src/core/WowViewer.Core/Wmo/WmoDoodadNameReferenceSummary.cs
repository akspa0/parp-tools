namespace WowViewer.Core.Wmo;

public sealed class WmoDoodadNameReferenceSummary
{
    public WmoDoodadNameReferenceSummary(
        string sourcePath,
        uint? version,
        int entryCount,
        int resolvedNameCount,
        int unresolvedNameCount,
        int distinctResolvedNameCount,
        int maxResolvedNameLength)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(resolvedNameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(unresolvedNameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctResolvedNameCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxResolvedNameLength);

        SourcePath = sourcePath;
        Version = version;
        EntryCount = entryCount;
        ResolvedNameCount = resolvedNameCount;
        UnresolvedNameCount = unresolvedNameCount;
        DistinctResolvedNameCount = distinctResolvedNameCount;
        MaxResolvedNameLength = maxResolvedNameLength;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int EntryCount { get; }
    public int ResolvedNameCount { get; }
    public int UnresolvedNameCount { get; }
    public int DistinctResolvedNameCount { get; }
    public int MaxResolvedNameLength { get; }
}
