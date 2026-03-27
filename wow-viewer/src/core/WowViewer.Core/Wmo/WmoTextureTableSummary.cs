namespace WowViewer.Core.Wmo;

public sealed class WmoTextureTableSummary
{
    public WmoTextureTableSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int textureCount,
        int longestEntryLength,
        int maxOffset,
        int distinctExtensionCount,
        int blpEntryCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(textureCount);
        ArgumentOutOfRangeException.ThrowIfNegative(longestEntryLength);
        ArgumentOutOfRangeException.ThrowIfNegative(maxOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctExtensionCount);
        ArgumentOutOfRangeException.ThrowIfNegative(blpEntryCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        TextureCount = textureCount;
        LongestEntryLength = longestEntryLength;
        MaxOffset = maxOffset;
        DistinctExtensionCount = distinctExtensionCount;
        BlpEntryCount = blpEntryCount;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int TextureCount { get; }

    public int LongestEntryLength { get; }

    public int MaxOffset { get; }

    public int DistinctExtensionCount { get; }

    public int BlpEntryCount { get; }
}
