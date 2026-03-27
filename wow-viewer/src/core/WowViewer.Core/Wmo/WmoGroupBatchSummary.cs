namespace WowViewer.Core.Wmo;

public sealed class WmoGroupBatchSummary
{
    public WmoGroupBatchSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int entryCount,
        bool hasMaterialIds,
        int distinctMaterialIdCount,
        int highestMaterialId,
        int totalIndexCount,
        int minFirstIndex,
        int maxFirstIndex,
        int maxIndexEnd,
        int flaggedBatchCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctMaterialIdCount);
        ArgumentOutOfRangeException.ThrowIfNegative(highestMaterialId);
        ArgumentOutOfRangeException.ThrowIfNegative(totalIndexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(minFirstIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(maxFirstIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(maxIndexEnd);
        ArgumentOutOfRangeException.ThrowIfNegative(flaggedBatchCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        EntryCount = entryCount;
        HasMaterialIds = hasMaterialIds;
        DistinctMaterialIdCount = distinctMaterialIdCount;
        HighestMaterialId = highestMaterialId;
        TotalIndexCount = totalIndexCount;
        MinFirstIndex = minFirstIndex;
        MaxFirstIndex = maxFirstIndex;
        MaxIndexEnd = maxIndexEnd;
        FlaggedBatchCount = flaggedBatchCount;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int EntryCount { get; }

    public bool HasMaterialIds { get; }

    public int DistinctMaterialIdCount { get; }

    public int HighestMaterialId { get; }

    public int TotalIndexCount { get; }

    public int MinFirstIndex { get; }

    public int MaxFirstIndex { get; }

    public int MaxIndexEnd { get; }

    public int FlaggedBatchCount { get; }
}
