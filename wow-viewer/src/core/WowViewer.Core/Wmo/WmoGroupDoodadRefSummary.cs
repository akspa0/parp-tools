namespace WowViewer.Core.Wmo;

public sealed class WmoGroupDoodadRefSummary
{
    public WmoGroupDoodadRefSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int refCount,
        int distinctRefCount,
        int minRef,
        int maxRef,
        int duplicateRefCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(refCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctRefCount);
        ArgumentOutOfRangeException.ThrowIfNegative(minRef);
        ArgumentOutOfRangeException.ThrowIfNegative(maxRef);
        ArgumentOutOfRangeException.ThrowIfNegative(duplicateRefCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        RefCount = refCount;
        DistinctRefCount = distinctRefCount;
        MinRef = minRef;
        MaxRef = maxRef;
        DuplicateRefCount = duplicateRefCount;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int RefCount { get; }

    public int DistinctRefCount { get; }

    public int MinRef { get; }

    public int MaxRef { get; }

    public int DuplicateRefCount { get; }
}
