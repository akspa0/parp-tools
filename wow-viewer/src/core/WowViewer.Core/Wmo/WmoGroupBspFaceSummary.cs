namespace WowViewer.Core.Wmo;

public sealed class WmoGroupBspFaceSummary
{
    public WmoGroupBspFaceSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int refCount,
        int distinctFaceRefCount,
        int minFaceRef,
        int maxFaceRef,
        int duplicateFaceRefCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(refCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctFaceRefCount);
        ArgumentOutOfRangeException.ThrowIfNegative(minFaceRef);
        ArgumentOutOfRangeException.ThrowIfNegative(maxFaceRef);
        ArgumentOutOfRangeException.ThrowIfNegative(duplicateFaceRefCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        RefCount = refCount;
        DistinctFaceRefCount = distinctFaceRefCount;
        MinFaceRef = minFaceRef;
        MaxFaceRef = maxFaceRef;
        DuplicateFaceRefCount = duplicateFaceRefCount;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int RefCount { get; }

    public int DistinctFaceRefCount { get; }

    public int MinFaceRef { get; }

    public int MaxFaceRef { get; }

    public int DuplicateFaceRefCount { get; }
}