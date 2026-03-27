namespace WowViewer.Core.Wmo;

public sealed class WmoGroupFaceMaterialSummary
{
    public WmoGroupFaceMaterialSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int entrySizeBytes,
        int faceCount,
        int distinctMaterialIdCount,
        int highestMaterialId,
        int hiddenFaceCount,
        int flaggedFaceCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entrySizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(faceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctMaterialIdCount);
        ArgumentOutOfRangeException.ThrowIfNegative(highestMaterialId);
        ArgumentOutOfRangeException.ThrowIfNegative(hiddenFaceCount);
        ArgumentOutOfRangeException.ThrowIfNegative(flaggedFaceCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        EntrySizeBytes = entrySizeBytes;
        FaceCount = faceCount;
        DistinctMaterialIdCount = distinctMaterialIdCount;
        HighestMaterialId = highestMaterialId;
        HiddenFaceCount = hiddenFaceCount;
        FlaggedFaceCount = flaggedFaceCount;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int EntrySizeBytes { get; }

    public int FaceCount { get; }

    public int DistinctMaterialIdCount { get; }

    public int HighestMaterialId { get; }

    public int HiddenFaceCount { get; }

    public int FlaggedFaceCount { get; }
}
