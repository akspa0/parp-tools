namespace WowViewer.Core.Wmo;

public sealed class WmoGroupNormalSummary
{
    public WmoGroupNormalSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int normalCount,
        float minX,
        float maxX,
        float minY,
        float maxY,
        float minZ,
        float maxZ,
        float minLength,
        float maxLength,
        float averageLength,
        int nearUnitCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(normalCount);
        ArgumentOutOfRangeException.ThrowIfNegative(nearUnitCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        NormalCount = normalCount;
        MinX = minX;
        MaxX = maxX;
        MinY = minY;
        MaxY = maxY;
        MinZ = minZ;
        MaxZ = maxZ;
        MinLength = minLength;
        MaxLength = maxLength;
        AverageLength = averageLength;
        NearUnitCount = nearUnitCount;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int NormalCount { get; }

    public float MinX { get; }

    public float MaxX { get; }

    public float MinY { get; }

    public float MaxY { get; }

    public float MinZ { get; }

    public float MaxZ { get; }

    public float MinLength { get; }

    public float MaxLength { get; }

    public float AverageLength { get; }

    public int NearUnitCount { get; }
}
