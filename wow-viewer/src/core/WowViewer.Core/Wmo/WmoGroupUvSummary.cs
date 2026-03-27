namespace WowViewer.Core.Wmo;

public sealed class WmoGroupUvSummary
{
    public WmoGroupUvSummary(
        string sourcePath,
        uint? version,
        int primaryPayloadSizeBytes,
        int primaryUvCount,
        float minU,
        float maxU,
        float minV,
        float maxV,
        int additionalUvSetCount,
        int totalAdditionalUvCount,
        int maxAdditionalUvCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(primaryPayloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(primaryUvCount);
        ArgumentOutOfRangeException.ThrowIfNegative(additionalUvSetCount);
        ArgumentOutOfRangeException.ThrowIfNegative(totalAdditionalUvCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxAdditionalUvCount);

        SourcePath = sourcePath;
        Version = version;
        PrimaryPayloadSizeBytes = primaryPayloadSizeBytes;
        PrimaryUvCount = primaryUvCount;
        MinU = minU;
        MaxU = maxU;
        MinV = minV;
        MaxV = maxV;
        AdditionalUvSetCount = additionalUvSetCount;
        TotalAdditionalUvCount = totalAdditionalUvCount;
        MaxAdditionalUvCount = maxAdditionalUvCount;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PrimaryPayloadSizeBytes { get; }

    public int PrimaryUvCount { get; }

    public float MinU { get; }

    public float MaxU { get; }

    public float MinV { get; }

    public float MaxV { get; }

    public int AdditionalUvSetCount { get; }

    public int TotalAdditionalUvCount { get; }

    public int MaxAdditionalUvCount { get; }
}
