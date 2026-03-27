using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoLightSummary
{
    public WmoLightSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int entryCount,
        int distinctTypeCount,
        int attenuatedCount,
        float minIntensity,
        float maxIntensity,
        float maxAttenEnd,
        Vector3 boundsMin,
        Vector3 boundsMax)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctTypeCount);
        ArgumentOutOfRangeException.ThrowIfNegative(attenuatedCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        EntryCount = entryCount;
        DistinctTypeCount = distinctTypeCount;
        AttenuatedCount = attenuatedCount;
        MinIntensity = minIntensity;
        MaxIntensity = maxIntensity;
        MaxAttenEnd = maxAttenEnd;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int PayloadSizeBytes { get; }
    public int EntryCount { get; }
    public int DistinctTypeCount { get; }
    public int AttenuatedCount { get; }
    public float MinIntensity { get; }
    public float MaxIntensity { get; }
    public float MaxAttenEnd { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
}
