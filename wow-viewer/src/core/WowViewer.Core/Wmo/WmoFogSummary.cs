using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoFogSummary
{
    public WmoFogSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int entryCount,
        int nonZeroFlagCount,
        float minSmallRadius,
        float maxLargeRadius,
        float maxFogEnd,
        Vector3 boundsMin,
        Vector3 boundsMax)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(nonZeroFlagCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        EntryCount = entryCount;
        NonZeroFlagCount = nonZeroFlagCount;
        MinSmallRadius = minSmallRadius;
        MaxLargeRadius = maxLargeRadius;
        MaxFogEnd = maxFogEnd;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int PayloadSizeBytes { get; }
    public int EntryCount { get; }
    public int NonZeroFlagCount { get; }
    public float MinSmallRadius { get; }
    public float MaxLargeRadius { get; }
    public float MaxFogEnd { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
}
