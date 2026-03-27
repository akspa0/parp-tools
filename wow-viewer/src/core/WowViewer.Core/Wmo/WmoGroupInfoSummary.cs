using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoGroupInfoSummary
{
    public WmoGroupInfoSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int entrySizeBytes,
        int entryCount,
        int distinctFlagCount,
        int nonZeroFlagCount,
        int minNameOffset,
        int maxNameOffset,
        Vector3 boundsMin,
        Vector3 boundsMax)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entrySizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctFlagCount);
        ArgumentOutOfRangeException.ThrowIfNegative(nonZeroFlagCount);
        ArgumentOutOfRangeException.ThrowIfNegative(minNameOffset);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNameOffset);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        EntrySizeBytes = entrySizeBytes;
        EntryCount = entryCount;
        DistinctFlagCount = distinctFlagCount;
        NonZeroFlagCount = nonZeroFlagCount;
        MinNameOffset = minNameOffset;
        MaxNameOffset = maxNameOffset;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int EntrySizeBytes { get; }

    public int EntryCount { get; }

    public int DistinctFlagCount { get; }

    public int NonZeroFlagCount { get; }

    public int MinNameOffset { get; }

    public int MaxNameOffset { get; }

    public Vector3 BoundsMin { get; }

    public Vector3 BoundsMax { get; }
}
