using System.Numerics;

namespace WowViewer.Core.Wmo;

public sealed class WmoDoodadPlacementSummary
{
    public WmoDoodadPlacementSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int entryCount,
        int distinctNameIndexCount,
        int maxNameIndex,
        float minScale,
        float maxScale,
        int minAlpha,
        int maxAlpha,
        Vector3 boundsMin,
        Vector3 boundsMax)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctNameIndexCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxNameIndex);
        ArgumentOutOfRangeException.ThrowIfNegative(minAlpha);
        ArgumentOutOfRangeException.ThrowIfNegative(maxAlpha);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        EntryCount = entryCount;
        DistinctNameIndexCount = distinctNameIndexCount;
        MaxNameIndex = maxNameIndex;
        MinScale = minScale;
        MaxScale = maxScale;
        MinAlpha = minAlpha;
        MaxAlpha = maxAlpha;
        BoundsMin = boundsMin;
        BoundsMax = boundsMax;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int EntryCount { get; }

    public int DistinctNameIndexCount { get; }

    public int MaxNameIndex { get; }

    public float MinScale { get; }

    public float MaxScale { get; }

    public int MinAlpha { get; }

    public int MaxAlpha { get; }

    public Vector3 BoundsMin { get; }

    public Vector3 BoundsMax { get; }
}
