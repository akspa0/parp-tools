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
        float minAttenStart,
        float maxAttenStart,
        float maxAttenEnd,
        int nonZeroHeaderFlagsWordCount,
        int distinctHeaderFlagsWordCount,
        ushort minHeaderFlagsWord,
        ushort maxHeaderFlagsWord,
        int rotationEntryCount,
        int nonIdentityRotationCount,
        float minRotationLength,
        float maxRotationLength,
        Vector3 boundsMin,
        Vector3 boundsMax)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctTypeCount);
        ArgumentOutOfRangeException.ThrowIfNegative(attenuatedCount);
        ArgumentOutOfRangeException.ThrowIfNegative(nonZeroHeaderFlagsWordCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctHeaderFlagsWordCount);
        ArgumentOutOfRangeException.ThrowIfNegative(rotationEntryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(nonIdentityRotationCount);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        EntryCount = entryCount;
        DistinctTypeCount = distinctTypeCount;
        AttenuatedCount = attenuatedCount;
        MinIntensity = minIntensity;
        MaxIntensity = maxIntensity;
        MinAttenStart = minAttenStart;
        MaxAttenStart = maxAttenStart;
        MaxAttenEnd = maxAttenEnd;
        NonZeroHeaderFlagsWordCount = nonZeroHeaderFlagsWordCount;
        DistinctHeaderFlagsWordCount = distinctHeaderFlagsWordCount;
        MinHeaderFlagsWord = minHeaderFlagsWord;
        MaxHeaderFlagsWord = maxHeaderFlagsWord;
        RotationEntryCount = rotationEntryCount;
        NonIdentityRotationCount = nonIdentityRotationCount;
        MinRotationLength = minRotationLength;
        MaxRotationLength = maxRotationLength;
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
    public float MinAttenStart { get; }
    public float MaxAttenStart { get; }
    public float MaxAttenEnd { get; }
    public int NonZeroHeaderFlagsWordCount { get; }
    public int DistinctHeaderFlagsWordCount { get; }
    public ushort MinHeaderFlagsWord { get; }
    public ushort MaxHeaderFlagsWord { get; }
    public int RotationEntryCount { get; }
    public int NonIdentityRotationCount { get; }
    public float MinRotationLength { get; }
    public float MaxRotationLength { get; }
    public Vector3 BoundsMin { get; }
    public Vector3 BoundsMax { get; }
}
