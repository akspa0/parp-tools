namespace WowViewer.Core.Wmo;

public sealed class WmoGroupVertexColorSummary
{
    public WmoGroupVertexColorSummary(
        string sourcePath,
        uint? version,
        int primaryPayloadSizeBytes,
        int primaryColorCount,
        int minRed,
        int maxRed,
        int minGreen,
        int maxGreen,
        int minBlue,
        int maxBlue,
        int minAlpha,
        int maxAlpha,
        int averageAlpha,
        int additionalColorSetCount,
        int totalAdditionalColorCount,
        int maxAdditionalColorCount)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(primaryPayloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(primaryColorCount);
        ArgumentOutOfRangeException.ThrowIfNegative(minRed);
        ArgumentOutOfRangeException.ThrowIfNegative(maxRed);
        ArgumentOutOfRangeException.ThrowIfNegative(minGreen);
        ArgumentOutOfRangeException.ThrowIfNegative(maxGreen);
        ArgumentOutOfRangeException.ThrowIfNegative(minBlue);
        ArgumentOutOfRangeException.ThrowIfNegative(maxBlue);
        ArgumentOutOfRangeException.ThrowIfNegative(minAlpha);
        ArgumentOutOfRangeException.ThrowIfNegative(maxAlpha);
        ArgumentOutOfRangeException.ThrowIfNegative(averageAlpha);
        ArgumentOutOfRangeException.ThrowIfNegative(additionalColorSetCount);
        ArgumentOutOfRangeException.ThrowIfNegative(totalAdditionalColorCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxAdditionalColorCount);

        SourcePath = sourcePath;
        Version = version;
        PrimaryPayloadSizeBytes = primaryPayloadSizeBytes;
        PrimaryColorCount = primaryColorCount;
        MinRed = minRed;
        MaxRed = maxRed;
        MinGreen = minGreen;
        MaxGreen = maxGreen;
        MinBlue = minBlue;
        MaxBlue = maxBlue;
        MinAlpha = minAlpha;
        MaxAlpha = maxAlpha;
        AverageAlpha = averageAlpha;
        AdditionalColorSetCount = additionalColorSetCount;
        TotalAdditionalColorCount = totalAdditionalColorCount;
        MaxAdditionalColorCount = maxAdditionalColorCount;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PrimaryPayloadSizeBytes { get; }

    public int PrimaryColorCount { get; }

    public int MinRed { get; }

    public int MaxRed { get; }

    public int MinGreen { get; }

    public int MaxGreen { get; }

    public int MinBlue { get; }

    public int MaxBlue { get; }

    public int MinAlpha { get; }

    public int MaxAlpha { get; }

    public int AverageAlpha { get; }

    public int AdditionalColorSetCount { get; }

    public int TotalAdditionalColorCount { get; }

    public int MaxAdditionalColorCount { get; }
}
