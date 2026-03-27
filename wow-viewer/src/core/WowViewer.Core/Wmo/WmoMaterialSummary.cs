namespace WowViewer.Core.Wmo;

public sealed class WmoMaterialSummary
{
    public WmoMaterialSummary(
        string sourcePath,
        uint? version,
        int payloadSizeBytes,
        int entrySizeBytes,
        int entryCount,
        int distinctShaderCount,
        int distinctBlendModeCount,
        int nonZeroFlagCount,
        int maxTexture1Offset,
        int maxTexture2Offset,
        int maxTexture3Offset)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(payloadSizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entrySizeBytes);
        ArgumentOutOfRangeException.ThrowIfNegative(entryCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctShaderCount);
        ArgumentOutOfRangeException.ThrowIfNegative(distinctBlendModeCount);
        ArgumentOutOfRangeException.ThrowIfNegative(nonZeroFlagCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxTexture1Offset);
        ArgumentOutOfRangeException.ThrowIfNegative(maxTexture2Offset);
        ArgumentOutOfRangeException.ThrowIfNegative(maxTexture3Offset);

        SourcePath = sourcePath;
        Version = version;
        PayloadSizeBytes = payloadSizeBytes;
        EntrySizeBytes = entrySizeBytes;
        EntryCount = entryCount;
        DistinctShaderCount = distinctShaderCount;
        DistinctBlendModeCount = distinctBlendModeCount;
        NonZeroFlagCount = nonZeroFlagCount;
        MaxTexture1Offset = maxTexture1Offset;
        MaxTexture2Offset = maxTexture2Offset;
        MaxTexture3Offset = maxTexture3Offset;
    }

    public string SourcePath { get; }

    public uint? Version { get; }

    public int PayloadSizeBytes { get; }

    public int EntrySizeBytes { get; }

    public int EntryCount { get; }

    public int DistinctShaderCount { get; }

    public int DistinctBlendModeCount { get; }

    public int NonZeroFlagCount { get; }

    public int MaxTexture1Offset { get; }

    public int MaxTexture2Offset { get; }

    public int MaxTexture3Offset { get; }
}
