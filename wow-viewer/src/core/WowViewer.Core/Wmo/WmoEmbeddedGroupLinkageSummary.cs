namespace WowViewer.Core.Wmo;

public sealed class WmoEmbeddedGroupLinkageSummary
{
    public WmoEmbeddedGroupLinkageSummary(
        string sourcePath,
        uint? version,
        int groupInfoCount,
        int embeddedGroupCount,
        int coveredPairCount,
        int missingEmbeddedGroupCount,
        int extraEmbeddedGroupCount,
        int flagMatchCount,
        int boundsMatchCount,
        float maxBoundsDelta)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(sourcePath);
        ArgumentOutOfRangeException.ThrowIfNegative(groupInfoCount);
        ArgumentOutOfRangeException.ThrowIfNegative(embeddedGroupCount);
        ArgumentOutOfRangeException.ThrowIfNegative(coveredPairCount);
        ArgumentOutOfRangeException.ThrowIfNegative(missingEmbeddedGroupCount);
        ArgumentOutOfRangeException.ThrowIfNegative(extraEmbeddedGroupCount);
        ArgumentOutOfRangeException.ThrowIfNegative(flagMatchCount);
        ArgumentOutOfRangeException.ThrowIfNegative(boundsMatchCount);
        ArgumentOutOfRangeException.ThrowIfNegative(maxBoundsDelta);

        SourcePath = sourcePath;
        Version = version;
        GroupInfoCount = groupInfoCount;
        EmbeddedGroupCount = embeddedGroupCount;
        CoveredPairCount = coveredPairCount;
        MissingEmbeddedGroupCount = missingEmbeddedGroupCount;
        ExtraEmbeddedGroupCount = extraEmbeddedGroupCount;
        FlagMatchCount = flagMatchCount;
        BoundsMatchCount = boundsMatchCount;
        MaxBoundsDelta = maxBoundsDelta;
    }

    public string SourcePath { get; }
    public uint? Version { get; }
    public int GroupInfoCount { get; }
    public int EmbeddedGroupCount { get; }
    public int CoveredPairCount { get; }
    public int MissingEmbeddedGroupCount { get; }
    public int ExtraEmbeddedGroupCount { get; }
    public int FlagMatchCount { get; }
    public int BoundsMatchCount { get; }
    public float MaxBoundsDelta { get; }
}
