namespace WowViewer.Core.Lit;

public sealed class LitSpatialSampleCandidate
{
    public LitSpatialSampleCandidate(
        LitListEntrySummary entry,
        float distance,
        float influence,
        bool withinCoreRadius,
        bool withinOuterRadius,
        bool isFallbackDefault)
    {
        ArgumentNullException.ThrowIfNull(entry);

        Entry = entry;
        Distance = distance;
        Influence = influence;
        WithinCoreRadius = withinCoreRadius;
        WithinOuterRadius = withinOuterRadius;
        IsFallbackDefault = isFallbackDefault;
    }

    public LitListEntrySummary Entry { get; }

    public float Distance { get; }

    public float Influence { get; }

    public bool WithinCoreRadius { get; }

    public bool WithinOuterRadius { get; }

    public bool IsFallbackDefault { get; }
}