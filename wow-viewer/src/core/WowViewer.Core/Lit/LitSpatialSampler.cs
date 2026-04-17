using System.Numerics;

namespace WowViewer.Core.Lit;

public static class LitSpatialSampler
{
    public static IReadOnlyList<LitSpatialSampleCandidate> Sample(LitSummary summary, Vector3 position, int maxResults = 8)
    {
        ArgumentNullException.ThrowIfNull(summary);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxResults);

        List<LitSpatialSampleCandidate> candidates = [];
        LitListEntrySummary? defaultEntry = null;

        foreach (LitListEntrySummary entry in summary.Entries)
        {
            if (entry.IsDefaultEntry)
            {
                defaultEntry ??= entry;
                continue;
            }

            float outerRadius = entry.OuterRadius;
            if (outerRadius <= 0f)
                continue;

            float distance = Vector3.Distance(position, entry.Position);
            if (distance > outerRadius)
                continue;

            bool withinCoreRadius = distance <= MathF.Max(entry.LightRadius, 0f);
            float influence = ComputeInfluence(entry, distance);
            candidates.Add(new LitSpatialSampleCandidate(entry, distance, influence, withinCoreRadius, withinOuterRadius: true, isFallbackDefault: false));
        }

        List<LitSpatialSampleCandidate> ordered = candidates
            .OrderByDescending(static candidate => candidate.Influence)
            .ThenBy(static candidate => candidate.Distance)
            .ThenBy(static candidate => candidate.Entry.Index)
            .Take(maxResults)
            .ToList();

        if (ordered.Count == 0 && defaultEntry is not null)
            ordered.Add(new LitSpatialSampleCandidate(defaultEntry, 0f, 0f, withinCoreRadius: true, withinOuterRadius: true, isFallbackDefault: true));

        return ordered;
    }

    private static float ComputeInfluence(LitListEntrySummary entry, float distance)
    {
        float coreRadius = MathF.Max(entry.LightRadius, 0f);
        if (distance <= coreRadius)
            return 1f;

        float outerRadius = entry.OuterRadius;
        if (outerRadius <= coreRadius)
            return 0f;

        float falloff = 1f - ((distance - coreRadius) / (outerRadius - coreRadius));
        return Math.Clamp(falloff, 0f, 1f);
    }
}