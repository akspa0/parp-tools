using System;
using System.Collections.Generic;

namespace WowViewer.Core.PM4.Matching;

/// <summary>
/// Marks shape-library assets that cannot be loaded. Unhealthy entries stay visible in the
/// catalog but are excluded from automatic proposals.
/// </summary>
public static class Pm4ShapeLibraryHealth
{
    public static IReadOnlyList<Pm4ShapeLibraryHealthEntry> Check(
        IEnumerable<string> modelPaths,
        Func<string, bool> exists)
    {
        ArgumentNullException.ThrowIfNull(modelPaths);
        ArgumentNullException.ThrowIfNull(exists);

        var entries = new List<Pm4ShapeLibraryHealthEntry>();
        foreach (string path in modelPaths)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                entries.Add(new Pm4ShapeLibraryHealthEntry(path ?? string.Empty, false, "empty path"));
                continue;
            }

            bool healthy;
            try
            {
                healthy = exists(path);
            }
            catch (Exception ex)
            {
                entries.Add(new Pm4ShapeLibraryHealthEntry(path, false, ex.Message));
                continue;
            }

            entries.Add(new Pm4ShapeLibraryHealthEntry(
                path,
                healthy,
                healthy ? null : "asset not found or unreadable"));
        }

        return entries;
    }

    public static IReadOnlyList<Pm4ModelCandidate> FilterHealthy(
        IReadOnlyList<Pm4ModelCandidate> candidates,
        IReadOnlyDictionary<string, bool> healthByPath)
    {
        ArgumentNullException.ThrowIfNull(candidates);
        ArgumentNullException.ThrowIfNull(healthByPath);

        var healthy = new List<Pm4ModelCandidate>(candidates.Count);
        foreach (Pm4ModelCandidate candidate in candidates)
        {
            if (healthByPath.TryGetValue(candidate.ModelPath, out bool isHealthy) && isHealthy)
                healthy.Add(candidate);
        }

        return healthy;
    }

    public static Pm4ShapeLibraryEntryHealth Check(
        Pm4ModelShapeCandidate candidate,
        Func<string, (bool Exists, bool Readable, string? Reason)> probe)
    {
        ArgumentNullException.ThrowIfNull(candidate);
        ArgumentNullException.ThrowIfNull(probe);

        if (string.IsNullOrWhiteSpace(candidate.ModelPath))
            return new Pm4ShapeLibraryEntryHealth(string.Empty, false, "Empty model path.");

        try
        {
            var (exists, readable, reason) = probe(candidate.ModelPath);
            if (!exists)
                return new Pm4ShapeLibraryEntryHealth(candidate.ModelPath, false, reason ?? "Asset not found.");
            if (!readable)
                return new Pm4ShapeLibraryEntryHealth(candidate.ModelPath, false, reason ?? "Asset unreadable.");
            return new Pm4ShapeLibraryEntryHealth(candidate.ModelPath, true, null);
        }
        catch (Exception ex)
        {
            return new Pm4ShapeLibraryEntryHealth(candidate.ModelPath, false, ex.Message);
        }
    }
}
