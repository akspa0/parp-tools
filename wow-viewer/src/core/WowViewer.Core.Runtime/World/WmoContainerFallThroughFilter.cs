using System.Numerics;

namespace WowViewer.Core.Runtime.World;

/// <summary>
/// Spec 211: Determines when a ray-intersected WMO is acting as an enclosing container
/// for interior objects (MDX, WMO doodads, or nested WMOs) along the same ray,
/// allowing clicks to fall through into the interior instead of locking onto the outer building envelope.
/// </summary>
public static class WmoContainerFallThroughFilter
{
    public readonly record struct CandidateObject(
        int Id,
        bool IsWmo,
        Vector3 BoundsMin,
        Vector3 BoundsMax,
        Vector3 SelectionPoint);

    public static bool IsPointInsideAabb(Vector3 point, Vector3 min, Vector3 max, float margin = 0.5f)
    {
        float minX = MathF.Min(min.X, max.X) - margin;
        float maxX = MathF.Max(min.X, max.X) + margin;
        float minY = MathF.Min(min.Y, max.Y) - margin;
        float maxY = MathF.Max(min.Y, max.Y) + margin;
        float minZ = MathF.Min(min.Z, max.Z) - margin;
        float maxZ = MathF.Max(min.Z, max.Z) + margin;

        return point.X >= minX && point.X <= maxX
            && point.Y >= minY && point.Y <= maxY
            && point.Z >= minZ && point.Z <= maxZ;
    }

    /// <summary>
    /// Filters candidate objects along a ray: if any candidate's selection point is inside
    /// a WMO candidate's bounding box, the enclosing WMO is deemed a container and removed
    /// so the interior object takes priority.
    /// If all candidates are removed or no interior objects exist, the original list is preserved.
    /// </summary>
    public static IReadOnlyList<CandidateObject> ApplyFallThrough(IReadOnlyList<CandidateObject> candidates)
    {
        if (candidates == null || candidates.Count <= 1)
            return candidates ?? [];

        var containerWmoIds = new HashSet<int>();
        for (int i = 0; i < candidates.Count; i++)
        {
            var wmo = candidates[i];
            if (!wmo.IsWmo)
                continue;

            for (int j = 0; j < candidates.Count; j++)
            {
                if (i == j)
                    continue;

                var other = candidates[j];
                if (IsPointInsideAabb(other.SelectionPoint, wmo.BoundsMin, wmo.BoundsMax))
                {
                    containerWmoIds.Add(wmo.Id);
                    break;
                }
            }
        }

        if (containerWmoIds.Count == 0)
            return candidates;

        var filtered = new List<CandidateObject>(candidates.Count);
        for (int i = 0; i < candidates.Count; i++)
        {
            var c = candidates[i];
            if (!c.IsWmo || !containerWmoIds.Contains(c.Id))
                filtered.Add(c);
        }

        return filtered.Count > 0 ? filtered : candidates;
    }
}
