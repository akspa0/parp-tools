namespace WowViewer.Core.Wmo;

/// <summary>
/// Resolution and range evaluation for WMO doodad sets (Spec 223 US3).
/// Resolves active sets, bounds, and doodad placement memberships without UI or OpenGL dependencies.
/// </summary>
public static class WmoDoodadSetResolver
{
    /// <summary>
    /// Resolves the effective doodad set index from a requested index, clamping to 0 if out of range,
    /// or returning -1 if no doodad sets are available.
    /// </summary>
    public static int ResolveActiveSetIndex(int requestedIndex, int setCount)
    {
        if (setCount <= 0)
            return -1;
        if (requestedIndex < 0 || requestedIndex >= setCount)
            return 0;
        return requestedIndex;
    }

    /// <summary>
    /// Checks whether a given placement index falls within the specified doodad set range.
    /// </summary>
    public static bool IsPlacementInSet(int placementIndex, int startIndex, int count)
    {
        if (placementIndex < 0 || count <= 0 || startIndex < 0)
            return false;
        return placementIndex >= startIndex && placementIndex < startIndex + count;
    }

    /// <summary>
    /// Checks whether a given placement index falls within the specified doodad set.
    /// </summary>
    public static bool IsPlacementInSet(int placementIndex, WmoDoodadSetDetail? set)
    {
        if (set == null)
            return false;
        return IsPlacementInSet(placementIndex, set.StartIndex, set.Count);
    }

    /// <summary>
    /// Computes the valid placement index range [start, endExclusive) for a doodad set,
    /// bounded by the total number of placements defined in the WMO.
    /// </summary>
    public static (int Start, int EndExclusive) GetPlacementRange(int startIndex, int count, int totalPlacementCount)
    {
        if (totalPlacementCount <= 0 || count <= 0 || startIndex < 0 || startIndex >= totalPlacementCount)
            return (0, 0);

        int start = startIndex;
        int endExclusive = Math.Min(startIndex + count, totalPlacementCount);
        return (start, endExclusive);
    }

    /// <summary>
    /// Computes the valid placement index range [start, endExclusive) for a doodad set detail,
    /// bounded by the total number of placements defined in the WMO.
    /// </summary>
    public static (int Start, int EndExclusive) GetPlacementRange(WmoDoodadSetDetail? set, int totalPlacementCount)
    {
        if (set == null)
            return (0, 0);
        return GetPlacementRange(set.StartIndex, set.Count, totalPlacementCount);
    }

    /// <summary>
    /// Resolves active doodad set summary information from strongly-typed set details.
    /// </summary>
    public static WmoActiveDoodadSetResolution ResolveActiveSet(
        IReadOnlyList<WmoDoodadSetDetail>? sets,
        int requestedIndex,
        int totalPlacementCount)
    {
        if (sets == null || sets.Count == 0)
        {
            return new WmoActiveDoodadSetResolution(
                ActiveIndex: -1,
                Name: string.Empty,
                StartIndex: 0,
                Count: 0,
                RangeEndExclusive: 0,
                TotalSets: 0,
                IsClamped: false,
                IsValid: false);
        }

        int resolvedIndex = ResolveActiveSetIndex(requestedIndex, sets.Count);
        bool isClamped = resolvedIndex != requestedIndex;
        WmoDoodadSetDetail set = sets[resolvedIndex];
        var (start, endExclusive) = GetPlacementRange(set.StartIndex, set.Count, totalPlacementCount);

        return new WmoActiveDoodadSetResolution(
            ActiveIndex: resolvedIndex,
            Name: set.Name,
            StartIndex: start,
            Count: Math.Max(0, endExclusive - start),
            RangeEndExclusive: endExclusive,
            TotalSets: sets.Count,
            IsClamped: isClamped,
            IsValid: true);
    }

    /// <summary>
    /// Resolves active doodad set summary information from lightweight name/start/count tuples.
    /// </summary>
    public static WmoActiveDoodadSetResolution ResolveActiveSet(
        IReadOnlyList<(string Name, int StartIndex, int Count)>? sets,
        int requestedIndex,
        int totalPlacementCount)
    {
        if (sets == null || sets.Count == 0)
        {
            return new WmoActiveDoodadSetResolution(
                ActiveIndex: -1,
                Name: string.Empty,
                StartIndex: 0,
                Count: 0,
                RangeEndExclusive: 0,
                TotalSets: 0,
                IsClamped: false,
                IsValid: false);
        }

        int resolvedIndex = ResolveActiveSetIndex(requestedIndex, sets.Count);
        bool isClamped = resolvedIndex != requestedIndex;
        var (name, startIndex, count) = sets[resolvedIndex];
        var (start, endExclusive) = GetPlacementRange(startIndex, count, totalPlacementCount);

        return new WmoActiveDoodadSetResolution(
            ActiveIndex: resolvedIndex,
            Name: name,
            StartIndex: start,
            Count: Math.Max(0, endExclusive - start),
            RangeEndExclusive: endExclusive,
            TotalSets: sets.Count,
            IsClamped: isClamped,
            IsValid: true);
    }

    /// <summary>
    /// Formats a concise human-readable summary for the resolved doodad set.
    /// </summary>
    public static string FormatSetSummary(WmoActiveDoodadSetResolution resolution)
    {
        if (!resolution.IsValid)
            return "No doodad sets";

        string clampMarker = resolution.IsClamped ? " (clamped)" : string.Empty;
        if (resolution.Count == 0)
            return $"[{resolution.ActiveIndex}] \"{resolution.Name}\": 0 doodads{clampMarker}";

        return $"[{resolution.ActiveIndex}] \"{resolution.Name}\": {resolution.Count} doodads ({resolution.StartIndex}..{resolution.RangeEndExclusive - 1}){clampMarker}";
    }
}

/// <summary>
/// Immutable resolution result for an active WMO doodad set.
/// </summary>
public sealed record WmoActiveDoodadSetResolution(
    int ActiveIndex,
    string Name,
    int StartIndex,
    int Count,
    int RangeEndExclusive,
    int TotalSets,
    bool IsClamped,
    bool IsValid);
