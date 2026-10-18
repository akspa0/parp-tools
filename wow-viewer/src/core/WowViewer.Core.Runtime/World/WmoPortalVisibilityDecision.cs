using System.Numerics;
using WowViewer.Core.Runtime.World.SceneGraph;

namespace WowViewer.Core.Runtime.World;

public enum WmoPortalVisibilityMode
{
    Exterior,
    Interior,
    ConservativeFallback,
}

public readonly record struct WmoPortalVisibilityGroup(
    int GroupIndex,
    uint Flags,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    string? Name = null,
    bool HasPortals = true)
{
    public const uint ExteriorFlag = 0x8;
    public const uint UsesExteriorLightingFlag = 0x40;
    public const uint IndoorFlag = 0x2000;
    public const uint AlwaysVisibleFlag = 0x10000;
    public const uint ShowSkyboxFlag = 0x40000;
    public const uint AntiportalFlag = 0x4000080;
    public const uint OccluderFlag = 0x4000000;

    public bool IsAntiportal => (Flags & OccluderFlag) != 0
        || (!string.IsNullOrEmpty(Name) && (
            Name.Contains("antiportal", StringComparison.OrdinalIgnoreCase) ||
            Name.Contains("anti_portal", StringComparison.OrdinalIgnoreCase) ||
            Name.Contains("collision", StringComparison.OrdinalIgnoreCase) ||
            Name.Contains("invis_wall", StringComparison.OrdinalIgnoreCase) ||
            Name.Contains("invisible", StringComparison.OrdinalIgnoreCase)));

    public bool IsExterior => !IsAntiportal && (
        (Flags & (ExteriorFlag | UsesExteriorLightingFlag | AlwaysVisibleFlag | ShowSkyboxFlag)) != 0
        || (!HasPortals && (Flags & IndoorFlag) == 0));

    public bool IsInterior => !IsExterior && !IsAntiportal;
}

public readonly record struct WmoPortalVisibilityReference(
    int GroupIndex,
    short Side);

public sealed record WmoPortalVisibilityPortal(
    int PortalIndex,
    IReadOnlyList<Vector3> Vertices,
    Vector3 Normal,
    float PlaneDistance,
    IReadOnlyList<WmoPortalVisibilityReference> References);

public sealed record WmoPortalVisibilityDiagnostics
{
    public WmoPortalVisibilityMode Mode { get; internal set; }

    public int SourceGroupIndex { get; internal set; } = -1;

    public int VisitedGroupCount { get; internal set; }

    public int AdmittedGroupCount { get; internal set; }

    public int TestedPortalCount { get; internal set; }

    public int RejectedPortalCount { get; internal set; }

    public int MaxDepthReached { get; internal set; }

    public bool UsedPortalClip { get; internal set; }

    public string? FallbackReason { get; internal set; }

    public static WmoPortalVisibilityDiagnostics CreateFallback(string reason)
        => new()
        {
            Mode = WmoPortalVisibilityMode.ConservativeFallback,
            FallbackReason = reason,
        };
}

public sealed record WmoPortalVisibilityDecision(
    IReadOnlyList<int> VisibleGroupIndices,
    WmoPortalVisibilityDiagnostics Diagnostics);

/// <summary>
/// Computes one bounded, fail-open group visibility decision from already-decoded WMO data.
/// The portal-volume construction follows the 0.5.3 Ghidra evidence without depending on the
/// original client implementation.
/// </summary>
public static class WmoPortalVisibilityEvaluator
{
    private const uint ExteriorGroupFlag = 0x8;
    private const uint AlwaysVisibleGroupFlag = 0x10000;
    private const float BoundsPadding = 0.1f;
    private const int DefaultMaxVisitsPerGroup = 4;

    public static WmoPortalVisibilityDecision Evaluate(
        IReadOnlyList<WmoPortalVisibilityGroup> groups,
        IReadOnlyList<WmoPortalVisibilityPortal> portals,
        Vector3 cameraLocalPosition,
        Func<int, bool>? isFrustumVisible = null,
        int interiorMaximumDepth = 4,
        int exteriorMaximumDepth = 1,
        int maxVisitsPerGroup = DefaultMaxVisitsPerGroup)
    {
        ArgumentNullException.ThrowIfNull(groups);
        ArgumentNullException.ThrowIfNull(portals);
        if (interiorMaximumDepth < 0)
            throw new ArgumentOutOfRangeException(nameof(interiorMaximumDepth));
        if (exteriorMaximumDepth < 0)
            throw new ArgumentOutOfRangeException(nameof(exteriorMaximumDepth));
        if (maxVisitsPerGroup < 1)
            throw new ArgumentOutOfRangeException(nameof(maxVisitsPerGroup));

        WmoPortalVisibilityDiagnostics diagnostics = new();
        int[] allGroups = groups
            .Select(static group => group.GroupIndex)
            .Distinct()
            .OrderBy(static groupIndex => groupIndex)
            .ToArray();

        string? groupError = null;
        string? portalError = null;

        if (!IsFinite(cameraLocalPosition)
            || !TryBuildGroupMap(groups, out Dictionary<int, WmoPortalVisibilityGroup> groupMap, out groupError)
            || !TryBuildPortalMap(portals, groupMap, out Dictionary<int, WmoPortalVisibilityPortal> portalMap, out portalError))
        {
            return Fallback(allGroups, diagnostics, groupError ?? portalError ?? "visibility_input_invalid");
        }

        if (allGroups.Length == 0)
        {
            diagnostics.Mode = WmoPortalVisibilityMode.ConservativeFallback;
            diagnostics.FallbackReason = "groups_absent";
            return new WmoPortalVisibilityDecision(Array.Empty<int>(), diagnostics);
        }

        if (portals.Count == 0)
            return Fallback(allGroups, diagnostics, "portal_data_absent");

        Dictionary<int, List<(int PortalIndex, int DestinationGroup, short SourceSide, short DestinationSide)>> adjacency =
            BuildAdjacency(portalMap);
        if (adjacency.Count == 0)
            return Fallback(allGroups, diagnostics, "portal_edges_absent");
        List<int> containingInteriorGroups = FindContainingInteriorGroups(groupMap, portalMap, adjacency, cameraLocalPosition);
        bool isInterior = containingInteriorGroups.Count > 0;
        int sourceGroupIndex = isInterior ? containingInteriorGroups[0] : -1;
        diagnostics.Mode = isInterior ? WmoPortalVisibilityMode.Interior : WmoPortalVisibilityMode.Exterior;
        diagnostics.SourceGroupIndex = sourceGroupIndex;

        var visible = new HashSet<int>();
        var queue = new Queue<(int GroupIndex, int FromGroupIndex, WorldScenePortalViewVolume Volume, int Depth)>();
        var visits = new Dictionary<int, int>();

        WorldScenePortalViewVolume rootVolume = WorldScenePortalViewVolume.CreateRoot();
        if (isInterior)
        {
            foreach (int groupIndex in containingInteriorGroups)
            {
                Enqueue(groupIndex, -1, rootVolume, 0);
            }
        }
        else
        {
            bool hasExteriorGroups = false;
            foreach (WmoPortalVisibilityGroup group in groupMap.Values.OrderBy(static group => group.GroupIndex))
            {
                if (group.IsExterior)
                {
                    hasExteriorGroups = true;
                    bool frustumVisible = isFrustumVisible?.Invoke(group.GroupIndex) ?? true;
                    if (frustumVisible || (group.Flags & WmoPortalVisibilityGroup.AlwaysVisibleFlag) != 0)
                        Enqueue(group.GroupIndex, -1, rootVolume, 0);
                }
            }

            if (queue.Count == 0)
            {
                if (hasExteriorGroups)
                {
                    // Exterior groups exist, but none intersect the camera frustum.
                    return new WmoPortalVisibilityDecision(Array.Empty<int>(), diagnostics with { AdmittedGroupCount = 0 });
                }

                return Fallback(allGroups, diagnostics, "exterior_seed_group_unknown");
            }
        }

        int maximumDepth = isInterior ? interiorMaximumDepth : exteriorMaximumDepth;
        while (queue.Count > 0)
        {
            (int groupIndex, int fromGroupIndex, WorldScenePortalViewVolume volume, int depth) = queue.Dequeue();
            diagnostics.VisitedGroupCount++;
            diagnostics.MaxDepthReached = Math.Max(diagnostics.MaxDepthReached, depth);
            visible.Add(groupIndex);

            if (depth >= maximumDepth)
                continue;
            if (!adjacency.TryGetValue(groupIndex, out List<(int PortalIndex, int DestinationGroup, short SourceSide, short DestinationSide)>? edges))
                continue;

            foreach ((int portalIndex, int destinationGroup, short sourceSide, short destinationSide) in edges)
            {
                // Native 3.3.5 MapObj::RRenderThruPortals: skip portal if destination is fromGroupIdx (do not backtrack).
                if (destinationGroup == fromGroupIndex)
                    continue;

                if (!portalMap.TryGetValue(portalIndex, out WmoPortalVisibilityPortal? portal)
                    || !groupMap.ContainsKey(destinationGroup))
                {
                    diagnostics.RejectedPortalCount++;
                    return Fallback(allGroups, diagnostics, "portal_edge_invalid");
                }

                diagnostics.TestedPortalCount++;
                if (depth == 0 && !IsOnPortalSide(portal, cameraLocalPosition, sourceSide))
                {
                    diagnostics.RejectedPortalCount++;
                    continue;
                }

                WorldScenePortalGeometry geometry = new(
                    portal.PortalIndex,
                    portal.Vertices,
                    portal.Normal,
                    portal.PlaneDistance,
                    portal.References.Select(static reference => reference.GroupIndex).Distinct().ToArray(),
                    portal.References
                        .Select(static reference => new WorldScenePortalGroupSide(reference.GroupIndex, reference.Side))
                        .ToArray());
                WorldScenePortalViewVolumeBuildResult child = WorldScenePortalViewVolumeBuilder.BuildChild(
                    volume,
                    geometry,
                    GroupNodeId(groupIndex),
                    GroupNodeId(destinationGroup),
                    cameraLocalPosition,
                    maximumDepth);
                if (child.FallbackRequired || child.Volume is null)
                {
                    diagnostics.RejectedPortalCount++;
                    return Fallback(allGroups, diagnostics, child.FallbackReason ?? "portal_volume_fallback");
                }

                diagnostics.UsedPortalClip = true;
                WmoPortalVisibilityGroup destination = groupMap[destinationGroup];
                if (!child.Volume.IntersectsBounds(destination.BoundsMin, destination.BoundsMax))
                    continue;

                if (visits.GetValueOrDefault(destinationGroup) >= maxVisitsPerGroup)
                {
                    diagnostics.RejectedPortalCount++;
                    return Fallback(allGroups, diagnostics, "portal_visit_capacity_reached");
                }

                Enqueue(destinationGroup, groupIndex, child.Volume, depth + 1);
            }
        }

        return new WmoPortalVisibilityDecision(
            visible.OrderBy(static groupIndex => groupIndex).ToArray(),
            diagnostics with { AdmittedGroupCount = visible.Count });

        void Enqueue(int groupIndex, int fromGroupIndex, WorldScenePortalViewVolume volume, int depth)
        {
            if (!groupMap.ContainsKey(groupIndex))
                return;

            int visitCount = visits.GetValueOrDefault(groupIndex);
            if (visitCount >= maxVisitsPerGroup)
                return;

            visits[groupIndex] = visitCount + 1;
            queue.Enqueue((groupIndex, fromGroupIndex, volume, depth));
        }
    }

    private static Dictionary<int, List<(int PortalIndex, int DestinationGroup, short SourceSide, short DestinationSide)>> BuildAdjacency(
        IReadOnlyDictionary<int, WmoPortalVisibilityPortal> portals)
    {
        var adjacency = new Dictionary<int, List<(int PortalIndex, int DestinationGroup, short SourceSide, short DestinationSide)>>();
        foreach (WmoPortalVisibilityPortal portal in portals.Values)
        {
            WmoPortalVisibilityReference[] references = portal.References
                .Where(static reference => reference.Side != 0)
                .Distinct()
                .ToArray();
            foreach (WmoPortalVisibilityReference source in references)
            {
                if (!adjacency.TryGetValue(source.GroupIndex, out List<(int, int, short, short)>? edges))
                {
                    edges = [];
                    adjacency.Add(source.GroupIndex, edges);
                }

                foreach (WmoPortalVisibilityReference destination in references)
                {
                    if (source.GroupIndex == destination.GroupIndex)
                        continue;

                    edges.Add((portal.PortalIndex, destination.GroupIndex, source.Side, destination.Side));
                }
            }
        }

        return adjacency;
    }

    private static bool IsOnPortalSide(
        WmoPortalVisibilityPortal portal,
        Vector3 cameraPosition,
        short side)
    {
        float signedDistance = Vector3.Dot(portal.Normal, cameraPosition) + portal.PlaneDistance;
        return side > 0 ? signedDistance >= -1.0f : signedDistance <= 1.0f;
    }

    private static List<int> FindContainingInteriorGroups(
        IReadOnlyDictionary<int, WmoPortalVisibilityGroup> groupMap,
        IReadOnlyDictionary<int, WmoPortalVisibilityPortal> portalMap,
        IReadOnlyDictionary<int, List<(int PortalIndex, int DestinationGroup, short SourceSide, short DestinationSide)>> adjacency,
        Vector3 cameraLocalPosition)
    {
        var candidates = new List<int>();
        foreach (WmoPortalVisibilityGroup group in groupMap.Values)
        {
            if (!group.IsInterior)
                continue;

            if (!ContainsExpanded(cameraLocalPosition, group.BoundsMin, group.BoundsMax))
                continue;

            // If this interior group connects to exterior groups via portals, verify that the
            // camera is actually on the interior side of the portal. If the camera is on the
            // exterior side of an exterior-facing portal, it is outdoors (e.g. standing in the
            // street outside a shop window/door), not inside this room.
            if (adjacency.TryGetValue(group.GroupIndex, out var edges))
            {
                bool isOutsideExteriorPortal = false;
                foreach (var edge in edges)
                {
                    if (groupMap.TryGetValue(edge.DestinationGroup, out var destGroup) && destGroup.IsExterior)
                    {
                        if (portalMap.TryGetValue(edge.PortalIndex, out var portal))
                        {
                            if (!IsOnPortalSide(portal, cameraLocalPosition, edge.SourceSide))
                            {
                                isOutsideExteriorPortal = true;
                                break;
                            }
                        }
                    }
                }

                if (isOutsideExteriorPortal)
                    continue;
            }

            candidates.Add(group.GroupIndex);
        }

        return candidates
            .OrderBy(groupIndex => DistanceSquaredPointToAabb(cameraLocalPosition, groupMap[groupIndex].BoundsMin, groupMap[groupIndex].BoundsMax))
            .ThenBy(static groupIndex => groupIndex)
            .ToList();
    }

    private static bool TryBuildGroupMap(
        IReadOnlyList<WmoPortalVisibilityGroup> groups,
        out Dictionary<int, WmoPortalVisibilityGroup> groupMap,
        out string? error)
    {
        groupMap = new Dictionary<int, WmoPortalVisibilityGroup>();
        error = null;
        foreach (WmoPortalVisibilityGroup group in groups)
        {
            if (!groupMap.TryAdd(group.GroupIndex, group))
            {
                error = "duplicate_group_index";
                return false;
            }

            if (!IsFiniteBounds(group.BoundsMin, group.BoundsMax))
            {
                error = "group_bounds_invalid";
                return false;
            }
        }

        return true;
    }

    private static bool TryBuildPortalMap(
        IReadOnlyList<WmoPortalVisibilityPortal> portals,
        IReadOnlyDictionary<int, WmoPortalVisibilityGroup> groups,
        out Dictionary<int, WmoPortalVisibilityPortal> portalMap,
        out string? error)
    {
        portalMap = new Dictionary<int, WmoPortalVisibilityPortal>();
        error = null;
        foreach (WmoPortalVisibilityPortal portal in portals)
        {
            if (!portalMap.TryAdd(portal.PortalIndex, portal))
            {
                error = "duplicate_portal_index";
                return false;
            }

            if (portal.Vertices is null
                || portal.Vertices.Count < 3
                || !IsFinite(portal.Normal)
                || !float.IsFinite(portal.PlaneDistance)
                || portal.Normal.LengthSquared() <= 0.000001f
                || portal.Vertices.Any(static vertex => !IsFinite(vertex))
                || !HasArea(portal.Vertices))
            {
                error = "portal_geometry_invalid";
                return false;
            }

            foreach (WmoPortalVisibilityReference reference in portal.References)
            {
                if (!groups.ContainsKey(reference.GroupIndex) || reference.Side == 0)
                {
                    error = "portal_reference_invalid";
                    return false;
                }
            }
        }

        return true;
    }

    private static WmoPortalVisibilityDecision Fallback(
        IReadOnlyList<int> allGroups,
        WmoPortalVisibilityDiagnostics diagnostics,
        string reason)
    {
        diagnostics.Mode = WmoPortalVisibilityMode.ConservativeFallback;
        diagnostics.FallbackReason = reason;
        return new WmoPortalVisibilityDecision(allGroups, diagnostics);
    }

    private static string GroupNodeId(int groupIndex) => $"wmo/group/{groupIndex:D4}";

    private static bool HasArea(IReadOnlyList<Vector3> vertices)
    {
        Vector3 edgeA = vertices[1] - vertices[0];
        for (int index = 2; index < vertices.Count; index++)
        {
            if (Vector3.Cross(edgeA, vertices[index] - vertices[0]).LengthSquared() > 0.000001f)
                return true;
        }

        return false;
    }

    private static bool ContainsExpanded(Vector3 point, Vector3 min, Vector3 max)
        => point.X >= MathF.Min(min.X, max.X) - BoundsPadding
            && point.X <= MathF.Max(min.X, max.X) + BoundsPadding
            && point.Y >= MathF.Min(min.Y, max.Y) - BoundsPadding
            && point.Y <= MathF.Max(min.Y, max.Y) + BoundsPadding
            && point.Z >= MathF.Min(min.Z, max.Z) - BoundsPadding
            && point.Z <= MathF.Max(min.Z, max.Z) + BoundsPadding;

    private static float DistanceSquaredPointToAabb(Vector3 point, Vector3 min, Vector3 max)
    {
        float dx = point.X < min.X ? min.X - point.X : point.X > max.X ? point.X - max.X : 0f;
        float dy = point.Y < min.Y ? min.Y - point.Y : point.Y > max.Y ? point.Y - max.Y : 0f;
        float dz = point.Z < min.Z ? min.Z - point.Z : point.Z > max.Z ? point.Z - max.Z : 0f;
        return dx * dx + dy * dy + dz * dz;
    }

    private static bool IsFiniteBounds(Vector3 min, Vector3 max)
        => IsFinite(min) && IsFinite(max)
            && min.X <= max.X && min.Y <= max.Y && min.Z <= max.Z;

    private static bool IsFinite(Vector3 value)
        => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
}
