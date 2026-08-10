using System.Numerics;
using WowViewer.Core.Wmo;

namespace WowViewer.Core.Runtime.World.SceneGraph;

/// <summary>
/// The stable identity of a WMO group as supplied to the scene graph adapter.
/// </summary>
public readonly record struct WorldSceneWmoPortalGroupReadModel(int GroupIndex);

/// <summary>
/// The group association of one existing WMO portal reference.
/// </summary>
public readonly record struct WorldSceneWmoPortalReferenceReadModel(
    int ReferenceIndex,
    int PortalIndex,
    int GroupIndex,
    short Side);

/// <summary>
/// Portal geometry and its existing group references, copied from an already-decoded WMO read
/// model. This type does not read client files and does not reinterpret portal coordinates.
/// </summary>
public sealed record WorldSceneWmoPortalReadModel(
    int PortalIndex,
    IReadOnlyList<Vector3>? Vertices,
    Vector3 Normal,
    float PlaneDistance,
    IReadOnlyList<WorldSceneWmoPortalReferenceReadModel>? References);

/// <summary>
/// Geometry retained for a graph portal edge. Geometry validation here only decides whether the
/// read model is safe to promote; clipping and nested view-volume construction remain later work.
/// </summary>
public sealed record WorldScenePortalGeometry(
    int PortalIndex,
    IReadOnlyList<Vector3> Vertices,
    Vector3 Normal,
    float PlaneDistance,
    IReadOnlyList<int> GroupIndices);

public sealed record WorldScenePortalAdapterResult(
    WorldScenePortalGraph Graph,
    IReadOnlyList<WorldScenePortalGeometry> Geometries,
    IReadOnlyList<int> RejectedPortalIndices,
    IReadOnlyList<WorldScenePortalLink> RejectedLinks,
    int DeclaredGroupCount,
    int DeclaredPortalCount,
    int AcceptedLinkCount);

/// <summary>
/// Adapts existing WMO portal read models into graph-owned group adjacency. Format readers and
/// the current WMO renderer remain untouched; malformed data produces graph fallback diagnostics.
/// </summary>
public static class WorldScenePortalAdapter
{
    public static WorldScenePortalAdapterResult Build(
        WmoRenderDocument document,
        string nodeIdPrefix = "wmo")
    {
        ArgumentNullException.ThrowIfNull(document);

        Dictionary<int, List<WorldSceneWmoPortalReferenceReadModel>> referencesByPortal = [];
        foreach (WmoPortalReferenceDetail reference in document.PortalReferences)
        {
            if (!referencesByPortal.TryGetValue(reference.PortalIndex, out List<WorldSceneWmoPortalReferenceReadModel>? references))
            {
                references = [];
                referencesByPortal.Add(reference.PortalIndex, references);
            }

            references.Add(new WorldSceneWmoPortalReferenceReadModel(
                reference.ReferenceIndex,
                reference.PortalIndex,
                reference.GroupIndex,
                reference.Side));
        }

        List<WorldSceneWmoPortalReadModel> portals = [];
        foreach (WmoPortalDetail portal in document.Portals)
        {
            referencesByPortal.Remove(portal.PortalIndex);
            portals.Add(new WorldSceneWmoPortalReadModel(
                portal.PortalIndex,
                portal.Vertices,
                portal.Normal,
                portal.PlaneDistance,
                document.PortalReferences
                    .Where(reference => reference.PortalIndex == portal.PortalIndex)
                    .OrderBy(reference => reference.ReferenceIndex)
                    .Select(reference => new WorldSceneWmoPortalReferenceReadModel(
                        reference.ReferenceIndex,
                        reference.PortalIndex,
                        reference.GroupIndex,
                        reference.Side))
                    .ToArray()));
        }

        // Keep references whose portal geometry is absent visible to the fail-open path instead
        // of silently dropping malformed read-model connectivity.
        foreach ((int portalIndex, List<WorldSceneWmoPortalReferenceReadModel> references) in referencesByPortal.OrderBy(pair => pair.Key))
        {
            portals.Add(new WorldSceneWmoPortalReadModel(
                portalIndex,
                Vertices: null,
                Normal: Vector3.Zero,
                PlaneDistance: 0f,
                references));
        }

        return Build(
            document.Groups
                .Select(group => new WorldSceneWmoPortalGroupReadModel(group.GroupIndex)),
            portals,
            nodeIdPrefix);
    }

    public static WorldScenePortalAdapterResult Build(
        IEnumerable<WorldSceneWmoPortalGroupReadModel> groups,
        IEnumerable<WorldSceneWmoPortalReadModel> portals,
        string nodeIdPrefix = "wmo")
    {
        ArgumentNullException.ThrowIfNull(groups);
        ArgumentNullException.ThrowIfNull(portals);
        ArgumentException.ThrowIfNullOrWhiteSpace(nodeIdPrefix);

        string normalizedPrefix = nodeIdPrefix.TrimEnd('/');
        if (normalizedPrefix.Length == 0)
            throw new ArgumentException("The portal node id prefix must contain a node name.", nameof(nodeIdPrefix));

        WorldSceneWmoPortalGroupReadModel[] orderedGroups = groups
            .OrderBy(group => group.GroupIndex)
            .ToArray();
        string[] nodeIds = orderedGroups
            .Select(group => GroupNodeId(normalizedPrefix, group.GroupIndex))
            .ToArray();

        List<WorldScenePortalLink> links = [];
        List<WorldScenePortalGeometry> geometries = [];
        List<int> rejectedPortalIndices = [];
        HashSet<int> seenPortalIndices = [];
        WorldSceneWmoPortalReadModel[] orderedPortals = portals
            .OrderBy(portal => portal.PortalIndex)
            .ToArray();

        foreach (WorldSceneWmoPortalReadModel portal in orderedPortals)
        {
            if (!seenPortalIndices.Add(portal.PortalIndex))
            {
                rejectedPortalIndices.Add(portal.PortalIndex);
                AddRejectedPortalLink(links, normalizedPrefix, portal.PortalIndex, rejectedPortalIndices.Count);
                continue;
            }

            WorldSceneWmoPortalReferenceReadModel[] references = portal.References?
                .OrderBy(reference => reference.ReferenceIndex)
                .ToArray()
                ?? [];
            int[] groupIndices = references
                .Select(reference => reference.GroupIndex)
                .Distinct()
                .OrderBy(groupIndex => groupIndex)
                .ToArray();

            if (!IsValidGeometry(portal.Vertices, portal.Normal, portal.PlaneDistance))
            {
                rejectedPortalIndices.Add(portal.PortalIndex);
                AddRejectedPortalLink(links, normalizedPrefix, portal.PortalIndex, rejectedPortalIndices.Count);
                continue;
            }

            geometries.Add(new WorldScenePortalGeometry(
                portal.PortalIndex,
                portal.Vertices!,
                portal.Normal,
                portal.PlaneDistance,
                groupIndices));

            for (int sourceIndex = 0; sourceIndex < groupIndices.Length; sourceIndex++)
            {
                for (int destinationIndex = sourceIndex + 1; destinationIndex < groupIndices.Length; destinationIndex++)
                {
                    int sourceGroup = groupIndices[sourceIndex];
                    int destinationGroup = groupIndices[destinationIndex];
                    links.Add(new WorldScenePortalLink(
                        LinkId(normalizedPrefix, portal.PortalIndex, sourceGroup, destinationGroup),
                        GroupNodeId(normalizedPrefix, sourceGroup),
                        GroupNodeId(normalizedPrefix, destinationGroup)));
                    links.Add(new WorldScenePortalLink(
                        LinkId(normalizedPrefix, portal.PortalIndex, destinationGroup, sourceGroup),
                        GroupNodeId(normalizedPrefix, destinationGroup),
                        GroupNodeId(normalizedPrefix, sourceGroup)));
                }
            }
        }

        WorldScenePortalGraphBuildResult graphResult = WorldScenePortalGraph.Build(nodeIds, links);
        return new WorldScenePortalAdapterResult(
            graphResult.Graph,
            geometries,
            rejectedPortalIndices,
            graphResult.RejectedLinks,
            nodeIds.Length,
            orderedPortals.Length,
            graphResult.AcceptedLinkCount);
    }

    private static bool IsValidGeometry(
        IReadOnlyList<Vector3>? vertices,
        Vector3 normal,
        float planeDistance)
    {
        if (vertices is null || vertices.Count < 3 || !IsFinite(normal) || !float.IsFinite(planeDistance))
            return false;

        foreach (Vector3 vertex in vertices)
        {
            if (!IsFinite(vertex))
                return false;
        }

        Vector3 edgeA = vertices[1] - vertices[0];
        for (int index = 2; index < vertices.Count; index++)
        {
            if (Vector3.Cross(edgeA, vertices[index] - vertices[0]).LengthSquared() > 0.000001f)
                return true;
        }

        return false;
    }

    private static bool IsFinite(Vector3 value)
        => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);

    private static void AddRejectedPortalLink(
        ICollection<WorldScenePortalLink> links,
        string prefix,
        int portalIndex,
        int rejectionOrdinal)
        => links.Add(new WorldScenePortalLink(
            $"{prefix}/portal-invalid/{portalIndex:D4}/{rejectionOrdinal:D4}",
            string.Empty,
            string.Empty));

    private static string GroupNodeId(string prefix, int groupIndex)
        => $"{prefix}/group/{groupIndex:D4}";

    private static string LinkId(string prefix, int portalIndex, int sourceGroup, int destinationGroup)
        => $"{prefix}/portal/{portalIndex:D4}/{sourceGroup:D4}-{destinationGroup:D4}";
}
