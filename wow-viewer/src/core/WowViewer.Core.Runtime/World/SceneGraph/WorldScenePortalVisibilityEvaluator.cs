using System.Numerics;

namespace WowViewer.Core.Runtime.World.SceneGraph;

public sealed class WorldScenePortalVisibilityDiagnostics
{
    public bool FallbackRequired { get; internal set; }

    public string? FallbackReason { get; internal set; }

    public int SourceGroupIndex { get; internal set; } = -1;

    public int VisitedGroupCount { get; internal set; }

    public int TestedPortalCount { get; internal set; }

    public int RejectedPortalCount { get; internal set; }

    public int MaxDepthReached { get; internal set; }
}

public sealed record WorldScenePortalVisibilityResult(
    IReadOnlyList<string> VisibleNodeIds,
    WorldScenePortalVisibilityDiagnostics Diagnostics);

/// <summary>
/// Evaluates the graph-side reachable WMO groups for one placement. This is a diagnostic and
/// traversal contract; the legacy renderer remains responsible for final group submission.
/// </summary>
public static class WorldScenePortalVisibilityEvaluator
{
    public static WorldScenePortalVisibilityResult Evaluate(
        WorldScenePortalAdapterResult adapter,
        WorldSceneNode placementNode,
        Vector3 cameraWorldPosition,
        int maximumDepth = 4)
    {
        ArgumentNullException.ThrowIfNull(adapter);
        ArgumentNullException.ThrowIfNull(placementNode);
        if (maximumDepth < 0)
            throw new ArgumentOutOfRangeException(nameof(maximumDepth));

        WorldScenePortalVisibilityDiagnostics diagnostics = new();
        if (!Matrix4x4.Invert(placementNode.WorldTransform, out Matrix4x4 inversePlacement))
            return Fallback(adapter, diagnostics, "placement_transform_invalid");

        Vector3 cameraLocal = Vector3.Transform(cameraWorldPosition, inversePlacement);
        if (!IsFinite(cameraLocal))
            return Fallback(adapter, diagnostics, "camera_position_invalid");

        if (!adapter.Graph.PortalDataPresent)
            return Fallback(adapter, diagnostics, "portal_data_absent");
        if (adapter.Graph.MalformedPortalData)
            return Fallback(adapter, diagnostics, "malformed_portal_edge");
        if (maximumDepth == 0)
            return Fallback(adapter, diagnostics, "maximum_depth_reached");

        Dictionary<int, WorldSceneNode> groupNodes = placementNode.Children
            .Where(static node => node.Kind == WorldSceneNodeKind.WmoGroup && node.PortalGroup.HasValue)
            .GroupBy(static node => node.PortalGroup!.Value)
            .ToDictionary(static group => group.Key, static group => group.First());
        if (groupNodes.Count == 0)
            return Fallback(adapter, diagnostics, "portal_group_nodes_absent");

        KeyValuePair<int, WorldSceneNode>? source = groupNodes
            .Where(pair => ContainsLocalBounds(pair.Value, cameraLocal))
            .OrderBy(pair => pair.Key)
            .Select(static pair => (KeyValuePair<int, WorldSceneNode>?)pair)
            .FirstOrDefault();
        if (!source.HasValue)
            return Fallback(adapter, diagnostics, "camera_group_unknown");

        diagnostics.SourceGroupIndex = source.Value.Key;
        HashSet<string> visibleNodeIds = new(StringComparer.Ordinal)
        {
            source.Value.Value.Id
        };
        HashSet<string> visitedNodeIds = new(StringComparer.Ordinal)
        {
            source.Value.Value.Id
        };
        Dictionary<int, WorldScenePortalGeometry> geometriesByPortalIndex = adapter.Geometries
            .GroupBy(static geometry => geometry.PortalIndex)
            .ToDictionary(static group => group.Key, static group => group.First());
        Queue<(string NodeId, WorldScenePortalViewVolume Volume)> queue = new();
        queue.Enqueue((source.Value.Value.Id, WorldScenePortalViewVolume.CreateRoot()));

        while (queue.Count > 0)
        {
            (string nodeId, WorldScenePortalViewVolume volume) = queue.Dequeue();
            diagnostics.VisitedGroupCount++;
            diagnostics.MaxDepthReached = Math.Max(diagnostics.MaxDepthReached, volume.Depth);

            foreach (WorldScenePortalLink link in adapter.Graph.GetOutgoingLinks(nodeId))
            {
                if (!TryGetPortalIndex(link.Id, out int portalIndex)
                    || !geometriesByPortalIndex.TryGetValue(portalIndex, out WorldScenePortalGeometry? geometry))
                {
                    diagnostics.RejectedPortalCount++;
                    return Fallback(adapter, diagnostics, "portal_geometry_missing");
                }

                if (!groupNodes.TryGetValue(ParseGroupIndex(link.DestinationNodeId), out WorldSceneNode? destinationNode))
                {
                    diagnostics.RejectedPortalCount++;
                    return Fallback(adapter, diagnostics, "portal_group_node_missing");
                }

                if (!visitedNodeIds.Add(destinationNode.Id))
                    continue;

                diagnostics.TestedPortalCount++;
                WorldScenePortalViewVolumeBuildResult child = WorldScenePortalViewVolumeBuilder.BuildChild(
                    volume,
                    geometry,
                    link.SourceNodeId,
                    link.DestinationNodeId,
                    cameraLocal,
                    maximumDepth);
                if (child.FallbackRequired || child.Volume is null)
                {
                    diagnostics.RejectedPortalCount++;
                    return Fallback(adapter, diagnostics, child.FallbackReason ?? "portal_volume_fallback");
                }

                if (!destinationNode.BoundsKnown || child.Volume.IntersectsBounds(
                        destinationNode.LocalBoundsMin,
                        destinationNode.LocalBoundsMax))
                {
                    visibleNodeIds.Add(destinationNode.Id);
                    queue.Enqueue((destinationNode.Id, child.Volume));
                }
            }
        }

        return new WorldScenePortalVisibilityResult(
            visibleNodeIds.OrderBy(static nodeId => nodeId, StringComparer.Ordinal).ToArray(),
            diagnostics);
    }

    private static WorldScenePortalVisibilityResult Fallback(
        WorldScenePortalAdapterResult adapter,
        WorldScenePortalVisibilityDiagnostics diagnostics,
        string reason)
    {
        diagnostics.FallbackRequired = true;
        diagnostics.FallbackReason = reason;
        return new WorldScenePortalVisibilityResult(adapter.Graph.NodeIds, diagnostics);
    }

    private static bool ContainsLocalBounds(WorldSceneNode node, Vector3 point)
        => node.BoundsKnown
            && point.X >= node.LocalBoundsMin.X
            && point.X <= node.LocalBoundsMax.X
            && point.Y >= node.LocalBoundsMin.Y
            && point.Y <= node.LocalBoundsMax.Y
            && point.Z >= node.LocalBoundsMin.Z
            && point.Z <= node.LocalBoundsMax.Z;

    private static int ParseGroupIndex(string nodeId)
    {
        int markerIndex = nodeId.LastIndexOf("/group/", StringComparison.Ordinal);
        return markerIndex >= 0
            && int.TryParse(nodeId[(markerIndex + "/group/".Length)..], out int groupIndex)
            ? groupIndex
            : int.MinValue;
    }

    private static bool TryGetPortalIndex(string linkId, out int portalIndex)
    {
        int markerIndex = linkId.LastIndexOf("/portal/", StringComparison.Ordinal);
        if (markerIndex < 0)
        {
            portalIndex = -1;
            return false;
        }

        int start = markerIndex + "/portal/".Length;
        int end = linkId.IndexOf('/', start);
        return int.TryParse(
            end < 0 ? linkId[start..] : linkId[start..end],
            out portalIndex);
    }

    private static bool IsFinite(Vector3 value)
        => float.IsFinite(value.X) && float.IsFinite(value.Y) && float.IsFinite(value.Z);
}
