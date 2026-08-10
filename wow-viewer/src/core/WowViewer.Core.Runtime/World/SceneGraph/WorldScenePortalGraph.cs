namespace WowViewer.Core.Runtime.World.SceneGraph;

public readonly record struct WorldScenePortalLink(
    string Id,
    string SourceNodeId,
    string DestinationNodeId);

public sealed record WorldScenePortalGraphBuildResult(
    WorldScenePortalGraph Graph,
    IReadOnlyList<WorldScenePortalLink> RejectedLinks,
    int DeclaredNodeCount,
    int AcceptedLinkCount);

public sealed class WorldScenePortalGraph
{
    private readonly HashSet<string> _nodeIds;
    private readonly Dictionary<string, List<WorldScenePortalLink>> _outgoing;

    private WorldScenePortalGraph(
        HashSet<string> nodeIds,
        Dictionary<string, List<WorldScenePortalLink>> outgoing,
        bool portalDataPresent,
        bool malformedPortalData)
    {
        _nodeIds = nodeIds;
        _outgoing = outgoing;
        PortalDataPresent = portalDataPresent;
        MalformedPortalData = malformedPortalData;
    }

    public bool PortalDataPresent { get; }

    public bool MalformedPortalData { get; }

    public int NodeCount => _nodeIds.Count;

    public int LinkCount => _outgoing.Values.Sum(static links => links.Count);

    public static WorldScenePortalGraphBuildResult Build(
        IEnumerable<string> nodeIds,
        IEnumerable<WorldScenePortalLink> links)
    {
        ArgumentNullException.ThrowIfNull(nodeIds);
        ArgumentNullException.ThrowIfNull(links);

        HashSet<string> declaredNodeIds = new(StringComparer.Ordinal);
        foreach (string nodeId in nodeIds)
        {
            if (string.IsNullOrWhiteSpace(nodeId))
                throw new ArgumentException("Portal graph node ids must not be empty.", nameof(nodeIds));
            if (!declaredNodeIds.Add(nodeId))
                throw new InvalidOperationException($"Portal graph contains duplicate node id '{nodeId}'.");
        }

        Dictionary<string, List<WorldScenePortalLink>> outgoing = new(StringComparer.Ordinal);
        HashSet<string> linkIds = new(StringComparer.Ordinal);
        List<WorldScenePortalLink> rejectedLinks = [];
        int declaredLinkCount = 0;

        foreach (WorldScenePortalLink link in links)
        {
            declaredLinkCount++;
            bool valid = !string.IsNullOrWhiteSpace(link.Id)
                && linkIds.Add(link.Id)
                && !string.IsNullOrWhiteSpace(link.SourceNodeId)
                && !string.IsNullOrWhiteSpace(link.DestinationNodeId)
                && declaredNodeIds.Contains(link.SourceNodeId)
                && declaredNodeIds.Contains(link.DestinationNodeId);
            if (!valid)
            {
                rejectedLinks.Add(link);
                continue;
            }

            if (!outgoing.TryGetValue(link.SourceNodeId, out List<WorldScenePortalLink>? sourceLinks))
            {
                sourceLinks = [];
                outgoing.Add(link.SourceNodeId, sourceLinks);
            }

            sourceLinks.Add(link);
        }

        foreach (List<WorldScenePortalLink> sourceLinks in outgoing.Values)
            sourceLinks.Sort(static (left, right) => string.CompareOrdinal(left.Id, right.Id));

        return new WorldScenePortalGraphBuildResult(
            new WorldScenePortalGraph(declaredNodeIds, outgoing, declaredLinkCount > 0, rejectedLinks.Count > 0),
            rejectedLinks,
            declaredNodeIds.Count,
            declaredLinkCount - rejectedLinks.Count);
    }

    public WorldScenePortalTraversalResult Traverse(string entryNodeId, int maximumDepth)
    {
        if (maximumDepth < 0)
            throw new ArgumentOutOfRangeException(nameof(maximumDepth), "Portal traversal depth must not be negative.");

        WorldScenePortalTraversalDiagnostics diagnostics = new()
        {
            PortalDataPresent = PortalDataPresent,
            FallbackRequired = !PortalDataPresent || MalformedPortalData,
            FallbackReason = !PortalDataPresent
                ? "portal_data_absent"
                : MalformedPortalData ? "malformed_portal_edge" : null
        };
        List<string> visibleNodeIds = [];
        List<WorldScenePortalLink> traversedLinks = [];

        if (!_nodeIds.Contains(entryNodeId))
        {
            diagnostics.FallbackRequired = true;
            diagnostics.FallbackReason = AppendReason(diagnostics.FallbackReason, "entry_node_missing");
            return new WorldScenePortalTraversalResult(visibleNodeIds, traversedLinks, diagnostics);
        }

        Queue<(string nodeId, int depth)> queue = new();
        HashSet<string> visited = new(StringComparer.Ordinal) { entryNodeId };
        queue.Enqueue((entryNodeId, 0));
        visibleNodeIds.Add(entryNodeId);

        while (queue.Count > 0)
        {
            (string nodeId, int depth) = queue.Dequeue();
            diagnostics.VisitedNodeCount++;
            diagnostics.MaxDepthReached = Math.Max(diagnostics.MaxDepthReached, depth);

            if (!_outgoing.TryGetValue(nodeId, out List<WorldScenePortalLink>? sourceLinks))
                continue;

            foreach (WorldScenePortalLink link in sourceLinks)
            {
                if (depth >= maximumDepth)
                {
                    diagnostics.DepthLimitHitCount++;
                    diagnostics.FallbackRequired = true;
                    diagnostics.FallbackReason = AppendReason(diagnostics.FallbackReason, "maximum_depth_reached");
                    continue;
                }

                if (!visited.Add(link.DestinationNodeId))
                {
                    diagnostics.CycleCount++;
                    continue;
                }

                traversedLinks.Add(link);
                diagnostics.TraversedLinkCount++;
                visibleNodeIds.Add(link.DestinationNodeId);
                queue.Enqueue((link.DestinationNodeId, depth + 1));
            }
        }

        return new WorldScenePortalTraversalResult(visibleNodeIds, traversedLinks, diagnostics);
    }

    private static string AppendReason(string? existing, string reason)
    {
        if (string.IsNullOrWhiteSpace(existing))
            return reason;
        if (existing.Split(',', StringSplitOptions.TrimEntries).Contains(reason, StringComparer.Ordinal))
            return existing;
        return $"{existing},{reason}";
    }
}

public sealed class WorldScenePortalTraversalDiagnostics
{
    public bool PortalDataPresent { get; internal set; }

    public bool FallbackRequired { get; internal set; }

    public string? FallbackReason { get; internal set; }

    public int VisitedNodeCount { get; internal set; }

    public int TraversedLinkCount { get; internal set; }

    public int CycleCount { get; internal set; }

    public int DepthLimitHitCount { get; internal set; }

    public int MaxDepthReached { get; internal set; }
}

public sealed record WorldScenePortalTraversalResult(
    IReadOnlyList<string> VisibleNodeIds,
    IReadOnlyList<WorldScenePortalLink> TraversedLinks,
    WorldScenePortalTraversalDiagnostics Diagnostics);
