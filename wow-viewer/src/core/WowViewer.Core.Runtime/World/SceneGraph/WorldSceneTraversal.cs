namespace WowViewer.Core.Runtime.World.SceneGraph;

public sealed class WorldSceneTraversalDiagnostics
{
    public int VisitedNodeCount { get; internal set; }

    public int IndividuallyTestedNodeCount { get; internal set; }

    public int NonRejectableNodeCount { get; internal set; }

    public int VisibleRenderableNodeCount { get; internal set; }

    public int RejectedNodeCount { get; internal set; }

    public int SkippedDescendantCount { get; internal set; }

    public int MaxDepthReached { get; internal set; }
}

public sealed record WorldSceneTraversalResult(
    IReadOnlyList<WorldSceneNode> VisibleNodes,
    IReadOnlyList<WorldSceneNode> RejectedNodes,
    WorldSceneTraversalDiagnostics Diagnostics);

public static class WorldSceneTraversal
{
    public static WorldSceneTraversalResult Traverse(
        WorldSceneGraph graph,
        Func<WorldSceneNode, bool> isVisible,
        Func<WorldSceneNode, bool>? includeNode = null)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(isVisible);

        graph.ValidateInvariants();
        includeNode ??= static node => node.IsRenderable;

        List<WorldSceneNode> visibleNodes = [];
        List<WorldSceneNode> rejectedNodes = [];
        WorldSceneTraversalDiagnostics diagnostics = new();
        Visit(graph.Root, isVisible, includeNode, visibleNodes, rejectedNodes, diagnostics);
        return new WorldSceneTraversalResult(visibleNodes, rejectedNodes, diagnostics);
    }

    private static void Visit(
        WorldSceneNode node,
        Func<WorldSceneNode, bool> isVisible,
        Func<WorldSceneNode, bool> includeNode,
        List<WorldSceneNode> visibleNodes,
        List<WorldSceneNode> rejectedNodes,
        WorldSceneTraversalDiagnostics diagnostics)
    {
        diagnostics.VisitedNodeCount++;
        diagnostics.MaxDepthReached = Math.Max(diagnostics.MaxDepthReached, node.Depth);

        if (node.CanRejectSubtree)
        {
            diagnostics.IndividuallyTestedNodeCount++;
            if (!isVisible(node))
            {
                diagnostics.RejectedNodeCount++;
                diagnostics.SkippedDescendantCount += CountDescendants(node);
                rejectedNodes.Add(node);
                return;
            }
        }
        else
        {
            diagnostics.NonRejectableNodeCount++;
        }

        if (includeNode(node))
        {
            visibleNodes.Add(node);
            diagnostics.VisibleRenderableNodeCount++;
        }

        foreach (WorldSceneNode child in node.Children)
            Visit(child, isVisible, includeNode, visibleNodes, rejectedNodes, diagnostics);
    }

    private static int CountDescendants(WorldSceneNode node)
    {
        int count = 0;
        foreach (WorldSceneNode child in node.Children)
            count += 1 + CountDescendants(child);
        return count;
    }
}
