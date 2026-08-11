namespace WowViewer.Core.Runtime.World.SceneGraph;

public sealed class WorldSceneTraversalDiagnostics
{
    private readonly Dictionary<WorldSceneNodeKind, int> _individuallyTestedNodeCountsByKind = [];
    private readonly Dictionary<WorldSceneNodeKind, int> _rejectedNodeCountsByKind = [];
    private readonly Dictionary<WorldSceneNodeKind, int> _skippedDescendantCountsByKind = [];
    private readonly Dictionary<WorldSceneNodeKind, int> _deferredVisibilityTestCountsByKind = [];

    public int VisitedNodeCount { get; internal set; }

    public int IndividuallyTestedNodeCount { get; internal set; }

    public int NonRejectableNodeCount { get; internal set; }

    public int VisibleRenderableNodeCount { get; internal set; }

    public int RejectedNodeCount { get; internal set; }

    public int SkippedDescendantCount { get; internal set; }

    public int MaxDepthReached { get; internal set; }

    public int DeferredVisibilityTestCount { get; internal set; }

    public IReadOnlyDictionary<WorldSceneNodeKind, int> IndividuallyTestedNodeCountsByKind =>
        _individuallyTestedNodeCountsByKind;

    public IReadOnlyDictionary<WorldSceneNodeKind, int> RejectedNodeCountsByKind =>
        _rejectedNodeCountsByKind;

    public IReadOnlyDictionary<WorldSceneNodeKind, int> SkippedDescendantCountsByKind =>
        _skippedDescendantCountsByKind;

    public IReadOnlyDictionary<WorldSceneNodeKind, int> DeferredVisibilityTestCountsByKind =>
        _deferredVisibilityTestCountsByKind;

    internal void RecordIndividualTest(WorldSceneNodeKind kind)
    {
        Increment(_individuallyTestedNodeCountsByKind, kind);
    }

    internal int RecordRejectedSubtree(WorldSceneNode node)
    {
        Increment(_rejectedNodeCountsByKind, node.Kind);
        return RecordDescendantKinds(node, _skippedDescendantCountsByKind);
    }

    internal void RecordDeferredVisibilityTest(WorldSceneNodeKind kind)
    {
        DeferredVisibilityTestCount++;
        Increment(_deferredVisibilityTestCountsByKind, kind);
    }

    private static int RecordDescendantKinds(
        WorldSceneNode node,
        Dictionary<WorldSceneNodeKind, int> counts)
    {
        int descendantCount = 0;
        foreach (WorldSceneNode child in node.Children)
        {
            descendantCount++;
            Increment(counts, child.Kind);
            descendantCount += RecordDescendantKinds(child, counts);
        }

        return descendantCount;
    }

    private static void Increment(Dictionary<WorldSceneNodeKind, int> counts, WorldSceneNodeKind kind)
    {
        counts[kind] = counts.GetValueOrDefault(kind) + 1;
    }
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
        Func<WorldSceneNode, bool>? includeNode = null,
        Func<WorldSceneNode, bool>? shouldEvaluateVisibility = null,
        bool validateGraph = true)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(isVisible);

        if (validateGraph)
            graph.ValidateInvariants();
        includeNode ??= static node => node.IsRenderable;

        List<WorldSceneNode> visibleNodes = [];
        List<WorldSceneNode> rejectedNodes = [];
        WorldSceneTraversalDiagnostics diagnostics = new();
        Visit(
            graph.Root,
            isVisible,
            includeNode,
            shouldEvaluateVisibility,
            visibleNodes,
            rejectedNodes,
            diagnostics);
        return new WorldSceneTraversalResult(visibleNodes, rejectedNodes, diagnostics);
    }

    private static void Visit(
        WorldSceneNode node,
        Func<WorldSceneNode, bool> isVisible,
        Func<WorldSceneNode, bool> includeNode,
        Func<WorldSceneNode, bool>? shouldEvaluateVisibility,
        List<WorldSceneNode> visibleNodes,
        List<WorldSceneNode> rejectedNodes,
        WorldSceneTraversalDiagnostics diagnostics)
    {
        diagnostics.VisitedNodeCount++;
        diagnostics.MaxDepthReached = Math.Max(diagnostics.MaxDepthReached, node.Depth);

        bool evaluateVisibility = node.CanRejectSubtree
            && (shouldEvaluateVisibility is null || shouldEvaluateVisibility(node));
        if (evaluateVisibility)
        {
            diagnostics.IndividuallyTestedNodeCount++;
            diagnostics.RecordIndividualTest(node.Kind);
            if (!isVisible(node))
            {
                diagnostics.RejectedNodeCount++;
                diagnostics.SkippedDescendantCount += diagnostics.RecordRejectedSubtree(node);
                rejectedNodes.Add(node);
                return;
            }
        }
        else if (!node.CanRejectSubtree)
        {
            diagnostics.NonRejectableNodeCount++;
        }
        else
        {
            diagnostics.RecordDeferredVisibilityTest(node.Kind);
        }

        if (includeNode(node))
        {
            visibleNodes.Add(node);
            diagnostics.VisibleRenderableNodeCount++;
        }

        foreach (WorldSceneNode child in node.Children)
            Visit(
                child,
                isVisible,
                includeNode,
                shouldEvaluateVisibility,
                visibleNodes,
                rejectedNodes,
                diagnostics);
    }

}
