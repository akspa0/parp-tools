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

    /// <summary>
    /// False when the traversal ran without per-kind attribution and rejected-node collection.
    /// The scalar counters remain valid; the per-kind dictionaries,
    /// <see cref="SkippedDescendantCount"/>, and the rejected-node list do not.
    /// </summary>
    public bool DetailedCollectionEnabled { get; set; } = true;

    /// <summary>
    /// Return to the zero state so one instance can be reused across frames instead of allocating a
    /// fresh diagnostics object (and its four dictionaries) per graph per frame.
    /// </summary>
    public void Reset()
    {
        VisitedNodeCount = 0;
        IndividuallyTestedNodeCount = 0;
        NonRejectableNodeCount = 0;
        VisibleRenderableNodeCount = 0;
        RejectedNodeCount = 0;
        SkippedDescendantCount = 0;
        MaxDepthReached = 0;
        DeferredVisibilityTestCount = 0;
        _individuallyTestedNodeCountsByKind.Clear();
        _rejectedNodeCountsByKind.Clear();
        _skippedDescendantCountsByKind.Clear();
        _deferredVisibilityTestCountsByKind.Clear();
    }

    public void Accumulate(WorldSceneTraversalDiagnostics other)
    {
        ArgumentNullException.ThrowIfNull(other);

        VisitedNodeCount += other.VisitedNodeCount;
        IndividuallyTestedNodeCount += other.IndividuallyTestedNodeCount;
        NonRejectableNodeCount += other.NonRejectableNodeCount;
        VisibleRenderableNodeCount += other.VisibleRenderableNodeCount;
        RejectedNodeCount += other.RejectedNodeCount;
        SkippedDescendantCount += other.SkippedDescendantCount;
        MaxDepthReached = Math.Max(MaxDepthReached, other.MaxDepthReached);
        DeferredVisibilityTestCount += other.DeferredVisibilityTestCount;
        AccumulateCounts(_individuallyTestedNodeCountsByKind, other._individuallyTestedNodeCountsByKind);
        AccumulateCounts(_rejectedNodeCountsByKind, other._rejectedNodeCountsByKind);
        AccumulateCounts(_skippedDescendantCountsByKind, other._skippedDescendantCountsByKind);
        AccumulateCounts(_deferredVisibilityTestCountsByKind, other._deferredVisibilityTestCountsByKind);
    }

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
        List<WorldSceneNode> children = node.ChildList;
        for (int i = 0; i < children.Count; i++)
        {
            WorldSceneNode child = children[i];
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

    private static void AccumulateCounts(
        Dictionary<WorldSceneNodeKind, int> destination,
        IReadOnlyDictionary<WorldSceneNodeKind, int> source)
    {
        foreach ((WorldSceneNodeKind kind, int count) in source)
            destination[kind] = destination.GetValueOrDefault(kind) + count;
    }
}

public sealed record WorldSceneTraversalResult(
    IReadOnlyList<WorldSceneNode> VisibleNodes,
    IReadOnlyList<WorldSceneNode> RejectedNodes,
    WorldSceneTraversalDiagnostics Diagnostics);

public static class WorldSceneTraversal
{
    /// <param name="collectDetailedDiagnostics">
    /// When true (the default, preserving existing callers and tests), per-kind attribution and the
    /// rejected-node list are gathered. That requires recursively walking every rejected subtree,
    /// so a production render loop should pass <c>false</c>: it keeps the cheap scalar counters and
    /// drops the work that scales with what was culled. Check
    /// <see cref="WorldSceneTraversalDiagnostics.DetailedCollectionEnabled"/> before reading the
    /// per-kind dictionaries or <see cref="WorldSceneTraversalResult.RejectedNodes"/>.
    /// </param>
    public static WorldSceneTraversalResult Traverse(
        WorldSceneGraph graph,
        Func<WorldSceneNode, bool> isVisible,
        Func<WorldSceneNode, bool>? includeNode = null,
        Func<WorldSceneNode, bool>? shouldEvaluateVisibility = null,
        bool validateGraph = true,
        bool collectDetailedDiagnostics = true)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(isVisible);

        if (validateGraph)
            graph.ValidateInvariants();
        includeNode ??= static node => node.IsRenderable;

        List<WorldSceneNode> visibleNodes = [];
        List<WorldSceneNode> rejectedNodes = [];
        WorldSceneTraversalDiagnostics diagnostics = new()
        {
            DetailedCollectionEnabled = collectDetailedDiagnostics,
        };
        TraverseInto(
            graph,
            isVisible,
            visibleNodes,
            rejectedNodes,
            diagnostics,
            includeNode,
            shouldEvaluateVisibility,
            validateGraph: false,
            collectDetailedDiagnostics);
        return new WorldSceneTraversalResult(visibleNodes, rejectedNodes, diagnostics);
    }

    /// <summary>
    /// Traverse into caller-owned buffers, which are cleared first and reused across frames.
    /// <para>
    /// <see cref="Traverse"/> allocates two lists, a diagnostics object holding four dictionaries,
    /// and a result record on every call. ADT tiles are isolated into independent graphs, so a
    /// per-frame render loop pays that per graph per frame — allocation proportional to the resident
    /// tile set. This overload exists so the hot path can own its buffers and allocate nothing.
    /// </para>
    /// </summary>
    public static void TraverseInto(
        WorldSceneGraph graph,
        Func<WorldSceneNode, bool> isVisible,
        List<WorldSceneNode> visibleNodes,
        List<WorldSceneNode> rejectedNodes,
        WorldSceneTraversalDiagnostics diagnostics,
        Func<WorldSceneNode, bool>? includeNode = null,
        Func<WorldSceneNode, bool>? shouldEvaluateVisibility = null,
        bool validateGraph = false,
        bool collectDetailedDiagnostics = true)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(isVisible);
        ArgumentNullException.ThrowIfNull(visibleNodes);
        ArgumentNullException.ThrowIfNull(rejectedNodes);
        ArgumentNullException.ThrowIfNull(diagnostics);

        if (validateGraph)
            graph.ValidateInvariants();
        includeNode ??= static node => node.IsRenderable;

        visibleNodes.Clear();
        rejectedNodes.Clear();
        diagnostics.Reset();
        diagnostics.DetailedCollectionEnabled = collectDetailedDiagnostics;

        Visit(
            graph.Root,
            isVisible,
            includeNode,
            shouldEvaluateVisibility,
            visibleNodes,
            rejectedNodes,
            diagnostics,
            collectDetailedDiagnostics);
    }

    private static void Visit(
        WorldSceneNode node,
        Func<WorldSceneNode, bool> isVisible,
        Func<WorldSceneNode, bool> includeNode,
        Func<WorldSceneNode, bool>? shouldEvaluateVisibility,
        List<WorldSceneNode> visibleNodes,
        List<WorldSceneNode> rejectedNodes,
        WorldSceneTraversalDiagnostics diagnostics,
        bool collectDetail)
    {
        diagnostics.VisitedNodeCount++;
        diagnostics.MaxDepthReached = Math.Max(diagnostics.MaxDepthReached, node.Depth);

        bool evaluateVisibility = node.CanRejectSubtree
            && (shouldEvaluateVisibility is null || shouldEvaluateVisibility(node));
        if (evaluateVisibility)
        {
            diagnostics.IndividuallyTestedNodeCount++;
            if (collectDetail)
                diagnostics.RecordIndividualTest(node.Kind);

            if (!isVisible(node))
            {
                diagnostics.RejectedNodeCount++;

                // Attributing the rejected subtree means recursively walking the very subtree that
                // culling just decided to skip, which makes rejecting a large region cost MORE than
                // accepting it. That is diagnostic-only work and must not run on production frames.
                if (collectDetail)
                {
                    diagnostics.SkippedDescendantCount += diagnostics.RecordRejectedSubtree(node);
                    rejectedNodes.Add(node);
                }

                return;
            }
        }
        else if (!node.CanRejectSubtree)
        {
            diagnostics.NonRejectableNodeCount++;
        }
        else if (collectDetail)
        {
            diagnostics.RecordDeferredVisibilityTest(node.Kind);
        }
        else
        {
            diagnostics.DeferredVisibilityTestCount++;
        }

        if (includeNode(node))
        {
            visibleNodes.Add(node);
            diagnostics.VisibleRenderableNodeCount++;
        }

        // Iterate the concrete list: foreach over IReadOnlyList<T> boxes the enumerator once per
        // node, and traversal touches every node every frame.
        List<WorldSceneNode> children = node.ChildList;
        for (int i = 0; i < children.Count; i++)
            Visit(
                children[i],
                isVisible,
                includeNode,
                shouldEvaluateVisibility,
                visibleNodes,
                rejectedNodes,
                diagnostics,
                collectDetail);
    }

}
