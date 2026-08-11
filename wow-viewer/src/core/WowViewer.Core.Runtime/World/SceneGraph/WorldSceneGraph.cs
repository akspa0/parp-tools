namespace WowViewer.Core.Runtime.World.SceneGraph;

public sealed class WorldSceneGraph
{
    private readonly Dictionary<string, WorldSceneNode> _nodes = new(StringComparer.Ordinal);

    public WorldSceneGraph(WorldSceneNode root)
    {
        ArgumentNullException.ThrowIfNull(root);
        if (root.Parent is not null)
            throw new ArgumentException("The scene graph root must not already have a parent.", nameof(root));

        Root = root;
        RegisterSubtree(root);
        ValidateInvariants();
    }

    public WorldSceneNode Root { get; }

    public int Count => _nodes.Count;

    public bool TryGetNode(string id, out WorldSceneNode? node)
    {
        if (string.IsNullOrWhiteSpace(id))
        {
            node = null;
            return false;
        }

        return _nodes.TryGetValue(id, out node);
    }

    public IEnumerable<WorldSceneNode> EnumerateDepthFirst() => Root.EnumerateDepthFirst();

    public void Attach(string parentId, WorldSceneNode child)
    {
        ArgumentNullException.ThrowIfNull(child);
        if (!_nodes.TryGetValue(parentId, out WorldSceneNode? parent))
            throw new KeyNotFoundException($"Scene graph parent '{parentId}' was not found.");
        if (child.Parent is not null)
            throw new InvalidOperationException($"Scene node '{child.Id}' already has a parent.");

        List<WorldSceneNode> subtree = child.EnumerateDepthFirst().ToList();
        HashSet<string> subtreeIds = new(StringComparer.Ordinal);
        foreach (WorldSceneNode node in subtree)
        {
            if (!subtreeIds.Add(node.Id))
                throw new InvalidOperationException($"Scene subtree contains duplicate node id '{node.Id}'.");
            if (_nodes.ContainsKey(node.Id))
                throw new InvalidOperationException($"Scene graph already contains node id '{node.Id}'.");
        }

        parent.AttachChild(child);
        foreach (WorldSceneNode node in subtree)
            _nodes.Add(node.Id, node);

        ValidateInvariants();
    }

    public bool Detach(string nodeId, out WorldSceneNode? detachedRoot)
    {
        detachedRoot = null;
        if (!_nodes.TryGetValue(nodeId, out WorldSceneNode? node) || ReferenceEquals(node, Root))
            return false;

        WorldSceneNode parent = node.Parent
            ?? throw new InvalidOperationException($"Reachable node '{node.Id}' has no parent.");
        if (!parent.DetachChild(node))
            throw new InvalidOperationException($"Parent '{parent.Id}' did not contain child '{node.Id}'.");

        foreach (WorldSceneNode descendant in node.EnumerateDepthFirst())
            _nodes.Remove(descendant.Id);

        detachedRoot = node;
        ValidateInvariants();
        return true;
    }

    public WorldSceneGraphSnapshot CreateSnapshot()
    {
        Dictionary<WorldSceneNodeKind, int> nodeKindCounts = new();
        Dictionary<WorldSceneRenderPass, int> renderPassCounts = new();
        int renderableCount = 0;
        int queryableCount = 0;
        int updateRequiredCount = 0;
        int nonRejectableCount = 0;
        int maxDepth = 0;

        List<string> nodeIds = [];
        foreach (WorldSceneNode node in EnumerateDepthFirst())
        {
            nodeIds.Add(node.Id);
            nodeKindCounts[node.Kind] = nodeKindCounts.GetValueOrDefault(node.Kind) + 1;
            if (node.RenderPassMask != WorldSceneRenderPass.None)
            {
                foreach (WorldSceneRenderPass pass in Enum.GetValues<WorldSceneRenderPass>())
                {
                    if (pass != WorldSceneRenderPass.None && node.RenderPassMask.HasFlag(pass))
                        renderPassCounts[pass] = renderPassCounts.GetValueOrDefault(pass) + 1;
                }
            }

            if (node.IsRenderable)
                renderableCount++;
            if (node.IsQueryable)
                queryableCount++;
            if (node.RequiresUpdate)
                updateRequiredCount++;
            if (!node.CanRejectSubtree)
                nonRejectableCount++;
            maxDepth = Math.Max(maxDepth, node.Depth);
        }

        return new WorldSceneGraphSnapshot(
            Count,
            renderableCount,
            queryableCount,
            updateRequiredCount,
            nonRejectableCount,
            maxDepth,
            nodeKindCounts,
            renderPassCounts,
            nodeIds,
            Root.WorldBoundsMin,
            Root.WorldBoundsMax);
    }

    public void ValidateInvariants()
    {
        HashSet<string> reachableIds = new(StringComparer.Ordinal);
        foreach (WorldSceneNode node in EnumerateDepthFirst())
        {
            if (!reachableIds.Add(node.Id))
                throw new InvalidOperationException($"Scene graph contains duplicate reachable node id '{node.Id}'.");
            if (!_nodes.TryGetValue(node.Id, out WorldSceneNode? indexed) || !ReferenceEquals(indexed, node))
                throw new InvalidOperationException($"Scene graph index is missing node '{node.Id}'.");
            if (!ReferenceEquals(node, Root) && node.Parent is null)
                throw new InvalidOperationException($"Non-root node '{node.Id}' is missing its parent.");
            if (node.CanRejectSubtree)
            {
                foreach (WorldSceneNode descendant in node.Children)
                {
                    if ((!descendant.CanRejectSubtree && !node.AllowsUnresolvedDescendants)
                        || (descendant.CanRejectSubtree && !Contains(node, descendant.WorldBoundsMin, descendant.WorldBoundsMax)))
                        throw new InvalidOperationException($"Rejectable node '{node.Id}' does not conservatively contain child '{descendant.Id}'.");
                }
            }
        }

        if (reachableIds.Count != _nodes.Count)
            throw new InvalidOperationException("Scene graph index contains an unreachable node.");
    }

    private void RegisterSubtree(WorldSceneNode root)
    {
        foreach (WorldSceneNode node in root.EnumerateDepthFirst())
        {
            if (!_nodes.TryAdd(node.Id, node))
                throw new InvalidOperationException($"Scene graph contains duplicate node id '{node.Id}'.");
        }
    }

    private static bool Contains(WorldSceneNode parent, System.Numerics.Vector3 boundsMin, System.Numerics.Vector3 boundsMax)
    {
        return boundsMin.X >= parent.WorldBoundsMin.X - 0.0001f
            && boundsMin.Y >= parent.WorldBoundsMin.Y - 0.0001f
            && boundsMin.Z >= parent.WorldBoundsMin.Z - 0.0001f
            && boundsMax.X <= parent.WorldBoundsMax.X + 0.0001f
            && boundsMax.Y <= parent.WorldBoundsMax.Y + 0.0001f
            && boundsMax.Z <= parent.WorldBoundsMax.Z + 0.0001f;
    }
}
