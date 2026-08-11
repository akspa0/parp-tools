using System.Numerics;

namespace WowViewer.Core.Runtime.World.SceneGraph;

public sealed class WorldSceneNode
{
    private const float BoundsEpsilon = 0.0001f;
    private readonly List<WorldSceneNode> _children = [];

    private Matrix4x4 _localTransform;
    private Vector3 _localBoundsMin;
    private Vector3 _localBoundsMax;

    public WorldSceneNode(
        string id,
        WorldSceneNodeKind kind,
        Matrix4x4 localTransform,
        Vector3 localBoundsMin,
        Vector3 localBoundsMax,
        bool boundsKnown = true,
        bool isRenderable = false,
        bool isQueryable = false,
        bool requiresUpdate = false,
        string? assetKey = null,
        WorldSceneRenderPass renderPassMask = WorldSceneRenderPass.None,
        int? portalGroup = null,
        bool allowsUnresolvedDescendants = false)
    {
        if (string.IsNullOrWhiteSpace(id))
            throw new ArgumentException("Scene node id must not be empty.", nameof(id));

        ValidateTransform(localTransform, nameof(localTransform));
        if (boundsKnown)
            ValidateBounds(localBoundsMin, localBoundsMax, nameof(localBoundsMin));

        Id = id;
        Kind = kind;
        _localTransform = localTransform;
        _localBoundsMin = boundsKnown ? localBoundsMin : Vector3.Zero;
        _localBoundsMax = boundsKnown ? localBoundsMax : Vector3.Zero;
        BoundsKnown = boundsKnown;
        IsRenderable = isRenderable;
        IsQueryable = isQueryable;
        RequiresUpdate = requiresUpdate;
        AssetKey = assetKey;
        RenderPassMask = renderPassMask;
        PortalGroup = portalGroup;
        AllowsUnresolvedDescendants = allowsUnresolvedDescendants;

        RefreshWorldState(Matrix4x4.Identity);
    }

    public string Id { get; }

    public WorldSceneNodeKind Kind { get; }

    public WorldSceneNode? Parent { get; private set; }

    public IReadOnlyList<WorldSceneNode> Children => _children;

    public Matrix4x4 LocalTransform => _localTransform;

    public Matrix4x4 WorldTransform { get; private set; }

    public Vector3 LocalBoundsMin => _localBoundsMin;

    public Vector3 LocalBoundsMax => _localBoundsMax;

    public Vector3 WorldBoundsMin { get; private set; }

    public Vector3 WorldBoundsMax { get; private set; }

    public bool BoundsKnown { get; private set; }

    public bool CanRejectSubtree { get; private set; }

    public bool IsRenderable { get; }

    public bool IsQueryable { get; }

    public bool RequiresUpdate { get; }

    public string? AssetKey { get; }

    public WorldSceneRenderPass RenderPassMask { get; }

    public int? PortalGroup { get; }

    /// <summary>
    /// Allows an authoritative spatial container (currently an ADT tile root)
    /// to reject its subtree even while streamed child assets have no bounds.
    /// Known descendants must still remain contained by this node's bounds.
    /// </summary>
    public bool AllowsUnresolvedDescendants { get; }

    public void UpdateLocalTransform(Matrix4x4 localTransform)
    {
        ValidateTransform(localTransform, nameof(localTransform));
        _localTransform = localTransform;
        RefreshRoot();
    }

    public void UpdateLocalBounds(Vector3 localBoundsMin, Vector3 localBoundsMax, bool boundsKnown = true)
    {
        if (boundsKnown)
            ValidateBounds(localBoundsMin, localBoundsMax, nameof(localBoundsMin));

        _localBoundsMin = boundsKnown ? localBoundsMin : Vector3.Zero;
        _localBoundsMax = boundsKnown ? localBoundsMax : Vector3.Zero;
        BoundsKnown = boundsKnown;
        RefreshRoot();
    }

    /// <summary>
    /// Updates a streamed placement's bounds without refreshing unrelated graph branches.
    /// Ancestor aggregate bounds are intentionally left unchanged; callers use this for a
    /// false-to-true placement bounds promotion while the authoritative tile root remains valid.
    /// </summary>
    public void UpdateLocalBoundsForStreaming(Vector3 localBoundsMin, Vector3 localBoundsMax, bool boundsKnown = true)
    {
        if (boundsKnown)
            ValidateBounds(localBoundsMin, localBoundsMax, nameof(localBoundsMin));

        _localBoundsMin = boundsKnown ? localBoundsMin : Vector3.Zero;
        _localBoundsMax = boundsKnown ? localBoundsMax : Vector3.Zero;
        BoundsKnown = boundsKnown;
        RefreshWorldState(Parent?.WorldTransform ?? Matrix4x4.Identity);
    }

    internal void AttachChild(WorldSceneNode child)
    {
        ArgumentNullException.ThrowIfNull(child);
        if (ReferenceEquals(this, child))
            throw new InvalidOperationException("A scene node cannot be attached to itself.");
        if (child.Parent is not null)
            throw new InvalidOperationException($"Scene node '{child.Id}' already has a parent.");
        if (child.ContainsDescendant(this))
            throw new InvalidOperationException($"Attaching '{child.Id}' below '{Id}' would create a cycle.");
        if (_children.Any(existing => existing.Id.Equals(child.Id, StringComparison.Ordinal)))
            throw new InvalidOperationException($"Scene node '{Id}' already has a child named '{child.Id}'.");

        child.Parent = this;
        _children.Add(child);
        RefreshRoot();
    }

    internal bool DetachChild(WorldSceneNode child)
    {
        ArgumentNullException.ThrowIfNull(child);
        if (!_children.Remove(child))
            return false;

        child.Parent = null;
        child.RefreshWorldState(Matrix4x4.Identity);
        RefreshRoot();
        return true;
    }

    internal IEnumerable<WorldSceneNode> EnumerateDepthFirst()
    {
        yield return this;
        foreach (WorldSceneNode child in _children)
        {
            foreach (WorldSceneNode descendant in child.EnumerateDepthFirst())
                yield return descendant;
        }
    }

    internal int Depth
    {
        get
        {
            int depth = 0;
            for (WorldSceneNode? node = Parent; node is not null; node = node.Parent)
                depth++;
            return depth;
        }
    }

    internal bool ContainsDescendant(WorldSceneNode candidate)
    {
        foreach (WorldSceneNode child in _children)
        {
            if (ReferenceEquals(child, candidate) || child.ContainsDescendant(candidate))
                return true;
        }

        return false;
    }

    internal bool ContainsWorldBounds(Vector3 boundsMin, Vector3 boundsMax)
    {
        if (!CanRejectSubtree)
            return false;

        return boundsMin.X >= WorldBoundsMin.X - BoundsEpsilon
            && boundsMin.Y >= WorldBoundsMin.Y - BoundsEpsilon
            && boundsMin.Z >= WorldBoundsMin.Z - BoundsEpsilon
            && boundsMax.X <= WorldBoundsMax.X + BoundsEpsilon
            && boundsMax.Y <= WorldBoundsMax.Y + BoundsEpsilon
            && boundsMax.Z <= WorldBoundsMax.Z + BoundsEpsilon;
    }

    internal void RefreshWorldState(Matrix4x4 parentWorld)
    {
        WorldTransform = _localTransform * parentWorld;
        if (BoundsKnown)
        {
            (WorldBoundsMin, WorldBoundsMax) = TransformBounds(_localBoundsMin, _localBoundsMax, WorldTransform);
            CanRejectSubtree = true;
        }
        else
        {
            WorldBoundsMin = Vector3.Zero;
            WorldBoundsMax = Vector3.Zero;
            CanRejectSubtree = false;
        }

        foreach (WorldSceneNode child in _children)
            child.RefreshWorldState(WorldTransform);

        if (_children.Any(child => (!child.CanRejectSubtree && !AllowsUnresolvedDescendants)
            || (child.CanRejectSubtree && !ContainsWorldBoundsUnchecked(child.WorldBoundsMin, child.WorldBoundsMax))))
            CanRejectSubtree = false;
    }

    private void RefreshRoot()
    {
        WorldSceneNode root = this;
        while (root.Parent is not null)
            root = root.Parent;

        root.RefreshWorldState(Matrix4x4.Identity);
    }

    private bool ContainsWorldBoundsUnchecked(Vector3 boundsMin, Vector3 boundsMax)
    {
        return boundsMin.X >= WorldBoundsMin.X - BoundsEpsilon
            && boundsMin.Y >= WorldBoundsMin.Y - BoundsEpsilon
            && boundsMin.Z >= WorldBoundsMin.Z - BoundsEpsilon
            && boundsMax.X <= WorldBoundsMax.X + BoundsEpsilon
            && boundsMax.Y <= WorldBoundsMax.Y + BoundsEpsilon
            && boundsMax.Z <= WorldBoundsMax.Z + BoundsEpsilon;
    }

    private static (Vector3 min, Vector3 max) TransformBounds(Vector3 boundsMin, Vector3 boundsMax, Matrix4x4 transform)
    {
        Span<Vector3> corners = stackalloc Vector3[8];
        corners[0] = new Vector3(boundsMin.X, boundsMin.Y, boundsMin.Z);
        corners[1] = new Vector3(boundsMax.X, boundsMin.Y, boundsMin.Z);
        corners[2] = new Vector3(boundsMin.X, boundsMax.Y, boundsMin.Z);
        corners[3] = new Vector3(boundsMax.X, boundsMax.Y, boundsMin.Z);
        corners[4] = new Vector3(boundsMin.X, boundsMin.Y, boundsMax.Z);
        corners[5] = new Vector3(boundsMax.X, boundsMin.Y, boundsMax.Z);
        corners[6] = new Vector3(boundsMin.X, boundsMax.Y, boundsMax.Z);
        corners[7] = new Vector3(boundsMax.X, boundsMax.Y, boundsMax.Z);

        Vector3 min = Vector3.Transform(corners[0], transform);
        Vector3 max = min;
        for (int index = 1; index < corners.Length; index++)
        {
            Vector3 transformed = Vector3.Transform(corners[index], transform);
            min = Vector3.Min(min, transformed);
            max = Vector3.Max(max, transformed);
        }

        return (min, max);
    }

    private static void ValidateBounds(Vector3 boundsMin, Vector3 boundsMax, string parameterName)
    {
        ValidateVector(boundsMin, parameterName);
        ValidateVector(boundsMax, parameterName);
        if (boundsMin.X > boundsMax.X || boundsMin.Y > boundsMax.Y || boundsMin.Z > boundsMax.Z)
            throw new ArgumentException("Scene node bounds must be ordered min-to-max.", parameterName);
    }

    private static void ValidateTransform(Matrix4x4 transform, string parameterName)
    {
        if (!float.IsFinite(transform.M11) || !float.IsFinite(transform.M12)
            || !float.IsFinite(transform.M13) || !float.IsFinite(transform.M14)
            || !float.IsFinite(transform.M21) || !float.IsFinite(transform.M22)
            || !float.IsFinite(transform.M23) || !float.IsFinite(transform.M24)
            || !float.IsFinite(transform.M31) || !float.IsFinite(transform.M32)
            || !float.IsFinite(transform.M33) || !float.IsFinite(transform.M34)
            || !float.IsFinite(transform.M41) || !float.IsFinite(transform.M42)
            || !float.IsFinite(transform.M43) || !float.IsFinite(transform.M44))
            throw new ArgumentException("Scene node transform must contain only finite values.", parameterName);
    }

    private static void ValidateVector(Vector3 value, string parameterName)
    {
        if (!float.IsFinite(value.X) || !float.IsFinite(value.Y) || !float.IsFinite(value.Z))
            throw new ArgumentException("Scene node bounds must contain only finite values.", parameterName);
    }
}
