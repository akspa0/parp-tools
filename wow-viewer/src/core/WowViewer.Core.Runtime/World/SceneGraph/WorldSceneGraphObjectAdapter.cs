using System.Collections.ObjectModel;
using System.Numerics;

namespace WowViewer.Core.Runtime.World.SceneGraph;

/// <summary>
/// Describes one already-resolved world placement that can be attached to the
/// shared scene graph. Format readers and renderers remain outside this adapter.
/// </summary>
public readonly record struct WorldSceneGraphObjectPlacement(
    string Id,
    WorldSceneNodeKind Kind,
    WorldObjectInstance Instance,
    bool IsExternal = false,
    WorldSceneRenderPass RenderPassMask = WorldSceneRenderPass.Opaque,
    bool IsQueryable = true,
    bool RequiresUpdate = false,
    bool IsSkybox = false,
    IReadOnlyList<WorldSceneGraphChildNode>? Children = null,
    WorldSceneGraphSpatialBucket? SpatialBucket = null);

public readonly record struct WorldSceneGraphSpatialBucket(
    WorldSceneNodeKind Kind,
    string Key);

public readonly record struct WorldSceneGraphChildNode(
    string Id,
    WorldSceneNodeKind Kind,
    Matrix4x4 LocalTransform,
    Vector3 LocalBoundsMin,
    Vector3 LocalBoundsMax,
    bool BoundsKnown = true,
    bool IsRenderable = true,
    bool IsQueryable = true,
    bool RequiresUpdate = false,
    string? AssetKey = null,
    WorldSceneRenderPass RenderPassMask = WorldSceneRenderPass.Opaque,
    int? PortalGroup = null);

public sealed class WorldSceneGraphAdapterOptions
{
    public string MapId { get; init; } = "world/map";

    public Vector3 MapBoundsMin { get; init; }

    public Vector3 MapBoundsMax { get; init; }

    public bool MapBoundsKnown { get; init; }
}

public sealed class WorldSceneGraphBuildResult
{
    private readonly Dictionary<string, WorldSceneGraphObjectPlacement> _placementsByNodeId;

    public WorldSceneGraphBuildResult(
        WorldSceneGraph graph,
        IReadOnlyDictionary<string, WorldSceneGraphObjectPlacement> placementsByNodeId)
    {
        ArgumentNullException.ThrowIfNull(graph);
        ArgumentNullException.ThrowIfNull(placementsByNodeId);

        Graph = graph;
        _placementsByNodeId = new Dictionary<string, WorldSceneGraphObjectPlacement>(
            placementsByNodeId,
            StringComparer.Ordinal);
        PlacementsByNodeId = new ReadOnlyDictionary<string, WorldSceneGraphObjectPlacement>(_placementsByNodeId);
    }

    public WorldSceneGraph Graph { get; }

    public IReadOnlyDictionary<string, WorldSceneGraphObjectPlacement> PlacementsByNodeId { get; }

    public bool TryUpdatePlacementInstance(string placementId, WorldObjectInstance instance)
    {
        if (!_placementsByNodeId.TryGetValue(placementId, out WorldSceneGraphObjectPlacement placement))
            return false;

        _placementsByNodeId[placementId] = placement with { Instance = instance };
        return true;
    }
}

public sealed record WorldSceneGraphBuildSet(
    IReadOnlyDictionary<(int TileX, int TileY), WorldSceneGraphBuildResult> AdtGraphs,
    WorldSceneGraphBuildResult? ExternalGraph,
    IReadOnlyDictionary<string, WorldSceneGraphBuildResult> GraphByPlacementId)
{
    public IEnumerable<WorldSceneGraphBuildResult> EnumerateGraphs()
    {
        foreach (WorldSceneGraphBuildResult build in AdtGraphs.Values)
            yield return build;

        if (ExternalGraph is not null)
            yield return ExternalGraph;
    }

    public bool TryGetGraphForPlacement(string placementId, out WorldSceneGraphBuildResult? build)
    {
        if (string.IsNullOrWhiteSpace(placementId))
        {
            build = null;
            return false;
        }

        return GraphByPlacementId.TryGetValue(placementId, out build);
    }

    public WorldSceneGraphSnapshot CreateSnapshot()
    {
        int nodeCount = 0;
        int renderableCount = 0;
        int queryableCount = 0;
        int updateRequiredCount = 0;
        int nonRejectableCount = 0;
        int maxDepth = 0;
        Dictionary<WorldSceneNodeKind, int> nodeKindCounts = [];
        Dictionary<WorldSceneRenderPass, int> renderPassCounts = [];
        List<string> nodeIds = [];
        Vector3 rootBoundsMin = new(float.MaxValue);
        Vector3 rootBoundsMax = new(float.MinValue);
        bool foundBounds = false;

        foreach (WorldSceneGraphBuildResult build in EnumerateGraphs())
        {
            WorldSceneGraphSnapshot snapshot = build.Graph.CreateSnapshot();
            nodeCount += snapshot.NodeCount;
            renderableCount += snapshot.RenderableCount;
            queryableCount += snapshot.QueryableCount;
            updateRequiredCount += snapshot.UpdateRequiredCount;
            nonRejectableCount += snapshot.NonRejectableCount;
            maxDepth = Math.Max(maxDepth, snapshot.MaxDepth);
            nodeIds.AddRange(snapshot.NodeIds);

            foreach ((WorldSceneNodeKind kind, int count) in snapshot.NodeKindCounts)
                nodeKindCounts[kind] = nodeKindCounts.GetValueOrDefault(kind) + count;

            foreach ((WorldSceneRenderPass pass, int count) in snapshot.RenderPassCounts)
                renderPassCounts[pass] = renderPassCounts.GetValueOrDefault(pass) + count;

            if (snapshot.NodeCount > 0)
            {
                rootBoundsMin = Vector3.Min(rootBoundsMin, snapshot.RootBoundsMin);
                rootBoundsMax = Vector3.Max(rootBoundsMax, snapshot.RootBoundsMax);
                foundBounds = true;
            }
        }

        return new WorldSceneGraphSnapshot(
            nodeCount,
            renderableCount,
            queryableCount,
            updateRequiredCount,
            nonRejectableCount,
            maxDepth,
            new ReadOnlyDictionary<WorldSceneNodeKind, int>(nodeKindCounts),
            new ReadOnlyDictionary<WorldSceneRenderPass, int>(renderPassCounts),
            nodeIds,
            foundBounds ? rootBoundsMin : Vector3.Zero,
            foundBounds ? rootBoundsMax : Vector3.Zero);
    }
}

/// <summary>
/// Adapts the current runtime object lists into the graph hierarchy:
/// map -> tile or external bucket -> placement.
///
/// The adapter intentionally does not invent WMO group bounds or attachment
/// transforms. Those must come from a later client-backed portal/asset adapter;
/// unknown child bounds therefore keep their ancestor fail-open.
/// </summary>
public static class WorldSceneGraphObjectAdapter
{
    // Terrain uses the native ADT grid: tile Y maps to world X and tile X maps
    // to world Y. These bounds are authoritative even while child model assets
    // are unresolved, allowing an off-camera ADT to reject its full subtree.
    private const float AdtTileWorldSize = 533.33333f;
    private const float AdtMapOrigin = 32f * AdtTileWorldSize;
    private const float AdtCullMinZ = -100_000f;
    private const float AdtCullMaxZ = 100_000f;

    public static WorldSceneGraphBuildResult Build(
        IEnumerable<WorldSceneGraphObjectPlacement> placements,
        WorldSceneGraphAdapterOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(placements);
        options ??= new WorldSceneGraphAdapterOptions();

        ValidateMapOptions(options);

        List<WorldSceneGraphObjectPlacement> orderedPlacements = placements
            .OrderBy(placement => placement.Id, StringComparer.Ordinal)
            .ToList();
        Dictionary<string, WorldSceneGraphObjectPlacement> placementIndex = new(StringComparer.Ordinal);
        foreach (WorldSceneGraphObjectPlacement placement in orderedPlacements)
        {
            if (string.IsNullOrWhiteSpace(placement.Id))
                throw new ArgumentException("Every scene placement must have a stable id.", nameof(placements));
            if (!placement.IsExternal && !placement.Instance.HasTileCoordinate)
                throw new ArgumentException(
                    $"Placement '{placement.Id}' is not external and has no tile coordinate.",
                    nameof(placements));
            if (!placementIndex.TryAdd(placement.Id, placement))
                throw new InvalidOperationException($"Duplicate scene placement id '{placement.Id}'.");

            ValidatePlacement(placement);
        }

        string rootId = options.MapId.Trim();
        if (placementIndex.ContainsKey(rootId))
            throw new InvalidOperationException($"Placement id '{rootId}' conflicts with the map root.");

        List<SceneBucket> buckets = BuildBuckets(orderedPlacements);
        foreach (SceneBucket bucket in buckets)
        {
            if (placementIndex.ContainsKey(bucket.Id))
                throw new InvalidOperationException($"Placement id '{bucket.Id}' conflicts with a scene bucket.");
        }

        bool rootBoundsKnown = options.MapBoundsKnown || buckets.All(bucket => bucket.BoundsKnown);
        (Vector3 rootMin, Vector3 rootMax) = options.MapBoundsKnown
            ? (options.MapBoundsMin, options.MapBoundsMax)
            : UnionBounds(buckets.Where(bucket => bucket.BoundsKnown).Select(bucket => (bucket.BoundsMin, bucket.BoundsMax)));

        WorldSceneNode root = new(
            rootId,
            WorldSceneNodeKind.Map,
            Matrix4x4.Identity,
            rootMin,
            rootMax,
            boundsKnown: rootBoundsKnown,
            isQueryable: true);
        WorldSceneGraph graph = new(root);

        foreach (SceneBucket bucket in buckets)
            AttachBucket(graph, rootId, bucket);

        graph.ValidateInvariants();
        return new WorldSceneGraphBuildResult(
            graph,
            new ReadOnlyDictionary<string, WorldSceneGraphObjectPlacement>(placementIndex));
    }

    public static WorldSceneGraphBuildSet BuildPerAdt(
        IEnumerable<WorldSceneGraphObjectPlacement> placements,
        WorldSceneGraphAdapterOptions? options = null)
    {
        ArgumentNullException.ThrowIfNull(placements);
        options ??= new WorldSceneGraphAdapterOptions();

        ValidateMapOptions(options);
        List<WorldSceneGraphObjectPlacement> orderedPlacements = OrderAndValidatePlacements(placements);

        Dictionary<(int TileX, int TileY), WorldSceneGraphBuildResult> adtGraphs = [];
        foreach (IGrouping<(int TileX, int TileY), WorldSceneGraphObjectPlacement> tileGroup in orderedPlacements
            .Where(placement => !placement.IsExternal && placement.Instance.HasTileCoordinate)
            .GroupBy(placement => (placement.Instance.TileX, placement.Instance.TileY))
            .OrderBy(group => group.Key.TileX)
            .ThenBy(group => group.Key.TileY))
        {
            WorldSceneGraphBuildResult build = BuildAdtGraph(tileGroup.Key, tileGroup.ToList());
            adtGraphs.Add(tileGroup.Key, build);
        }

        List<WorldSceneGraphObjectPlacement> externalPlacements = orderedPlacements
            .Where(placement => placement.IsExternal)
            .ToList();
        WorldSceneGraphBuildResult? externalGraph = externalPlacements.Count == 0
            ? null
            : Build(externalPlacements, options);

        Dictionary<string, WorldSceneGraphBuildResult> graphByPlacementId = new(StringComparer.Ordinal);
        foreach (WorldSceneGraphBuildResult build in adtGraphs.Values)
        {
            foreach (string placementId in build.PlacementsByNodeId.Keys)
                graphByPlacementId.Add(placementId, build);
        }

        if (externalGraph is not null)
        {
            foreach (string placementId in externalGraph.PlacementsByNodeId.Keys)
                graphByPlacementId.Add(placementId, externalGraph);
        }

        return new WorldSceneGraphBuildSet(
            new ReadOnlyDictionary<(int TileX, int TileY), WorldSceneGraphBuildResult>(adtGraphs),
            externalGraph,
            new ReadOnlyDictionary<string, WorldSceneGraphBuildResult>(graphByPlacementId));
    }

    private static WorldSceneGraphBuildResult BuildAdtGraph(
        (int TileX, int TileY) tileKey,
        IReadOnlyList<WorldSceneGraphObjectPlacement> placements)
    {
        string rootId = $"world/tile/{tileKey.TileX:D2}/{tileKey.TileY:D2}";
        (Vector3 tileBoundsMin, Vector3 tileBoundsMax) = GetAdtTileBounds(tileKey);
        (Vector3 boundsMin, Vector3 boundsMax) = UnionBounds(
        [
            (tileBoundsMin, tileBoundsMax),
            .. placements
                .Where(placement => placement.Instance.BoundsResolved)
                .Select(placement => (placement.Instance.BoundsMin, placement.Instance.BoundsMax)),
        ]);

        WorldSceneNode root = new(
            rootId,
            WorldSceneNodeKind.Tile,
            Matrix4x4.Identity,
            boundsMin,
            boundsMax,
            boundsKnown: true,
            isQueryable: true,
            allowsUnresolvedDescendants: true);
        WorldSceneGraph graph = new(root);

        IEnumerable<IGrouping<(WorldSceneNodeKind Kind, string Key), WorldSceneGraphObjectPlacement>> bucketGroups =
            placements
                .Where(placement => placement.SpatialBucket is not null)
                .GroupBy(placement =>
                {
                    WorldSceneGraphSpatialBucket bucket = placement.SpatialBucket!.Value;
                    return (bucket.Kind, bucket.Key.Trim('/'));
                })
                .OrderBy(group => group.Key.Kind)
                .ThenBy(group => group.Key.Item2, StringComparer.Ordinal);

        foreach (IGrouping<(WorldSceneNodeKind Kind, string Key), WorldSceneGraphObjectPlacement> bucketGroup in bucketGroups)
        {
            List<WorldSceneGraphObjectPlacement> bucketPlacements = bucketGroup.ToList();
            bool bucketBoundsKnown = bucketPlacements.All(placement => placement.Instance.BoundsResolved);
            (Vector3 bucketBoundsMin, Vector3 bucketBoundsMax) = bucketBoundsKnown
                ? UnionBounds(bucketPlacements.Select(placement => (placement.Instance.BoundsMin, placement.Instance.BoundsMax)))
                : (Vector3.Zero, Vector3.Zero);
            SceneBucket bucket = new(
                $"{rootId}/{GetKindToken(bucketGroup.Key.Item1)}/{bucketGroup.Key.Item2}",
                bucketGroup.Key.Item1,
                bucketBoundsKnown,
                bucketBoundsMin,
                bucketBoundsMax,
                bucketPlacements,
                []);
            AttachBucket(graph, rootId, bucket);
        }

        foreach (WorldSceneGraphObjectPlacement placement in placements.Where(placement => placement.SpatialBucket is null))
            AttachPlacement(graph, rootId, placement);

        graph.ValidateInvariants();
        Dictionary<string, WorldSceneGraphObjectPlacement> placementIndex = placements
            .ToDictionary(placement => placement.Id, StringComparer.Ordinal);
        return new WorldSceneGraphBuildResult(
            graph,
            new ReadOnlyDictionary<string, WorldSceneGraphObjectPlacement>(placementIndex));
    }

    private static (Vector3 min, Vector3 max) GetAdtTileBounds((int TileX, int TileY) tileKey)
    {
        float minX = AdtMapOrigin - ((tileKey.TileY + 1) * AdtTileWorldSize);
        float maxX = AdtMapOrigin - (tileKey.TileY * AdtTileWorldSize);
        float minY = AdtMapOrigin - ((tileKey.TileX + 1) * AdtTileWorldSize);
        float maxY = AdtMapOrigin - (tileKey.TileX * AdtTileWorldSize);
        return (new Vector3(minX, minY, AdtCullMinZ), new Vector3(maxX, maxY, AdtCullMaxZ));
    }

    private static List<SceneBucket> BuildBuckets(
        IReadOnlyList<WorldSceneGraphObjectPlacement> placements)
    {
        Dictionary<string, List<WorldSceneGraphObjectPlacement>> placementGroups = new(StringComparer.Ordinal);
        foreach (WorldSceneGraphObjectPlacement placement in placements)
        {
            string bucketId = GetBucketId(placement);
            if (!placementGroups.TryGetValue(bucketId, out List<WorldSceneGraphObjectPlacement>? group))
            {
                group = [];
                placementGroups.Add(bucketId, group);
            }

            group.Add(placement);
        }

        List<SceneBucket> buckets = [];
        foreach ((string bucketId, List<WorldSceneGraphObjectPlacement> group) in placementGroups.OrderBy(pair => pair.Key, StringComparer.Ordinal))
        {
            bool boundsKnown = group.All(placement => placement.Instance.BoundsResolved);
            (Vector3 boundsMin, Vector3 boundsMax) = boundsKnown
                ? UnionBounds(group.Select(placement => (placement.Instance.BoundsMin, placement.Instance.BoundsMax)))
                : (Vector3.Zero, Vector3.Zero);
            WorldSceneNodeKind kind = group[0].IsExternal
                ? WorldSceneNodeKind.SyntheticProxy
                : WorldSceneNodeKind.Tile;

            List<WorldSceneGraphObjectPlacement> directPlacements = group
                .Where(placement => placement.SpatialBucket is null)
                .ToList();
            List<SceneBucket> childBuckets = [];
            foreach (IGrouping<(WorldSceneNodeKind Kind, string Key), WorldSceneGraphObjectPlacement> childGroup
                in group
                    .Where(placement => placement.SpatialBucket is not null)
                    .GroupBy(placement =>
                    {
                        WorldSceneGraphSpatialBucket spatialBucket = placement.SpatialBucket!.Value;
                        return (spatialBucket.Kind, spatialBucket.Key.Trim());
                    })
                    .OrderBy(pair => pair.Key.Kind)
                    .ThenBy(pair => pair.Key.Item2, StringComparer.Ordinal))
            {
                List<WorldSceneGraphObjectPlacement> childPlacements = childGroup.ToList();
                bool childBoundsKnown = childPlacements.All(placement => placement.Instance.BoundsResolved);
                (Vector3 childBoundsMin, Vector3 childBoundsMax) = childBoundsKnown
                    ? UnionBounds(childPlacements.Select(placement => (placement.Instance.BoundsMin, placement.Instance.BoundsMax)))
                    : (Vector3.Zero, Vector3.Zero);
                string childId = $"{bucketId}/{GetKindToken(childGroup.Key.Kind)}/{childGroup.Key.Item2.Trim('/')}";
                childBuckets.Add(new SceneBucket(
                    childId,
                    childGroup.Key.Kind,
                    childBoundsKnown,
                    childBoundsMin,
                    childBoundsMax,
                    childPlacements,
                    []));
            }

            buckets.Add(new SceneBucket(bucketId, kind, boundsKnown, boundsMin, boundsMax, directPlacements, childBuckets));
        }

        return buckets;
    }

    private static List<WorldSceneGraphObjectPlacement> OrderAndValidatePlacements(
        IEnumerable<WorldSceneGraphObjectPlacement> placements)
    {
        List<WorldSceneGraphObjectPlacement> orderedPlacements = placements
            .OrderBy(placement => placement.Id, StringComparer.Ordinal)
            .ToList();
        HashSet<string> placementIds = new(StringComparer.Ordinal);
        foreach (WorldSceneGraphObjectPlacement placement in orderedPlacements)
        {
            if (string.IsNullOrWhiteSpace(placement.Id))
                throw new ArgumentException("Every scene placement must have a stable id.", nameof(placements));
            if (!placement.IsExternal && !placement.Instance.HasTileCoordinate)
                throw new ArgumentException(
                    $"Placement '{placement.Id}' is not external and has no tile coordinate.",
                    nameof(placements));
            if (!placementIds.Add(placement.Id))
                throw new InvalidOperationException($"Duplicate scene placement id '{placement.Id}'.");

            ValidatePlacement(placement);
        }

        return orderedPlacements;
    }

    private static string GetBucketId(WorldSceneGraphObjectPlacement placement)
    {
        if (placement.IsExternal || !placement.Instance.HasTileCoordinate)
            return $"world/external/{GetKindToken(placement.Kind)}";

        return $"world/tile/{placement.Instance.TileX:D2}/{placement.Instance.TileY:D2}";
    }

    private static void AttachBucket(WorldSceneGraph graph, string parentId, SceneBucket bucket)
    {
        WorldSceneNode bucketNode = new(
            bucket.Id,
            bucket.Kind,
            Matrix4x4.Identity,
            bucket.BoundsMin,
            bucket.BoundsMax,
            boundsKnown: bucket.BoundsKnown,
            isQueryable: true);
        graph.Attach(parentId, bucketNode);

        foreach (SceneBucket childBucket in bucket.Children)
            AttachBucket(graph, bucket.Id, childBucket);

        foreach (WorldSceneGraphObjectPlacement placement in bucket.Placements)
            AttachPlacement(graph, bucket.Id, placement);
    }

    private static void AttachPlacement(
        WorldSceneGraph graph,
        string parentId,
        WorldSceneGraphObjectPlacement placement)
    {
        WorldSceneNode objectNode = new(
            placement.Id,
            placement.Kind,
            placement.Instance.Transform,
            placement.Instance.LocalBoundsMin,
            placement.Instance.LocalBoundsMax,
            boundsKnown: placement.Instance.BoundsResolved,
            isRenderable: true,
            isQueryable: placement.IsQueryable,
            requiresUpdate: placement.RequiresUpdate,
            assetKey: placement.Instance.ModelKey,
            renderPassMask: placement.RenderPassMask);
        graph.Attach(parentId, objectNode);

        if (placement.Children is null)
            return;

        foreach (WorldSceneGraphChildNode child in placement.Children)
        {
            WorldSceneNode childNode = new(
                child.Id,
                child.Kind,
                child.LocalTransform,
                child.LocalBoundsMin,
                child.LocalBoundsMax,
                boundsKnown: child.BoundsKnown,
                isRenderable: child.IsRenderable,
                isQueryable: child.IsQueryable,
                requiresUpdate: child.RequiresUpdate,
                assetKey: child.AssetKey,
                renderPassMask: child.RenderPassMask,
                portalGroup: child.PortalGroup);
            graph.Attach(objectNode.Id, childNode);
        }
    }

    private static string GetKindToken(WorldSceneNodeKind kind)
    {
        return kind switch
        {
            WorldSceneNodeKind.WmoPlacement => "wmo",
            WorldSceneNodeKind.M2Placement => "m2",
            _ => kind.ToString().ToLowerInvariant(),
        };
    }

    private static void ValidateMapOptions(WorldSceneGraphAdapterOptions options)
    {
        if (string.IsNullOrWhiteSpace(options.MapId))
            throw new ArgumentException("The scene map id must not be empty.", nameof(options));
        if (options.MapBoundsKnown)
            ValidateBounds(options.MapBoundsMin, options.MapBoundsMax, "map bounds");
    }

    private static void ValidatePlacement(WorldSceneGraphObjectPlacement placement)
    {
        ValidateVector(placement.Instance.Transform.Translation, $"placement '{placement.Id}' translation");
        if (placement.SpatialBucket is WorldSceneGraphSpatialBucket spatialBucket)
        {
            if (spatialBucket.Kind is WorldSceneNodeKind.Map
                or WorldSceneNodeKind.Tile
                or WorldSceneNodeKind.SyntheticProxy)
            {
                throw new ArgumentException(
                    $"Placement '{placement.Id}' uses invalid spatial bucket kind '{spatialBucket.Kind}'.",
                    nameof(placement));
            }

            if (string.IsNullOrWhiteSpace(spatialBucket.Key)
                || spatialBucket.Key.Contains('\\')
                || spatialBucket.Key.Contains("..", StringComparison.Ordinal))
            {
                throw new ArgumentException(
                    $"Placement '{placement.Id}' uses an invalid spatial bucket key.",
                    nameof(placement));
            }
        }
        if (placement.Instance.BoundsResolved)
        {
            ValidateBounds(placement.Instance.LocalBoundsMin, placement.Instance.LocalBoundsMax, $"placement '{placement.Id}' local bounds");
            ValidateBounds(placement.Instance.BoundsMin, placement.Instance.BoundsMax, $"placement '{placement.Id}' world bounds");
        }
    }

    private static (Vector3 min, Vector3 max) UnionBounds(IEnumerable<(Vector3 min, Vector3 max)> bounds)
    {
        Vector3 min = new(float.MaxValue);
        Vector3 max = new(float.MinValue);
        bool found = false;
        foreach ((Vector3 candidateMin, Vector3 candidateMax) in bounds)
        {
            min = Vector3.Min(min, candidateMin);
            max = Vector3.Max(max, candidateMax);
            found = true;
        }

        return found ? (min, max) : (Vector3.Zero, Vector3.Zero);
    }

    private static void ValidateBounds(Vector3 min, Vector3 max, string description)
    {
        ValidateVector(min, $"{description} minimum");
        ValidateVector(max, $"{description} maximum");
        if (min.X > max.X || min.Y > max.Y || min.Z > max.Z)
            throw new ArgumentException($"{description} must be ordered min-to-max.");
    }

    private static void ValidateVector(Vector3 value, string description)
    {
        if (!float.IsFinite(value.X) || !float.IsFinite(value.Y) || !float.IsFinite(value.Z))
            throw new ArgumentException($"{description} must contain only finite values.");
    }

    private sealed record SceneBucket(
        string Id,
        WorldSceneNodeKind Kind,
        bool BoundsKnown,
        Vector3 BoundsMin,
        Vector3 BoundsMax,
        IReadOnlyList<WorldSceneGraphObjectPlacement> Placements,
        IReadOnlyList<SceneBucket> Children);
}
