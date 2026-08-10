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
    IReadOnlyList<WorldSceneGraphChildNode>? Children = null);

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

public sealed record WorldSceneGraphBuildResult(
    WorldSceneGraph Graph,
    IReadOnlyDictionary<string, WorldSceneGraphObjectPlacement> PlacementsByNodeId);

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
        {
            WorldSceneNode bucketNode = new(
                bucket.Id,
                bucket.Kind,
                Matrix4x4.Identity,
                bucket.BoundsMin,
                bucket.BoundsMax,
                boundsKnown: bucket.BoundsKnown,
                isQueryable: true);
            graph.Attach(rootId, bucketNode);

            foreach (WorldSceneGraphObjectPlacement placement in bucket.Placements)
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
                graph.Attach(bucket.Id, objectNode);

                if (placement.Children is null)
                    continue;

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
        }

        graph.ValidateInvariants();
        return new WorldSceneGraphBuildResult(
            graph,
            new ReadOnlyDictionary<string, WorldSceneGraphObjectPlacement>(placementIndex));
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
            buckets.Add(new SceneBucket(bucketId, kind, boundsKnown, boundsMin, boundsMax, group));
        }

        return buckets;
    }

    private static string GetBucketId(WorldSceneGraphObjectPlacement placement)
    {
        if (placement.IsExternal || !placement.Instance.HasTileCoordinate)
            return $"world/external/{GetKindToken(placement.Kind)}";

        return $"world/tile/{placement.Instance.TileX:D2}/{placement.Instance.TileY:D2}";
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
        IReadOnlyList<WorldSceneGraphObjectPlacement> Placements);
}
