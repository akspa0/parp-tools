using System.Numerics;

namespace WowViewer.Core.Runtime.World.SceneGraph;

public static class SyntheticWorldWorkloadBuilder
{
    private const float RegionSize = 1024f;
    private const float RegionHeight = 128f;

    public static SyntheticWorldWorkloadBuildResult Build(SyntheticWorldWorkloadDefinition definition)
    {
        ArgumentNullException.ThrowIfNull(definition);
        definition.Validate();

        int regionColumns = CeilingSquareRoot(definition.ResidentRegionCount);
        int regionRows = (definition.ResidentRegionCount + regionColumns - 1) / regionColumns;
        Vector3 rootMin = new(-16f, -16f, -32f);
        Vector3 rootMax = new(regionColumns * RegionSize + 16f, regionRows * RegionSize + 16f, RegionHeight + 32f);
        WorldSceneNode root = CreateNode(
            "synthetic/map",
            null,
            WorldSceneNodeKind.Map,
            Matrix4x4.Identity,
            rootMin,
            rootMax,
            boundsKnown: true,
            isRenderable: false,
            isQueryable: false,
            requiresUpdate: false,
            assetKey: null,
            WorldSceneRenderPass.None,
            portalGroup: null);

        WorldSceneGraph graph = new(root);
        List<string> wmoGroupIds = [];
        List<(string id, string parentId)> chunkIds = [];

        for (int regionIndex = 0; regionIndex < definition.ResidentRegionCount; regionIndex++)
        {
            int regionX = regionIndex % regionColumns;
            int regionY = regionIndex / regionColumns;
            Vector3 regionOrigin = new(regionX * RegionSize, regionY * RegionSize, 0f);
            string regionId = $"synthetic/region/{regionIndex:D4}";
            graph.Attach(
                root.Id,
                CreateNode(
                    regionId,
                    root.Id,
                    WorldSceneNodeKind.Tile,
                    Matrix4x4.CreateTranslation(regionOrigin),
                    Vector3.Zero,
                    new Vector3(RegionSize, RegionSize, RegionHeight),
                    boundsKnown: true,
                    isRenderable: false,
                    isQueryable: true,
                    requiresUpdate: false,
                    assetKey: null,
                    WorldSceneRenderPass.None,
                    portalGroup: null));

            int chunkColumns = CeilingSquareRoot(definition.ChunksPerRegion);
            float chunkSize = RegionSize / chunkColumns;
            for (int chunkIndex = 0; chunkIndex < definition.ChunksPerRegion; chunkIndex++)
            {
                int chunkX = chunkIndex % chunkColumns;
                int chunkY = chunkIndex / chunkColumns;
                Vector3 chunkOrigin = new(chunkX * chunkSize, chunkY * chunkSize, 0f);
                string chunkId = $"{regionId}/chunk/{chunkIndex:D4}";
                graph.Attach(
                    regionId,
                    CreateNode(
                        chunkId,
                        regionId,
                        WorldSceneNodeKind.Chunk,
                        Matrix4x4.CreateTranslation(chunkOrigin),
                        Vector3.Zero,
                        new Vector3(chunkSize, chunkSize, RegionHeight),
                        boundsKnown: true,
                        isRenderable: true,
                        isQueryable: true,
                        requiresUpdate: false,
                        assetKey: null,
                        WorldSceneRenderPass.Opaque,
                        portalGroup: null));
                chunkIds.Add((chunkId, regionId));
            }

            for (int placementIndex = 0; placementIndex < definition.WmoPlacements; placementIndex++)
            {
                Vector3 placementOrigin = new(
                    128f + StableFraction(definition.Seed, regionIndex, placementIndex, 0) * 512f,
                    128f + StableFraction(definition.Seed, regionIndex, placementIndex, 1) * 512f,
                    0f);
                string placementId = $"{regionId}/wmo/{placementIndex:D4}";
                graph.Attach(
                    regionId,
                    CreateNode(
                        placementId,
                        regionId,
                        WorldSceneNodeKind.WmoPlacement,
                        Matrix4x4.CreateTranslation(placementOrigin),
                        Vector3.Zero,
                        new Vector3(160f, 160f, 112f),
                        boundsKnown: true,
                        isRenderable: false,
                        isQueryable: true,
                        requiresUpdate: false,
                        assetKey: $"wmo/asset-{placementIndex % Math.Max(1, definition.WmoPlacements):D4}",
                        WorldSceneRenderPass.None,
                        portalGroup: null));

                int groupColumns = Math.Max(1, CeilingSquareRoot(definition.WmoGroupsPerPlacement));
                float groupSize = 160f / groupColumns;
                for (int groupIndex = 0; groupIndex < definition.WmoGroupsPerPlacement; groupIndex++)
                {
                    int groupX = groupIndex % groupColumns;
                    int groupY = groupIndex / groupColumns;
                    string groupId = $"{placementId}/group/{groupIndex:D4}";
                    graph.Attach(
                        placementId,
                        CreateNode(
                            groupId,
                            placementId,
                            WorldSceneNodeKind.WmoGroup,
                            Matrix4x4.CreateTranslation(groupX * groupSize, groupY * groupSize, 0f),
                            Vector3.Zero,
                            new Vector3(groupSize * 0.9f, groupSize * 0.9f, 96f),
                            boundsKnown: true,
                            isRenderable: true,
                            isQueryable: true,
                            requiresUpdate: false,
                            assetKey: $"wmo/asset-{placementIndex % Math.Max(1, definition.WmoPlacements):D4}",
                            PassForIndex(definition.RenderPassMix, groupIndex),
                            portalGroup: groupIndex));
                    wmoGroupIds.Add(groupId);
                }
            }

            for (int overlayIndex = 0; overlayIndex < definition.Pm4OverlayCount; overlayIndex++)
            {
                string overlayId = $"{regionId}/pm4/{overlayIndex:D4}";
                graph.Attach(
                    regionId,
                    CreateNode(
                        overlayId,
                        regionId,
                        WorldSceneNodeKind.Pm4Structure,
                        Matrix4x4.CreateTranslation(32f + overlayIndex * 96f, 32f, 0f),
                        Vector3.Zero,
                        new Vector3(64f, 64f, 48f),
                        boundsKnown: true,
                        isRenderable: true,
                        isQueryable: true,
                        requiresUpdate: false,
                        assetKey: $"pm4/overlay-{overlayIndex:D4}",
                        WorldSceneRenderPass.Overlay,
                        portalGroup: null));
            }
        }

        for (int placementIndex = 0; placementIndex < definition.M2Placements; placementIndex++)
        {
            (string chunkId, _) = chunkIds[placementIndex % chunkIds.Count];
            int chunkIndex = placementIndex % definition.ChunksPerRegion;
            int chunkColumns = CeilingSquareRoot(definition.ChunksPerRegion);
            float chunkSize = RegionSize / chunkColumns;
            float objectSize = MathF.Min(24f, chunkSize * 0.18f);
            Vector3 placementOrigin = new(
                StableFraction(definition.Seed, placementIndex, chunkIndex, 2) * MathF.Max(1f, chunkSize - objectSize),
                StableFraction(definition.Seed, placementIndex, chunkIndex, 3) * MathF.Max(1f, chunkSize - objectSize),
                StableFraction(definition.Seed, placementIndex, chunkIndex, 4) * 32f);
            string placementId = $"{chunkId}/m2/{placementIndex:D5}";
            graph.Attach(
                chunkId,
                CreateNode(
                    placementId,
                    chunkId,
                    WorldSceneNodeKind.M2Placement,
                    Matrix4x4.CreateTranslation(placementOrigin),
                    Vector3.Zero,
                    new Vector3(objectSize, objectSize, objectSize),
                    boundsKnown: true,
                    isRenderable: true,
                    isQueryable: true,
                    requiresUpdate: placementIndex % 3 == 0,
                    assetKey: $"m2/asset-{placementIndex % definition.RepeatedAssetCount:D4}",
                    PassForIndex(definition.RenderPassMix, placementIndex + 1),
                    portalGroup: null));
        }

        List<SyntheticPortalLink> portalLinks = [];
        if (wmoGroupIds.Count > 1)
        {
            for (int linkIndex = 0; linkIndex < definition.PortalLinkCount; linkIndex++)
            {
                string source = wmoGroupIds[linkIndex % wmoGroupIds.Count];
                string destination = wmoGroupIds[(linkIndex + 1) % wmoGroupIds.Count];
                if (!source.Equals(destination, StringComparison.Ordinal))
                    portalLinks.Add(new SyntheticPortalLink(source, destination));
            }
        }

        SyntheticWorldWorkload manifest = new SyntheticWorldWorkload(
            SyntheticWorldWorkload.CurrentSchema,
            SyntheticWorldWorkload.CurrentWorkloadClass,
            definition.FixtureName,
            definition.Seed,
            definition.ResidentRegionCount,
            definition.ChunksPerRegion,
            definition.WmoPlacements,
            definition.WmoGroupsPerPlacement,
            definition.M2Placements,
            definition.RepeatedAssetCount,
            definition.Pm4OverlayCount,
            definition.PortalLinkCount,
            definition.RenderPassMix,
            definition.Camera,
            graph.EnumerateDepthFirst().Select(CreateDescriptor).ToList(),
            portalLinks,
            string.Empty).WithComputedManifestHash();

        manifest.Validate();
        return new SyntheticWorldWorkloadBuildResult(graph, manifest);
    }

    private static WorldSceneNode CreateNode(
        string id,
        string? parentId,
        WorldSceneNodeKind kind,
        Matrix4x4 localTransform,
        Vector3 localBoundsMin,
        Vector3 localBoundsMax,
        bool boundsKnown,
        bool isRenderable,
        bool isQueryable,
        bool requiresUpdate,
        string? assetKey,
        WorldSceneRenderPass renderPassMask,
        int? portalGroup)
    {
        return new WorldSceneNode(
            id,
            kind,
            localTransform,
            localBoundsMin,
            localBoundsMax,
            boundsKnown,
            isRenderable,
            isQueryable,
            requiresUpdate,
            assetKey,
            renderPassMask,
            portalGroup);
    }

    private static SyntheticWorldNodeDescriptor CreateDescriptor(WorldSceneNode node)
    {
        return new SyntheticWorldNodeDescriptor(
            node.Id,
            node.Parent?.Id,
            node.Kind,
            MatrixToArray(node.LocalTransform),
            VectorToArray(node.LocalBoundsMin),
            VectorToArray(node.LocalBoundsMax),
            node.BoundsKnown,
            node.CanRejectSubtree,
            node.IsRenderable,
            node.IsQueryable,
            node.RequiresUpdate,
            node.AssetKey,
            node.RenderPassMask,
            node.PortalGroup);
    }

    private static WorldSceneRenderPass PassForIndex(SyntheticRenderPassMix mix, int index)
    {
        List<WorldSceneRenderPass> passes = [];
        AddPasses(passes, WorldSceneRenderPass.Opaque, mix.Opaque);
        AddPasses(passes, WorldSceneRenderPass.AlphaTested, mix.AlphaTested);
        AddPasses(passes, WorldSceneRenderPass.Transparent, mix.Transparent);
        AddPasses(passes, WorldSceneRenderPass.Liquid, mix.Liquid);
        AddPasses(passes, WorldSceneRenderPass.Overlay, mix.Overlay);
        return passes[index % passes.Count];
    }

    private static void AddPasses(List<WorldSceneRenderPass> passes, WorldSceneRenderPass pass, int count)
    {
        for (int index = 0; index < count; index++)
            passes.Add(pass);
    }

    private static int CeilingSquareRoot(int value)
    {
        int root = (int)MathF.Ceiling(MathF.Sqrt(value));
        return Math.Max(1, root);
    }

    private static float StableFraction(int seed, int first, int second, int salt)
    {
        unchecked
        {
            uint value = (uint)seed;
            value ^= (uint)(first + 0x9E3779B9);
            value = (value * 16777619u) ^ (uint)(second + 0x85EBCA6B);
            value = (value * 16777619u) ^ (uint)(salt + 0xC2B2AE35);
            value ^= value >> 16;
            return (value % 10000u) / 10000f;
        }
    }

    private static float[] MatrixToArray(Matrix4x4 matrix) =>
    [
        matrix.M11, matrix.M12, matrix.M13, matrix.M14,
        matrix.M21, matrix.M22, matrix.M23, matrix.M24,
        matrix.M31, matrix.M32, matrix.M33, matrix.M34,
        matrix.M41, matrix.M42, matrix.M43, matrix.M44
    ];

    private static float[] VectorToArray(Vector3 vector) => [vector.X, vector.Y, vector.Z];
}
