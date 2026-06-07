using System.Numerics;
using System.Security.Cryptography;
using System.Text;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Research;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Matching;

public static class Pm4ObjectSegmentBuilder
{
    public static IReadOnlyList<Pm4BuiltObjectSegment> Build(string pm4Path)
    {
        if (!Pm4CoordinateService.TryParseTileCoordinates(pm4Path, out int tileX, out int tileY))
            throw new InvalidOperationException($"Could not parse tile coordinates from '{pm4Path}'.");

        return Build(Pm4ResearchReader.ReadFile(pm4Path), tileX, tileY);
    }

    public static IReadOnlyList<Pm4BuiltObjectSegment> Build(Pm4ResearchDocument document, int tileX, int tileY)
    {
        IReadOnlyList<Vector3> meshVertices = document.KnownChunks.Msvt;
        IReadOnlyList<uint> meshIndices = document.KnownChunks.Msvi;
        List<IndexedSurface> indexedSurfaces = document.KnownChunks.Msur
            .Select(static (surface, surfaceIndex) => new IndexedSurface(surfaceIndex, surface))
            .Where(static indexed => indexed.Surface.IndexCount >= 3)
            .ToList();
        if (indexedSurfaces.Count == 0)
            return Array.Empty<Pm4BuiltObjectSegment>();

        HashSet<ushort> reusedLow16ObjectIds = document.KnownChunks.Msur
            .Where(static surface => surface.Ck24 != 0u && surface.Ck24ObjectId != 0)
            .GroupBy(static surface => surface.Ck24ObjectId)
            .Where(static group => group.Select(static surface => surface.Ck24).Distinct().Count() > 1)
            .Select(static group => group.Key)
            .ToHashSet();

        List<Pm4CorrelationObjectInput> inputs = [];
        List<PendingSegment> pendingSegments = [];
        IReadOnlyList<SeedGroup> seedGroups = BuildSeedGroups(indexedSurfaces);
        bool fallbackTileLocal = Pm4PlacementMath.IsLikelyTileLocal(meshVertices);
        uint? field04 = document.KnownChunks.Mshd?.Field04;
        string tileCoordinate = $"{tileX}_{tileY}";
        int nextObjectPartId = 0;

        foreach (SeedGroup seedGroup in seedGroups)
        {
            List<Pm4MsurEntry> seedSurfaces = seedGroup.Surfaces.Select(static surface => surface.Surface).ToList();
            Pm4AxisConvention axisConvention = Pm4PlacementMath.DetectAxisConventionBySurfaceNormals(meshVertices, meshIndices, seedSurfaces);
            List<Pm4MprlEntry> seedRefs = CollectLinkedPositionRefs(document, seedGroup.Surfaces);
            Pm4CoordinateMode fallbackMode = fallbackTileLocal ? Pm4CoordinateMode.TileLocal : Pm4CoordinateMode.WorldSpace;
            Pm4CoordinateModeResolution coordinateModeResolution = Pm4PlacementMath.ResolveCoordinateMode(
                meshVertices,
                meshIndices,
                seedSurfaces,
                seedRefs,
                seedRefs,
                tileX,
                tileY,
                axisConvention,
                fallbackMode);
            Pm4PlacementSolution seedPlacement = Pm4PlacementMath.ResolvePlacementSolution(
                meshVertices,
                meshIndices,
                seedSurfaces,
                seedRefs,
                seedRefs,
                tileX,
                tileY,
                coordinateModeResolution.CoordinateMode,
                axisConvention);

            List<List<IndexedSurface>> linkedGroups = seedGroup.RequiresConnectivitySeedSplit
                ? SplitByConnectivity(document, seedGroup.Surfaces)
                : SplitByMslkGroupObjectId(document, seedGroup.Surfaces);

            foreach (List<IndexedSurface> linkedGroup in linkedGroups)
            {
                if (linkedGroup.Count == 0)
                    continue;

                uint dominantLinkGroupObjectId = SelectDominantGroupObjectId(document, linkedGroup);
                List<Pm4MsurEntry> linkedSurfaces = linkedGroup.Select(static surface => surface.Surface).ToList();
                List<Pm4MprlEntry> linkedRefs = CollectLinkedPositionRefs(document, linkedGroup);
                Pm4CoordinateModeResolution linkedModeResolution = Pm4PlacementMath.ResolveCoordinateMode(
                    meshVertices,
                    meshIndices,
                    linkedSurfaces,
                    linkedRefs,
                    linkedRefs,
                    tileX,
                    tileY,
                    axisConvention,
                    coordinateModeResolution.CoordinateMode);
                Pm4PlacementSolution linkedPlacement = Pm4PlacementMath.ResolvePlacementSolution(
                    meshVertices,
                    meshIndices,
                    linkedSurfaces,
                    linkedRefs,
                    linkedRefs,
                    tileX,
                    tileY,
                    linkedModeResolution.CoordinateMode,
                    axisConvention);

                List<List<Pm4MsurEntry>> mscnGroups = !seedGroup.RequiresConnectivitySeedSplit
                    ? linkedSurfaces.GroupBy(static surface => surface.MscnRefIndex).Select(static group => group.ToList()).ToList()
                    : [linkedSurfaces];

                foreach (List<Pm4MsurEntry> mscnGroup in mscnGroups)
                {
                    List<IndexedSurface> matchingIndexed = linkedGroup.Where(surface => mscnGroup.Contains(surface.Surface)).ToList();
                    List<List<Pm4MsurEntry>> components = !seedGroup.RequiresConnectivitySeedSplit
                        ? SplitByConnectivity(document, matchingIndexed).Select(static component => component.Select(static indexed => indexed.Surface).ToList()).ToList()
                        : [mscnGroup];

                    foreach (List<Pm4MsurEntry> component in components)
                    {
                        List<Vector3> vertices = Pm4PlacementMath.CollectSurfaceVertices(meshVertices, meshIndices, component);
                        if (vertices.Count == 0)
                            continue;

                        List<IndexedSurface> componentIndexed = matchingIndexed.Where(surface => component.Contains(surface.Surface)).OrderBy(static surface => surface.SurfaceIndex).ToList();
                        List<Pm4MprlEntry> componentRefs = CollectLinkedPositionRefs(document, componentIndexed);
                        List<Vector3> worldPoints = new(vertices.Count);
                        for (int index = 0; index < vertices.Count; index++)
                            worldPoints.Add(Pm4PlacementMath.ConvertPm4VertexToWorld(vertices[index], linkedPlacement));

                        byte dominantGroupKey = SelectDominantSurfaceValue(component, static surface => surface.GroupKey);
                        byte dominantAttributeMask = SelectDominantSurfaceValue(component, static surface => surface.AttributeMask);
                        uint dominantMscnRefIndex = SelectDominantSurfaceValue(component, static surface => surface.MscnRefIndex);
                        float averageSurfaceHeight = component.Average(static surface => surface.Height);
                        int objectPartId = nextObjectPartId++;
                        uint internalGroupCk24 = seedGroup.DisplayCk24 == 0u ? 0x80000000u | (uint)objectPartId : seedGroup.DisplayCk24;

                        Pm4CorrelationObjectInput input = new(
                            tileX,
                            tileY,
                            new Pm4ObjectGroupKey(tileX, tileY, internalGroupCk24),
                            new Pm4CorrelationObjectDescriptor(
                                seedGroup.DisplayCk24,
                                seedGroup.DisplayCk24Type,
                                objectPartId,
                                dominantLinkGroupObjectId,
                                component.Count,
                                componentRefs.Count,
                                dominantGroupKey,
                                dominantAttributeMask,
                                dominantMscnRefIndex,
                                averageSurfaceHeight),
                            worldPoints,
                            seedPlacement.WorldPivot);

                        IReadOnlyList<uint> linkGroupIds = CollectDistinctLinkGroupIds(document, componentIndexed);
                        IReadOnlyList<Pm4ObjectSegmentSurface> segmentSurfaces = componentIndexed
                            .Select(static surface => new Pm4ObjectSegmentSurface(
                                surface.SurfaceIndex,
                                surface.Surface.GroupKey,
                                surface.Surface.AttributeMask,
                                surface.Surface.IndexCount,
                                surface.Surface.Height,
                                surface.Surface.MsviFirstIndex,
                                surface.Surface.MscnRefIndex,
                                surface.Surface._0x1C,
                                surface.Surface.Ck24,
                                surface.Surface.Ck24Type,
                                surface.Surface.Ck24ObjectId,
                                surface.Surface.Normal))
                            .ToList();
                        Pm4LinkedPositionRefSummary anchorSummary = Pm4PlacementMath.SummarizeLinkedPositionRefs(componentRefs);
                        IReadOnlyList<Vector2> anchorPlanarPoints = BuildAnchorPlanarPoints(componentRefs);
                        Pm4SegmentConfidenceFlags confidenceFlags = ResolveConfidenceFlags(
                            seedGroup,
                            linkGroupIds,
                            componentRefs,
                            field04,
                            reusedLow16ObjectIds.Contains(seedGroup.DisplayCk24ObjectId));
                        IReadOnlyList<int> surfaceIndices = componentIndexed.Select(static surface => surface.SurfaceIndex).ToList();
                        string segmentId = BuildSegmentId(
                            tileCoordinate,
                            field04,
                            seedGroup.DisplayCk24,
                            seedGroup.DisplayCk24Type,
                            seedGroup.DisplayCk24ObjectId,
                            surfaceIndices,
                            linkGroupIds);

                        Pm4ObjectSegment segment = new(
                            segmentId,
                            seedGroup.DisplayCk24,
                            seedGroup.DisplayCk24Type,
                            seedGroup.DisplayCk24ObjectId,
                            [tileCoordinate],
                            field04.HasValue ? [field04.Value] : Array.Empty<uint>(),
                            component.Count,
                            component.Sum(static surface => surface.IndexCount),
                            linkGroupIds,
                            dominantLinkGroupObjectId,
                            confidenceFlags);

                        inputs.Add(input);
                        pendingSegments.Add(new PendingSegment(
                            segment,
                            anchorSummary,
                            anchorPlanarPoints,
                            segmentSurfaces,
                            linkedPlacement.CoordinateMode,
                            axisConvention,
                            linkedPlacement.PlanarTransform,
                            linkedPlacement.WorldYawCorrectionRadians * 180f / MathF.PI));
                    }
                }
            }
        }

        IReadOnlyList<Pm4CorrelationObjectState> states = Pm4CorrelationMath.BuildObjectStates(inputs);
        List<Pm4BuiltObjectSegment> builtSegments = new(states.Count);
        for (int index = 0; index < states.Count; index++)
        {
            PendingSegment pending = pendingSegments[index];
            Pm4SegmentSignalRecord signal = Pm4SegmentSignalExtractor.Extract(pending.Segment, states[index], pending.AnchorSummary, pending.Surfaces);
            builtSegments.Add(new Pm4BuiltObjectSegment(
                pending.Segment,
                signal,
                states[index],
                pending.AnchorSummary,
                pending.AnchorPlanarPoints,
                pending.Surfaces,
                pending.CoordinateMode,
                pending.AxisConvention,
                pending.PlanarTransform,
                pending.FrameYawDegrees));
        }

        return builtSegments;
    }

    private static Pm4SegmentConfidenceFlags ResolveConfidenceFlags(
        SeedGroup seedGroup,
        IReadOnlyList<uint> linkGroupIds,
        IReadOnlyList<Pm4MprlEntry> positionRefs,
        uint? field04,
        bool reusedLow16ObjectId)
    {
        Pm4SegmentConfidenceFlags flags = Pm4SegmentConfidenceFlags.None;
        if (seedGroup.DisplayCk24 == 0u)
            flags |= Pm4SegmentConfidenceFlags.ZeroCk24Seed;
        if (seedGroup.RequiresConnectivitySeedSplit)
            flags |= Pm4SegmentConfidenceFlags.UsedConnectivityFallback;
        if (linkGroupIds.Count > 1)
            flags |= Pm4SegmentConfidenceFlags.MultipleLinkGroupIds;
        if (linkGroupIds.Count == 0)
            flags |= Pm4SegmentConfidenceFlags.HasUnlinkedSurfaces;
        if (positionRefs.Count == 0)
            flags |= Pm4SegmentConfidenceFlags.MissingPositionRefs;
        if (reusedLow16ObjectId)
            flags |= Pm4SegmentConfidenceFlags.ReusedLow16ObjectId;
        if (!field04.HasValue)
            flags |= Pm4SegmentConfidenceFlags.SpansMultipleField04Values;

        return flags;
    }

    private static string BuildSegmentId(
        string tileCoordinate,
        uint? field04,
        uint ck24,
        byte ck24Type,
        ushort ck24ObjectId,
        IReadOnlyList<int> surfaceIndices,
        IReadOnlyList<uint> linkGroupIds)
    {
        string canonicalIdentity =
            $"tile={tileCoordinate}|field04={(field04.HasValue ? field04.Value.ToString() : "none")}|ck24={ck24:X6}|type={ck24Type:X2}|low16={ck24ObjectId}|surfaces={string.Join(",", surfaceIndices)}|groups={string.Join(",", linkGroupIds)}";
        byte[] bytes = SHA256.HashData(Encoding.UTF8.GetBytes(canonicalIdentity));
        return $"pm4seg-{Convert.ToHexString(bytes.AsSpan(0, 8)).ToLowerInvariant()}";
    }

    private static IReadOnlyList<SeedGroup> BuildSeedGroups(IReadOnlyList<IndexedSurface> indexedSurfaces)
    {
        List<SeedGroup> groups = [];

        foreach (IGrouping<uint, IndexedSurface> group in indexedSurfaces
            .Where(static surface => surface.Surface.Ck24 != 0u)
            .GroupBy(static surface => surface.Surface.Ck24)
            .OrderBy(static group => group.Key))
        {
            IndexedSurface exemplar = group.First();
            groups.Add(new SeedGroup(
                exemplar.Surface.Ck24,
                exemplar.Surface.Ck24Type,
                exemplar.Surface.Ck24ObjectId,
                false,
                group.OrderBy(static item => item.SurfaceIndex).ToList()));
        }

        foreach (IGrouping<(byte GroupKey, byte AttributeMask), IndexedSurface> group in indexedSurfaces
            .Where(static surface => surface.Surface.Ck24 == 0u)
            .GroupBy(static surface => (surface.Surface.GroupKey, surface.Surface.AttributeMask))
            .OrderBy(static group => group.Key.GroupKey)
            .ThenBy(static group => group.Key.AttributeMask))
        {
            groups.Add(new SeedGroup(
                0u,
                0,
                0,
                true,
                group.OrderBy(static item => item.SurfaceIndex).ToList()));
        }

        return groups;
    }

    private static IReadOnlyList<uint> CollectDistinctLinkGroupIds(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        if (document.KnownChunks.Mslk.Count == 0 || surfaces.Count == 0)
            return Array.Empty<uint>();

        HashSet<int> surfaceIndices = surfaces.Select(static surface => surface.SurfaceIndex).ToHashSet();
        return document.KnownChunks.Mslk
            .Where(link => link.GroupObjectId != 0 && surfaceIndices.Contains(link.RefIndex))
            .Select(static link => link.GroupObjectId)
            .Distinct()
            .OrderBy(static groupId => groupId)
            .ToList();
    }

    private static IReadOnlyList<Vector2> BuildAnchorPlanarPoints(IReadOnlyList<Pm4MprlEntry> refs)
    {
        if (refs.Count == 0)
            return Array.Empty<Vector2>();

        List<Vector2> points = new(refs.Count);
        for (int index = 0; index < refs.Count; index++)
        {
            Vector3 position = refs[index].Position;
            points.Add(new Vector2(position.X, position.Z));
        }

        return points;
    }

    private static List<Pm4MprlEntry> CollectLinkedPositionRefs(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        if (document.KnownChunks.Mprl.Count == 0 || document.KnownChunks.Mslk.Count == 0 || surfaces.Count == 0)
            return [];

        HashSet<int> surfaceIndices = surfaces.Select(static surface => surface.SurfaceIndex).ToHashSet();
        HashSet<int> seenRefs = [];
        List<Pm4MprlEntry> refs = [];

        for (int index = 0; index < document.KnownChunks.Mslk.Count; index++)
        {
            Pm4MslkEntry link = document.KnownChunks.Mslk[index];
            if (!surfaceIndices.Contains(link.RefIndex) || (uint)link.RefIndex >= (uint)document.KnownChunks.Mprl.Count || !seenRefs.Add(link.RefIndex))
                continue;

            refs.Add(document.KnownChunks.Mprl[link.RefIndex]);
        }

        return refs;
    }

    private static uint SelectDominantGroupObjectId(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        HashSet<int> surfaceIndices = surfaces.Select(static surface => surface.SurfaceIndex).ToHashSet();
        Dictionary<uint, int> counts = [];
        uint bestValue = 0;
        int bestCount = 0;

        for (int index = 0; index < document.KnownChunks.Mslk.Count; index++)
        {
            Pm4MslkEntry link = document.KnownChunks.Mslk[index];
            if (link.GroupObjectId == 0 || !surfaceIndices.Contains(link.RefIndex))
                continue;

            int nextCount = counts.TryGetValue(link.GroupObjectId, out int existing) ? existing + 1 : 1;
            counts[link.GroupObjectId] = nextCount;
            if (nextCount > bestCount)
            {
                bestCount = nextCount;
                bestValue = link.GroupObjectId;
            }
        }

        return bestValue;
    }

    private static List<List<IndexedSurface>> SplitByMslkGroupObjectId(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        if (surfaces.Count <= 1 || document.KnownChunks.Mslk.Count == 0)
            return [surfaces.OrderBy(static surface => surface.SurfaceIndex).ToList()];

        Dictionary<int, int> localIndices = new(surfaces.Count);
        for (int index = 0; index < surfaces.Count; index++)
            localIndices[surfaces[index].SurfaceIndex] = index;

        Dictionary<uint, HashSet<int>> membersByGroupId = [];
        for (int index = 0; index < document.KnownChunks.Mslk.Count; index++)
        {
            Pm4MslkEntry link = document.KnownChunks.Mslk[index];
            if (link.GroupObjectId == 0 || !localIndices.TryGetValue(link.RefIndex, out int localIndex))
                continue;

            if (!membersByGroupId.TryGetValue(link.GroupObjectId, out HashSet<int>? members))
            {
                members = [];
                membersByGroupId[link.GroupObjectId] = members;
            }

            members.Add(localIndex);
        }

        if (membersByGroupId.Count == 0)
            return [surfaces.OrderBy(static surface => surface.SurfaceIndex).ToList()];

        int[] parent = new int[surfaces.Count];
        for (int index = 0; index < parent.Length; index++)
            parent[index] = index;

        HashSet<int> linked = [];
        foreach (HashSet<int> members in membersByGroupId.Values)
        {
            if (members.Count < 2)
                continue;

            int first = members.First();
            linked.Add(first);
            foreach (int member in members)
            {
                linked.Add(member);
                Union(parent, first, member);
            }
        }

        if (linked.Count < 2)
            return [surfaces.OrderBy(static surface => surface.SurfaceIndex).ToList()];

        Dictionary<int, List<IndexedSurface>> components = [];
        for (int index = 0; index < surfaces.Count; index++)
        {
            if (!linked.Contains(index))
                continue;

            int root = Find(parent, index);
            if (!components.TryGetValue(root, out List<IndexedSurface>? component))
            {
                component = [];
                components[root] = component;
            }

            component.Add(surfaces[index]);
        }

        List<List<IndexedSurface>> result = components.Values
            .Select(static component => component.OrderBy(static item => item.SurfaceIndex).ToList())
            .OrderBy(static component => component.Min(static item => item.SurfaceIndex))
            .ToList();
        List<IndexedSurface> unlinked = surfaces.Where((_, localIndex) => !linked.Contains(localIndex)).OrderBy(static surface => surface.SurfaceIndex).ToList();
        if (unlinked.Count > 0)
            result.Add(unlinked);

        return result.Count > 0 ? result : [surfaces.OrderBy(static surface => surface.SurfaceIndex).ToList()];
    }

    private static List<List<IndexedSurface>> SplitByConnectivity(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        if (surfaces.Count <= 1)
            return [surfaces.OrderBy(static surface => surface.SurfaceIndex).ToList()];

        IReadOnlyList<uint> meshIndices = document.KnownChunks.Msvi;
        IReadOnlyList<Vector3> meshVertices = document.KnownChunks.Msvt;
        List<List<int>> surfaceVertices = new(surfaces.Count);
        Dictionary<int, List<int>> vertexToSurfaceIndices = [];

        for (int surfaceIndex = 0; surfaceIndex < surfaces.Count; surfaceIndex++)
        {
            Pm4MsurEntry surface = surfaces[surfaceIndex].Surface;
            int firstIndex = checked((int)surface.MsviFirstIndex);
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            List<int> vertices = [];
            HashSet<int> unique = [];

            if (surface.IndexCount > 0 && firstIndex >= 0 && endExclusive > firstIndex)
            {
                for (int index = firstIndex; index < endExclusive; index++)
                {
                    int vertexIndex = checked((int)meshIndices[index]);
                    if ((uint)vertexIndex >= (uint)meshVertices.Count || !unique.Add(vertexIndex))
                        continue;

                    vertices.Add(vertexIndex);
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? owners))
                    {
                        owners = [];
                        vertexToSurfaceIndices[vertexIndex] = owners;
                    }

                    owners.Add(surfaceIndex);
                }
            }

            surfaceVertices.Add(vertices);
        }

        bool[] visited = new bool[surfaces.Count];
        Queue<int> queue = new();
        List<List<IndexedSurface>> components = [];

        for (int start = 0; start < surfaces.Count; start++)
        {
            if (visited[start])
                continue;

            visited[start] = true;
            queue.Enqueue(start);
            List<IndexedSurface> component = [];

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                component.Add(surfaces[current]);

                foreach (int vertexIndex in surfaceVertices[current])
                {
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? neighbors))
                        continue;

                    foreach (int neighbor in neighbors)
                    {
                        if (visited[neighbor])
                            continue;

                        visited[neighbor] = true;
                        queue.Enqueue(neighbor);
                    }
                }
            }

            components.Add(component.OrderBy(static item => item.SurfaceIndex).ToList());
        }

        return components.OrderBy(static component => component.Min(static item => item.SurfaceIndex)).ToList();
    }

    private static T SelectDominantSurfaceValue<T>(IReadOnlyList<Pm4MsurEntry> surfaces, Func<Pm4MsurEntry, T> selector) where T : notnull
    {
        return surfaces
            .GroupBy(selector)
            .OrderByDescending(static group => group.Count())
            .Select(static group => group.Key)
            .First();
    }

    private static int Find(int[] parent, int index)
    {
        while (parent[index] != index)
        {
            parent[index] = parent[parent[index]];
            index = parent[index];
        }

        return index;
    }

    private static void Union(int[] parent, int left, int right)
    {
        int rootLeft = Find(parent, left);
        int rootRight = Find(parent, right);
        if (rootLeft != rootRight)
            parent[rootRight] = rootLeft;
    }

    private sealed record IndexedSurface(int SurfaceIndex, Pm4MsurEntry Surface);

    private sealed record SeedGroup(
        uint DisplayCk24,
        byte DisplayCk24Type,
        ushort DisplayCk24ObjectId,
        bool RequiresConnectivitySeedSplit,
        List<IndexedSurface> Surfaces);

    private sealed record PendingSegment(
        Pm4ObjectSegment Segment,
        Pm4LinkedPositionRefSummary AnchorSummary,
        IReadOnlyList<Vector2> AnchorPlanarPoints,
        IReadOnlyList<Pm4ObjectSegmentSurface> Surfaces,
        Pm4CoordinateMode CoordinateMode,
        Pm4AxisConvention AxisConvention,
        Pm4PlanarTransform PlanarTransform,
        float FrameYawDegrees);
}
