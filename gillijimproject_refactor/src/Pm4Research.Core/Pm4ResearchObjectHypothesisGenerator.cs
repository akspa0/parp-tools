using System.Numerics;
using System.Text.RegularExpressions;

namespace Pm4Research.Core;

public static partial class Pm4ResearchObjectHypothesisGenerator
{
    private sealed record IndexedSurface(int SurfaceIndex, Pm4MsurEntry Surface);

    private delegate List<List<IndexedSurface>> SurfaceSplitter(Pm4ResearchFile file, IReadOnlyList<IndexedSurface> surfaces);

    private sealed record SplitFamily(string Name, IReadOnlyList<string> Axes, IReadOnlyList<SurfaceSplitter> Splitters);

    private static readonly SplitFamily[] Families =
    {
        new("ck24", Array.Empty<string>(), Array.Empty<SurfaceSplitter>()),
        new("ck24_mslk_refindex", new[] { "mslk_refindex" }, new SurfaceSplitter[] { SplitByMslkRefIndex }),
        new("ck24_mdos", new[] { "mdos" }, new SurfaceSplitter[] { SplitByMdos }),
        new("ck24_connectivity", new[] { "connectivity" }, new SurfaceSplitter[] { SplitByConnectivity }),
        new("ck24_mslk_refindex_mdos", new[] { "mslk_refindex", "mdos" }, new SurfaceSplitter[] { SplitByMslkRefIndex, SplitByMdos }),
        new("ck24_mslk_refindex_connectivity", new[] { "mslk_refindex", "connectivity" }, new SurfaceSplitter[] { SplitByMslkRefIndex, SplitByConnectivity }),
        new("ck24_mdos_connectivity", new[] { "mdos", "connectivity" }, new SurfaceSplitter[] { SplitByMdos, SplitByConnectivity }),
        new("ck24_mslk_refindex_mdos_connectivity", new[] { "mslk_refindex", "mdos", "connectivity" }, new SurfaceSplitter[] { SplitByMslkRefIndex, SplitByMdos, SplitByConnectivity }),
    };

    public static Pm4TileObjectHypothesisReport Analyze(Pm4ResearchFile file)
    {
        List<IndexedSurface> allCk24Surfaces = file.KnownChunks.Msur
            .Select(static (surface, surfaceIndex) => new IndexedSurface(surfaceIndex, surface))
            .Where(static indexedSurface => indexedSurface.Surface.Ck24 != 0)
            .ToList();

        var objects = new List<Pm4ObjectHypothesis>();
        foreach (IGrouping<uint, IndexedSurface> ck24Group in allCk24Surfaces.GroupBy(static surface => surface.Surface.Ck24).OrderBy(static group => group.Key))
        {
            List<IndexedSurface> seed = ck24Group.OrderBy(static surface => surface.SurfaceIndex).ToList();
            foreach (SplitFamily family in Families)
            {
                List<List<IndexedSurface>> groups = ApplySplitFamily(file, seed, family.Splitters);
                for (int i = 0; i < groups.Count; i++)
                {
                    List<IndexedSurface> group = groups[i]
                        .OrderBy(static surface => surface.SurfaceIndex)
                        .ToList();
                    if (group.Count == 0)
                        continue;

                    objects.Add(BuildHypothesis(file, family.Name, i, ck24Group.Key, group));
                }
            }
        }

        (int? tileX, int? tileY) = TryParseTileCoordinates(file.SourcePath);

        return new Pm4TileObjectHypothesisReport(
            file.SourcePath,
            tileX,
            tileY,
            file.Version,
            allCk24Surfaces.Select(static surface => surface.Surface.Ck24).Distinct().Count(),
            objects.Count,
            objects,
            file.Diagnostics);
    }

    private static List<List<IndexedSurface>> ApplySplitFamily(
        Pm4ResearchFile file,
        IReadOnlyList<IndexedSurface> seed,
        IReadOnlyList<SurfaceSplitter> splitters)
    {
        List<List<IndexedSurface>> groups = new() { seed.ToList() };
        for (int i = 0; i < splitters.Count; i++)
        {
            var next = new List<List<IndexedSurface>>();
            foreach (List<IndexedSurface> group in groups)
            {
                if (group.Count == 0)
                    continue;

                List<List<IndexedSurface>> split = splitters[i](file, group);
                if (split.Count == 0)
                {
                    next.Add(group);
                    continue;
                }

                foreach (List<IndexedSurface> candidate in split)
                {
                    if (candidate.Count > 0)
                        next.Add(candidate);
                }
            }

            groups = next.Count > 0 ? next : groups;
        }

        return groups;
    }

    private static Pm4ObjectHypothesis BuildHypothesis(
        Pm4ResearchFile file,
        string family,
        int familyObjectIndex,
        uint ck24,
        IReadOnlyList<IndexedSurface> surfaces)
    {
        IReadOnlyList<int> surfaceIndices = surfaces.Select(static surface => surface.SurfaceIndex).ToList();
        IReadOnlyList<uint> mdosIndices = surfaces
            .Select(static surface => surface.Surface.MdosIndex)
            .Distinct()
            .OrderBy(static value => value)
            .ToList();
        IReadOnlyList<byte> groupKeys = surfaces
            .Select(static surface => surface.Surface.GroupKey)
            .Distinct()
            .OrderBy(static value => value)
            .ToList();
        IReadOnlyList<uint> mslkGroupObjectIds = CollectMslkGroupObjectIds(file, surfaceIndices);
        IReadOnlyList<ushort> mslkRefIndices = CollectMslkRefIndices(file, surfaceIndices);

        byte ck24Type = surfaces[0].Surface.Ck24Type;
        ushort ck24ObjectId = surfaces[0].Surface.Ck24ObjectId;
        int totalIndexCount = surfaces.Sum(static surface => surface.Surface.IndexCount);
        Pm4Bounds3? bounds = ComputeBounds(file, surfaces);
        Pm4MprlFootprintSummary mprlFootprint = SummarizeMprlFootprint(file, mslkGroupObjectIds, bounds);

        return new Pm4ObjectHypothesis(
            family,
            familyObjectIndex,
            ck24,
            ck24Type,
            ck24ObjectId,
            surfaces.Count,
            totalIndexCount,
            surfaceIndices,
            mdosIndices,
            groupKeys,
            mslkGroupObjectIds,
            mslkRefIndices,
                bounds,
                mprlFootprint);
    }

    private static IReadOnlyList<uint> CollectMslkGroupObjectIds(Pm4ResearchFile file, IReadOnlyList<int> surfaceIndices)
    {
        HashSet<int> surfaceSet = surfaceIndices.ToHashSet();
        return file.KnownChunks.Mslk
            .Where(static link => link.GroupObjectId != 0)
            .Where(link => surfaceSet.Contains(link.RefIndex))
            .Select(static link => link.GroupObjectId)
            .Distinct()
            .OrderBy(static value => value)
            .ToList();
    }

    private static IReadOnlyList<ushort> CollectMslkRefIndices(Pm4ResearchFile file, IReadOnlyList<int> surfaceIndices)
    {
        HashSet<int> surfaceSet = surfaceIndices.ToHashSet();
        return file.KnownChunks.Mslk
            .Where(link => surfaceSet.Contains(link.RefIndex))
            .Select(static link => link.RefIndex)
            .Distinct()
            .OrderBy(static value => value)
            .ToList();
    }

    private static List<List<IndexedSurface>> SplitByMslkRefIndex(Pm4ResearchFile file, IReadOnlyList<IndexedSurface> surfaces)
    {
        var groups = new List<List<IndexedSurface>>();
        if (surfaces.Count <= 1 || file.KnownChunks.Mslk.Count == 0)
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        var surfaceIndexToLocal = new Dictionary<int, int>(surfaces.Count);
        for (int i = 0; i < surfaces.Count; i++)
            surfaceIndexToLocal[surfaces[i].SurfaceIndex] = i;

        var groupToMembers = new Dictionary<uint, HashSet<int>>();
        for (int i = 0; i < file.KnownChunks.Mslk.Count; i++)
        {
            Pm4MslkEntry link = file.KnownChunks.Mslk[i];
            if (link.GroupObjectId == 0)
                continue;

            if (!surfaceIndexToLocal.TryGetValue(link.RefIndex, out int localIndex))
                continue;

            if (!groupToMembers.TryGetValue(link.GroupObjectId, out HashSet<int>? members))
            {
                members = new HashSet<int>();
                groupToMembers[link.GroupObjectId] = members;
            }

            members.Add(localIndex);
        }

        if (groupToMembers.Count == 0)
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        int[] parent = new int[surfaces.Count];
        for (int i = 0; i < parent.Length; i++)
            parent[i] = i;

        static int Find(int[] parentArray, int index)
        {
            while (parentArray[index] != index)
            {
                parentArray[index] = parentArray[parentArray[index]];
                index = parentArray[index];
            }

            return index;
        }

        static void Union(int[] parentArray, int a, int b)
        {
            int rootA = Find(parentArray, a);
            int rootB = Find(parentArray, b);
            if (rootA != rootB)
                parentArray[rootB] = rootA;
        }

        HashSet<int> linkedLocalIndices = new();
        foreach (HashSet<int> members in groupToMembers.Values)
        {
            if (members.Count < 2)
                continue;

            int first = members.First();
            linkedLocalIndices.Add(first);
            foreach (int member in members)
            {
                linkedLocalIndices.Add(member);
                Union(parent, first, member);
            }
        }

        if (linkedLocalIndices.Count < 2)
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        var linkedComponents = new Dictionary<int, List<IndexedSurface>>();
        for (int i = 0; i < surfaces.Count; i++)
        {
            if (!linkedLocalIndices.Contains(i))
                continue;

            int root = Find(parent, i);
            if (!linkedComponents.TryGetValue(root, out List<IndexedSurface>? component))
            {
                component = new List<IndexedSurface>();
                linkedComponents[root] = component;
            }

            component.Add(surfaces[i]);
        }

        foreach (List<IndexedSurface> component in linkedComponents.Values.OrderBy(static component => component.Min(static entry => entry.SurfaceIndex)))
            groups.Add(component);

        List<IndexedSurface> unlinked = surfaces.Where((_, localIndex) => !linkedLocalIndices.Contains(localIndex)).ToList();
        if (unlinked.Count > 0)
            groups.Add(unlinked);

        return groups.Count > 0 ? groups : new List<List<IndexedSurface>> { surfaces.ToList() };
    }

    private static List<List<IndexedSurface>> SplitByMdos(Pm4ResearchFile file, IReadOnlyList<IndexedSurface> surfaces)
    {
        if (surfaces.Count <= 1)
            return new List<List<IndexedSurface>> { surfaces.ToList() };

        List<List<IndexedSurface>> groups = surfaces
            .GroupBy(static surface => surface.Surface.MdosIndex)
            .Select(static group => group.OrderBy(static surface => surface.SurfaceIndex).ToList())
            .ToList();

        return groups.Count > 0 ? groups : new List<List<IndexedSurface>> { surfaces.ToList() };
    }

    private static List<List<IndexedSurface>> SplitByConnectivity(Pm4ResearchFile file, IReadOnlyList<IndexedSurface> surfaces)
    {
        var components = new List<List<IndexedSurface>>();
        if (surfaces.Count <= 1)
        {
            components.Add(surfaces.ToList());
            return components;
        }

        IReadOnlyList<uint> meshIndices = file.KnownChunks.Msvi;
        IReadOnlyList<Vector3> meshVertices = file.KnownChunks.Msvt;
        var surfaceVertices = new List<List<int>>(surfaces.Count);
        var vertexToSurfaceIndices = new Dictionary<int, List<int>>();

        for (int s = 0; s < surfaces.Count; s++)
        {
            Pm4MsurEntry surface = surfaces[s].Surface;
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            var vertices = new List<int>();
            var unique = new HashSet<int>();

            if (surface.IndexCount > 0 && firstIndex >= 0 && endExclusive > firstIndex)
            {
                for (int idx = firstIndex; idx < endExclusive; idx++)
                {
                    int vertexIndex = (int)meshIndices[idx];
                    if ((uint)vertexIndex >= (uint)meshVertices.Count)
                        continue;
                    if (!unique.Add(vertexIndex))
                        continue;

                    vertices.Add(vertexIndex);
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? owners))
                    {
                        owners = new List<int>();
                        vertexToSurfaceIndices[vertexIndex] = owners;
                    }

                    owners.Add(s);
                }
            }

            surfaceVertices.Add(vertices);
        }

        bool[] visited = new bool[surfaces.Count];
        Queue<int> queue = new();
        for (int start = 0; start < surfaces.Count; start++)
        {
            if (visited[start])
                continue;

            visited[start] = true;
            queue.Enqueue(start);
            var component = new List<IndexedSurface>();

            while (queue.Count > 0)
            {
                int current = queue.Dequeue();
                component.Add(surfaces[current]);

                List<int> vertices = surfaceVertices[current];
                for (int v = 0; v < vertices.Count; v++)
                {
                    int vertexIndex = vertices[v];
                    if (!vertexToSurfaceIndices.TryGetValue(vertexIndex, out List<int>? neighbors))
                        continue;

                    for (int n = 0; n < neighbors.Count; n++)
                    {
                        int neighborSurface = neighbors[n];
                        if (visited[neighborSurface])
                            continue;

                        visited[neighborSurface] = true;
                        queue.Enqueue(neighborSurface);
                    }
                }
            }

            components.Add(component.OrderBy(static surface => surface.SurfaceIndex).ToList());
        }

        return components;
    }

    private static Pm4Bounds3? ComputeBounds(Pm4ResearchFile file, IReadOnlyList<IndexedSurface> surfaces)
    {
        IReadOnlyList<uint> meshIndices = file.KnownChunks.Msvi;
        IReadOnlyList<Vector3> meshVertices = file.KnownChunks.Msvt;
        bool hasPoint = false;
        Vector3 min = Vector3.Zero;
        Vector3 max = Vector3.Zero;

        for (int s = 0; s < surfaces.Count; s++)
        {
            Pm4MsurEntry surface = surfaces[s].Surface;
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            for (int idx = Math.Max(0, firstIndex); idx < endExclusive; idx++)
            {
                int vertexIndex = (int)meshIndices[idx];
                if ((uint)vertexIndex >= (uint)meshVertices.Count)
                    continue;

                Vector3 point = meshVertices[vertexIndex];
                if (!hasPoint)
                {
                    min = point;
                    max = point;
                    hasPoint = true;
                }
                else
                {
                    min = Vector3.Min(min, point);
                    max = Vector3.Max(max, point);
                }
            }
        }

        return hasPoint ? new Pm4Bounds3(min, max) : null;
    }

    private static Pm4MprlFootprintSummary SummarizeMprlFootprint(
        Pm4ResearchFile file,
        IReadOnlyList<uint> mslkGroupObjectIds,
        Pm4Bounds3? bounds)
    {
        IReadOnlyList<Pm4MprlEntry> tileRefs = file.KnownChunks.Mprl;
        IReadOnlyList<Pm4MprlEntry> linkedRefs = CollectLinkedMprlRefs(file, mslkGroupObjectIds);

        int linkedNormalCount = 0;
        int linkedTerminatorCount = 0;
        short? linkedFloorMin = null;
        short? linkedFloorMax = null;
        for (int i = 0; i < linkedRefs.Count; i++)
        {
            Pm4MprlEntry entry = linkedRefs[i];
            if (entry.Unk16 == 0)
                linkedNormalCount++;
            else
                linkedTerminatorCount++;

            if (!linkedFloorMin.HasValue || entry.Unk14 < linkedFloorMin.Value)
                linkedFloorMin = entry.Unk14;
            if (!linkedFloorMax.HasValue || entry.Unk14 > linkedFloorMax.Value)
                linkedFloorMax = entry.Unk14;
        }

        return new Pm4MprlFootprintSummary(
            tileRefs.Count,
            linkedRefs.Count,
            linkedNormalCount,
            linkedTerminatorCount,
            CountRefsInBounds(tileRefs, bounds, 0f),
            CountRefsInBounds(tileRefs, bounds, 2f),
            CountRefsInBounds(linkedRefs, bounds, 0f),
            CountRefsInBounds(linkedRefs, bounds, 2f),
            linkedFloorMin,
            linkedFloorMax);
    }

    private static IReadOnlyList<Pm4MprlEntry> CollectLinkedMprlRefs(Pm4ResearchFile file, IReadOnlyList<uint> mslkGroupObjectIds)
    {
        if (mslkGroupObjectIds.Count == 0 || file.KnownChunks.Mprl.Count == 0)
            return Array.Empty<Pm4MprlEntry>();

        HashSet<uint> groupIds = mslkGroupObjectIds.ToHashSet();
        HashSet<int> seen = new();
        List<Pm4MprlEntry> refs = new();
        for (int i = 0; i < file.KnownChunks.Mslk.Count; i++)
        {
            Pm4MslkEntry link = file.KnownChunks.Mslk[i];
            if (!groupIds.Contains(link.GroupObjectId))
                continue;
            if ((uint)link.RefIndex >= (uint)file.KnownChunks.Mprl.Count)
                continue;
            if (!seen.Add(link.RefIndex))
                continue;

            refs.Add(file.KnownChunks.Mprl[link.RefIndex]);
        }

        return refs;
    }

    private static int CountRefsInBounds(IReadOnlyList<Pm4MprlEntry> refs, Pm4Bounds3? bounds, float expansion)
    {
        if (bounds == null || refs.Count == 0)
            return 0;

        Vector3 margin = new(expansion, expansion, expansion);
        Vector3 min = bounds.Min - margin;
        Vector3 max = bounds.Max + margin;
        int count = 0;
        for (int i = 0; i < refs.Count; i++)
        {
            Vector3 point = refs[i].Position;
            if (point.X < min.X || point.X > max.X)
                continue;
            if (point.Y < min.Y || point.Y > max.Y)
                continue;
            if (point.Z < min.Z || point.Z > max.Z)
                continue;

            count++;
        }

        return count;
    }

    private static (int? tileX, int? tileY) TryParseTileCoordinates(string? sourcePath)
    {
        if (string.IsNullOrWhiteSpace(sourcePath))
            return (null, null);

        Match match = TilePattern().Match(Path.GetFileNameWithoutExtension(sourcePath));
        if (!match.Success)
            return (null, null);

        return (
            int.Parse(match.Groups[1].Value),
            int.Parse(match.Groups[2].Value));
    }

    [GeneratedRegex(@"_(\d+)_(\d+)$", RegexOptions.CultureInvariant)]
    private static partial Regex TilePattern();
}