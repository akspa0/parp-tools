using System.Numerics;
using System.Text.RegularExpressions;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public static partial class Pm4ResearchHierarchyAnalyzer
{
    private sealed record IndexedSurface(int SurfaceIndex, Pm4MsurEntry Surface);

    private delegate List<List<IndexedSurface>> SurfaceSplitter(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces);

    private sealed record SplitFamily(string Name, IReadOnlyList<SurfaceSplitter> Splitters);

    private static readonly SplitFamily[] Families =
    [
        new("ck24", Array.Empty<SurfaceSplitter>()),
        new("ck24_mslk_refindex", [SplitByMslkRefIndex]),
        new("ck24_mdos", [SplitByMdos]),
        new("ck24_connectivity", [SplitByConnectivity]),
        new("ck24_mslk_refindex_mdos", [SplitByMslkRefIndex, SplitByMdos]),
        new("ck24_mslk_refindex_connectivity", [SplitByMslkRefIndex, SplitByConnectivity]),
        new("ck24_mdos_connectivity", [SplitByMdos, SplitByConnectivity]),
        new("ck24_mslk_refindex_mdos_connectivity", [SplitByMslkRefIndex, SplitByMdos, SplitByConnectivity])
    ];

    public static Pm4TileObjectHypothesisReport Analyze(Pm4ResearchDocument document)
    {
        List<IndexedSurface> allCk24Surfaces = document.KnownChunks.Msur
            .Select(static (surface, surfaceIndex) => new IndexedSurface(surfaceIndex, surface))
            .Where(static indexedSurface => indexedSurface.Surface.Ck24 != 0)
            .ToList();

        List<Pm4ObjectHypothesis> objects = [];
        foreach (IGrouping<uint, IndexedSurface> ck24Group in allCk24Surfaces.GroupBy(static surface => surface.Surface.Ck24).OrderBy(static group => group.Key))
        {
            List<IndexedSurface> seed = ck24Group.OrderBy(static surface => surface.SurfaceIndex).ToList();
            foreach (SplitFamily family in Families)
            {
                List<List<IndexedSurface>> groups = ApplySplitFamily(document, seed, family.Splitters);
                for (int i = 0; i < groups.Count; i++)
                {
                    List<IndexedSurface> group = groups[i]
                        .OrderBy(static surface => surface.SurfaceIndex)
                        .ToList();
                    if (group.Count == 0)
                        continue;

                    objects.Add(BuildHypothesis(document, family.Name, i, ck24Group.Key, group));
                }
            }
        }

        (int? tileX, int? tileY) = TryParseTileCoordinates(document.SourcePath);
        List<string> notes =
        [
            "Research-only PM4 hierarchy report. MSLK.GroupObjectId, MPRL heading evidence, and split-family object candidates are exploratory scene-graph signals, not confirmed final runtime ownership.",
            "PlacementComparison exposes the current shared PM4 solver result for each candidate object family so local-vs-world placement and frame-yaw drift can be compared without re-deriving those guesses in each consumer."
        ];

        if (tileX is null || tileY is null)
            notes.Add("Tile coordinates could not be parsed from the PM4 source path. PlacementComparison used tile (0,0) as a fallback anchor.");

        return new Pm4TileObjectHypothesisReport(
            document.SourcePath,
            tileX,
            tileY,
            document.Version,
            allCk24Surfaces.Select(static surface => surface.Surface.Ck24).Distinct().Count(),
            objects.Count,
            objects,
            notes,
            document.Diagnostics);
    }

    private static List<List<IndexedSurface>> ApplySplitFamily(
        Pm4ResearchDocument document,
        IReadOnlyList<IndexedSurface> seed,
        IReadOnlyList<SurfaceSplitter> splitters)
    {
        List<List<IndexedSurface>> groups = [seed.ToList()];
        for (int i = 0; i < splitters.Count; i++)
        {
            List<List<IndexedSurface>> next = [];
            foreach (List<IndexedSurface> group in groups)
            {
                if (group.Count == 0)
                    continue;

                List<List<IndexedSurface>> split = splitters[i](document, group);
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
        Pm4ResearchDocument document,
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
        IReadOnlyList<uint> mslkGroupObjectIds = CollectMslkGroupObjectIds(document, surfaceIndices);
        IReadOnlyList<ushort> mslkRefIndices = CollectMslkRefIndices(document, surfaceIndices);
        IReadOnlyList<Pm4MprlEntry> linkedRefs = CollectLinkedMprlRefs(document, surfaceIndices, mslkGroupObjectIds);

        byte ck24Type = surfaces[0].Surface.Ck24Type;
        ushort ck24ObjectId = surfaces[0].Surface.Ck24ObjectId;
        int totalIndexCount = surfaces.Sum(static surface => surface.Surface.IndexCount);
        Pm4Bounds3? bounds = ComputeBounds(document, surfaces);
        Pm4MprlFootprintSummary mprlFootprint = SummarizeMprlFootprint(document, linkedRefs, bounds);
        Pm4ForensicsPlacementComparison placementComparison = BuildPlacementComparison(document, surfaceIndices, linkedRefs);
        uint dominantLinkGroupObjectId = SelectDominantLinkGroupObjectId(document, surfaceIndices);

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
            dominantLinkGroupObjectId,
            placementComparison,
            bounds,
            mprlFootprint);
    }

    private static IReadOnlyList<uint> CollectMslkGroupObjectIds(Pm4ResearchDocument document, IReadOnlyList<int> surfaceIndices)
    {
        HashSet<int> surfaceSet = surfaceIndices.ToHashSet();
        return document.KnownChunks.Mslk
            .Where(static link => link.GroupObjectId != 0)
            .Where(link => surfaceSet.Contains(link.RefIndex))
            .Select(static link => link.GroupObjectId)
            .Distinct()
            .OrderBy(static value => value)
            .ToList();
    }

    private static IReadOnlyList<ushort> CollectMslkRefIndices(Pm4ResearchDocument document, IReadOnlyList<int> surfaceIndices)
    {
        HashSet<int> surfaceSet = surfaceIndices.ToHashSet();
        return document.KnownChunks.Mslk
            .Where(link => surfaceSet.Contains(link.RefIndex))
            .Select(static link => link.RefIndex)
            .Distinct()
            .OrderBy(static value => value)
            .ToList();
    }

    private static uint SelectDominantLinkGroupObjectId(Pm4ResearchDocument document, IReadOnlyList<int> surfaceIndices)
    {
        HashSet<int> surfaceSet = surfaceIndices.ToHashSet();
        Dictionary<uint, int> counts = [];
        uint bestGroupObjectId = 0;
        int bestCount = 0;

        for (int i = 0; i < document.KnownChunks.Mslk.Count; i++)
        {
            Pm4MslkEntry link = document.KnownChunks.Mslk[i];
            if (link.GroupObjectId == 0 || !surfaceSet.Contains(link.RefIndex))
                continue;

            int nextCount = counts.TryGetValue(link.GroupObjectId, out int existingCount)
                ? existingCount + 1
                : 1;
            counts[link.GroupObjectId] = nextCount;
            if (nextCount > bestCount)
            {
                bestCount = nextCount;
                bestGroupObjectId = link.GroupObjectId;
            }
        }

        return bestGroupObjectId;
    }

    private static IReadOnlyList<Pm4MprlEntry> CollectLinkedMprlRefs(
        Pm4ResearchDocument document,
        IReadOnlyList<int> surfaceIndices,
        IReadOnlyList<uint> mslkGroupObjectIds)
    {
        HashSet<int> seen = [];
        List<Pm4MprlEntry> refs = [];

        if (mslkGroupObjectIds.Count > 0 && document.KnownChunks.Mprl.Count > 0)
        {
            HashSet<uint> groupIds = mslkGroupObjectIds.ToHashSet();
            for (int i = 0; i < document.KnownChunks.Mslk.Count; i++)
            {
                Pm4MslkEntry link = document.KnownChunks.Mslk[i];
                if (!groupIds.Contains(link.GroupObjectId))
                    continue;
                if ((uint)link.RefIndex >= (uint)document.KnownChunks.Mprl.Count)
                    continue;
                if (!seen.Add(link.RefIndex))
                    continue;

                refs.Add(document.KnownChunks.Mprl[link.RefIndex]);
            }
        }

        if (refs.Count > 0)
            return refs;

        if (surfaceIndices.Count == 0 || document.KnownChunks.Mprl.Count == 0 || document.KnownChunks.Mslk.Count == 0)
            return Array.Empty<Pm4MprlEntry>();

        HashSet<int> surfaceSet = surfaceIndices.ToHashSet();
        for (int i = 0; i < document.KnownChunks.Mslk.Count; i++)
        {
            Pm4MslkEntry link = document.KnownChunks.Mslk[i];
            if (!surfaceSet.Contains(link.RefIndex))
                continue;
            if ((uint)link.RefIndex >= (uint)document.KnownChunks.Mprl.Count)
                continue;
            if (!seen.Add(link.RefIndex))
                continue;

            refs.Add(document.KnownChunks.Mprl[link.RefIndex]);
        }

        return refs;
    }

    private static List<List<IndexedSurface>> SplitByMslkRefIndex(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        List<List<IndexedSurface>> groups = [];
        if (surfaces.Count <= 1 || document.KnownChunks.Mslk.Count == 0)
        {
            groups.Add(surfaces.ToList());
            return groups;
        }

        Dictionary<int, int> surfaceIndexToLocal = new(surfaces.Count);
        for (int i = 0; i < surfaces.Count; i++)
            surfaceIndexToLocal[surfaces[i].SurfaceIndex] = i;

        Dictionary<uint, HashSet<int>> groupToMembers = [];
        for (int i = 0; i < document.KnownChunks.Mslk.Count; i++)
        {
            Pm4MslkEntry link = document.KnownChunks.Mslk[i];
            if (link.GroupObjectId == 0)
                continue;

            if (!surfaceIndexToLocal.TryGetValue(link.RefIndex, out int localIndex))
                continue;

            if (!groupToMembers.TryGetValue(link.GroupObjectId, out HashSet<int>? members))
            {
                members = [];
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

        HashSet<int> linkedLocalIndices = [];
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

        Dictionary<int, List<IndexedSurface>> linkedComponents = [];
        for (int i = 0; i < surfaces.Count; i++)
        {
            if (!linkedLocalIndices.Contains(i))
                continue;

            int root = Find(parent, i);
            if (!linkedComponents.TryGetValue(root, out List<IndexedSurface>? component))
            {
                component = [];
                linkedComponents[root] = component;
            }

            component.Add(surfaces[i]);
        }

        foreach (List<IndexedSurface> component in linkedComponents.Values.OrderBy(static component => component.Min(static entry => entry.SurfaceIndex)))
            groups.Add(component);

        List<IndexedSurface> unlinked = surfaces.Where((_, localIndex) => !linkedLocalIndices.Contains(localIndex)).ToList();
        if (unlinked.Count > 0)
            groups.Add(unlinked);

        return groups.Count > 0 ? groups : [surfaces.ToList()];
    }

    private static List<List<IndexedSurface>> SplitByMdos(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        if (surfaces.Count <= 1)
            return [surfaces.ToList()];

        List<List<IndexedSurface>> groups = surfaces
            .GroupBy(static surface => surface.Surface.MdosIndex)
            .Select(static group => group.OrderBy(static surface => surface.SurfaceIndex).ToList())
            .ToList();

        return groups.Count > 0 ? groups : [surfaces.ToList()];
    }

    private static List<List<IndexedSurface>> SplitByConnectivity(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        List<List<IndexedSurface>> components = [];
        if (surfaces.Count <= 1)
        {
            components.Add(surfaces.ToList());
            return components;
        }

        IReadOnlyList<uint> meshIndices = document.KnownChunks.Msvi;
        IReadOnlyList<Vector3> meshVertices = document.KnownChunks.Msvt;
        List<List<int>> surfaceVertices = new(surfaces.Count);
        Dictionary<int, List<int>> vertexToSurfaceIndices = [];

        for (int s = 0; s < surfaces.Count; s++)
        {
            Pm4MsurEntry surface = surfaces[s].Surface;
            int firstIndex = (int)surface.MsviFirstIndex;
            int endExclusive = Math.Min(firstIndex + surface.IndexCount, meshIndices.Count);
            List<int> vertices = [];
            HashSet<int> unique = [];

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
                        owners = [];
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
            List<IndexedSurface> component = [];

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

    private static Pm4Bounds3? ComputeBounds(Pm4ResearchDocument document, IReadOnlyList<IndexedSurface> surfaces)
    {
        IReadOnlyList<uint> meshIndices = document.KnownChunks.Msvi;
        IReadOnlyList<Vector3> meshVertices = document.KnownChunks.Msvt;
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
        Pm4ResearchDocument document,
        IReadOnlyList<Pm4MprlEntry> linkedRefs,
        Pm4Bounds3? bounds)
    {
        IReadOnlyList<Pm4MprlEntry> tileRefs = document.KnownChunks.Mprl;
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

    private static Pm4ForensicsPlacementComparison BuildPlacementComparison(
        Pm4ResearchDocument document,
        IReadOnlyList<int> surfaceIndices,
        IReadOnlyList<Pm4MprlEntry> linkedPositionRefs)
    {
        IReadOnlyList<Pm4MsurEntry> surfaces = surfaceIndices.Select(index => document.KnownChunks.Msur[index]).ToList();
        Pm4AxisConvention axisConvention = Pm4PlacementMath.DetectAxisConventionBySurfaceNormals(document.KnownChunks.Msvt, document.KnownChunks.Msvi, surfaces);
        Pm4CoordinateMode fallbackCoordinateMode = Pm4PlacementMath.IsLikelyTileLocal(document.KnownChunks.Msvt)
            ? Pm4CoordinateMode.TileLocal
            : Pm4CoordinateMode.WorldSpace;

        (int? tileX, int? tileY) = TryParseTileCoordinates(document.SourcePath);
        int resolvedTileX = tileX ?? 0;
        int resolvedTileY = tileY ?? 0;

        Pm4CoordinateModeResolution resolution = Pm4PlacementMath.ResolveCoordinateMode(
            document.KnownChunks.Msvt,
            document.KnownChunks.Msvi,
            surfaces,
            linkedPositionRefs,
            anchorPositionRefs: linkedPositionRefs,
            resolvedTileX,
            resolvedTileY,
            axisConvention,
            fallbackCoordinateMode);

        Pm4PlacementSolution placement = Pm4PlacementMath.ResolvePlacementSolution(
            document.KnownChunks.Msvt,
            document.KnownChunks.Msvi,
            surfaces,
            linkedPositionRefs,
            anchorPositionRefs: linkedPositionRefs,
            resolvedTileX,
            resolvedTileY,
            resolution.CoordinateMode,
            axisConvention);

        Pm4LinkedPositionRefSummary summary = SanitizeSummary(Pm4PlacementMath.SummarizeLinkedPositionRefs(linkedPositionRefs));
        float? meanHeading = summary.HasNormalHeadings ? summary.HeadingMeanDegrees : null;
        float frameYawDegrees = placement.WorldYawCorrectionRadians * (180f / MathF.PI);

        return new Pm4ForensicsPlacementComparison(
            tileX,
            tileY,
            axisConvention,
            resolution.CoordinateMode,
            resolution.PlanarTransform,
            resolution.UsedFallback,
            SanitizeFinite(resolution.TileLocalScore),
            SanitizeFinite(resolution.WorldSpaceScore),
            placement.WorldPivot,
            frameYawDegrees,
            meanHeading,
            meanHeading.HasValue ? NormalizeDegrees(meanHeading.Value - frameYawDegrees) : null);
    }

    private static Pm4LinkedPositionRefSummary SanitizeSummary(Pm4LinkedPositionRefSummary summary)
    {
        return new Pm4LinkedPositionRefSummary(
            summary.TotalCount,
            summary.NormalCount,
            summary.TerminatorCount,
            summary.FloorMin,
            summary.FloorMax,
            SanitizeFinite(summary.HeadingMinDegrees) ?? 0f,
            SanitizeFinite(summary.HeadingMaxDegrees) ?? 0f,
            SanitizeFinite(summary.HeadingMeanDegrees) ?? 0f);
    }

    private static float? SanitizeFinite(float value) => float.IsFinite(value) ? value : null;

    private static float NormalizeDegrees(float degrees)
    {
        while (degrees > 180f)
            degrees -= 360f;
        while (degrees < -180f)
            degrees += 360f;
        return degrees;
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