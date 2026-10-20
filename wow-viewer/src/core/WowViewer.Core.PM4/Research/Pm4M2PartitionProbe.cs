using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4M2ValueCount(string Value, long Count);

public sealed record Pm4M2ComponentRow(
    string FileId,
    int ComponentIndex,
    int Surfaces,
    int Placements,
    string PaddingValues,
    string WindowStartValues,
    string ModelPaths);

public sealed record Pm4M2PartitionReport(
    int Files,
    int FilesWithM2,
    long M2Surfaces,
    long Components,
    long ComponentsWithZeroPlacements,
    long ComponentsWithOnePlacement,
    long ComponentsWithMultiplePlacements,
    long ComponentsWithMultipleSameModel,
    long ComponentsWithMultipleMixedModels,
    int MaxSurfacesPerComponent,
    int MaxPlacementsPerComponent,
    double MeanSurfacesPerComponent,
    double MeanPlacementsPerComponent,
    long PlacementsWithSingleWindowStart,
    long PlacementsWithMultipleWindowStarts,
    long PlacementsWithSingleNormal,
    long PlacementsWithMultipleNormals,
    long PlacementsWithSingleHeight,
    long PlacementsWithMultipleHeights,
    long PlacementsWithDisjointVertices,
    long PlacementsSharingVertices,
    long PlacementsWithNoCoveringSurface,
    long PlacementsWithDisjointSurfaceRanges,
    long PlacementsWithOverlappingSurfaceRanges,
    long PlacementsWithSingleGroupObjectId,
    long PlacementsWithMultipleGroupObjectIds,
    long GroupObjectIdsWithSingleModel,
    long GroupObjectIdsWithMultipleModels,
    IReadOnlyList<Pm4M2ValueCount> PaddingHistogram,
    IReadOnlyList<Pm4M2ValueCount> WindowStartHistogram,
    IReadOnlyList<Pm4M2ComponentRow> LargestComponents);

/// <summary>
/// Tests whether the M2 surface class (<c>Ck24Type == 0</c>) splits into per-object pieces by
/// topological adjacency (surfaces sharing an MSVT vertex) and by the remaining MSUR bytes.
/// </summary>
/// <remarks>
/// A component that covers exactly one authored M2 origin is a candidate per-object piece. A
/// component that covers several origins is a merged mesh; a component that covers none may be
/// geometry whose origin lies outside it. This is a correlation instrument, not an identity claim.
/// </remarks>
public static class Pm4M2PartitionProbe
{
    public static Pm4M2PartitionReport Analyze(IReadOnlyList<Pm4PlacementAuditInput> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        long m2Surfaces = 0, components = 0, zero = 0, one = 0, multiple = 0;
        long multipleSameModel = 0, multipleMixedModels = 0;
        long singleWindow = 0, multipleWindow = 0, singleNormal = 0, multipleNormal = 0, singleHeight = 0, multipleHeight = 0;
        long disjointVertices = 0, sharingVertices = 0, noCoveringSurface = 0;
        long disjointRanges = 0, overlappingRanges = 0;
        long singleGroupId = 0, multipleGroupIds = 0, groupIdSingleModel = 0, groupIdMultipleModels = 0;
        long componentSurfaceTotal = 0, componentPlacementTotal = 0;
        int maxSurfaces = 0, maxPlacements = 0;
        int filesWithM2 = 0;
        var padding = new Dictionary<string, long>(StringComparer.Ordinal);
        var windowStart = new Dictionary<string, long>(StringComparer.Ordinal);
        var largest = new List<Pm4M2ComponentRow>();

        foreach (var file in input)
        {
            if (file.Placements is null)
                continue;

            var chunks = file.Document.KnownChunks;
            var surfaces = BuildM2Surfaces(chunks);
            if (surfaces.Count == 0)
                continue;
            filesWithM2++;
            m2Surfaces += surfaces.Count;

            foreach (var surface in surfaces)
            {
                Increment(padding, surface.Padding.ToString());
                Increment(windowStart, surface.WindowStart.ToString());
            }

            // Per-placement vertex sets: if two placements never share an MSVT vertex, their
            // geometry is separable by vertex range even when a topological component merges them.
            var placementVertices = new List<HashSet<uint>>(file.Placements.ModelPlacements.Count);
            var vertexToPlacements = new Dictionary<uint, List<int>>();
            for (int p = 0; p < file.Placements.ModelPlacements.Count; p++)
            {
                var placement = file.Placements.ModelPlacements[p];
                var set = new HashSet<uint>();
                foreach (var surface in surfaces)
                {
                    if (surface.Contains(placement.Position))
                        foreach (uint vertex in surface.Vertices)
                            set.Add(vertex);
                }
                placementVertices.Add(set);
                if (set.Count == 0)
                {
                    noCoveringSurface++;
                    continue;
                }
                foreach (uint vertex in set)
                {
                    if (!vertexToPlacements.TryGetValue(vertex, out var list))
                        vertexToPlacements[vertex] = list = [];
                    list.Add(p);
                }
            }
            for (int p = 0; p < placementVertices.Count; p++)
            {
                if (placementVertices[p].Count == 0)
                    continue;
                if (placementVertices[p].Any(v => vertexToPlacements[v].Count > 1))
                    sharingVertices++;
                else
                    disjointVertices++;
            }

            // MSUR index ranges: a contiguous run per placement would partition objects even when
            // seam vertices are shared. Overlap means the surface order interleaves objects.
            var ranges = new List<(int Min, int Max)>();
            foreach (var placement in file.Placements.ModelPlacements)
            {
                int min = int.MaxValue, max = int.MinValue;
                foreach (var surface in surfaces)
                {
                    if (!surface.Contains(placement.Position))
                        continue;
                    min = Math.Min(min, surface.Index);
                    max = Math.Max(max, surface.Index);
                }
                if (min != int.MaxValue)
                    ranges.Add((min, max));
            }
            for (int a = 0; a < ranges.Count; a++)
            {
                bool overlaps = false;
                for (int b = 0; b < ranges.Count && !overlaps; b++)
                {
                    if (a == b) continue;
                    if (ranges[a].Min <= ranges[b].Max && ranges[b].Min <= ranges[a].Max)
                        overlaps = true;
                }
                if (overlaps) overlappingRanges++; else disjointRanges++;
            }

            // MSLK GroupObjectId (+0x04): does one value select one placed model?
            var groupIdModels = new Dictionary<uint, HashSet<string>>();
            foreach (var placement in file.Placements.ModelPlacements)
            {
                var ids = new HashSet<uint>();
                foreach (var surface in surfaces)
                {
                    if (!surface.Contains(placement.Position))
                        continue;
                    foreach (uint id in surface.GroupObjectIds)
                        ids.Add(id);
                }
                if (ids.Count == 0)
                    continue;
                if (ids.Count == 1) singleGroupId++; else multipleGroupIds++;
                foreach (uint id in ids)
                {
                    if (!groupIdModels.TryGetValue(id, out var models))
                        groupIdModels[id] = models = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    models.Add(placement.ModelPath);
                }
            }
            foreach (var models in groupIdModels.Values)
            {
                if (models.Count == 1) groupIdSingleModel++; else groupIdMultipleModels++;
            }

            var parent = new int[surfaces.Count];
            for (int i = 0; i < parent.Length; i++) parent[i] = i;

            var firstByVertex = new Dictionary<uint, int>();
            for (int i = 0; i < surfaces.Count; i++)
            {
                foreach (uint vertex in surfaces[i].Vertices)
                {
                    if (firstByVertex.TryGetValue(vertex, out int other))
                        Union(parent, i, other);
                    else
                        firstByVertex[vertex] = i;
                }
            }

            var groups = new Dictionary<int, List<int>>();
            for (int i = 0; i < surfaces.Count; i++)
            {
                int root = Find(parent, i);
                if (!groups.TryGetValue(root, out var list))
                    groups[root] = list = [];
                list.Add(i);
            }

            int componentIndex = 0;
            foreach (var group in groups.Values.OrderByDescending(g => g.Count))
            {
                components++;
                componentSurfaceTotal += group.Count;
                var covered = new List<AdtModelPlacement>();
                foreach (var placement in file.Placements.ModelPlacements)
                {
                    if (group.Any(i => surfaces[i].Contains(placement.Position)))
                        covered.Add(placement);
                }
                int placements = covered.Count;
                componentPlacementTotal += placements;
                foreach (var placement in covered)
                {
                    var covering = group.Where(i => surfaces[i].Contains(placement.Position)).Select(i => surfaces[i]).ToArray();
                    if (covering.Select(s => s.WindowStart).Distinct().Count() == 1) singleWindow++; else multipleWindow++;
                    if (covering.Select(s => s.NormalKey).Distinct().Count() == 1) singleNormal++; else multipleNormal++;
                    if (covering.Select(s => s.HeightKey).Distinct().Count() == 1) singleHeight++; else multipleHeight++;
                }
                if (placements == 0) zero++;
                else if (placements == 1) one++;
                else
                {
                    multiple++;
                    if (covered.Select(p => p.ModelPath).Distinct(StringComparer.OrdinalIgnoreCase).Count() == 1)
                        multipleSameModel++;
                    else
                        multipleMixedModels++;
                }
                maxSurfaces = Math.Max(maxSurfaces, group.Count);
                maxPlacements = Math.Max(maxPlacements, placements);

                if (largest.Count < 40)
                {
                    largest.Add(new Pm4M2ComponentRow(
                        file.FileId, componentIndex, group.Count, placements,
                        string.Join(',', group.Select(i => surfaces[i].Padding).Distinct().Order()),
                        string.Join(',', group.Select(i => surfaces[i].WindowStart).Distinct().Order()),
                        string.Join('|', covered.Select(p => p.ModelPath).Distinct(StringComparer.OrdinalIgnoreCase))));
                }
                componentIndex++;
            }
        }

        return new Pm4M2PartitionReport(
            input.Count,
            filesWithM2,
            m2Surfaces,
            components,
            zero,
            one,
            multiple,
            multipleSameModel,
            multipleMixedModels,
            maxSurfaces,
            maxPlacements,
            components == 0 ? 0 : (double)componentSurfaceTotal / components,
            components == 0 ? 0 : (double)componentPlacementTotal / components,
            singleWindow,
            multipleWindow,
            singleNormal,
            multipleNormal,
            singleHeight,
            multipleHeight,
            disjointVertices,
            sharingVertices,
            noCoveringSurface,
            disjointRanges,
            overlappingRanges,
            singleGroupId,
            multipleGroupIds,
            groupIdSingleModel,
            groupIdMultipleModels,
            Rows(padding),
            Rows(windowStart),
            largest);
    }

    private static List<M2Surface> BuildM2Surfaces(Pm4KnownChunkSet chunks)
    {
        var result = new List<M2Surface>();
        for (int i = 0; i < chunks.Msur.Count; i++)
        {
            var surface = chunks.Msur[i];
            if (surface.Ck24Type != 0)
                continue;
            if ((long)surface.MsviFirstIndex + surface.IndexCount > chunks.Msvi.Count)
                throw new InvalidDataException("Invalid surface geometry window.");

            var vertices = new uint[surface.IndexCount];
            var polygon = new List<Vector3>(surface.IndexCount);
            for (int j = 0; j < surface.IndexCount; j++)
            {
                uint vi = chunks.Msvi[(int)surface.MsviFirstIndex + j];
                if (vi >= chunks.Msvt.Count)
                    throw new InvalidDataException("Invalid surface vertex.");
                vertices[j] = vi;
                polygon.Add(Pm4CoordinateService.Pm4LocalToAdtPlacement(chunks.Msvt[(int)vi]));
            }

            long end = (long)surface._0x18 + surface.AttributeMask;
            if (end > chunks.Mslk.Count)
                throw new InvalidDataException("Invalid proposed MSUR/MSLK window.");
            var groupObjectIds = new HashSet<uint>();
            for (long k = surface._0x18; k < end; k++)
                groupObjectIds.Add(chunks.Mslk[(int)k].GroupObjectId);

            result.Add(new M2Surface(i, vertices, polygon.ToArray(), surface.Padding, surface._0x18,
                surface.Normal, surface.Height, groupObjectIds.ToArray()));
        }
        return result;
    }

    private static void Increment(Dictionary<string, long> map, string key)
        => map[key] = map.TryGetValue(key, out long n) ? n + 1 : 1;

    private static IReadOnlyList<Pm4M2ValueCount> Rows(Dictionary<string, long> map)
        => map.OrderByDescending(p => p.Value).ThenBy(p => p.Key, StringComparer.Ordinal)
            .Select(p => new Pm4M2ValueCount(p.Key, p.Value)).ToArray();

    private static int Find(int[] parent, int x)
    {
        while (parent[x] != x)
        {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    private static void Union(int[] parent, int a, int b)
    {
        int ra = Find(parent, a), rb = Find(parent, b);
        if (ra != rb) parent[rb] = ra;
    }

    private sealed class M2Surface(int index, uint[] vertices, Vector3[] polygon, byte padding, uint windowStart,
        Vector3 normal, float height, uint[] groupObjectIds)
    {
        public int Index { get; } = index;
        public uint[] Vertices { get; } = vertices;
        public uint[] GroupObjectIds { get; } = groupObjectIds;
        public byte Padding { get; } = padding;
        public uint WindowStart { get; } = windowStart;
        public string NormalKey { get; } = $"{normal.X:F3},{normal.Y:F3},{normal.Z:F3}";
        public string HeightKey { get; } = height.ToString("F3");

        public bool Contains(Vector3 p)
        {
            if (polygon.Length < 3)
                return false;
            bool inside = false;
            for (int i = 0, j = polygon.Length - 1; i < polygon.Length; j = i++)
            {
                Vector3 a = polygon[i], b = polygon[j];
                if (DistanceToSegmentSquared(p, a, b) <= 0.05f * 0.05f)
                    return true;
                bool crosses = (a.Y > p.Y) != (b.Y > p.Y)
                    && p.X < (b.X - a.X) * (p.Y - a.Y) / (b.Y - a.Y) + a.X;
                if (crosses)
                    inside = !inside;
            }
            return inside;
        }

        private static float DistanceToSegmentSquared(Vector3 p, Vector3 a, Vector3 b)
        {
            float dx = b.X - a.X, dy = b.Y - a.Y;
            float lengthSquared = dx * dx + dy * dy;
            if (lengthSquared <= 1e-12f)
                return (p.X - a.X) * (p.X - a.X) + (p.Y - a.Y) * (p.Y - a.Y);
            float t = Math.Clamp(((p.X - a.X) * dx + (p.Y - a.Y) * dy) / lengthSquared, 0, 1);
            float cx = a.X + t * dx, cy = a.Y + t * dy;
            return (p.X - cx) * (p.X - cx) + (p.Y - cy) * (p.Y - cy);
        }
    }
}
