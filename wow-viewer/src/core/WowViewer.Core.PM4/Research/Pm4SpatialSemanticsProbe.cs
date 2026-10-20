using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4SpatialSemanticsReport(
    int Files,
    long MscnPoints,
    long MscnWithSurface,
    double MscnMedianBoundaryDistance,
    double MscnMedianInteriorDistance,
    long MscnCloserToBoundary,
    long MscnCloserToInterior,
    long UnattributedGroups,
    long UnattributedGroupsOneModel,
    long UnattributedGroupsMultipleModels,
    long UnattributedGroupsNoModel,
    long MsurVertices,
    long MsurVerticesNearMprl,
    double MsurVertexMedianHeightError);

/// <summary>
/// Spatial tests for the open questions: MSCN contour placement, unattributed object groups, and
/// MSUR vertices as terrain-height samples. All distances are in placement space.
/// </summary>
public static class Pm4SpatialSemanticsProbe
{
    private const float SurfaceCell = 64f;
    private const float MprlCell = 8f;

    public static Pm4SpatialSemanticsReport Analyze(IReadOnlyList<Pm4PlacementAuditInput> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        long mscnPoints = 0, mscnWithSurface = 0, closerBoundary = 0, closerInterior = 0;
        var boundaryDistances = new List<double>();
        var interiorDistances = new List<double>();
        long unattributedGroups = 0, unattributedOne = 0, unattributedMultiple = 0, unattributedNone = 0;
        long msurVertices = 0, msurNearMprl = 0;
        var heightErrors = new List<double>();

        foreach (var file in input)
        {
            var chunks = file.Document.KnownChunks;
            var polygons = BuildPolygons(chunks);
            var centroids = new Vector2[polygons.Length];
            var radii = new float[polygons.Length];
            for (int s = 0; s < polygons.Length; s++)
            {
                if (polygons[s].Length == 0) continue;
                Vector2 centroid = Vector2.Zero;
                foreach (var vertex in polygons[s])
                {
                    var p = Pm4CoordinateService.Pm4LocalToAdtPlacement(vertex);
                    centroid += new Vector2(p.X, p.Y);
                }
                centroid /= polygons[s].Length;
                centroids[s] = centroid;
                radii[s] = polygons[s].Max(v =>
                {
                    var p = Pm4CoordinateService.Pm4LocalToAdtPlacement(v);
                    return Vector2.Distance(centroid, new Vector2(p.X, p.Y));
                });
            }

            var surfaceGrid = BuildSurfaceGrid(centroids, polygons);

            foreach (var point in chunks.Mscn)
            {
                mscnPoints++;
                var placement = Pm4CoordinateService.Pm4LocalToAdtPlacement(point);
                var xy = new Vector2(placement.X, placement.Y);
                int surface = FindNearestSurface(surfaceGrid, centroids, polygons, xy);
                if (surface < 0 || radii[surface] < 2f) continue;
                mscnWithSurface++;
                double boundary = DistanceToBoundary(polygons[surface], xy);
                double interior = Vector2.Distance(xy, centroids[surface]);
                boundaryDistances.Add(boundary);
                interiorDistances.Add(interior);
                if (boundary < interior) closerBoundary++;
                else closerInterior++;
            }

            var mprlGrid = BuildMprlGrid(chunks.Mprl);
            foreach (var surface in polygons)
            {
                foreach (var vertex in surface)
                {
                    msurVertices++;
                    var placement = Pm4CoordinateService.Pm4LocalToAdtPlacement(vertex);
                    if (TryNearestMprl(mprlGrid, chunks.Mprl, placement, out float mprlZ))
                    {
                        msurNearMprl++;
                        heightErrors.Add(Math.Abs(placement.Z - mprlZ));
                    }
                }
            }

            if (file.Placements is not null)
            {
                var models = file.Placements.ModelPlacements.Select(p => (p.Position, p.ModelPath))
                    .Concat(file.Placements.WorldModelPlacements.Select(p => (p.Position, p.ModelPath)))
                    .ToArray();
                var surfacesByGroup = BuildSurfacesByGroup(chunks);
                foreach (var group in chunks.Mprl.GroupBy(p => p.Unk04))
                {
                    if (!surfacesByGroup.TryGetValue(group.Key, out var surfaces)) continue;
                    bool allUnattributed = surfaces.All(s => chunks.Msur[s].PackedParams == 0u);
                    if (!allUnattributed) continue;
                    unattributedGroups++;
                    var modelSet = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    foreach (var point in group)
                    {
                        var placement = Pm4CoordinateService.MprlToAdtPlacement(point.Position);
                        foreach (var model in models)
                        {
                            if (Vector3.Distance(placement, model.Position) <= 25f)
                                modelSet.Add(model.ModelPath);
                        }
                    }
                    if (modelSet.Count == 0) unattributedNone++;
                    else if (modelSet.Count == 1) unattributedOne++;
                    else unattributedMultiple++;
                }
            }
        }

        return new Pm4SpatialSemanticsReport(
            input.Count,
            mscnPoints,
            mscnWithSurface,
            Median(boundaryDistances),
            Median(interiorDistances),
            closerBoundary,
            closerInterior,
            unattributedGroups,
            unattributedOne,
            unattributedMultiple,
            unattributedNone,
            msurVertices,
            msurNearMprl,
            Median(heightErrors));
    }

    private static Dictionary<(int X, int Y), List<int>> BuildSurfaceGrid(Vector2[] centroids, Vector3[][] polygons)
    {
        var grid = new Dictionary<(int X, int Y), List<int>>();
        for (int s = 0; s < polygons.Length; s++)
        {
            if (polygons[s].Length == 0) continue;
            var key = Cell(centroids[s], SurfaceCell);
            if (!grid.TryGetValue(key, out var list)) grid[key] = list = new List<int>();
            list.Add(s);
        }
        return grid;
    }

    private static int FindNearestSurface(
        Dictionary<(int X, int Y), List<int>> grid,
        Vector2[] centroids,
        Vector3[][] polygons,
        Vector2 point)
    {
        var key = Cell(point, SurfaceCell);
        int best = -1;
        float bestDistance = float.MaxValue;
        for (int radius = 0; radius <= 2; radius++)
        {
            for (int dx = -radius; dx <= radius; dx++)
            {
                for (int dy = -radius; dy <= radius; dy++)
                {
                    if (Math.Max(Math.Abs(dx), Math.Abs(dy)) != radius) continue;
                    if (!grid.TryGetValue((key.X + dx, key.Y + dy), out var candidates)) continue;
                    foreach (int s in candidates)
                    {
                        float distance = Vector2.Distance(point, centroids[s]);
                        if (distance < bestDistance) { bestDistance = distance; best = s; }
                    }
                }
            }
            if (best >= 0) return best;
        }
        return best;
    }

    private static double DistanceToBoundary(Vector3[] polygon, Vector2 point)
    {
        double best = double.MaxValue;
        for (int i = 0; i < polygon.Length; i++)
        {
            var a = Pm4CoordinateService.Pm4LocalToAdtPlacement(polygon[i]);
            var b = Pm4CoordinateService.Pm4LocalToAdtPlacement(polygon[(i + 1) % polygon.Length]);
            best = Math.Min(best, DistanceToSegment(point, new Vector2(a.X, a.Y), new Vector2(b.X, b.Y)));
        }
        return best;
    }

    private static double DistanceToSegment(Vector2 p, Vector2 a, Vector2 b)
    {
        Vector2 ab = b - a;
        float lengthSquared = ab.LengthSquared();
        if (lengthSquared <= 1e-6f) return Vector2.Distance(p, a);
        float t = Math.Clamp(Vector2.Dot(p - a, ab) / lengthSquared, 0f, 1f);
        return Vector2.Distance(p, a + ab * t);
    }

    private static Dictionary<(int X, int Y), List<int>> BuildMprlGrid(IReadOnlyList<Pm4MprlEntry> points)
    {
        var grid = new Dictionary<(int X, int Y), List<int>>();
        for (int i = 0; i < points.Count; i++)
        {
            var placement = Pm4CoordinateService.MprlToAdtPlacement(points[i].Position);
            var key = Cell(new Vector2(placement.X, placement.Y), MprlCell);
            if (!grid.TryGetValue(key, out var list)) grid[key] = list = new List<int>();
            list.Add(i);
        }
        return grid;
    }

    private static bool TryNearestMprl(
        Dictionary<(int X, int Y), List<int>> grid,
        IReadOnlyList<Pm4MprlEntry> points,
        Vector3 placement,
        out float z)
    {
        z = 0f;
        var key = Cell(new Vector2(placement.X, placement.Y), MprlCell);
        float best = 1f;
        bool found = false;
        for (int dx = -1; dx <= 1; dx++)
        {
            for (int dy = -1; dy <= 1; dy++)
            {
                if (!grid.TryGetValue((key.X + dx, key.Y + dy), out var candidates)) continue;
                foreach (int index in candidates)
                {
                    var candidate = Pm4CoordinateService.MprlToAdtPlacement(points[index].Position);
                    float distance = Vector2.Distance(new Vector2(placement.X, placement.Y), new Vector2(candidate.X, candidate.Y));
                    if (distance <= best)
                    {
                        best = distance;
                        z = candidate.Z;
                        found = true;
                    }
                }
            }
        }
        return found;
    }

    private static Dictionary<ushort, List<int>> BuildSurfacesByGroup(Pm4KnownChunkSet chunks)
    {
        var result = new Dictionary<ushort, List<int>>();
        for (int s = 0; s < chunks.Msur.Count; s++)
        {
            var surface = chunks.Msur[s];
            int start = (int)surface._0x18;
            int count = surface.AttributeMask;
            if (start < 0 || start + count > chunks.Mslk.Count) continue;
            for (int i = start; i < start + count; i++)
            {
                ushort group = (ushort)chunks.Mslk[i].GroupObjectId;
                if (!result.TryGetValue(group, out var list)) result[group] = list = new List<int>();
                if (!list.Contains(s)) list.Add(s);
            }
        }
        return result;
    }

    private static Vector3[][] BuildPolygons(Pm4KnownChunkSet chunks)
    {
        var polygons = new Vector3[chunks.Msur.Count][];
        for (int s = 0; s < chunks.Msur.Count; s++)
        {
            var surface = chunks.Msur[s];
            int start = (int)surface.MsviFirstIndex;
            int count = surface.IndexCount;
            if (start < 0 || count < 3 || start + count > chunks.Msvi.Count)
            {
                polygons[s] = [];
                continue;
            }
            var vertices = new Vector3[count];
            bool valid = true;
            for (int i = 0; i < count; i++)
            {
                uint index = chunks.Msvi[start + i];
                if (index >= chunks.Msvt.Count) { valid = false; break; }
                vertices[i] = chunks.Msvt[(int)index];
            }
            polygons[s] = valid ? vertices : [];
        }
        return polygons;
    }

    private static (int X, int Y) Cell(Vector2 point, float size)
        => ((int)MathF.Floor(point.X / size), (int)MathF.Floor(point.Y / size));

    private static double Median(List<double> values)
    {
        if (values.Count == 0) return 0d;
        values.Sort();
        return values[values.Count / 2];
    }
}
