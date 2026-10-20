using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>One MPRL object group (<c>Unk04</c> = <c>MSLK.GroupObjectId</c>) for a single tile.</summary>
public sealed record Pm4ObjectGroupEntry(
    uint GroupObjectId,
    int PointCount,
    int SurfaceCount,
    Vector3 Centroid,
    Vector3 BoundsMin,
    Vector3 BoundsMax,
    string SurfaceClass,
    string? NearestModel,
    float NearestDistance,
    bool Consistent);

/// <summary>
/// Builds the per-tile object-group catalog used by the viewer: each MPRL <c>Unk04</c> group with
/// its point/surface counts, ground-surface class, nearest authored model and consistency flag.
/// </summary>
/// <remarks>
/// This is the single-tile counterpart of <see cref="Pm4ObjectModelProbe"/>. It is read-only and
/// does not change the working WMO baseline.
/// </remarks>
public static class Pm4ObjectGroupCatalog
{
    public static IReadOnlyList<Pm4ObjectGroupEntry> Build(
        Pm4ResearchDocument document,
        AdtPlacementCatalog? placements)
    {
        ArgumentNullException.ThrowIfNull(document);

        var chunks = document.KnownChunks;
        var polygons = BuildPolygons(chunks);
        var authored = placements is null
            ? []
            : placements.ModelPlacements
                .Select(p => (Position: p.Position, Model: p.ModelPath))
                .Concat(placements.WorldModelPlacements.Select(p => (Position: p.Position, Model: p.ModelPath)))
                .ToArray();

        var entries = new List<Pm4ObjectGroupEntry>();
        foreach (var group in chunks.Mprl.GroupBy(p => p.Unk04))
        {
            var points = group.ToArray();
            Vector3 centroid = Vector3.Zero;
            Vector3 min = new(float.PositiveInfinity);
            Vector3 max = new(float.NegativeInfinity);
            var surfaces = new HashSet<int>();
            foreach (var point in points)
            {
                Vector3 placement = Pm4CoordinateService.MprlToAdtPlacement(point.Position);
                centroid += placement;
                min = Vector3.Min(min, placement);
                max = Vector3.Max(max, placement);
                for (int s = 0; s < polygons.Count; s++)
                {
                    if (Contains(polygons[s], placement))
                        surfaces.Add(s);
                }
            }
            centroid /= points.Length;

            bool hasM2 = false, hasWmo = false;
            foreach (int s in surfaces)
            {
                byte type = chunks.Msur[s].Ck24Type;
                if (type == 0) hasM2 = true;
                else if (type is >= 64 and <= 67) hasWmo = true;
            }
            string surfaceClass = hasM2 && hasWmo ? "mixed" : hasM2 ? "m2" : hasWmo ? "wmo" : "other";

            string? nearestModel = null;
            float nearestDistance = float.PositiveInfinity;
            foreach (var placement in authored)
            {
                float d = Vector3.Distance(placement.Position, centroid);
                if (d < nearestDistance)
                {
                    nearestDistance = d;
                    nearestModel = placement.Model;
                }
            }
            string modelClass = nearestModel is null ? "none"
                : nearestModel.EndsWith(".M2", StringComparison.OrdinalIgnoreCase) ? "m2"
                : nearestModel.EndsWith(".WMO", StringComparison.OrdinalIgnoreCase) ? "wmo" : "other";
            bool consistent = nearestModel is not null && nearestDistance <= 25 && surfaceClass == modelClass;

            entries.Add(new Pm4ObjectGroupEntry(
                group.Key, points.Length, surfaces.Count, centroid, min, max,
                surfaceClass, nearestModel, nearestDistance, consistent));
        }

        return entries
            .OrderByDescending(e => e.PointCount)
            .ThenBy(e => e.GroupObjectId)
            .ToArray();
    }

    private static List<Vector3[]> BuildPolygons(Pm4KnownChunkSet chunks)
    {
        var result = new List<Vector3[]>(chunks.Msur.Count);
        for (int i = 0; i < chunks.Msur.Count; i++)
        {
            var surface = chunks.Msur[i];
            if ((long)surface.MsviFirstIndex + surface.IndexCount > chunks.Msvi.Count)
                throw new InvalidDataException("Invalid surface geometry window.");
            var polygon = new List<Vector3>(surface.IndexCount);
            for (int j = 0; j < surface.IndexCount; j++)
            {
                uint vi = chunks.Msvi[(int)surface.MsviFirstIndex + j];
                if (vi >= chunks.Msvt.Count)
                    throw new InvalidDataException("Invalid surface vertex.");
                polygon.Add(Pm4CoordinateService.Pm4LocalToAdtPlacement(chunks.Msvt[(int)vi]));
            }
            result.Add(polygon.ToArray());
        }
        return result;
    }

    private static bool Contains(Vector3[] polygon, Vector3 p)
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
