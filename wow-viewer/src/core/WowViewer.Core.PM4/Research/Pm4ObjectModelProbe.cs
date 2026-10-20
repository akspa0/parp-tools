using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4ObjectModelRow(
    string FileId,
    uint GroupObjectId,
    int Points,
    int Surfaces,
    int Placements,
    string ModelPaths);

public sealed record Pm4ObjectModelReport(
    int Files,
    int FilesWithPlacements,
    long Groups,
    long GroupsWithOneModel,
    long GroupsWithMultipleModels,
    long GroupsWithNoModel,
    long GroupsWithNoSurface,
    IReadOnlyList<Pm4ObjectModelRow> Samples);

/// <summary>
/// Joins the MPRL object groups (<c>Unk04</c> = <c>MSLK.GroupObjectId</c>) to the authored ADT
/// placements whose origin lies on the ground surfaces containing the group's points.
/// </summary>
/// <remarks>
/// A group that maps to one model path is a candidate object-to-asset match. A group that maps to
/// several model paths is either a merged mesh or a reused group id. This is a correlation
/// instrument, not an identity claim.
/// </remarks>
public static class Pm4ObjectModelProbe
{
    public static Pm4ObjectModelReport Analyze(IReadOnlyList<Pm4PlacementAuditInput> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        long groups = 0, oneModel = 0, multipleModels = 0, noModel = 0, noSurface = 0;
        int filesWithPlacements = 0;
        var samples = new List<Pm4ObjectModelRow>();

        foreach (var file in input)
        {
            if (file.Placements is null)
                continue;
            filesWithPlacements++;

            var chunks = file.Document.KnownChunks;
            var polygons = BuildPolygons(chunks);
            var placements = file.Placements.ModelPlacements
                .Concat<object>(file.Placements.WorldModelPlacements)
                .Select(p => p switch
                {
                    AdtModelPlacement m => (Position: m.Position, Model: m.ModelPath),
                    AdtWorldModelPlacement w => (Position: w.Position, Model: w.ModelPath),
                    _ => throw new InvalidDataException("Unknown placement type."),
                })
                .ToArray();

            // Surface index -> model paths of placements whose origin is inside it.
            var modelsBySurface = new Dictionary<int, HashSet<string>>();
            for (int s = 0; s < polygons.Count; s++)
            {
                foreach (var placement in placements)
                {
                    if (!Contains(polygons[s], placement.Position))
                        continue;
                    if (!modelsBySurface.TryGetValue(s, out var models))
                        modelsBySurface[s] = models = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                    models.Add(placement.Model);
                }
            }

            foreach (var group in chunks.Mprl.GroupBy(p => p.Unk04))
            {
                groups++;
                var surfaces = new HashSet<int>();
                foreach (var point in group)
                {
                    Vector3 placement = Pm4CoordinateService.MprlToAdtPlacement(point.Position);
                    for (int s = 0; s < polygons.Count; s++)
                    {
                        if (Contains(polygons[s], placement))
                            surfaces.Add(s);
                    }
                }
                if (surfaces.Count == 0)
                {
                    noSurface++;
                    continue;
                }
                var models = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
                foreach (int s in surfaces)
                {
                    if (modelsBySurface.TryGetValue(s, out var surfaceModels))
                        models.UnionWith(surfaceModels);
                }
                if (models.Count == 0) noModel++;
                else if (models.Count == 1) oneModel++;
                else multipleModels++;

                if (samples.Count < 60)
                {
                    samples.Add(new Pm4ObjectModelRow(
                        file.FileId, group.Key, group.Count(), surfaces.Count, models.Count,
                        string.Join('|', models.Order(StringComparer.OrdinalIgnoreCase))));
                }
            }
        }

        return new Pm4ObjectModelReport(
            input.Count, filesWithPlacements, groups,
            oneModel, multipleModels, noModel, noSurface, samples);
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
