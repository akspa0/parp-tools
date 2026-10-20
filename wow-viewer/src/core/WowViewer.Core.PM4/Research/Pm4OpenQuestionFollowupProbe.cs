using System.Numerics;
using WowViewer.Core.Maps;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4MprrNodeRow(int Width, int Height, int Runs);

public sealed record Pm4OpenQuestionFollowupReport(
    int Files,
    long SurfacesWithPolygon,
    long PlaneWorldMatches,
    double PlaneWorldMedianError,
    long PlaneNegatedMatches,
    double PlaneNegatedMedianError,
    double MshdField00VsRawPlacementX,
    double MshdField00VsRawPlacementY,
    double MshdField08VsRawPlacementX,
    double MshdField08VsRawPlacementY,
    long NonzeroPlacementZ,
    long NonzeroPlacementZGlobalMatched,
    long MprlCounterMatchesRunIndex,
    long MprlCounterMatchesNodeLevel,
    long MprlCounterOther,
    long MprrRuns,
    long MprrFirstTwoAreSpatialBounds,
    long MprrLastIsTerminator,
    long MprrMiddleInsideBounds,
    long MprrMiddleTotal,
    IReadOnlyList<Pm4MprrNodeRow> MprrNodeSizes);

/// <summary>
/// Follow-up tests for the questions the first sweep left open or partial: the MSUR plane frame,
/// the MSHD span frame, the global placement-Z match, the MPRL counter, and the MPRR node bounds.
/// </summary>
public static class Pm4OpenQuestionFollowupProbe
{
    private const float MapOrigin = Pm4CoordinateService.MapOrigin;
    // The MPRR node bounds run 0..767 over a 533.333-unit tile: a 768x768 grid.
    private const float Grid = 533.333f / 768f;

    public static Pm4OpenQuestionFollowupReport Analyze(IReadOnlyList<Pm4PlacementAuditInput> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        var globalPlacementZ = new HashSet<uint>();
        foreach (var file in input)
        {
            if (file.Placements is null) continue;
            foreach (var placement in file.Placements.ModelPlacements)
                globalPlacementZ.Add(BitConverter.SingleToUInt32Bits(placement.Position.Z));
            foreach (var placement in file.Placements.WorldModelPlacements)
                globalPlacementZ.Add(BitConverter.SingleToUInt32Bits(placement.Position.Z));
        }

        long surfacesWithPolygon = 0, planeWorldMatches = 0, planeNegatedMatches = 0;
        var planeWorldErrors = new List<double>();
        var planeNegatedErrors = new List<double>();
        long nonzeroPlacementZ = 0, nonzeroPlacementZGlobalMatched = 0;
        long counterRunIndex = 0, counterNodeLevel = 0, counterOther = 0;
        long mprrRuns = 0, mprrFirstTwoBounds = 0, mprrLastTerminator = 0;
        long mprrMiddleInside = 0, mprrMiddleTotal = 0;
        var nodeSizes = new Dictionary<(int W, int H), int>();

        var mshdRows = new List<(uint F00, uint F08, float RawX, float RawY)>();

        foreach (var file in input)
        {
            var chunks = file.Document.KnownChunks;
            var polygons = BuildPolygons(chunks);

            if (chunks.Mshd is not null && file.Placements is not null)
            {
                var positions = file.Placements.ModelPlacements.Select(p => p.Position)
                    .Concat(file.Placements.WorldModelPlacements.Select(p => p.Position))
                    .ToArray();
                if (positions.Length > 0)
                {
                    float rawX = positions.Max(p => MapOrigin - p.X) - positions.Min(p => MapOrigin - p.X);
                    float rawY = positions.Max(p => MapOrigin - p.Y) - positions.Min(p => MapOrigin - p.Y);
                    mshdRows.Add((chunks.Mshd.Field00, chunks.Mshd.Field08, rawX, rawY));
                }
            }

            for (int s = 0; s < chunks.Msur.Count; s++)
            {
                var surface = chunks.Msur[s];
                if (polygons[s].Length < 3) continue;
                surfacesWithPolygon++;

                Vector3 world = Pm4CoordinateService.Pm4LocalToAdtPlacement(polygons[s][0]);
                double worldPlane = Vector3.Dot(surface.Normal, world);
                double worldError = Math.Abs(worldPlane - surface.Height);
                planeWorldErrors.Add(worldError);
                if (worldError <= 0.01d) planeWorldMatches++;

                double negated = -Vector3.Dot(surface.Normal, polygons[s][0]);
                double negatedError = Math.Abs(negated - surface.Height);
                planeNegatedErrors.Add(negatedError);
                if (negatedError <= 0.01d) planeNegatedMatches++;

                if (surface.PackedParams != 0u)
                {
                    nonzeroPlacementZ++;
                    if (globalPlacementZ.Contains(surface.PackedParams))
                        nonzeroPlacementZGlobalMatched++;
                }
            }

            var mprr = AnalyzeMprr(chunks, nodeSizes);
            mprrRuns += mprr.Runs;
            mprrFirstTwoBounds += mprr.FirstTwoBounds;
            mprrLastTerminator += mprr.LastTerminator;
            mprrMiddleInside += mprr.MiddleInside;
            mprrMiddleTotal += mprr.MiddleTotal;
            counterRunIndex += mprr.CounterRunIndex;
            counterNodeLevel += mprr.CounterNodeLevel;
            counterOther += mprr.CounterOther;
        }

        return new Pm4OpenQuestionFollowupReport(
            input.Count,
            surfacesWithPolygon,
            planeWorldMatches,
            Median(planeWorldErrors),
            planeNegatedMatches,
            Median(planeNegatedErrors),
            Pearson(mshdRows.Select(r => (double)r.F00), mshdRows.Select(r => (double)r.RawX)),
            Pearson(mshdRows.Select(r => (double)r.F00), mshdRows.Select(r => (double)r.RawY)),
            Pearson(mshdRows.Select(r => (double)r.F08), mshdRows.Select(r => (double)r.RawX)),
            Pearson(mshdRows.Select(r => (double)r.F08), mshdRows.Select(r => (double)r.RawY)),
            nonzeroPlacementZ,
            nonzeroPlacementZGlobalMatched,
            counterRunIndex,
            counterNodeLevel,
            counterOther,
            mprrRuns,
            mprrFirstTwoBounds,
            mprrLastTerminator,
            mprrMiddleInside,
            mprrMiddleTotal,
            nodeSizes.OrderBy(kv => kv.Key.W).ThenBy(kv => kv.Key.H)
                .Select(kv => new Pm4MprrNodeRow(kv.Key.W, kv.Key.H, kv.Value)).ToList());
    }

    private static (long Runs, long FirstTwoBounds, long LastTerminator, long MiddleInside, long MiddleTotal,
        long CounterRunIndex, long CounterNodeLevel, long CounterOther) AnalyzeMprr(
        Pm4KnownChunkSet chunks,
        Dictionary<(int W, int H), int> nodeSizes)
    {
        long runs = 0, firstTwoBounds = 0, lastTerminator = 0, middleInside = 0, middleTotal = 0;
        long counterRunIndex = 0, counterNodeLevel = 0, counterOther = 0;
        var current = new List<Pm4MprrEntry>();
        void Flush()
        {
            if (current.Count == 0) return;
            runs++;
            if (current.Count >= 3)
            {
                int minX = current[0].Value1, minY = current[0].Value2;
                int maxX = current[1].Value1, maxY = current[1].Value2;
                bool ordered = minX <= maxX && minY <= maxY;
                if (ordered) firstTwoBounds++;
                if (current[^1].Value1 == 0 && current[^1].Value2 == 768) lastTerminator++;

                int width = maxX - minX + 1;
                int height = maxY - minY + 1;
                if (width > 0 && height > 0)
                {
                    var key = (width, height);
                    nodeSizes[key] = nodeSizes.GetValueOrDefault(key) + 1;
                }
                int level = width > 0 ? (int)MathF.Round(MathF.Log2(width)) : 0;

                for (int i = 2; i < current.Count - 1; i++)
                {
                    int index = current[i].Value1;
                    if (index >= chunks.Mprl.Count) continue;
                    middleTotal++;
                    var placement = Pm4CoordinateService.MprlToAdtPlacement(chunks.Mprl[index].Position);
                    int gx = (int)MathF.Floor(placement.X / Grid);
                    int gy = (int)MathF.Floor(placement.Y / Grid);
                    if (gx >= minX && gx <= maxX && gy >= minY && gy <= maxY)
                        middleInside++;

                    short value = chunks.Mprl[index].Unk14;
                    if (value == i - 2) counterRunIndex++;
                    else if (value == level) counterNodeLevel++;
                    else counterOther++;
                }
            }
            current.Clear();
        }
        foreach (var record in chunks.Mprr)
        {
            if (record.IsSentinel) Flush();
            else current.Add(record);
        }
        Flush();
        return (runs, firstTwoBounds, lastTerminator, middleInside, middleTotal,
            counterRunIndex, counterNodeLevel, counterOther);
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

    private static double Median(List<double> values)
    {
        if (values.Count == 0) return 0d;
        values.Sort();
        return values[values.Count / 2];
    }

    private static double Pearson(IEnumerable<double> x, IEnumerable<double> y)
    {
        var xs = x.ToArray();
        var ys = y.ToArray();
        if (xs.Length < 2 || xs.Length != ys.Length) return 0d;
        double mx = xs.Average(), my = ys.Average();
        double numerator = 0d, dx = 0d, dy = 0d;
        for (int i = 0; i < xs.Length; i++)
        {
            double a = xs[i] - mx, b = ys[i] - my;
            numerator += a * b; dx += a * a; dy += b * b;
        }
        return dx > 0 && dy > 0 ? numerator / Math.Sqrt(dx * dy) : 0d;
    }
}
