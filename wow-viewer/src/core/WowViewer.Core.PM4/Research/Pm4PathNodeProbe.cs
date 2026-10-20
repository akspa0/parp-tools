using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public sealed record Pm4PathNodeValueCount(string Value, long Count);

public sealed record Pm4PathNodeReport(
    int Files,
    long MprlCount,
    long MprrCount,
    long MprrSentinels,
    long MprrValue1InMprlRange,
    long MprrValue2InMprlRange,
    long MprrValue1InMscnRange,
    long MprrValue2InMscnRange,
    long MprlInsideMesh,
    long MprlOutsideMesh,
    long MprlWithFinitePosition,
    IReadOnlyList<Pm4PathNodeValueCount> MprlUnk00,
    IReadOnlyList<Pm4PathNodeValueCount> MprlUnk02,
    IReadOnlyList<Pm4PathNodeValueCount> MprlUnk04,
    IReadOnlyList<Pm4PathNodeValueCount> MprlUnk06,
    IReadOnlyList<Pm4PathNodeValueCount> MprlUnk14,
    IReadOnlyList<Pm4PathNodeValueCount> MprlUnk16,
    IReadOnlyList<Pm4PathNodeValueCount> MprrValue1,
    IReadOnlyList<Pm4PathNodeValueCount> MprrValue2,
    IReadOnlyList<Pm4PathNodeValueCount> MprrValue1HighBytes,
    IReadOnlyList<Pm4PathNodeValueCount> MprrValue1LowBytes,
    IReadOnlyList<Pm4PathNodeValueCount> MprrValue2HighBytes,
    IReadOnlyList<Pm4PathNodeValueCount> MprrValue2LowBytes,
    IReadOnlyList<Pm4PathNodeValueCount> MprrRunLengths);

/// <summary>
/// Tests whether MPRL positions and MPRR records form the authored path-node layer: are MPRL
/// points inside the PM4 mesh, and do MPRR values index into MPRL or MSCN?
/// </summary>
/// <remarks>
/// A value in range is a candidate index, not proof of an edge. The sentinel 0xFFFF is retained
/// rather than silently dropped. This is a correlation instrument, not an editor-node identity claim.
/// </remarks>
public static class Pm4PathNodeProbe
{
    public static Pm4PathNodeReport Analyze(IReadOnlyList<Pm4MapAuditInput> input)
    {
        ArgumentNullException.ThrowIfNull(input);

        long mprl = 0, mprr = 0, sentinels = 0;
        long v1Mprl = 0, v2Mprl = 0, v1Mscn = 0, v2Mscn = 0;
        long inside = 0, outside = 0, finite = 0;
        var unk00 = new Dictionary<string, long>(StringComparer.Ordinal);
        var unk02 = new Dictionary<string, long>(StringComparer.Ordinal);
        var unk04 = new Dictionary<string, long>(StringComparer.Ordinal);
        var unk06 = new Dictionary<string, long>(StringComparer.Ordinal);
        var unk14 = new Dictionary<string, long>(StringComparer.Ordinal);
        var unk16 = new Dictionary<string, long>(StringComparer.Ordinal);
        var value1 = new Dictionary<string, long>(StringComparer.Ordinal);
        var value2 = new Dictionary<string, long>(StringComparer.Ordinal);
        var v1High = new Dictionary<string, long>(StringComparer.Ordinal);
        var v1Low = new Dictionary<string, long>(StringComparer.Ordinal);
        var v2High = new Dictionary<string, long>(StringComparer.Ordinal);
        var v2Low = new Dictionary<string, long>(StringComparer.Ordinal);
        var runLengths = new Dictionary<string, long>(StringComparer.Ordinal);

        foreach (var file in input)
        {
            var chunks = file.Document.KnownChunks;
            mprl += chunks.Mprl.Count;
            mprr += chunks.Mprr.Count;

            var polygons = BuildPolygons(chunks);
            foreach (var point in chunks.Mprl)
            {
                Increment(unk00, point.Unk00.ToString());
                Increment(unk02, point.Unk02.ToString());
                Increment(unk04, point.Unk04.ToString());
                Increment(unk06, point.Unk06.ToString());
                Increment(unk14, point.Unk14.ToString());
                Increment(unk16, point.Unk16.ToString());

                Vector3 placement = Pm4CoordinateService.MprlToAdtPlacement(point.Position);
                if (!float.IsFinite(placement.X) || !float.IsFinite(placement.Y) || !float.IsFinite(placement.Z))
                    continue;
                finite++;
                if (polygons.Any(p => Contains(p, placement))) inside++; else outside++;
            }

            int run = 0;
            foreach (var record in chunks.Mprr)
            {
                Increment(value1, record.Value1.ToString());
                Increment(value2, record.Value2.ToString());
                Increment(v1High, (record.Value1 >> 8).ToString());
                Increment(v1Low, (record.Value1 & 0xFF).ToString());
                Increment(v2High, (record.Value2 >> 8).ToString());
                Increment(v2Low, (record.Value2 & 0xFF).ToString());
                if (record.IsSentinel)
                {
                    sentinels++;
                    Increment(runLengths, run.ToString());
                    run = 0;
                }
                else
                {
                    run++;
                }
                if (record.Value1 < chunks.Mprl.Count) v1Mprl++;
                if (record.Value2 < chunks.Mprl.Count) v2Mprl++;
                if (record.Value1 < chunks.Mscn.Count) v1Mscn++;
                if (record.Value2 < chunks.Mscn.Count) v2Mscn++;
            }
            if (run > 0)
                Increment(runLengths, run.ToString());
        }

        return new Pm4PathNodeReport(
            input.Count, mprl, mprr, sentinels,
            v1Mprl, v2Mprl, v1Mscn, v2Mscn,
            inside, outside, finite,
            Rows(unk00), Rows(unk02), Rows(unk04), Rows(unk06), Rows(unk14), Rows(unk16),
            Rows(value1), Rows(value2),
            Rows(v1High), Rows(v1Low), Rows(v2High), Rows(v2Low), Rows(runLengths));
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

    private static void Increment(Dictionary<string, long> map, string key)
        => map[key] = map.TryGetValue(key, out long n) ? n + 1 : 1;

    private static IReadOnlyList<Pm4PathNodeValueCount> Rows(Dictionary<string, long> map)
        => map.OrderByDescending(p => p.Value).ThenBy(p => p.Key, StringComparer.Ordinal)
            .Take(20).Select(p => new Pm4PathNodeValueCount(p.Key, p.Value)).ToArray();
}
