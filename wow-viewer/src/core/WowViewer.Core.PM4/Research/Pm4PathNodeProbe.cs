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
    long MprrValue1InMsurRange,
    long MprrValue2InMsurRange,
    long MprrValue1InMsvtRange,
    long MprrValue2InMsvtRange,
    long MprrValue1InMsviRange,
    long MprrValue2InMsviRange,
    long MprrValue1InMspvRange,
    long MprrValue2InMspvRange,
    long MprrValue1InMspiRange,
    long MprrValue2InMspiRange,
    long MprrValue1InMslkRange,
    long MprrValue2InMslkRange,
    long MprrRunsWithValidBox,
    long MprrRunsWithInvalidBox,
    long MprrMiddleValuesInsideBox,
    long MprrMiddleValuesOutsideBox,
    long MprrRunsWith32x32Bounds,
    long MprrRunsAlignedTo32,
    long MprrMiddleValuesInChunkRange,
    long MprrMiddleValuesOutsideChunkRange,
    long MprrMiddleValuesZeroV2,
    long MprrMiddleValuesNonZeroV2,
    long MprrRunsWithValidMsurRange,
    long MprrRunsWithValidMprlIndices,
    long MprrRunsWithZeroMiddleV2,
    long MprrMsurRangeWidthSum,
    long MprrMsurRangeOverlaps,
    long MprrRunsWithSingleMprlUnk04,
    long MprrRunsWithMultipleMprlUnk04,
    long MprrRunsWithSingleMprlUnk00,
    long MprrRunsWithMultipleMprlUnk00,
    long MprlUnk04MatchesMslkGroupObjectId,
    long MprlUnk04MatchesMsurCk24ObjectId,
    long MprlUnk04MatchesMsurGroupKey,
    long MprlUnk04MatchesMscnIndex,
    long MprlPointsWithGroupSurfaces,
    long MprlPointsInsideGroupSurfaces,
    long MprlPointsOutsideGroupSurfaces,
    long MprlPointsInsideM2Surfaces,
    long MprlPointsInsideWmoSurfaces,
    long MprlPointsInsideOtherSurfaces,
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
        long v1Msur = 0, v2Msur = 0, v1Msvt = 0, v2Msvt = 0, v1Msvi = 0, v2Msvi = 0;
        long v1Mspv = 0, v2Mspv = 0, v1Mspi = 0, v2Mspi = 0, v1Mslk = 0, v2Mslk = 0;
        long validBox = 0, invalidBox = 0, middleInside = 0, middleOutside = 0;
        long bounds32 = 0, aligned32 = 0, middleInChunk = 0, middleOutChunk = 0, middleZeroV2 = 0, middleNonZeroV2 = 0;
        long validMsurRange = 0, validMprlIndices = 0, zeroMiddleV2 = 0, msurWidthSum = 0, msurOverlaps = 0;
        long singleUnk04 = 0, multipleUnk04 = 0, singleUnk00 = 0, multipleUnk00 = 0;
        long unk04Mslk = 0, unk04Ck24 = 0, unk04GroupKey = 0, unk04Mscn = 0;
        long withGroupSurfaces = 0, insideGroupSurfaces = 0, outsideGroupSurfaces = 0;
        long insideM2 = 0, insideWmo = 0, insideOther = 0;
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
            // Map each GroupObjectId to the surfaces whose MSLK window references it.
            var surfacesByGroup = new Dictionary<uint, List<int>>();
            for (int s = 0; s < chunks.Msur.Count; s++)
            {
                var surface = chunks.Msur[s];
                long end = (long)surface._0x18 + surface.AttributeMask;
                if (end > chunks.Mslk.Count)
                    throw new InvalidDataException("Invalid proposed MSUR/MSLK window.");
                for (long k = surface._0x18; k < end; k++)
                {
                    uint id = chunks.Mslk[(int)k].GroupObjectId;
                    if (!surfacesByGroup.TryGetValue(id, out var list))
                        surfacesByGroup[id] = list = [];
                    if (!list.Contains(s)) list.Add(s);
                }
            }
            var mslkIds = new HashSet<uint>(chunks.Mslk.Select(l => l.GroupObjectId));
            var ck24Ids = new HashSet<uint>(chunks.Msur.Select(s => (uint)s.Ck24ObjectId));
            var groupKeys = new HashSet<uint>(chunks.Msur.Select(s => (uint)s.GroupKey));
            foreach (var point in chunks.Mprl)
            {
                if (mslkIds.Contains(point.Unk04)) unk04Mslk++;
                if (ck24Ids.Contains(point.Unk04)) unk04Ck24++;
                if (groupKeys.Contains(point.Unk04)) unk04GroupKey++;
                if (point.Unk04 < chunks.Mscn.Count) unk04Mscn++;
                if (surfacesByGroup.TryGetValue(point.Unk04, out var groupSurfaces))
                {
                    withGroupSurfaces++;
                    Vector3 groupPlacement = Pm4CoordinateService.MprlToAdtPlacement(point.Position);
                    if (groupSurfaces.Any(s => Contains(polygons[s], groupPlacement)))
                        insideGroupSurfaces++;
                    else
                        outsideGroupSurfaces++;
                }
                Vector3 pointPlacement = Pm4CoordinateService.MprlToAdtPlacement(point.Position);
                bool inM2 = false, inWmo = false, inOther = false;
                for (int s = 0; s < polygons.Count; s++)
                {
                    if (!Contains(polygons[s], pointPlacement))
                        continue;
                    byte type = chunks.Msur[s].Ck24Type;
                    if (type == 0) inM2 = true;
                    else if (type is >= 64 and <= 67) inWmo = true;
                    else inOther = true;
                }
                if (inM2) insideM2++;
                if (inWmo) insideWmo++;
                if (inOther) insideOther++;
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
            var current = new List<Pm4MprrEntry>();
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
                    AnalyzeRun(current, ref validBox, ref invalidBox, ref middleInside, ref middleOutside,
                        ref bounds32, ref aligned32, ref middleInChunk, ref middleOutChunk, ref middleZeroV2, ref middleNonZeroV2,
                        chunks, ref validMsurRange, ref validMprlIndices, ref zeroMiddleV2, ref msurWidthSum, ref msurOverlaps,
                        ref singleUnk04, ref multipleUnk04, ref singleUnk00, ref multipleUnk00);
                    current.Clear();
                    run = 0;
                }
                else
                {
                    current.Add(record);
                    run++;
                }
                if (record.Value1 < chunks.Mprl.Count) v1Mprl++;
                if (record.Value2 < chunks.Mprl.Count) v2Mprl++;
                if (record.Value1 < chunks.Mscn.Count) v1Mscn++;
                if (record.Value2 < chunks.Mscn.Count) v2Mscn++;
                if (record.Value1 < chunks.Msur.Count) v1Msur++;
                if (record.Value2 < chunks.Msur.Count) v2Msur++;
                if (record.Value1 < chunks.Msvt.Count) v1Msvt++;
                if (record.Value2 < chunks.Msvt.Count) v2Msvt++;
                if (record.Value1 < chunks.Msvi.Count) v1Msvi++;
                if (record.Value2 < chunks.Msvi.Count) v2Msvi++;
                if (record.Value1 < chunks.Mspv.Count) v1Mspv++;
                if (record.Value2 < chunks.Mspv.Count) v2Mspv++;
                if (record.Value1 < chunks.Mspi.Count) v1Mspi++;
                if (record.Value2 < chunks.Mspi.Count) v2Mspi++;
                if (record.Value1 < chunks.Mslk.Count) v1Mslk++;
                if (record.Value2 < chunks.Mslk.Count) v2Mslk++;
            }
            if (run > 0)
            {
                Increment(runLengths, run.ToString());
                AnalyzeRun(current, ref validBox, ref invalidBox, ref middleInside, ref middleOutside,
                    ref bounds32, ref aligned32, ref middleInChunk, ref middleOutChunk, ref middleZeroV2, ref middleNonZeroV2,
                    chunks, ref validMsurRange, ref validMprlIndices, ref zeroMiddleV2, ref msurWidthSum, ref msurOverlaps,
                    ref singleUnk04, ref multipleUnk04, ref singleUnk00, ref multipleUnk00);
            }
        }

        return new Pm4PathNodeReport(
            input.Count, mprl, mprr, sentinels,
            v1Mprl, v2Mprl, v1Mscn, v2Mscn,
            v1Msur, v2Msur, v1Msvt, v2Msvt, v1Msvi, v2Msvi,
            v1Mspv, v2Mspv, v1Mspi, v2Mspi, v1Mslk, v2Mslk,
            validBox, invalidBox, middleInside, middleOutside,
            bounds32, aligned32, middleInChunk, middleOutChunk, middleZeroV2, middleNonZeroV2,
            validMsurRange, validMprlIndices, zeroMiddleV2, msurWidthSum, msurOverlaps,
            singleUnk04, multipleUnk04, singleUnk00, multipleUnk00,
            unk04Mslk, unk04Ck24, unk04GroupKey, unk04Mscn,
            withGroupSurfaces, insideGroupSurfaces, outsideGroupSurfaces,
            insideM2, insideWmo, insideOther,
            inside, outside, finite,
            Rows(unk00), Rows(unk02), Rows(unk04), Rows(unk06), Rows(unk14), Rows(unk16),
            Rows(value1), Rows(value2),
            Rows(v1High), Rows(v1Low), Rows(v2High), Rows(v2Low), Rows(runLengths));
    }

    private static void AnalyzeRun(
        List<Pm4MprrEntry> run,
        ref long validBox, ref long invalidBox, ref long middleInside, ref long middleOutside,
        ref long bounds32, ref long aligned32, ref long middleInChunk, ref long middleOutChunk,
        ref long middleZeroV2, ref long middleNonZeroV2,
        Pm4KnownChunkSet chunks,
        ref long validMsurRange, ref long validMprlIndices, ref long zeroMiddleV2,
        ref long msurWidthSum, ref long msurOverlaps,
        ref long singleUnk04, ref long multipleUnk04, ref long singleUnk00, ref long multipleUnk00)
    {
        if (run.Count < 2)
            return;
        // Hypothesis: the first two records are the (min, max) corners of a box and the rest are
        // cells inside it. A valid box has min <= max on both axes.
        var first = run[0];
        var second = run[1];
        bool valid = first.Value1 <= second.Value1 && first.Value2 <= second.Value2;
        if (valid) validBox++; else invalidBox++;
        if (!valid)
            return;
        for (int i = 2; i < run.Count; i++)
        {
            var r = run[i];
            if (r.Value1 >= first.Value1 && r.Value1 <= second.Value1
                && r.Value2 >= first.Value2 && r.Value2 <= second.Value2)
                middleInside++;
            else
                middleOutside++;
        }

        // Chunk hypothesis: the first two records are a 32x32 chunk's world bounds and the middle
        // records are chunk-local linear cell indices (0..1023) with a zero second word.
        if (second.Value1 - first.Value1 == 31 && second.Value2 - first.Value2 == 31)
            bounds32++;
        if (first.Value1 % 32 == 0 && first.Value2 % 32 == 0)
            aligned32++;
        for (int i = 2; i < run.Count; i++)
        {
            var r = run[i];
            if (r.Value1 < 1024) middleInChunk++; else middleOutChunk++;
            if (r.Value2 == 0) middleZeroV2++; else middleNonZeroV2++;
        }

        // Range hypothesis: first = (MSUR start, 0), second = (MSUR end, width), middle = MPRL indices.
        if (first.Value1 <= second.Value1 && second.Value1 < chunks.Msur.Count)
        {
            validMsurRange++;
            msurWidthSum += second.Value1 - first.Value1 + 1;
            if (second.Value1 - first.Value1 != second.Value2)
                msurOverlaps++;
        }
        bool allMprl = true, allZero = true;
        for (int i = 2; i < run.Count; i++)
        {
            if (run[i].Value1 >= chunks.Mprl.Count) allMprl = false;
            if (run[i].Value2 != 0) allZero = false;
        }
        if (allMprl) validMprlIndices++;
        if (allZero) zeroMiddleV2++;

        // Object-partition hypothesis: the middle records are MPRL indices and all share one
        // MPRL.Unk04 (or Unk00) group id.
        var unk04 = new HashSet<ushort>();
        var unk00 = new HashSet<ushort>();
        bool anyMprl = false;
        for (int i = 2; i < run.Count; i++)
        {
            if (run[i].Value1 >= chunks.Mprl.Count)
                continue;
            anyMprl = true;
            var point = chunks.Mprl[run[i].Value1];
            unk04.Add(point.Unk04);
            unk00.Add(point.Unk00);
        }
        if (anyMprl)
        {
            if (unk04.Count == 1) singleUnk04++; else multipleUnk04++;
            if (unk00.Count == 1) singleUnk00++; else multipleUnk00++;
        }
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
