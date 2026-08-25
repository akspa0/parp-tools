using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Characterises what <c>MSLK</c>'s enumerated values actually select, by measuring the geometry each
/// value carries rather than by reading its name.
/// </summary>
/// <remarks>
/// <c>_0x00</c> has 10 values and <c>_0x01</c> has 19, and neither has ever been tied to anything
/// observable. The method that worked on <c>MSUR._0x00</c> works here: a value that selects a KIND of
/// thing separates on properties the field itself knows nothing about.
///
/// <para>Each record either carries a wall window into <c>MSPI</c>/<c>MSPV</c> (<c>MspiFirstIndex</c>
/// non-negative) or does not. So for every value this measures how often it carries geometry, how long
/// the window is, and what that geometry looks like - vertical extent, and whether the face is
/// Z-dominant, which separates a floor from a wall without the field being consulted.</para>
///
/// <para>It also settles what a <c>_0x04</c> PAIR is. Pairs share <c>TypeFlags</c> 99.82% of the time
/// and do not reference each other, so the live hypothesis is that one member carries the wall and the
/// other an anchor. That is a two-by-two the data can answer directly.</para>
/// </remarks>
internal static class Pm4MslkValueSupport
{
    public static Pm4MslkValueReport Analyze(string inputDirectory, int maxFiles = 200)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        var byType = new Dictionary<byte, ValueAccumulator>();
        var bySubtype = new Dictionary<byte, ValueAccumulator>();
        int files = 0;

        long pairs = 0, pairBothWall = 0, pairBothAnchor = 0, pairOneEach = 0;
        long pairSameSubtype = 0;

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (files >= maxFiles)
                break;

            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (c.Mslk.Count == 0)
                continue;

            files++;

            for (int i = 0; i < c.Mslk.Count; i++)
            {
                Pm4MslkEntry e = c.Mslk[i];
                Observe(byType, e.TypeFlags, e, c);
                Observe(bySubtype, e.Subtype, e, c);
            }

            // What is a _0x04 pair?
            var members = new Dictionary<uint, List<int>>();
            for (int i = 0; i < c.Mslk.Count; i++)
            {
                if (!members.TryGetValue(c.Mslk[i].GroupObjectId, out List<int>? list))
                {
                    list = [];
                    members[c.Mslk[i].GroupObjectId] = list;
                }

                list.Add(i);
            }

            foreach ((uint _, List<int> list) in members)
            {
                if (list.Count != 2)
                    continue;

                pairs++;
                bool aWall = c.Mslk[list[0]].MspiFirstIndex >= 0;
                bool bWall = c.Mslk[list[1]].MspiFirstIndex >= 0;
                if (aWall && bWall) pairBothWall++;
                else if (!aWall && !bWall) pairBothAnchor++;
                else pairOneEach++;

                if (c.Mslk[list[0]].Subtype == c.Mslk[list[1]].Subtype)
                    pairSameSubtype++;
            }
        }

        return new Pm4MslkValueReport(
            resolved, files,
            [.. byType.OrderByDescending(static kv => kv.Value.Records).Select(static kv => kv.Value.ToResult($"0x{kv.Key:X2}"))],
            [.. bySubtype.OrderByDescending(static kv => kv.Value.Records).Select(static kv => kv.Value.ToResult($"0x{kv.Key:X2}"))],
            pairs,
            pairs == 0 ? 0 : (double)pairBothWall / pairs,
            pairs == 0 ? 0 : (double)pairBothAnchor / pairs,
            pairs == 0 ? 0 : (double)pairOneEach / pairs,
            pairs == 0 ? 0 : (double)pairSameSubtype / pairs);
    }

    private static void Observe(Dictionary<byte, ValueAccumulator> map, byte key, Pm4MslkEntry e, Pm4KnownChunkSet c)
    {
        if (!map.TryGetValue(key, out ValueAccumulator? acc))
        {
            acc = new ValueAccumulator();
            map[key] = acc;
        }

        acc.Records++;
        acc.SubtypeValues.Add(e.Subtype);
        acc.TypeValues.Add(e.TypeFlags);

        if (e.MspiFirstIndex < 0)
        {
            acc.Anchors++;
            return;
        }

        acc.WithWindow++;
        acc.WindowLengthSum += e.MspiIndexCount;

        long start = e.MspiFirstIndex;
        long end = start + e.MspiIndexCount;
        if (e.MspiIndexCount == 0 || end > c.Mspi.Count)
            return;

        Vector3 lo = new(float.MaxValue), hi = new(float.MinValue);
        int n = 0;
        for (long k = start; k < end; k++)
        {
            uint vi = c.Mspi[(int)k];
            if (vi >= c.Mspv.Count)
                continue;

            Vector3 p = c.Mspv[(int)vi];
            lo = Vector3.Min(lo, p);
            hi = Vector3.Max(hi, p);
            n++;
        }

        if (n < 3)
            return;

        acc.GeometryMeasured++;
        acc.ZExtentSum += hi.Z - lo.Z;

        // A face whose footprint is much wider than it is tall reads as a floor; the reverse reads as
        // a wall. Measured from the points, so the field being characterised never informs it.
        float horizontal = MathF.Max(hi.X - lo.X, hi.Y - lo.Y);
        float vertical = hi.Z - lo.Z;
        if (vertical > horizontal)
            acc.TallerThanWide++;
    }

    private sealed class ValueAccumulator
    {
        public long Records;
        public long WithWindow;
        public long Anchors;
        public long WindowLengthSum;
        public long GeometryMeasured;
        public double ZExtentSum;
        public long TallerThanWide;
        public readonly HashSet<byte> SubtypeValues = [];
        public readonly HashSet<byte> TypeValues = [];

        public Pm4MslkValueResult ToResult(string name) => new(
            name,
            Records,
            Records == 0 ? 0 : (double)WithWindow / Records,
            WithWindow == 0 ? 0 : (double)WindowLengthSum / WithWindow,
            GeometryMeasured == 0 ? 0 : ZExtentSum / GeometryMeasured,
            GeometryMeasured == 0 ? 0 : (double)TallerThanWide / GeometryMeasured,
            SubtypeValues.Count,
            TypeValues.Count);
    }
}

internal sealed record Pm4MslkValueResult(
    string Value,
    long Records,
    double CarriesWindowFraction,
    double MeanWindowLength,
    double MeanZExtent,
    double TallerThanWideFraction,
    int DistinctSubtypes,
    int DistinctTypeFlags);

internal sealed record Pm4MslkValueReport(
    string InputDirectory,
    int Files,
    IReadOnlyList<Pm4MslkValueResult> TypeFlags,
    IReadOnlyList<Pm4MslkValueResult> Subtypes,
    long Pairs,
    double PairBothWallFraction,
    double PairBothAnchorFraction,
    double PairOneEachFraction,
    double PairSameSubtypeFraction);
