using System.Numerics;
using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

/// <summary>
/// Works out what the unknown <c>MPRL</c> fields are, now that the points themselves are known to be
/// terrain contacts around object footprints.
/// </summary>
/// <remarks>
/// That context says what to look for. A set of points tracing footprints wants a key saying WHICH
/// footprint a point belongs to, so the first question for any index-like field here is whether its
/// groups are spatially coherent.
///
/// <para>The heading reading of <c>Unk04</c>, live in <c>Pm4ObjectPositionDecoder</c>, is tested the
/// cheap decisive way first: a 16-bit angle uses its whole range. If the observed maximum sits far
/// below 65535 then the field cannot be an angle scaled that way, whatever else it turns out to be.</para>
///
/// <para>Spatial coherence is mean pairwise distance inside a group, against a control of randomly
/// assembled groups of the same size from the same file. Every point in a file is already within a
/// tile of every other, so a raw distance says nothing without that floor under it.</para>
/// </remarks>
internal static class Pm4MprlValueSupport
{
    public static Pm4MprlValueReport Analyze(string inputDirectory, int maxFiles = 200)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        var unk00 = new FieldStats("MPRL.Unk00");
        var unk04 = new FieldStats("MPRL.Unk04");
        var unk14 = new FieldStats("MPRL.Unk14");
        var unk16 = new FieldStats("MPRL.Unk16");

        var byUnk14 = new Dictionary<uint, EnumStats>();
        var byUnk16 = new Dictionary<uint, EnumStats>();

        int files = 0;
        var rng = new Random(7);

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (files >= maxFiles)
                break;

            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (c.Mprl.Count < 8)
                continue;

            files++;

            var placed = new List<Vector3>(c.Mprl.Count);
            foreach (Pm4MprlEntry e in c.Mprl)
                placed.Add(Pm4CoordinateService.MprlToAdtPlacement(e.Position));

            var v00 = new List<uint>();
            var v04 = new List<uint>();
            for (int i = 0; i < c.Mprl.Count; i++)
            {
                Pm4MprlEntry e = c.Mprl[i];
                uint u14 = unchecked((uint)(ushort)e.Unk14);

                v00.Add(e.Unk00);
                v04.Add(e.Unk04);
                unk00.Observe(e.Unk00);
                unk04.Observe(e.Unk04);
                unk14.Observe(u14);
                unk16.Observe(e.Unk16);

                Accumulate(byUnk14, u14, placed[i]);
                Accumulate(byUnk16, e.Unk16, placed[i]);
            }

            unk00.MeasureGrouping(v00, placed, rng);
            unk04.MeasureGrouping(v04, placed, rng);
        }

        return new Pm4MprlValueReport(
            resolved, files,
            unk00.ToResult(), unk04.ToResult(), unk14.ToResult(), unk16.ToResult(),
            ToEnumResults(byUnk14), ToEnumResults(byUnk16));
    }

    private static void Accumulate(Dictionary<uint, EnumStats> map, uint key, Vector3 position)
    {
        if (!map.TryGetValue(key, out EnumStats? stats))
        {
            stats = new EnumStats();
            map[key] = stats;
        }

        stats.Count++;
        stats.HeightSum += position.Z;
        if (position.Z < stats.MinHeight) stats.MinHeight = position.Z;
        if (position.Z > stats.MaxHeight) stats.MaxHeight = position.Z;
    }

    private static IReadOnlyList<Pm4MprlEnumResult> ToEnumResults(Dictionary<uint, EnumStats> map) =>
        [.. map.OrderByDescending(static kv => kv.Value.Count)
            .Select(static kv => new Pm4MprlEnumResult(
                $"0x{kv.Key:X4}", kv.Value.Count,
                kv.Value.Count == 0 ? 0 : kv.Value.HeightSum / kv.Value.Count,
                kv.Value.MinHeight, kv.Value.MaxHeight))];

    private sealed class EnumStats
    {
        public long Count;
        public double HeightSum;
        public float MinHeight = float.MaxValue;
        public float MaxHeight = float.MinValue;
    }

    private sealed class FieldStats(string name)
    {
        private readonly Dictionary<int, long> _groupSizes = [];
        private readonly List<double> _withinGroup = [];
        private readonly List<double> _controlGroup = [];
        private long _distinctSum;
        private int _files;

        public string Name { get; } = name;

        public long Total;
        public uint Min = uint.MaxValue;
        public uint Max;

        public void Observe(uint v)
        {
            Total++;
            if (v < Min) Min = v;
            if (v > Max) Max = v;
        }

        /// <summary>
        /// Group-size distribution, plus whether a group is spatially tight compared with a random
        /// group of the same size drawn from the same file.
        /// </summary>
        public void MeasureGrouping(List<uint> values, List<Vector3> positions, Random rng)
        {
            var members = new Dictionary<uint, List<int>>();
            for (int i = 0; i < values.Count; i++)
            {
                if (!members.TryGetValue(values[i], out List<int>? list))
                {
                    list = [];
                    members[values[i]] = list;
                }

                list.Add(i);
            }

            _files++;
            _distinctSum += members.Count;

            foreach ((uint _, List<int> list) in members)
            {
                _groupSizes[list.Count] = _groupSizes.GetValueOrDefault(list.Count) + 1;
                if (list.Count is < 2 or > 64)
                    continue;

                _withinGroup.Add(MeanPairwise(positions, list));

                var control = new List<int>(list.Count);
                for (int k = 0; k < list.Count; k++)
                    control.Add(rng.Next(positions.Count));
                _controlGroup.Add(MeanPairwise(positions, control));
            }
        }

        private static double MeanPairwise(List<Vector3> positions, List<int> members)
        {
            double sum = 0;
            int n = 0;
            for (int a = 0; a < members.Count; a++)
            {
                for (int b = a + 1; b < members.Count; b++)
                {
                    sum += Vector3.Distance(positions[members[a]], positions[members[b]]);
                    n++;
                }
            }

            return n == 0 ? 0 : sum / n;
        }

        public Pm4MprlFieldResult ToResult()
        {
            IReadOnlyList<Pm4ValueFrequency> sizes = [.. _groupSizes
                .OrderByDescending(static kv => kv.Value).Take(8)
                .Select(static kv => new Pm4ValueFrequency(kv.Key.ToString(), (int)kv.Value))];

            return new Pm4MprlFieldResult(
                Name, Total, Min == uint.MaxValue ? 0 : Min, Max,
                _files == 0 ? 0 : (double)_distinctSum / _files,
                sizes,
                Median(_withinGroup), Median(_controlGroup));
        }

        private static double Median(List<double> v)
        {
            if (v.Count == 0)
                return 0;

            v.Sort();
            return v[v.Count / 2];
        }
    }
}

internal sealed record Pm4MprlFieldResult(
    string Name,
    long Total,
    uint Min,
    uint Max,
    double MeanGroupsPerFile,
    IReadOnlyList<Pm4ValueFrequency> GroupSizes,
    double MedianWithinGroupDistance,
    double MedianControlGroupDistance);

internal sealed record Pm4MprlEnumResult(
    string Value,
    long Count,
    double MeanHeight,
    float MinHeight,
    float MaxHeight);

internal sealed record Pm4MprlValueReport(
    string InputDirectory,
    int Files,
    Pm4MprlFieldResult Unk00,
    Pm4MprlFieldResult Unk04,
    Pm4MprlFieldResult Unk14,
    Pm4MprlFieldResult Unk16,
    IReadOnlyList<Pm4MprlEnumResult> ByUnk14,
    IReadOnlyList<Pm4MprlEnumResult> ByUnk16);
