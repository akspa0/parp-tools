using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Characterises every PM4 record field by BEHAVIOUR rather than by the name it carries.
/// </summary>
/// <remarks>
/// Every field this project got wrong was adopted from a name and never tested: "AttributeMask" was
/// a window length, "GroupObjectId" was near-unique per link, and the "CK24" key was a float. Each
/// took a separate accident to catch. This sweep asks the same questions of every field at once.
///
/// <para><b>Detector power comes first.</b> The sweep is only trustworthy on unknown fields if it
/// re-derives the settled ones unaided, so it is run over those too: <c>MSUR._0x02</c> must show as a
/// small enumerated count, <c>MSUR._0x1C</c> as float-like, and <c>MSLK.GroupObjectId</c> as
/// near-unique. If those three do not come out right, nothing else in the output is worth
/// reading.</para>
/// </remarks>
public static class Pm4FieldSweepAnalyzer
{
    public static Pm4FieldSweepReport AnalyzeDirectory(string inputDirectory, int maxFiles = 120)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);
        var fields = new Dictionary<string, FieldAccumulator>(StringComparer.Ordinal);
        int files = 0;
        long msurCount = 0, mslkCount = 0, mscnCount = 0, msvtCount = 0, mspvCount = 0, mprlCount = 0;

        FieldAccumulator Get(string name)
        {
            if (!fields.TryGetValue(name, out FieldAccumulator? a))
            {
                a = new FieldAccumulator(name);
                fields[name] = a;
            }
            return a;
        }

        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            if (files >= maxFiles)
                break;

            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (c.Msur.Count == 0)
                continue;

            files++;
            msurCount += c.Msur.Count; mslkCount += c.Mslk.Count; mscnCount += c.Mscn.Count;
            msvtCount += c.Msvt.Count; mspvCount += c.Mspv.Count; mprlCount += c.Mprl.Count;

            var perFile = new Dictionary<string, HashSet<uint>>(StringComparer.Ordinal);
            void File(string n, uint v)
            {
                if (!perFile.TryGetValue(n, out HashSet<uint>? set))
                {
                    set = [];
                    perFile[n] = set;
                }
                set.Add(v);
            }

            foreach (Pm4MsurEntry e in c.Msur)
            {
                // Controls: three fields already settled, included so the sweep proves it can see them.
                Get("MSUR._0x00 surface class").Observe(e.GroupKey); File("MSUR._0x00 surface class", e.GroupKey);
                Get("MSUR._0x02 [CONTROL: window len]").Observe(e.AttributeMask); File("MSUR._0x02 [CONTROL: window len]", e.AttributeMask);
                Get("MSUR._0x1C [CONTROL: float]").Observe(e.PackedParams); File("MSUR._0x1C [CONTROL: float]", e.PackedParams);
                Get("MSUR._0x03 padding").Observe(e.Padding); File("MSUR._0x03 padding", e.Padding);
            }

            foreach (Pm4MslkEntry e in c.Mslk)
            {
                Get("MSLK._0x00 TypeFlags").Observe(e.TypeFlags); File("MSLK._0x00 TypeFlags", e.TypeFlags);
                Get("MSLK._0x01 Subtype").Observe(e.Subtype); File("MSLK._0x01 Subtype", e.Subtype);
                Get("MSLK._0x02 Padding").Observe(e.Padding); File("MSLK._0x02 Padding", e.Padding);
                Get("MSLK._0x04 [CONTROL: near-unique]").Observe(e.GroupObjectId); File("MSLK._0x04 [CONTROL: near-unique]", e.GroupObjectId);
                Get("MSLK.LinkId").Observe(e.LinkId); File("MSLK.LinkId", e.LinkId);
                Get("MSLK.SystemFlag").Observe(e.SystemFlag); File("MSLK.SystemFlag", e.SystemFlag);
            }

            foreach (Pm4MprlEntry e in c.Mprl)
            {
                Get("MPRL.Unk00").Observe(e.Unk00); File("MPRL.Unk00", e.Unk00);
                Get("MPRL.Unk02").Observe(unchecked((uint)(ushort)e.Unk02)); File("MPRL.Unk02", unchecked((uint)(ushort)e.Unk02));
                Get("MPRL.Unk04").Observe(e.Unk04); File("MPRL.Unk04", e.Unk04);
                Get("MPRL.Unk06").Observe(e.Unk06); File("MPRL.Unk06", e.Unk06);
                Get("MPRL.Unk14").Observe(unchecked((uint)(ushort)e.Unk14)); File("MPRL.Unk14", unchecked((uint)(ushort)e.Unk14));
                Get("MPRL.Unk16").Observe(e.Unk16); File("MPRL.Unk16", e.Unk16);
            }

            // Per-file uniqueness is the meaningful measure: an id that is unique WITHIN a file
            // repeats across files, so a corpus-wide distinct ratio hides exactly the property being
            // tested. This is the error that let MSLK._0x04 be read as an index.
            int recordsThisFile;
            foreach ((string name, HashSet<uint> set) in perFile)
            {
                recordsThisFile = name.StartsWith("MSUR", StringComparison.Ordinal) ? c.Msur.Count
                    : name.StartsWith("MSLK", StringComparison.Ordinal) ? c.Mslk.Count
                    : c.Mprl.Count;
                if (recordsThisFile > 0)
                    Get(name).AddPerFileRatio((double)set.Count / recordsThisFile);
            }
        }

        var counts = new Dictionary<string, long>(StringComparer.Ordinal)
        {
            ["MSUR"] = msurCount, ["MSLK"] = mslkCount, ["MSCN"] = mscnCount,
            ["MSVT"] = msvtCount, ["MSPV"] = mspvCount, ["MPRL"] = mprlCount,
        };

        List<Pm4FieldSweepResult> results = fields.Values
            .OrderBy(static a => a.Name, StringComparer.Ordinal)
            .Select(a => a.ToResult(counts))
            .ToList();

        return new Pm4FieldSweepReport(resolved, files, results);
    }

    private sealed class FieldAccumulator(string name)
    {
        private readonly HashSet<uint> _distinct = [];
        private bool _distinctCapped;
        public string Name { get; } = name;
        public long Total;
        public uint Min = uint.MaxValue;
        public uint Max;
        public long Zero;
        private readonly Dictionary<byte, long> _highByte = [];
        private readonly List<double> _perFileRatios = [];

        public void AddPerFileRatio(double r) => _perFileRatios.Add(r);

        public void Observe(uint v)
        {
            Total++;
            if (v == 0) Zero++;
            if (v < Min) Min = v;
            if (v > Max) Max = v;
            if (!_distinctCapped)
            {
                _distinct.Add(v);
                if (_distinct.Count > 200_000)
                    _distinctCapped = true;
            }
            _highByte[(byte)(v >> 24)] = _highByte.GetValueOrDefault((byte)(v >> 24)) + 1;
        }

        public Pm4FieldSweepResult ToResult(Dictionary<string, long> chunkCounts)
        {
            // Mean per-file distinct ratio, not the corpus-wide one.
            double distinctRatio = _perFileRatios.Count == 0 ? 0 : _perFileRatios.Average();

            // Float-like: a 32-bit float population concentrates its high byte in exponent bands and
            // their sign-set mirrors. A count or an index spreads its high byte over almost nothing
            // (small values) or uniformly (hashes).
            int bands = _highByte.Count(static kv => kv.Value > 0);
            bool highByteAlwaysZero = _highByte.Count == 1 && _highByte.ContainsKey(0);
            long signSet = _highByte.Where(static kv => kv.Key >= 0xB0 && kv.Key <= 0xC7).Sum(static kv => kv.Value);
            long expBand = _highByte.Where(static kv => kv.Key >= 0x38 && kv.Key <= 0x47).Sum(static kv => kv.Value);

            // Measured over NON-ZERO values. A float field that uses 0.0 as "absent" would otherwise
            // be dragged below the threshold by its own sentinel - which is what hid MSUR._0x1C,
            // where 27% of values are the zero that means "no placement height".
            long nonZero = Total - Zero;
            double floatLike = nonZero <= 0 ? 0 : (double)(signSet + expBand) / nonZero;

            // Index-like: every value fits inside some chunk's entry count.
            string indexFits = string.Join(",", chunkCounts
                .Where(kv => kv.Value > 0 && Max < kv.Value)
                .Select(static kv => kv.Key));

            string shape =
                Total == 0 ? "empty"
                : Min == Max ? $"CONSTANT {Min}"
                : floatLike > 0.90 && !highByteAlwaysZero ? "FLOAT-like"
                : distinctRatio > 0.5 ? "NEAR-UNIQUE (per-record id)"
                : _distinct.Count <= 32 ? $"ENUM ({_distinct.Count} values)"
                : indexFits.Length > 0 ? "index-like"
                : "spread";

            return new Pm4FieldSweepResult(
                Name, Total, _distinctCapped ? -1 : _distinct.Count, distinctRatio,
                Min, Max, Total == 0 ? 0 : (double)Zero / Total,
                floatLike, indexFits, shape);
        }
    }
}

public sealed record Pm4FieldSweepResult(
    string Name,
    long Total,
    int DistinctValues,
    double DistinctRatio,
    uint Min,
    uint Max,
    double ZeroFraction,
    double FloatLikeFraction,
    string IndexFitsChunks,
    string Shape);

public sealed record Pm4FieldSweepReport(
    string InputDirectory,
    int Files,
    IReadOnlyList<Pm4FieldSweepResult> Fields);
