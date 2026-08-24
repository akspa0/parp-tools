using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Dissects the out-of-range values in PM4's index relationships instead of merely counting them.
/// </summary>
/// <remarks>
/// Every bounds test in this codebase reports "fits N, misses M" and stops. That treats M as error.
/// The recurring pattern in these chunks says otherwise: <c>MPRR</c> interleaves an explicit sentinel,
/// <c>MSUR._0x1C</c> read as a key looked malformed until it was read as a float, and
/// <c>MSUR._0x18</c> looked like a partial index until it was read as a window that partitions
/// exactly. So the misses are worth characterising: a single repeated value is a sentinel, a tight
/// band just past the array end is a header or a reserved block, and a broad spread is genuine
/// disorder. Those three cases point at completely different readings.
/// </remarks>
public static class Pm4MissAnatomyAnalyzer
{
    public static Pm4MissAnatomyReport AnalyzeDirectory(string inputDirectory)
    {
        string resolved = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        var mslkRef = new MissAccumulator("MSLK.RefIndex -> MSUR");
        var msurMscn = new MissAccumulator("MSUR._0x18 (as a single index) -> MSCN");
        var mslkGroup = new MissAccumulator("MSLK.GroupObjectId -> MSUR");

        int files = 0;
        foreach (string path in Directory
            .EnumerateFiles(resolved, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            Pm4KnownChunkSet c = Pm4ResearchReader.ReadFile(path).KnownChunks;
            if (c.Msur.Count == 0)
                continue;

            files++;
            int msurCount = c.Msur.Count;
            int mscnCount = c.Mscn.Count;

            foreach (Pm4MslkEntry e in c.Mslk)
            {
                mslkRef.Observe(e.RefIndex, msurCount);
                mslkGroup.Observe(e.GroupObjectId, msurCount);
            }

            foreach (Pm4MsurEntry e in c.Msur)
                msurMscn.Observe(e._0x18, mscnCount);
        }

        return new Pm4MissAnatomyReport(
            resolved, files,
            mslkRef.ToResult(), msurMscn.ToResult(), mslkGroup.ToResult());
    }

    private sealed class MissAccumulator(string name)
    {
        private readonly Dictionary<uint, long> _missValues = [];
        private readonly Dictionary<long, long> _overflow = [];
        public string Name { get; } = name;
        public long Total;
        public long Fits;
        public long Misses;

        public void Observe(uint value, int bound)
        {
            Total++;
            if (value < bound)
            {
                Fits++;
                return;
            }

            Misses++;
            _missValues[value] = _missValues.GetValueOrDefault(value) + 1;

            // How far past the end does it land? A tight band means a header or reserved block
            // sitting just after the array; a broad spread means the value is not an index at all.
            long over = value - bound;
            long bucket = over < 16 ? over : over < 256 ? 100 + (over / 16) : 1000;
            _overflow[bucket] = _overflow.GetValueOrDefault(bucket) + 1;
        }

        public Pm4MissResult ToResult()
        {
            List<Pm4ValueFrequency> top = _missValues
                .OrderByDescending(static kv => kv.Value)
                .Take(8)
                .Select(static kv => new Pm4ValueFrequency($"0x{kv.Key:X} ({kv.Key})", checked((int)Math.Min(kv.Value, int.MaxValue))))
                .ToList();

            long sentinel16 = _missValues.GetValueOrDefault(0xFFFFu);
            long sentinel32 = _missValues.GetValueOrDefault(0xFFFFFFFFu);
            long withinOne = _overflow.Where(static kv => kv.Key is >= 0 and < 16).Sum(static kv => kv.Value);
            long far = _overflow.GetValueOrDefault(1000);

            return new Pm4MissResult(
                Name, Total, Fits, Misses,
                Total == 0 ? 0 : (double)Misses / Total,
                _missValues.Count,
                sentinel16, sentinel32,
                withinOne, Misses == 0 ? 0 : (double)withinOne / Misses,
                far, Misses == 0 ? 0 : (double)far / Misses,
                top);
        }
    }
}

public sealed record Pm4MissResult(
    string Name,
    long Total,
    long Fits,
    long Misses,
    double MissFraction,
    int DistinctMissValues,
    long Equals0xFFFF,
    long Equals0xFFFFFFFF,
    long WithinSixteenPastEnd,
    double WithinSixteenFraction,
    long FarPastEnd,
    double FarPastEndFraction,
    IReadOnlyList<Pm4ValueFrequency> TopMissValues);

public sealed record Pm4MissAnatomyReport(
    string InputDirectory,
    int Files,
    Pm4MissResult MslkRefIndex,
    Pm4MissResult MsurMscnSingleIndex,
    Pm4MissResult MslkGroupObjectId);
