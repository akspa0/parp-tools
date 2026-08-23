using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Tests whether <c>MSUR</c> carries a SECOND running window — <c>(_0x18, AttributeMask)</c> into
/// <c>MSCN</c> — parallel to the known <c>(MsviFirstIndex, IndexCount)</c> window into <c>MSVI</c>.
/// </summary>
/// <remarks>
/// <para>
/// Motivation: <c>_0x18</c> is documented on wowdev as "index into MSCN" and was previously measured
/// as a single index, fitting 511,891 / 6,201 but reaching only 34.5% of MSCN points. A single index
/// per surface cannot reach a stream larger than the surface count, so "MSCN is not simply one
/// boundary vertex per surface" was the correct conclusion from the wrong model. The byte at +0x02
/// (locally <c>AttributeMask</c>) was never paired with it as a window length.
/// </para>
/// <para>
/// Detector power is established before any claim, per the rule that a measurement which cannot
/// separate the interpretations it tests is not evidence. Three chains are run over the same pairs:
/// a POSITIVE control (<c>MsviFirstIndex + IndexCount</c>, known-true, must fit near 100%), the
/// HYPOTHESIS (<c>_0x18 + AttributeMask</c>), and two NEGATIVE controls that reuse the hypothesis's
/// start field with a wrong length (<c>_0x18 + IndexCount</c>, <c>_0x18 + GroupKey</c>). If a
/// negative control fits as well as the hypothesis, the chain test discriminates nothing and the
/// result must be reported as uninformative rather than as support.
/// </para>
/// </remarks>
public static class Pm4MsurWindowAnalyzer
{
    public static Pm4MsurWindowReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4ResearchDocument> files = Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .ToList();

        return Analyze(resolvedDirectory, files);
    }

    public static Pm4MsurWindowReport AnalyzeFile(string path)
        => Analyze(path, [Pm4ResearchReader.ReadFile(path)]);

    /// <summary>
    /// Analyzes already-read documents. Public so the chain test can be exercised on constructed
    /// cases — a fabricated document with a known-good and a known-broken chain must produce a
    /// 100% and a 0% fit respectively before any corpus number is quoted.
    /// </summary>
    public static Pm4MsurWindowReport Analyze(string inputDirectory, IReadOnlyList<Pm4ResearchDocument> files)
    {
        var msvi = new ChainAccumulator("MsviFirstIndex + IndexCount -> MSVI (positive control)");
        var mslk = new ChainAccumulator("_0x18 + AttributeMask -> MSLK (hypothesis)");
        var mscn = new ChainAccumulator("_0x18 + AttributeMask -> MSCN (prior reading's target)");
        var wrongLenIndexCount = new ChainAccumulator("_0x18 + IndexCount -> MSLK (negative control)");
        var wrongLenGroupKey = new ChainAccumulator("_0x18 + GroupKey -> MSLK (negative control)");

        List<Pm4MsurWindowFile> fileReports = [];
        int nonEmptyFileCount = 0;
        long roundTripEntries = 0;
        long roundTripFits = 0;
        long roundTripFilesExact = 0;
        long adjacencyEdges = 0;
        long adjacencyReciprocated = 0;
        long adjacencySelfEdges = 0;
        long adjacencyEdgesWithWallGeometry = 0;

        foreach (Pm4ResearchDocument document in files)
        {
            IReadOnlyList<Pm4MsurEntry> surfaces = document.KnownChunks.Msur;
            if (surfaces.Count == 0)
                continue;

            nonEmptyFileCount++;

            int mslkCount = document.KnownChunks.Mslk.Count;
            int mscnCount = document.KnownChunks.Mscn.Count;
            int msviCount = document.KnownChunks.Msvi.Count;

            var fileMslk = new ChainAccumulator(mslk.Name);
            var fileMsvi = new ChainAccumulator(msvi.Name);

            AccumulateChain(surfaces, msviCount, static s => s.MsviFirstIndex, static s => s.IndexCount, msvi, fileMsvi);
            AccumulateChain(surfaces, mslkCount, static s => s._0x18, static s => s.AttributeMask, mslk, fileMslk);
            AccumulateChain(surfaces, mscnCount, static s => s._0x18, static s => s.AttributeMask, mscn, null);
            AccumulateChain(surfaces, mslkCount, static s => s._0x18, static s => s.IndexCount, wrongLenIndexCount, null);
            AccumulateChain(surfaces, mslkCount, static s => s._0x18, static s => s.GroupKey, wrongLenGroupKey, null);

            (long entries, long fits) = MeasureRoundTrip(surfaces, document.KnownChunks.Mslk);
            roundTripEntries += entries;
            roundTripFits += fits;
            if (entries > 0 && entries == fits)
                roundTripFilesExact++;

            (long edges, long reciprocated, long selfEdges, long withWall) =
                MeasureAdjacencySymmetry(surfaces, document.KnownChunks.Mslk);
            adjacencyEdges += edges;
            adjacencyReciprocated += reciprocated;
            adjacencySelfEdges += selfEdges;
            adjacencyEdgesWithWallGeometry += withWall;

            fileReports.Add(new Pm4MsurWindowFile(
                SourcePath: document.SourcePath ?? "(memory)",
                SurfaceCount: surfaces.Count,
                MslkCount: mslkCount,
                MscnCount: mscnCount,
                MsviCount: msviCount,
                MslkChainFits: fileMslk.Fits,
                MslkChainPairs: fileMslk.Pairs,
                MslkSumOfCounts: fileMslk.SumOfCounts,
                MslkMaxWindowEnd: fileMslk.MaxWindowEnd,
                MslkCoverage: fileMslk.Coverage(mslkCount),
                RoundTripEntries: entries,
                RoundTripFits: fits,
                MsviChainFits: fileMsvi.Fits,
                MsviChainPairs: fileMsvi.Pairs));
        }

        return new Pm4MsurWindowReport(
            InputDirectory: inputDirectory,
            FileCount: files.Count,
            NonEmptyFileCount: nonEmptyFileCount,
            MsviWindow: msvi.ToResult(),
            MslkWindow: mslk.ToResult(),
            MscnWindow: mscn.ToResult(),
            WrongLengthIndexCount: wrongLenIndexCount.ToResult(),
            WrongLengthGroupKey: wrongLenGroupKey.ToResult(),
            RoundTripEntries: roundTripEntries,
            RoundTripFits: roundTripFits,
            RoundTripFilesExact: roundTripFilesExact,
            AdjacencyEdges: adjacencyEdges,
            AdjacencyReciprocated: adjacencyReciprocated,
            AdjacencySelfEdges: adjacencySelfEdges,
            AdjacencyEdgesWithWallGeometry: adjacencyEdgesWithWallGeometry,
            Files: fileReports);
    }

    /// <summary>
    /// Tests the adjacency reading: if MSUR[i]'s MSLK run lists i's neighbours by
    /// <see cref="Pm4MslkEntry.RefIndex"/>, then <c>j in neighbours(i)</c> implies
    /// <c>i in neighbours(j)</c>. Reciprocity is what separates an adjacency graph from an
    /// owner back-pointer or an arbitrary index; a directed or accidental relation does not
    /// reciprocate. Also counts how many edges carry MSPV/MSPI wall geometry, which is the
    /// open-portal versus blocked-edge split.
    /// </summary>
    private static (long Edges, long Reciprocated, long SelfEdges, long WithWallGeometry) MeasureAdjacencySymmetry(
        IReadOnlyList<Pm4MsurEntry> surfaces,
        IReadOnlyList<Pm4MslkEntry> links)
    {
        var neighbours = new HashSet<long>[surfaces.Count];
        var hasWall = new List<bool>[surfaces.Count];

        for (int i = 0; i < surfaces.Count; i++)
        {
            neighbours[i] = [];
            hasWall[i] = [];

            Pm4MsurEntry surface = surfaces[i];
            long windowStart = surface._0x18;
            long windowEnd = windowStart + surface.AttributeMask;
            if (windowEnd > links.Count)
                continue;

            for (long j = windowStart; j < windowEnd; j++)
            {
                neighbours[i].Add(links[(int)j].RefIndex);
                hasWall[i].Add(links[(int)j].MspiFirstIndex >= 0 && links[(int)j].MspiIndexCount > 0);
            }
        }

        long edges = 0;
        long reciprocated = 0;
        long selfEdges = 0;
        long withWall = 0;

        for (int i = 0; i < surfaces.Count; i++)
        {
            foreach (long neighbour in neighbours[i])
            {
                edges++;
                if (neighbour == i)
                {
                    selfEdges++;
                    continue;
                }

                if (neighbour >= 0 && neighbour < surfaces.Count && neighbours[(int)neighbour].Contains(i))
                    reciprocated++;
            }

            foreach (bool wall in hasWall[i])
            {
                if (wall)
                    withWall++;
            }
        }

        return (edges, reciprocated, selfEdges, withWall);
    }

    /// <summary>
    /// The decisive bidirectional test. If surface <c>i</c> owns the MSLK run
    /// <c>[_0x18, _0x18 + AttributeMask)</c>, then every MSLK entry in that run must name <c>i</c>
    /// back through its own <c>RefIndex</c>. A forward chain that merely partitions the stream can
    /// arise from any monotone running sum; agreement with an independently stored back-pointer
    /// cannot.
    /// </summary>
    private static (long Entries, long Fits) MeasureRoundTrip(
        IReadOnlyList<Pm4MsurEntry> surfaces,
        IReadOnlyList<Pm4MslkEntry> links)
    {
        long entries = 0;
        long fits = 0;

        for (int i = 0; i < surfaces.Count; i++)
        {
            Pm4MsurEntry surface = surfaces[i];
            long windowStart = surface._0x18;
            long windowEnd = windowStart + surface.AttributeMask;
            if (windowEnd > links.Count)
                continue;

            for (long j = windowStart; j < windowEnd; j++)
            {
                entries++;
                if (links[(int)j].RefIndex == i)
                    fits++;
            }
        }

        return (entries, fits);
    }

    /// <summary>
    /// Describes what an MSUR window's MSLK entries actually name, so a failed round trip is
    /// diagnosed rather than merely reported. An exactly-zero round trip is systematic, not random:
    /// over 12,820 entries against 4,110 surfaces, chance alone would land roughly three hits.
    /// </summary>
    public static Pm4MsurWindowContentReport DescribeWindowContents(string path, int sampleWindows = 6)
    {
        Pm4ResearchDocument document = Pm4ResearchReader.ReadFile(path);
        IReadOnlyList<Pm4MsurEntry> surfaces = document.KnownChunks.Msur;
        IReadOnlyList<Pm4MslkEntry> links = document.KnownChunks.Mslk;

        long windows = 0;
        long constantRefWindows = 0;
        long constantGroupWindows = 0;
        var deltaHistogram = new Dictionary<long, long>();
        var samples = new List<string>();

        for (int i = 0; i < surfaces.Count; i++)
        {
            Pm4MsurEntry surface = surfaces[i];
            long windowStart = surface._0x18;
            int windowLength = surface.AttributeMask;
            long windowEnd = windowStart + windowLength;
            if (windowLength == 0 || windowEnd > links.Count)
                continue;

            windows++;

            ushort firstRef = links[(int)windowStart].RefIndex;
            uint firstGroup = links[(int)windowStart].GroupObjectId;
            bool constantRef = true;
            bool constantGroup = true;

            for (long j = windowStart; j < windowEnd; j++)
            {
                if (links[(int)j].RefIndex != firstRef)
                    constantRef = false;
                if (links[(int)j].GroupObjectId != firstGroup)
                    constantGroup = false;

                long delta = links[(int)j].RefIndex - i;
                deltaHistogram[delta] = deltaHistogram.GetValueOrDefault(delta) + 1;
            }

            if (constantRef)
                constantRefWindows++;
            if (constantGroup)
                constantGroupWindows++;

            if (samples.Count < sampleWindows)
            {
                IEnumerable<string> refs = Enumerable
                    .Range((int)windowStart, windowLength)
                    .Select(j => $"{links[j].RefIndex}/g{links[j].GroupObjectId}/t0x{links[j].TypeFlags:X2}");
                samples.Add($"  MSUR[{i}] window=[{windowStart}..{windowEnd}) -> {string.Join(", ", refs)}");
            }
        }

        List<Pm4MsurWindowDelta> topDeltas = deltaHistogram
            .OrderByDescending(static kvp => kvp.Value)
            .Take(8)
            .Select(static kvp => new Pm4MsurWindowDelta(kvp.Key, kvp.Value))
            .ToList();

        return new Pm4MsurWindowContentReport(
            path,
            surfaces.Count,
            links.Count,
            windows,
            constantRefWindows,
            constantGroupWindows,
            topDeltas,
            samples);
    }

    /// <summary>
    /// Walks consecutive MSUR pairs testing <c>start[n] + count[n] == start[n + 1]</c>, and marks
    /// the covered span of the target stream so window coverage is measured rather than assumed.
    /// A break is classified by whether the two surfaces belong to the same CK24 group, which
    /// separates "the chain is wrong" from "the chain restarts per object".
    /// </summary>
    private static void AccumulateChain(
        IReadOnlyList<Pm4MsurEntry> surfaces,
        int streamCount,
        Func<Pm4MsurEntry, uint> start,
        Func<Pm4MsurEntry, uint> length,
        ChainAccumulator total,
        ChainAccumulator? perFile)
    {
        bool[] covered = streamCount > 0 ? new bool[streamCount] : [];

        for (int i = 0; i < surfaces.Count; i++)
        {
            Pm4MsurEntry surface = surfaces[i];
            uint windowStart = start(surface);
            uint windowLength = length(surface);
            long windowEnd = (long)windowStart + windowLength;

            total.SumOfCounts += windowLength;
            perFile?.AddCount(windowLength);

            if (windowEnd > total.MaxWindowEnd)
                total.MaxWindowEnd = windowEnd;
            perFile?.RaiseMaxWindowEnd(windowEnd);

            if (windowEnd > streamCount)
            {
                total.OutOfRangeWindows++;
                perFile?.AddOutOfRange();
            }
            else
            {
                for (long c = windowStart; c < windowEnd; c++)
                    covered[c] = true;
            }

            if (i + 1 >= surfaces.Count)
                continue;

            Pm4MsurEntry next = surfaces[i + 1];
            total.Pairs++;
            perFile?.AddPair();

            if (windowEnd == start(next))
            {
                total.Fits++;
                perFile?.AddFit();
            }
            else if (surface.Ck24 == next.Ck24)
            {
                total.MissesWithinCk24++;
            }
            else
            {
                total.MissesAtCk24Boundary++;
            }
        }

        int coveredCount = 0;
        foreach (bool c in covered)
        {
            if (c)
                coveredCount++;
        }

        total.StreamCount += streamCount;
        total.CoveredCount += coveredCount;
        perFile?.AddCoverage(coveredCount);
    }

    private sealed class ChainAccumulator(string name)
    {
        public string Name { get; } = name;
        public long Pairs;
        public long Fits;
        public long MissesWithinCk24;
        public long MissesAtCk24Boundary;
        public long SumOfCounts;
        public long MaxWindowEnd;
        public long OutOfRangeWindows;
        public long StreamCount;
        public long CoveredCount;

        public void AddPair() => Pairs++;
        public void AddFit() => Fits++;
        public void AddCount(uint c) => SumOfCounts += c;
        public void AddOutOfRange() => OutOfRangeWindows++;
        public void AddCoverage(int c) => CoveredCount += c;
        public void RaiseMaxWindowEnd(long end)
        {
            if (end > MaxWindowEnd)
                MaxWindowEnd = end;
        }

        public double Coverage(int streamCount) => streamCount > 0 ? (double)CoveredCount / streamCount : 0d;

        public Pm4RunningWindowResult ToResult() => new(
            Name,
            Pairs,
            Fits,
            MissesWithinCk24,
            MissesAtCk24Boundary,
            Pairs > 0 ? (double)Fits / Pairs : 0d,
            SumOfCounts,
            MaxWindowEnd,
            OutOfRangeWindows,
            StreamCount,
            CoveredCount,
            StreamCount > 0 ? (double)CoveredCount / StreamCount : 0d);
    }
}

public sealed record Pm4RunningWindowResult(
    string Name,
    long Pairs,
    long Fits,
    long MissesWithinCk24,
    long MissesAtCk24Boundary,
    double FitFraction,
    long SumOfCounts,
    long MaxWindowEnd,
    long OutOfRangeWindows,
    long StreamCount,
    long CoveredCount,
    double StreamCoverage);

public sealed record Pm4MsurWindowFile(
    string SourcePath,
    int SurfaceCount,
    int MslkCount,
    int MscnCount,
    int MsviCount,
    long MslkChainFits,
    long MslkChainPairs,
    long MslkSumOfCounts,
    long MslkMaxWindowEnd,
    double MslkCoverage,
    long RoundTripEntries,
    long RoundTripFits,
    long MsviChainFits,
    long MsviChainPairs);

public sealed record Pm4MsurWindowReport(
    string InputDirectory,
    int FileCount,
    int NonEmptyFileCount,
    Pm4RunningWindowResult MsviWindow,
    Pm4RunningWindowResult MslkWindow,
    Pm4RunningWindowResult MscnWindow,
    Pm4RunningWindowResult WrongLengthIndexCount,
    Pm4RunningWindowResult WrongLengthGroupKey,
    long RoundTripEntries,
    long RoundTripFits,
    long RoundTripFilesExact,
    long AdjacencyEdges,
    long AdjacencyReciprocated,
    long AdjacencySelfEdges,
    long AdjacencyEdgesWithWallGeometry,
    IReadOnlyList<Pm4MsurWindowFile> Files);

public sealed record Pm4MsurWindowDelta(long Delta, long Count);

public sealed record Pm4MsurWindowContentReport(
    string SourcePath,
    int SurfaceCount,
    int MslkCount,
    long Windows,
    long ConstantRefIndexWindows,
    long ConstantGroupObjectIdWindows,
    IReadOnlyList<Pm4MsurWindowDelta> TopRefIndexDeltas,
    IReadOnlyList<string> SampleWindows);
