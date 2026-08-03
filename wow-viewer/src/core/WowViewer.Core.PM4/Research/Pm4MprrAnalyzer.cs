using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

/// <summary>
/// Sweeps MPRR against every chunk domain and tests its sentinel-delimited run structure.
/// </summary>
/// <remarks>
/// The existing unknowns analyzer tests MPRR.Value1 against MPRL and MSVT only, and neither
/// explains it. It also skips sentinels rather than using them: <c>Value1 == 0xFFFF</c> is already
/// modelled as <see cref="Pm4MprrEntry.IsSentinel"/>, and a sentinel scattered through a large flat
/// array is the classic shape of a run-delimited list. If MPRR is a list of runs, a global bound
/// test is asking the wrong question — the target may be local to the run.
///
/// The most decisive cheap test is structural rather than numeric: if the run count equals another
/// chunk's entry count file after file, MPRR is a per-entry list for that chunk, and that identifies
/// the owner without decoding a single value.
/// </remarks>
public static class Pm4MprrAnalyzer
{
    private static readonly string[] DomainNames =
        ["MSLK", "MSPI", "MSPV", "MSVI", "MSVT", "MSUR", "MSCN", "MPRL", "MPRR"];

    public static Pm4MprrReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        int filesWithMprr = 0;
        long entries = 0, sentinels = 0, nonSentinel = 0;
        long runs = 0;

        Dictionary<string, long> value1Fits = DomainNames.ToDictionary(static n => n, static _ => 0L, StringComparer.Ordinal);
        Dictionary<string, long> value2Fits = DomainNames.ToDictionary(static n => n, static _ => 0L, StringComparer.Ordinal);
        Dictionary<string, int> runCountMatches = DomainNames.ToDictionary(static n => n, static _ => 0, StringComparer.Ordinal);
        Dictionary<int, int> runLengths = [];
        long runsWhereValue1FitsRunLength = 0;
        long runsWhereValue1FitsRunIndex = 0;

        foreach (string path in Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName))
        {
            Pm4KnownChunkSet chunks = Pm4ResearchReader.ReadFile(path).KnownChunks;
            IReadOnlyList<Pm4MprrEntry> mprr = chunks.Mprr;
            if (mprr.Count == 0)
                continue;

            filesWithMprr++;
            entries += mprr.Count;

            Dictionary<string, int> counts = new(StringComparer.Ordinal)
            {
                ["MSLK"] = chunks.Mslk.Count,
                ["MSPI"] = chunks.Mspi.Count,
                ["MSPV"] = chunks.Mspv.Count,
                ["MSVI"] = chunks.Msvi.Count,
                ["MSVT"] = chunks.Msvt.Count,
                ["MSUR"] = chunks.Msur.Count,
                ["MSCN"] = chunks.Mscn.Count,
                ["MPRL"] = chunks.Mprl.Count,
                ["MPRR"] = mprr.Count,
            };

            int fileRuns = 0;
            int runStart = 0;
            for (int i = 0; i < mprr.Count; i++)
            {
                if (mprr[i].IsSentinel)
                {
                    sentinels++;
                    int length = i - runStart;
                    AddCount(runLengths, length);
                    fileRuns++;

                    for (int j = runStart; j < i; j++)
                    {
                        if (mprr[j].Value1 < length)
                            runsWhereValue1FitsRunLength++;
                        if (mprr[j].Value1 < fileRuns)
                            runsWhereValue1FitsRunIndex++;
                    }

                    runStart = i + 1;
                    continue;
                }

                nonSentinel++;
                foreach (string domain in DomainNames)
                {
                    if (mprr[i].Value1 < counts[domain])
                        value1Fits[domain]++;
                    if (mprr[i].Value2 < counts[domain])
                        value2Fits[domain]++;
                }
            }

            runs += fileRuns;

            // The structural test: does the run count equal some chunk's entry count?
            foreach (string domain in DomainNames)
            {
                if (fileRuns == counts[domain])
                    runCountMatches[domain]++;
            }
        }

        IReadOnlyList<Pm4MprrDomainFit> value1 = DomainNames
            .Select(name => new Pm4MprrDomainFit(name, value1Fits[name], nonSentinel - value1Fits[name],
                nonSentinel == 0 ? 0d : (double)value1Fits[name] / nonSentinel))
            .OrderByDescending(static fit => fit.Fits)
            .ToList();

        IReadOnlyList<Pm4MprrDomainFit> value2 = DomainNames
            .Select(name => new Pm4MprrDomainFit(name, value2Fits[name], nonSentinel - value2Fits[name],
                nonSentinel == 0 ? 0d : (double)value2Fits[name] / nonSentinel))
            .OrderByDescending(static fit => fit.Fits)
            .ToList();

        IReadOnlyList<Pm4MprrRunCountMatch> structural = DomainNames
            .Select(name => new Pm4MprrRunCountMatch(name, runCountMatches[name], filesWithMprr,
                filesWithMprr == 0 ? 0d : (double)runCountMatches[name] / filesWithMprr))
            .OrderByDescending(static match => match.FilesMatching)
            .ToList();

        IReadOnlyList<Pm4ValueFrequency> lengths = runLengths
            .OrderByDescending(static kv => kv.Value)
            .ThenBy(static kv => kv.Key)
            .Take(16)
            .Select(static kv => new Pm4ValueFrequency(kv.Key.ToString(), kv.Value))
            .ToList();

        return new Pm4MprrReport(
            resolvedDirectory,
            filesWithMprr,
            entries,
            sentinels,
            nonSentinel,
            runs,
            nonSentinel == 0 ? 0d : (double)runsWhereValue1FitsRunLength / nonSentinel,
            nonSentinel == 0 ? 0d : (double)runsWhereValue1FitsRunIndex / nonSentinel,
            value1,
            value2,
            structural,
            lengths,
            [
                "Value1/Value2 fits are BOUND tests only: a value in range does not prove the domain owns it.",
                "The run-count match is structural — if the sentinel-delimited run count equals a chunk's entry count file after file, MPRR is a per-entry list for that chunk.",
                "Sentinel is Value1 == 0xFFFF, already modelled by Pm4MprrEntry.IsSentinel."
            ]);
    }

    private static void AddCount(Dictionary<int, int> counts, int key)
    {
        counts.TryGetValue(key, out int existing);
        counts[key] = existing + 1;
    }
}
