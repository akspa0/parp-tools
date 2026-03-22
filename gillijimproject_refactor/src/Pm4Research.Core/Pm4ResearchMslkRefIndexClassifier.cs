namespace Pm4Research.Core;

public static class Pm4ResearchMslkRefIndexClassifier
{
    private static readonly string[] CandidateDomains =
    {
        "MSLK",
        "MSPI",
        "MSVI",
        "MSCN",
        "MPRL",
        "MSPV",
        "MSVT",
        "MPRR"
    };

    public static Pm4RefIndexClassifierReport AnalyzeDirectory(string inputDirectory)
    {
        List<Pm4ResearchFile> files = Directory
            .EnumerateFiles(inputDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .ToList();

        int filesWithMismatches = 0;
        int totalMismatchCount = 0;
        Dictionary<string, int> baselineFitCounts = CandidateDomains.ToDictionary(static domain => domain, static _ => 0, StringComparer.Ordinal);
        Dictionary<string, FamilyAccumulator> families = new(StringComparer.Ordinal);

        foreach (Pm4ResearchFile file in files)
        {
            bool fileHasMismatch = false;
            int msurCount = file.KnownChunks.Msur.Count;

            foreach (Pm4MslkEntry link in file.KnownChunks.Mslk)
            {
                if (link.RefIndex < msurCount)
                    continue;

                fileHasMismatch = true;
                totalMismatchCount++;
                IReadOnlyList<string> fits = GetFitDomains(file, link.RefIndex);
                foreach (string domain in fits)
                    baselineFitCounts[domain]++;

                string familyKey = BuildFamilyKey(link);
                if (!families.TryGetValue(familyKey, out FamilyAccumulator? family))
                {
                    family = new FamilyAccumulator(familyKey);
                    families.Add(familyKey, family);
                }

                family.EntryCount++;
                family.FilePaths.Add(file.SourcePath ?? string.Empty);
                family.MinRefIndex = Math.Min(family.MinRefIndex, link.RefIndex);
                family.MaxRefIndex = Math.Max(family.MaxRefIndex, link.RefIndex);
                AddCount(family.RefIndexCounts, link.RefIndex.ToString());
                foreach (string domain in fits)
                    family.DomainFitCounts[domain]++;
            }

            if (fileHasMismatch)
                filesWithMismatches++;
        }

        List<Pm4RefIndexDomainBaseline> baselines = CandidateDomains
            .Select(domain => new Pm4RefIndexDomainBaseline(
                domain,
                baselineFitCounts[domain],
                totalMismatchCount > 0 ? (float)baselineFitCounts[domain] / totalMismatchCount : 0f))
            .OrderByDescending(static item => item.Coverage)
            .ThenBy(static item => item.Domain)
            .ToList();

        Dictionary<string, int> classificationCounts = new(StringComparer.Ordinal);
        int resolvedFamilyCount = 0;
        int ambiguousFamilyCount = 0;
        int resolvedEntryCount = 0;

        List<Pm4RefIndexClassifiedFamily> topFamilies = families.Values
            .Select(family => ClassifyFamily(family, baselines))
            .OrderByDescending(static family => family.EntryCount)
            .ThenBy(static family => family.FamilyKey)
            .ToList();

        foreach (Pm4RefIndexClassifiedFamily family in topFamilies)
        {
            AddCount(classificationCounts, family.Classification);
            if (family.Classification == "ambiguous")
            {
                ambiguousFamilyCount++;
            }
            else
            {
                resolvedFamilyCount++;
                resolvedEntryCount += family.EntryCount;
            }
        }

        Pm4RefIndexClassificationSummary summary = new(
            resolvedFamilyCount,
            ambiguousFamilyCount,
            resolvedEntryCount,
            classificationCounts
                .OrderByDescending(static kv => kv.Value)
                .ThenBy(static kv => kv.Key)
                .Select(static kv => new Pm4ValueFrequency(kv.Key, kv.Value))
                .ToList());

        List<string> notes =
        [
            "This classifier scores mismatch families against corpus baseline coverage so huge domains like MPRR do not win merely because they fit almost everything by size.",
            "A resolved family here is a likely target-domain family, not final semantic proof.",
            "Families are still keyed by LinkId + TypeFlags + Subtype because current evidence says those seams cluster the unresolved RefIndex population best.",
        ];

        return new Pm4RefIndexClassifierReport(
            inputDirectory,
            files.Count,
            filesWithMismatches,
            totalMismatchCount,
            baselines,
            summary,
            topFamilies.Take(48).ToList(),
            notes);
    }

    private static Pm4RefIndexClassifiedFamily ClassifyFamily(FamilyAccumulator family, IReadOnlyList<Pm4RefIndexDomainBaseline> baselines)
    {
        List<Pm4RefIndexFamilyDomainScore> scores = baselines
            .Select(baseline =>
            {
                family.DomainFitCounts.TryGetValue(baseline.Domain, out int fitCount);
                float coverage = family.EntryCount > 0 ? (float)fitCount / family.EntryCount : 0f;
                float delta = coverage - baseline.Coverage;
                float lift = baseline.Coverage > 0f ? coverage / baseline.Coverage : (coverage > 0f ? float.PositiveInfinity : 0f);
                return new Pm4RefIndexFamilyDomainScore(baseline.Domain, fitCount, coverage, baseline.Coverage, delta, lift);
            })
            .OrderByDescending(static score => score.CoverageDelta)
            .ThenByDescending(static score => score.Coverage)
            .ThenBy(static score => score.Domain)
            .ToList();

        Pm4RefIndexFamilyDomainScore best = scores[0];
        Pm4RefIndexFamilyDomainScore second = scores.Count > 1 ? scores[1] : best;

        string classification = "ambiguous";
        string confidence = "low";
        float deltaGap = best.CoverageDelta - second.CoverageDelta;

        if (best.Coverage >= 0.9f && best.CoverageDelta >= 0.2f && deltaGap >= 0.1f)
        {
            classification = $"probable-{best.Domain}";
            confidence = "high";
        }
        else if (best.Coverage >= 0.75f && best.CoverageDelta >= 0.12f && deltaGap >= 0.05f)
        {
            classification = $"probable-{best.Domain}";
            confidence = "medium";
        }
        else if (best.Coverage >= 0.5f && best.CoverageDelta >= 0.08f && deltaGap >= 0.02f)
        {
            classification = $"possible-{best.Domain}";
            confidence = "low";
        }

        return new Pm4RefIndexClassifiedFamily(
            family.FamilyKey,
            classification,
            confidence,
            family.FilePaths.Count,
            family.EntryCount,
            family.MinRefIndex == ushort.MaxValue ? (ushort)0 : family.MinRefIndex,
            family.MaxRefIndex,
            scores.Take(6).ToList(),
            family.RefIndexCounts
                .OrderByDescending(static kv => kv.Value)
                .ThenBy(static kv => kv.Key)
                .Take(8)
                .Select(static kv => new Pm4ValueFrequency(kv.Key, kv.Value))
                .ToList());
    }

    private static IReadOnlyList<string> GetFitDomains(Pm4ResearchFile file, ushort refIndex)
    {
        List<string> fits = new(CandidateDomains.Length);
        if (refIndex < file.KnownChunks.Mslk.Count)
            fits.Add("MSLK");
        if (refIndex < file.KnownChunks.Mspi.Count)
            fits.Add("MSPI");
        if (refIndex < file.KnownChunks.Msvi.Count)
            fits.Add("MSVI");
        if (refIndex < file.KnownChunks.Mscn.Count)
            fits.Add("MSCN");
        if (refIndex < file.KnownChunks.Mprl.Count)
            fits.Add("MPRL");
        if (refIndex < file.KnownChunks.Mspv.Count)
            fits.Add("MSPV");
        if (refIndex < file.KnownChunks.Msvt.Count)
            fits.Add("MSVT");
        if (refIndex < file.KnownChunks.Mprr.Count)
            fits.Add("MPRR");
        return fits;
    }

    private static string BuildFamilyKey(Pm4MslkEntry link)
    {
        string linkKey = TryDecodeTileLink(link.LinkId, out string? tileKey)
            ? $"tile={tileKey}"
            : $"link=0x{link.LinkId:X8}";
        return $"{linkKey}|flags=0x{link.TypeFlags:X2}|subtype={link.Subtype}";
    }

    private static bool TryDecodeTileLink(uint linkId, out string tileKey)
    {
        ushort high = (ushort)(linkId >> 16);
        if (high != 0xFFFF)
        {
            tileKey = string.Empty;
            return false;
        }

        ushort low = (ushort)(linkId & 0xFFFF);
        byte tileY = (byte)(low >> 8);
        byte tileX = (byte)(low & 0xFF);
        tileKey = $"{tileX}_{tileY}";
        return true;
    }

    private static void AddCount(Dictionary<string, int> counts, string key)
    {
        counts.TryGetValue(key, out int existing);
        counts[key] = existing + 1;
    }

    private sealed class FamilyAccumulator
    {
        public FamilyAccumulator(string familyKey)
        {
            FamilyKey = familyKey;
            DomainFitCounts = CandidateDomains.ToDictionary(static domain => domain, static _ => 0, StringComparer.Ordinal);
        }

        public string FamilyKey { get; }
        public int EntryCount { get; set; }
        public HashSet<string> FilePaths { get; } = new(StringComparer.Ordinal);
        public ushort MinRefIndex { get; set; } = ushort.MaxValue;
        public ushort MaxRefIndex { get; set; }
        public Dictionary<string, int> DomainFitCounts { get; }
        public Dictionary<string, int> RefIndexCounts { get; } = new(StringComparer.Ordinal);
    }
}