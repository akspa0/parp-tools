using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public static class Pm4ResearchMshdAnalyzer
{
    private sealed record Sample(
        string TileCoordinate,
        bool HasTileCoordinates,
        int TileX,
        int TileY,
        uint Field00,
        uint Field04,
        uint Field08,
        uint Field0C,
        uint Field10,
        uint Field14,
        uint Field18,
        uint Field1C,
        int MslkCount,
        int MsurCount,
        int MprlCount,
        int MscnCount,
        int DistinctNonZeroLinkGroupObjectIdCount,
        int DistinctNonZeroCk24Count,
        int DistinctNonZeroCk24ObjectIdCount,
        int DistinctCk24TypeCount,
        int DistinctMscnRefCount,
        int DistinctGroupKeyCount,
        int DistinctAttributeMaskCount,
        int NonEmptyChunkFamilyCount);

    public static Pm4MshdReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4ResearchDocument> files = Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .ToList();

        List<Sample> samples = files
            .Where(static file => file.KnownChunks.Mshd is not null)
            .Select(BuildSample)
            .ToList();

        IReadOnlyList<Pm4MshdFieldSummary> fields =
        [
            BuildFieldSummary("MSHD.Field00", samples, static sample => sample.Field00),
            BuildFieldSummary("MSHD.Field04", samples, static sample => sample.Field04),
            BuildFieldSummary("MSHD.Field08", samples, static sample => sample.Field08),
            BuildFieldSummary("MSHD.Field0C", samples, static sample => sample.Field0C),
            BuildFieldSummary("MSHD.Field10", samples, static sample => sample.Field10),
            BuildFieldSummary("MSHD.Field14", samples, static sample => sample.Field14),
            BuildFieldSummary("MSHD.Field18", samples, static sample => sample.Field18),
            BuildFieldSummary("MSHD.Field1C", samples, static sample => sample.Field1C)
        ];

        IReadOnlyList<Pm4MshdRelationshipSummary> relationships =
        [
            BuildRelationship("Field00 == Field08", samples, static sample => sample.Field00 == sample.Field08, "Checks whether the first and third header fields are acting like the same count or duplicated slot."),
            BuildRelationship("Field0C == 0", samples, static sample => sample.Field0C == 0, "Tests the older placeholder hypothesis for the first trailing field."),
            BuildRelationship("Field10 == 0", samples, static sample => sample.Field10 == 0, "Tests the older placeholder hypothesis for the second trailing field."),
            BuildRelationship("Field14 == 0", samples, static sample => sample.Field14 == 0, "Tests the older placeholder hypothesis for the third trailing field."),
            BuildRelationship("Field18 == 0", samples, static sample => sample.Field18 == 0, "Tests the older placeholder hypothesis for the fourth trailing field."),
            BuildRelationship("Field1C == 0", samples, static sample => sample.Field1C == 0, "Tests the older placeholder hypothesis for the fifth trailing field."),
            BuildRelationship("Field04 == 1", samples, static sample => sample.Field04 == 1, "Checks whether the second field behaves like a common constant or version-like flag in the current corpus.")
        ];

        List<Sample> tileSamples = samples
            .Where(static sample => sample.HasTileCoordinates)
            .ToList();

        IReadOnlyList<Pm4MshdTilePackingSummary> tilePackingHypotheses =
        [
            BuildTilePackingHypothesis("Field04 == (TileX << 8) | TileY", tileSamples, static sample => sample.Field04 == PackTileBytes(sample.TileX, sample.TileY), "Tests a packed XXYY low-word hypothesis using tile X as the high byte and tile Y as the low byte."),
            BuildTilePackingHypothesis("Field04 == (TileY << 8) | TileX", tileSamples, static sample => sample.Field04 == PackTileBytes(sample.TileY, sample.TileX), "Tests a packed YYXX low-word hypothesis in case the current XX/YY assumption is byte-swapped."),
            BuildTilePackingHypothesis("Field04 low byte == TileX", tileSamples, static sample => (sample.Field04 & 0xFFu) == (uint)sample.TileX, "Checks whether the low byte alone acts like the PM4 tile X coordinate."),
            BuildTilePackingHypothesis("Field04 low byte == TileY", tileSamples, static sample => (sample.Field04 & 0xFFu) == (uint)sample.TileY, "Checks whether the low byte alone acts like the PM4 tile Y coordinate."),
            BuildTilePackingHypothesis("Field04 high byte == TileX", tileSamples, static sample => ((sample.Field04 >> 8) & 0xFFu) == (uint)sample.TileX, "Checks whether the next byte acts like the PM4 tile X coordinate."),
            BuildTilePackingHypothesis("Field04 high byte == TileY", tileSamples, static sample => ((sample.Field04 >> 8) & 0xFFu) == (uint)sample.TileY, "Checks whether the next byte acts like the PM4 tile Y coordinate.")
        ];

        Pm4MshdTileReuseSummary tileReuse = BuildTileReuseSummary(tileSamples);

        List<string> notes = new();
        if (samples.Count == 0)
        {
            notes.Add("No files with MSHD were found in the input directory.");
        }
        else
        {
            if (relationships.First(static relation => relation.Relationship == "Field0C == 0").MatchCount == samples.Count
                && relationships.First(static relation => relation.Relationship == "Field10 == 0").MatchCount == samples.Count
                && relationships.First(static relation => relation.Relationship == "Field14 == 0").MatchCount == samples.Count
                && relationships.First(static relation => relation.Relationship == "Field18 == 0").MatchCount == samples.Count
                && relationships.First(static relation => relation.Relationship == "Field1C == 0").MatchCount == samples.Count)
            {
                notes.Add("MSHD.Field0C..Field1C are all zero across the current development PM4 corpus, which weakens the idea that those trailing slots actively encode per-file root-bucket counts in this dataset.");
            }

            notes.Add("If MSHD is driving root-group or type-bucket splitting, at least one field should show a strong exact-match or high-correlation signal against distinct link-group, CK24, MDOS, or surface-group metrics. This report measures those signals directly.");
            notes.Add("Treat high correlation without exact matches carefully. It can indicate file-size or scene-density coupling rather than direct semantic ownership.");

            if (tileSamples.Count > 0)
            {
                Pm4MshdTilePackingSummary bestPacking = tilePackingHypotheses
                    .OrderByDescending(static summary => summary.MatchCount)
                    .ThenBy(static summary => summary.Hypothesis)
                    .First();

                if (bestPacking.MatchCount == 0)
                {
                    notes.Add("None of the tested byte-packed tile-coordinate hypotheses matched MSHD.Field04 anywhere in the current corpus, which weakens the idea that Field04 directly stores XX_YY tile coordinates.");
                }
                else
                {
                    notes.Add($"Best tile-coordinate packing fit was '{bestPacking.Hypothesis}' at {bestPacking.MatchCount}/{bestPacking.FileCount} files. Treat partial matches cautiously until the non-matching files are explained.");
                }

                if (tileReuse.MultiTileField04Count > 0)
                {
                    notes.Add($"MSHD.Field04 is reused across multiple tiles for {tileReuse.MultiTileField04Count} distinct values in the current corpus, so it does not behave like a one-value-per-tile key.");
                }
                else
                {
                    notes.Add("Each observed MSHD.Field04 value stayed on a single tile in the current corpus.");
                }
            }
        }

        return new Pm4MshdReport(
            resolvedDirectory,
            files.Count,
            samples.Count,
            fields,
            relationships,
            tilePackingHypotheses,
            tileReuse,
            notes);
    }

    private static Sample BuildSample(Pm4ResearchDocument file)
    {
        Pm4MshdHeader header = file.KnownChunks.Mshd!;
        IReadOnlyList<Pm4MslkEntry> links = file.KnownChunks.Mslk;
        IReadOnlyList<Pm4MsurEntry> surfaces = file.KnownChunks.Msur;

        int nonEmptyChunkFamilyCount = 0;
        if (file.KnownChunks.Mslk.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Mspv.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Mspi.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Msvt.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Msvi.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Msur.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Mscn.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Mprl.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Mprr.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Mdbh is not null) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Mdbi.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Mdbf.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Mdos.Count > 0) nonEmptyChunkFamilyCount++;
        if (file.KnownChunks.Mdsf.Count > 0) nonEmptyChunkFamilyCount++;

        bool hasTileCoordinates = Pm4CoordinateService.TryParseTileCoordinates(file.SourcePath ?? string.Empty, out int tileX, out int tileY);
        string tileCoordinate = hasTileCoordinates ? $"{tileX}_{tileY}" : "<none>";

        return new Sample(
            tileCoordinate,
            hasTileCoordinates,
            tileX,
            tileY,
            header.Field00,
            header.Field04,
            header.Field08,
            header.Field0C,
            header.Field10,
            header.Field14,
            header.Field18,
            header.Field1C,
            links.Count,
            surfaces.Count,
            file.KnownChunks.Mprl.Count,
            file.KnownChunks.Mscn.Count,
            links.Select(static link => link.GroupObjectId).Where(static value => value != 0).Distinct().Count(),
            surfaces.Select(static surface => surface.Ck24).Where(static value => value != 0).Distinct().Count(),
            surfaces.Select(static surface => surface.Ck24ObjectId).Where(static value => value != 0).Distinct().Count(),
            surfaces.Select(static surface => surface.Ck24Type).Distinct().Count(),
            surfaces.Select(static surface => surface.MscnRefIndex).Distinct().Count(),
            surfaces.Select(static surface => surface.GroupKey).Distinct().Count(),
            surfaces.Select(static surface => surface.AttributeMask).Distinct().Count(),
            nonEmptyChunkFamilyCount);
    }

    private static Pm4MshdFieldSummary BuildFieldSummary(string fieldName, IReadOnlyList<Sample> samples, Func<Sample, uint> selector)
    {
        Dictionary<string, int> counts = new(StringComparer.Ordinal);
        foreach (Sample sample in samples)
            AddCount(counts, selector(sample).ToString());

        IReadOnlyList<Pm4MshdMetricCorrelation> metricCorrelations = new[]
        {
            BuildMetricCorrelation("MSLK.Count", samples, selector, static sample => sample.MslkCount),
            BuildMetricCorrelation("MSUR.Count", samples, selector, static sample => sample.MsurCount),
            BuildMetricCorrelation("MPRL.Count", samples, selector, static sample => sample.MprlCount),
            BuildMetricCorrelation("MSCN.Count", samples, selector, static sample => sample.MscnCount),
            BuildMetricCorrelation("DistinctNonZeroLinkGroupObjectId.Count", samples, selector, static sample => sample.DistinctNonZeroLinkGroupObjectIdCount),
            BuildMetricCorrelation("DistinctNonZeroCK24.Count", samples, selector, static sample => sample.DistinctNonZeroCk24Count),
            BuildMetricCorrelation("DistinctNonZeroCk24ObjectId.Count", samples, selector, static sample => sample.DistinctNonZeroCk24ObjectIdCount),
            BuildMetricCorrelation("DistinctCk24Type.Count", samples, selector, static sample => sample.DistinctCk24TypeCount),
            BuildMetricCorrelation("DistinctMscnRef.Count", samples, selector, static sample => sample.DistinctMscnRefCount),
            BuildMetricCorrelation("DistinctGroupKey.Count", samples, selector, static sample => sample.DistinctGroupKeyCount),
            BuildMetricCorrelation("DistinctAttributeMask.Count", samples, selector, static sample => sample.DistinctAttributeMaskCount),
            BuildMetricCorrelation("NonEmptyChunkFamily.Count", samples, selector, static sample => sample.NonEmptyChunkFamilyCount)
        }
            .OrderByDescending(static metric => metric.ExactMatchCount)
            .ThenByDescending(static metric => metric.WithinOneCount)
            .ThenByDescending(static metric => Math.Abs(metric.PearsonCorrelation))
            .ToList();

        int zeroCount = samples.Count(sample => selector(sample) == 0);
        return new Pm4MshdFieldSummary(
            fieldName,
            counts.Count,
            zeroCount,
            samples.Count - zeroCount,
            ToTopFrequencies(counts),
            metricCorrelations);
    }

    private static Pm4MshdMetricCorrelation BuildMetricCorrelation(string metricName, IReadOnlyList<Sample> samples, Func<Sample, uint> fieldSelector, Func<Sample, int> metricSelector)
    {
        int exactMatchCount = 0;
        int withinOneCount = 0;
        double[] fieldValues = new double[samples.Count];
        double[] metricValues = new double[samples.Count];

        for (int index = 0; index < samples.Count; index++)
        {
            Sample sample = samples[index];
            uint fieldValue = fieldSelector(sample);
            int metricValue = metricSelector(sample);
            fieldValues[index] = fieldValue;
            metricValues[index] = metricValue;

            long diff = (long)fieldValue - metricValue;
            if (diff == 0)
                exactMatchCount++;
            if (Math.Abs(diff) <= 1)
                withinOneCount++;
        }

        return new Pm4MshdMetricCorrelation(metricName, exactMatchCount, withinOneCount, ComputePearsonCorrelation(fieldValues, metricValues));
    }

    private static Pm4MshdRelationshipSummary BuildRelationship(string name, IReadOnlyList<Sample> samples, Func<Sample, bool> predicate, string notes)
    {
        return new Pm4MshdRelationshipSummary(name, samples.Count(predicate), samples.Count, notes);
    }

    private static Pm4MshdTilePackingSummary BuildTilePackingHypothesis(string name, IReadOnlyList<Sample> samples, Func<Sample, bool> predicate, string notes)
    {
        return new Pm4MshdTilePackingSummary(name, samples.Count(predicate), samples.Count, notes);
    }

    private static Pm4MshdTileReuseSummary BuildTileReuseSummary(IReadOnlyList<Sample> tileSamples)
    {
        if (tileSamples.Count == 0)
        {
            return new Pm4MshdTileReuseSummary(0, 0, 0, 0, 0, Array.Empty<Pm4MshdField04TileReuseCase>());
        }

        Dictionary<uint, HashSet<string>> tilesByField04 = new();
        foreach (Sample sample in tileSamples)
        {
            if (!tilesByField04.TryGetValue(sample.Field04, out HashSet<string>? tileCoordinates))
            {
                tileCoordinates = new HashSet<string>(StringComparer.Ordinal);
                tilesByField04[sample.Field04] = tileCoordinates;
            }

            tileCoordinates.Add(sample.TileCoordinate);
        }

        List<Pm4MshdField04TileReuseCase> topMultiTileField04Values = tilesByField04
            .Where(static kv => kv.Value.Count > 1)
            .OrderByDescending(static kv => kv.Value.Count)
            .ThenBy(static kv => kv.Key)
            .Take(12)
            .Select(static kv => new Pm4MshdField04TileReuseCase(
                kv.Key,
                kv.Value.Count,
                kv.Value.OrderBy(static tile => tile).ToList()))
            .ToList();

        int singleTileField04Count = tilesByField04.Count(static kv => kv.Value.Count == 1);
        return new Pm4MshdTileReuseSummary(
            tileSamples.Count,
            tileSamples.Select(static sample => sample.TileCoordinate).Distinct(StringComparer.Ordinal).Count(),
            tilesByField04.Count,
            singleTileField04Count,
            tilesByField04.Count - singleTileField04Count,
            topMultiTileField04Values);
    }

    private static double ComputePearsonCorrelation(IReadOnlyList<double> x, IReadOnlyList<double> y)
    {
        if (x.Count != y.Count || x.Count == 0)
            return 0d;

        double meanX = x.Average();
        double meanY = y.Average();
        double covariance = 0d;
        double varianceX = 0d;
        double varianceY = 0d;

        for (int index = 0; index < x.Count; index++)
        {
            double dx = x[index] - meanX;
            double dy = y[index] - meanY;
            covariance += dx * dy;
            varianceX += dx * dx;
            varianceY += dy * dy;
        }

        if (varianceX <= 0d || varianceY <= 0d)
            return 0d;

        return covariance / Math.Sqrt(varianceX * varianceY);
    }

    private static IReadOnlyList<Pm4ValueFrequency> ToTopFrequencies(Dictionary<string, int> counts)
    {
        return counts
            .OrderByDescending(static kv => kv.Value)
            .ThenBy(static kv => kv.Key)
            .Take(12)
            .Select(static kv => new Pm4ValueFrequency(kv.Key, kv.Value))
            .ToList();
    }

    private static void AddCount(Dictionary<string, int> counts, string key)
    {
        counts.TryGetValue(key, out int existing);
        counts[key] = existing + 1;
    }

    private static uint PackTileBytes(int first, int second)
    {
        return ((uint)(byte)first << 8) | (byte)second;
    }
}
