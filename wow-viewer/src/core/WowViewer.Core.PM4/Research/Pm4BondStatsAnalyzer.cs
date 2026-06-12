using WowViewer.Core.PM4.Models;
using WowViewer.Core.PM4.Services;

namespace WowViewer.Core.PM4.Research;

public static class Pm4BondStatsAnalyzer
{
    private static readonly Dictionary<byte, string> KnownTypeLabels = new()
    {
        [0x03] = "M2 top",
        [0x10] = "interior WMO floor",
        [0x12] = "exterior WMO solid",
    };

    public static Pm4BondStatsReport AnalyzeDirectory(string inputDirectory)
    {
        string resolvedDirectory = Pm4CoordinateService.ResolveMapDirectory(inputDirectory);

        List<Pm4ResearchDocument> files = Directory
            .EnumerateFiles(resolvedDirectory, "*.pm4", SearchOption.TopDirectoryOnly)
            .OrderBy(Path.GetFileName)
            .Select(Pm4ResearchReader.ReadFile)
            .ToList();

        List<Pm4BondStatsPerFileEntry> perFileEntries = [];
        Dictionary<string, int> globalHighByteReuse = new(StringComparer.Ordinal);
        Dictionary<string, int> globalLowByteReuse = new(StringComparer.Ordinal);
        Dictionary<string, int> globalCombinedReuse = new(StringComparer.Ordinal);
        Dictionary<(byte high, byte low), (int surfaces, int files, HashSet<byte> types)> pairCounts = new();
        HashSet<uint> distinctCk24 = new();
        HashSet<byte> distinctCk24Types = new();
        int totalSurfaceCount = 0;
        int zeroCk24Count = 0;

        foreach (Pm4ResearchDocument file in files)
        {
            IReadOnlyList<Pm4MsurEntry> msur = file.KnownChunks.Msur;
            (int? tileX, int? tileY) = TryParseTileCoordinates(file.SourcePath);

            Dictionary<string, int> fileHighByteReuse = new(StringComparer.Ordinal);
            Dictionary<string, int> fileLowByteReuse = new(StringComparer.Ordinal);
            Dictionary<string, int> fileCombinedReuse = new(StringComparer.Ordinal);
            Dictionary<byte, List<(byte high, byte low)>> typeBucketPairs = new();
            int fileSurfaceCount = 0;

            foreach (Pm4MsurEntry surface in msur)
            {
                distinctCk24.Add(surface.Ck24);
                distinctCk24Types.Add(surface.Ck24Type);

                if (surface.Ck24 == 0)
                {
                    zeroCk24Count++;
                    continue;
                }

                totalSurfaceCount++;
                fileSurfaceCount++;

                byte high = surface.Ck24HighByte;
                byte low = surface.Ck24LowByte;
                string highKey = $"0x{high:X2}";
                string lowKey = $"0x{low:X2}";
                string combinedKey = $"0x{(high << 8 | low):X4}";

                AddCount(fileHighByteReuse, highKey);
                AddCount(fileLowByteReuse, lowKey);
                AddCount(fileCombinedReuse, combinedKey);
                AddCount(globalHighByteReuse, highKey);
                AddCount(globalLowByteReuse, lowKey);
                AddCount(globalCombinedReuse, combinedKey);

                var pairKey = (high, low);
                if (!pairCounts.TryGetValue(pairKey, out var pair))
                {
                    pair = (0, 0, new HashSet<byte>());
                    pairCounts[pairKey] = pair;
                }
                pairCounts[pairKey] = (pair.surfaces + 1, pair.files, pair.types);
                pair.types.Add(surface.Ck24Type);

                if (!typeBucketPairs.TryGetValue(surface.Ck24Type, out var pairs))
                {
                    pairs = [];
                    typeBucketPairs[surface.Ck24Type] = pairs;
                }
                pairs.Add((high, low));
            }

            foreach (var kv in pairCounts.ToList())
            {
                var (high, low) = kv.Key;
                var entry = kv.Value;
                if (fileHighByteReuse.ContainsKey($"0x{high:X2}"))
                {
                    pairCounts[kv.Key] = (entry.surfaces, entry.files + 1, entry.types);
                }
            }

            List<Pm4BondStatsTypeBucketEntry> typeBuckets = [];
            foreach (var kv in typeBucketPairs.OrderBy(kv => kv.Key))
            {
                var pairs = kv.Value;
                typeBuckets.Add(new Pm4BondStatsTypeBucketEntry(
                    Ck24Type: kv.Key,
                    TypeLabel: KnownTypeLabels.TryGetValue(kv.Key, out var label) ? label : $"0x{kv.Key:X2}",
                    SurfaceCount: pairs.Count,
                    DistinctHighBytes: pairs.Select(p => p.high).Distinct().Count(),
                    DistinctLowBytes: pairs.Select(p => p.low).Distinct().Count(),
                    DistinctCombinedIds: pairs.Distinct().Count(),
                    TopHighByteValues: pairs.GroupBy(p => p.high)
                        .OrderByDescending(g => g.Count())
                        .Take(8)
                        .Select(g => new Pm4ValueFrequency($"0x{g.Key:X2}", g.Count()))
                        .ToList(),
                    TopLowByteValues: pairs.GroupBy(p => p.low)
                        .OrderByDescending(g => g.Count())
                        .Take(8)
                        .Select(g => new Pm4ValueFrequency($"0x{g.Key:X2}", g.Count()))
                        .ToList()
                ));
            }

            perFileEntries.Add(new Pm4BondStatsPerFileEntry(
                SourcePath: file.SourcePath ?? "<unknown>",
                TileX: tileX,
                TileY: tileY,
                SurfaceCount: fileSurfaceCount,
                DistinctCk24Types: typeBucketPairs.Count,
                HighByteReuseDistribution: ToValueFrequencies(fileHighByteReuse),
                LowByteReuseDistribution: ToValueFrequencies(fileLowByteReuse),
                CombinedReuseDistribution: ToValueFrequencies(fileCombinedReuse),
                TypeBucketBreakdown: typeBuckets
            ));
        }

        var topPairsByCount = pairCounts
            .OrderByDescending(kv => kv.Value.surfaces)
            .Take(32)
            .Select(kv => new Pm4BondStatsBytePair(
                HighByte: kv.Key.high,
                LowByte: kv.Key.low,
                SurfaceCount: kv.Value.surfaces,
                FileCount: kv.Value.files,
                AssociatedCk24Types: kv.Value.types.OrderBy(t => t).ToList()
            ))
            .ToList();

        var topPairsByHighByte = topPairsByCount
            .GroupBy(p => p.HighByte)
            .OrderByDescending(g => g.Sum(p => p.SurfaceCount))
            .SelectMany(g => g.OrderByDescending(p => p.SurfaceCount))
            .ToList();

        var topPairsByLowByte = topPairsByCount
            .GroupBy(p => p.LowByte)
            .OrderByDescending(g => g.Sum(p => p.SurfaceCount))
            .SelectMany(g => g.OrderByDescending(p => p.SurfaceCount))
            .ToList();

        var crossTabulation = new Pm4BondStatsCrossTabulation(
            TotalPairs: pairCounts.Count,
            DistinctHighByteValues: pairCounts.Keys.Select(k => k.high).Distinct().Count(),
            DistinctLowByteValues: pairCounts.Keys.Select(k => k.low).Distinct().Count(),
            TopPairsByCount: topPairsByCount,
            TopPairsByHighByte: topPairsByHighByte,
            TopPairsByLowByte: topPairsByLowByte
        );

        List<string> notes = [];
        notes.Add($"Analyzed {files.Count} PM4 files from '{resolvedDirectory}'.");
        notes.Add($"Found {distinctCk24.Count} distinct CK24 values across {totalSurfaceCount} non-zero surfaces.");
        notes.Add($"Zero-pad CK24 (0x000000) excluded: {zeroCk24Count} surfaces.");
        notes.Add($"Distinct CK24 types: {distinctCk24Types.Count}.");

        return new Pm4BondStatsReport(
            InputDirectory: resolvedDirectory,
            FileCount: files.Count,
            TotalSurfaceCount: totalSurfaceCount,
            ZeroCk24SurfaceCount: zeroCk24Count,
            DistinctCk24Values: distinctCk24.Count,
            DistinctCk24Types: distinctCk24Types.Count,
            CrossTabulation: crossTabulation,
            PerFileEntries: perFileEntries,
            Notes: notes
        );
    }

    private static void AddCount(Dictionary<string, int> dict, string key)
    {
        dict.TryGetValue(key, out int current);
        dict[key] = current + 1;
    }

    private static IReadOnlyList<Pm4ValueFrequency> ToValueFrequencies(Dictionary<string, int> dict)
    {
        return dict
            .OrderByDescending(kv => kv.Value)
            .Take(16)
            .Select(kv => new Pm4ValueFrequency(kv.Key, kv.Value))
            .ToList();
    }

    private static (int?, int?) TryParseTileCoordinates(string? path)
    {
        if (string.IsNullOrWhiteSpace(path))
            return (null, null);

        string name = Path.GetFileNameWithoutExtension(path);
        string[] parts = name.Split('_');
        if (parts.Length >= 3
            && int.TryParse(parts[^2], out int x)
            && int.TryParse(parts[^1], out int y))
        {
            return (x, y);
        }

        return (null, null);
    }
}
