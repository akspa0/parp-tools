using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text.Json;

namespace WoWRollback.Core.Services;

public static class StatsService
{
    public sealed class HeatmapStats
    {
        public int MinUnique { get; set; }
        public int MaxUnique { get; set; }
        public Dictionary<string, MapRange> PerMap { get; set; } = new(StringComparer.OrdinalIgnoreCase);
        public DateTime GeneratedAt { get; set; }

        public sealed class MapRange
        {
            public int Min { get; set; }
            public int Max { get; set; }
        }
    }

    public sealed class HeatmapStatsResult
    {
        public required string OutputPath { get; init; }
        public int MinUnique { get; init; }
        public int MaxUnique { get; init; }
        public int PerMapCount { get; init; }
        public bool Skipped { get; init; }
    }

    /// <summary>
    /// Scans <buildRoot> recursively for tile_layers.csv files and writes heatmap_stats.json at buildRoot.
    /// Recomputes when any tile_layers.csv is newer than the existing stats file (unless force==true).
    /// </summary>
    public static HeatmapStatsResult GenerateHeatmapStats(string buildRoot, bool force = false)
    {
        if (string.IsNullOrWhiteSpace(buildRoot) || !Directory.Exists(buildRoot))
            throw new DirectoryNotFoundException(buildRoot);

        var statsPath = Path.Combine(buildRoot, "heatmap_stats.json");

        // Find candidate CSVs: <map>/tile_layers.csv and fallback *_tile_layers.csv
        var csvs = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
        foreach (var f in Directory.EnumerateFiles(buildRoot, "tile_layers.csv", SearchOption.AllDirectories)) csvs.Add(f);
        foreach (var f in Directory.EnumerateFiles(buildRoot, "*_tile_layers.csv", SearchOption.AllDirectories)) csvs.Add(f);

        DateTime latestCsvWrite = DateTime.MinValue;
        foreach (var p in csvs)
        {
            try { var t = File.GetLastWriteTimeUtc(p); if (t > latestCsvWrite) latestCsvWrite = t; } catch { }
        }

        var statsExists = File.Exists(statsPath);
        if (!force && statsExists)
        {
            try
            {
                var statsWrite = File.GetLastWriteTimeUtc(statsPath);
                if (latestCsvWrite != DateTime.MinValue && statsWrite >= latestCsvWrite)
                {
                    // Nothing newer than stats; return existing summary (lightweight)
                    var json = File.ReadAllText(statsPath);
                    var existing = JsonSerializer.Deserialize<HeatmapStats>(json);
                    return new HeatmapStatsResult
                    {
                        OutputPath = statsPath,
                        MinUnique = existing?.MinUnique ?? 0,
                        MaxUnique = existing?.MaxUnique ?? 0,
                        PerMapCount = existing?.PerMap?.Count ?? 0,
                        Skipped = true
                    };
                }
            }
            catch { /* fall through and recompute */ }
        }

        var perMap = new Dictionary<string, (int min, int max)>(StringComparer.OrdinalIgnoreCase);
        foreach (var csv in csvs)
        {
            try
            {
                using var sr = new StreamReader(csv);
                var headerLine = sr.ReadLine();
                if (string.IsNullOrWhiteSpace(headerLine)) continue;
                var headers = headerLine.Split(',');

                int idxMap = IndexOf(headers, "map");
                int idxStart = IndexOf(headers, "range_start");
                int idxEnd = IndexOf(headers, "range_end");
                if (idxStart < 0 || idxEnd < 0) continue;

                string? mapHintFromPath = TryMapFromPath(csv);

                while (!sr.EndOfStream)
                {
                    var line = sr.ReadLine();
                    if (string.IsNullOrWhiteSpace(line)) continue;
                    var parts = line.Split(',');
                    if (parts.Length <= Math.Max(idxEnd, idxMap)) continue;

                    var map = idxMap >= 0 ? parts[idxMap].Trim().Trim('"') : (mapHintFromPath ?? "");
                    if (string.IsNullOrWhiteSpace(map)) map = mapHintFromPath ?? "unknown";

                    if (!int.TryParse(parts[idxStart].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var rStart)) continue;
                    if (!int.TryParse(parts[idxEnd].Trim(), NumberStyles.Integer, CultureInfo.InvariantCulture, out var rEnd)) continue;

                    if (!perMap.TryGetValue(map, out var agg)) agg = (int.MaxValue, int.MinValue);
                    agg.min = Math.Min(agg.min, rStart);
                    agg.max = Math.Max(agg.max, rEnd);
                    perMap[map] = agg;
                }
            }
            catch { /* ignore file-level errors */ }
        }

        int globalMin = int.MaxValue, globalMax = int.MinValue;
        var dto = new HeatmapStats
        {
            MinUnique = 0,
            MaxUnique = 0,
            GeneratedAt = DateTime.UtcNow
        };
        foreach (var kv in perMap.OrderBy(k => k.Key, StringComparer.OrdinalIgnoreCase))
        {
            if (kv.Value.min == int.MaxValue || kv.Value.max == int.MinValue) continue;
            globalMin = Math.Min(globalMin, kv.Value.min);
            globalMax = Math.Max(globalMax, kv.Value.max);
            dto.PerMap[kv.Key] = new HeatmapStats.MapRange { Min = kv.Value.min, Max = kv.Value.max };
        }
        dto.MinUnique = globalMin == int.MaxValue ? 0 : globalMin;
        dto.MaxUnique = globalMax == int.MinValue ? 0 : globalMax;

        var options = new JsonSerializerOptions { WriteIndented = true, PropertyNamingPolicy = JsonNamingPolicy.CamelCase };
        Directory.CreateDirectory(buildRoot);
        File.WriteAllText(statsPath, JsonSerializer.Serialize(dto, options));

        return new HeatmapStatsResult
        {
            OutputPath = statsPath,
            MinUnique = dto.MinUnique,
            MaxUnique = dto.MaxUnique,
            PerMapCount = dto.PerMap.Count,
            Skipped = false
        };

        static int IndexOf(string[] headers, string name)
        {
            for (int i = 0; i < headers.Length; i++)
            {
                if (string.Equals(headers[i].Trim().Trim('"'), name, StringComparison.OrdinalIgnoreCase)) return i;
            }
            return -1;
        }

        static string? TryMapFromPath(string csvPath)
        {
            try
            {
                var dir = Path.GetDirectoryName(csvPath);
                if (string.IsNullOrWhiteSpace(dir)) return null;
                var map = Path.GetFileName(dir);
                if (!string.IsNullOrWhiteSpace(map) && !map.Contains("_tile_layers", StringComparison.OrdinalIgnoreCase)) return map;
                // fallback: filename like <map>_tile_layers.csv
                var name = Path.GetFileName(csvPath);
                if (name.EndsWith("_tile_layers.csv", StringComparison.OrdinalIgnoreCase))
                {
                    return name.Substring(0, name.Length - "_tile_layers.csv".Length);
                }
            }
            catch { }
            return null;
        }
    }
}
