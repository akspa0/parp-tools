using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;

namespace AlphaWdtAnalyzer.Core;

public static class CsvReportWriter
{
    public static void WriteAssetsByType(string outDir, IEnumerable<string> wmo, IEnumerable<string> m2, IEnumerable<string> blp)
    {
        Directory.CreateDirectory(outDir);
        File.WriteAllLines(Path.Combine(outDir, "assets_wmo.csv"), wmo.OrderBy(x => x));
        File.WriteAllLines(Path.Combine(outDir, "assets_m2.csv"), m2.OrderBy(x => x));
        File.WriteAllLines(Path.Combine(outDir, "assets_blp.csv"), blp.OrderBy(x => x));
    }

    public static void WritePlacements(string outDir, IEnumerable<PlacementRecord> placements)
    {
        Directory.CreateDirectory(outDir);
        var path = Path.Combine(outDir, "placements.csv");
        using var sw = new StreamWriter(path);
        sw.WriteLine("type,asset_path,map,tile_x,tile_y,unique_id");
        foreach (var p in placements)
        {
            sw.WriteLine($"{p.Type},{Escape(p.AssetPath)},{Escape(p.MapName)},{p.TileX},{p.TileY},{p.UniqueId?.ToString(CultureInfo.InvariantCulture) ?? string.Empty}");
        }
    }

    public static void WriteMissing(string outDir, IEnumerable<string> missingWmo, IEnumerable<string> missingM2, IEnumerable<string> missingBlp)
    {
        Directory.CreateDirectory(outDir);
        File.WriteAllLines(Path.Combine(outDir, "missing-assets-wmo.csv"), missingWmo.OrderBy(x => x));
        File.WriteAllLines(Path.Combine(outDir, "missing-assets-m2.csv"), missingM2.OrderBy(x => x));
        File.WriteAllLines(Path.Combine(outDir, "missing-assets-blp.csv"), missingBlp.OrderBy(x => x));
    }

    public static void WriteIdRanges(string outDir, IEnumerable<UniqueIdClusterer.Cluster> clusters)
    {
        Directory.CreateDirectory(outDir);
        var path = Path.Combine(outDir, "id_ranges.csv");
        using var sw = new StreamWriter(path);
        sw.WriteLine("min_id,max_id,count");
        foreach (var c in clusters)
        {
            sw.WriteLine($"{c.MinId},{c.MaxId},{c.Count}");
        }
    }

    public static void WriteUniqueIds(string outDir, IEnumerable<PlacementRecord> placements)
    {
        Directory.CreateDirectory(outDir);
        var path = Path.Combine(outDir, "unique_ids_all.csv");
        using var sw = new StreamWriter(path);
        sw.WriteLine("type,asset_path,map,tile_x,tile_y,unique_id");
        foreach (var p in placements)
        {
            if (!p.UniqueId.HasValue) continue;
            sw.WriteLine($"{p.Type},{Escape(p.AssetPath)},{Escape(p.MapName)},{p.TileX},{p.TileY},{p.UniqueId.Value.ToString(CultureInfo.InvariantCulture)}");
        }
    }

    public static void WriteTimeline(string outDir, IEnumerable<PlacementRecord> placements)
    {
        Directory.CreateDirectory(outDir);
        var path = Path.Combine(outDir, "unique_id_timeline.csv");
        var sorted = placements.Where(p => p.UniqueId.HasValue).OrderBy(p => p.UniqueId!.Value).ToList();
        using var sw = new StreamWriter(path);
        sw.WriteLine("type,asset_path,map,tile_x,tile_y,unique_id");
        foreach (var p in sorted)
        {
            sw.WriteLine($"{p.Type},{Escape(p.AssetPath)},{Escape(p.MapName)},{p.TileX},{p.TileY},{p.UniqueId!.Value.ToString(CultureInfo.InvariantCulture)}");
        }
    }

    public static void WriteMapUniqueIds(string mapDir, string mapName, IEnumerable<PlacementRecord> placements)
    {
        Directory.CreateDirectory(mapDir);
        var path = Path.Combine(mapDir, "unique_ids.csv");
        using var sw = new StreamWriter(path);
        sw.WriteLine("type,asset_path,tile_x,tile_y,unique_id");
        foreach (var p in placements)
        {
            if (!p.UniqueId.HasValue) continue;
            sw.WriteLine($"{p.Type},{Escape(p.AssetPath)},{p.TileX},{p.TileY},{p.UniqueId.Value.ToString(CultureInfo.InvariantCulture)}");
        }
    }

    public static void WriteMapTimeline(string mapDir, string mapName, IEnumerable<PlacementRecord> placements)
    {
        Directory.CreateDirectory(mapDir);
        var path = Path.Combine(mapDir, "unique_id_timeline.csv");
        var sorted = placements.Where(p => p.UniqueId.HasValue).OrderBy(p => p.UniqueId!.Value).ToList();
        using var sw = new StreamWriter(path);
        sw.WriteLine("type,asset_path,tile_x,tile_y,unique_id");
        foreach (var p in sorted)
        {
            sw.WriteLine($"{p.Type},{Escape(p.AssetPath)},{p.TileX},{p.TileY},{p.UniqueId!.Value.ToString(CultureInfo.InvariantCulture)}");
        }
    }

    public static void WriteIdRangesByMap(string outDir, IEnumerable<(string MapName, UniqueIdClusterer.Cluster Cluster)> entries)
    {
        Directory.CreateDirectory(outDir);
        var list = entries.ToList();

        var path = Path.Combine(outDir, "id_ranges_by_map.csv");
        using (var sw = new StreamWriter(path))
        {
            sw.WriteLine("map,min_id,max_id,count");
            foreach (var e in list.OrderBy(e => e.MapName).ThenBy(e => e.Cluster.MinId))
            {
                sw.WriteLine($"{Escape(e.MapName)},{e.Cluster.MinId},{e.Cluster.MaxId},{e.Cluster.Count}");
            }
        }

        var perMapDir = Path.Combine(outDir, "id_ranges_by_map");
        Directory.CreateDirectory(perMapDir);
        foreach (var grouping in list.GroupBy(e => e.MapName, StringComparer.OrdinalIgnoreCase))
        {
            var fileName = $"id_ranges_{SanitizeFileSegment(grouping.Key)}.csv";
            var mapPath = Path.Combine(perMapDir, fileName);
            using var mapWriter = new StreamWriter(mapPath);
            mapWriter.WriteLine("min_id,max_id,count");
            foreach (var entry in grouping.OrderBy(e => e.Cluster.MinId))
            {
                mapWriter.WriteLine($"{entry.Cluster.MinId},{entry.Cluster.MaxId},{entry.Cluster.Count}");
            }
        }
    }

    public static void WriteIdRangeSummaryByMap(string outDir, IEnumerable<(string MapName, int MinId, int MaxId, int Count)> summaries)
    {
        Directory.CreateDirectory(outDir);
        var path = Path.Combine(outDir, "id_range_summary_by_map.csv");
        using var sw = new StreamWriter(path);
        sw.WriteLine("map,min_id,max_id,count");
        foreach (var s in summaries.OrderBy(s => s.MapName))
        {
            sw.WriteLine($"{Escape(s.MapName)},{s.MinId},{s.MaxId},{s.Count}");
        }
    }

    public static void WriteIdRangeSummaryGlobal(string outDir, int minId, int maxId, int count)
    {
        Directory.CreateDirectory(outDir);
        var path = Path.Combine(outDir, "id_range_summary_global.csv");
        using var sw = new StreamWriter(path);
        sw.WriteLine("min_id,max_id,count");
        sw.WriteLine($"{minId},{maxId},{count}");
    }

    private static string Escape(string s)
    {
        if (s.Contains(',') || s.Contains('"'))
        {
            return '"' + s.Replace("\"", "\"\"") + '"';
        }
        return s;
    }

    private static string SanitizeFileSegment(string value)
    {
        if (string.IsNullOrWhiteSpace(value)) return "map";
        var invalid = Path.GetInvalidFileNameChars();
        var chars = value.Trim().Select(ch => invalid.Contains(ch) ? '_' : ch).ToArray();
        var sanitized = new string(chars).Replace(' ', '_');
        return string.IsNullOrWhiteSpace(sanitized) ? "map" : sanitized;
    }
}
