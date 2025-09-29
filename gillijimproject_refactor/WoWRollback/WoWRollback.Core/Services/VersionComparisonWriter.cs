using System.Globalization;
using System.Text;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

public static class VersionComparisonWriter
{
    public static ComparisonOutputPaths WriteOutputs(string? rootDirectory, VersionComparisonResult result)
    {
        var baseDirectory = string.IsNullOrWhiteSpace(rootDirectory)
            ? Path.Combine(Directory.GetCurrentDirectory(), "rollback_outputs")
            : rootDirectory!;

        var comparisonDirectory = Path.Combine(baseDirectory, "comparisons", result.ComparisonKey);
        Directory.CreateDirectory(comparisonDirectory);

        var versionRangesPath = Path.Combine(comparisonDirectory, $"version_ranges_{result.ComparisonKey}.csv");
        var mapSummaryPath = Path.Combine(comparisonDirectory, $"map_summary_{result.ComparisonKey}.csv");
        var overlapPath = Path.Combine(comparisonDirectory, $"range_overlaps_{result.ComparisonKey}.csv");
        var assetFirstSeenPath = Path.Combine(comparisonDirectory, $"asset_first_seen_{result.ComparisonKey}.csv");
        var assetFolderSummaryPath = Path.Combine(comparisonDirectory, $"asset_folder_summary_{result.ComparisonKey}.csv");
        var assetFolderTimelinePath = Path.Combine(comparisonDirectory, $"asset_folder_timeline_{result.ComparisonKey}.csv");
        var assetTimelinePath = Path.Combine(comparisonDirectory, $"asset_timeline_{result.ComparisonKey}.csv");

        WriteVersionRanges(versionRangesPath, result.RangeEntries);
        WriteMapSummaries(mapSummaryPath, result.MapSummaries);
        WriteOverlaps(overlapPath, result.Overlaps);
        WriteAssetFirstSeen(assetFirstSeenPath, result.AssetFirstSeen);
        WriteAssetFolderSummary(assetFolderSummaryPath, result.AssetFolderSummaries);
        WriteAssetFolderTimeline(assetFolderTimelinePath, result.AssetFolderTimeline);
        WriteAssetTimeline(assetTimelinePath, result.AssetTimeline);

        if (result.Warnings.Count > 0)
        {
            var warningPath = Path.Combine(comparisonDirectory, $"warnings_{result.ComparisonKey}.txt");
            File.WriteAllLines(warningPath, result.Warnings);
        }

        return new ComparisonOutputPaths(
            comparisonDirectory,
            versionRangesPath,
            mapSummaryPath,
            overlapPath,
            assetFirstSeenPath,
            assetFolderSummaryPath,
            assetFolderTimelinePath,
            assetTimelinePath);
    }

    private static void WriteVersionRanges(string path, IReadOnlyList<VersionRangeEntry> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version,map,tile_row,tile_col,kind,min_unique_id,max_unique_id,file,assets");
        foreach (var entry in entries)
        {
            var kind = entry.Kind == PlacementKind.M2 ? "M2" : "WMO";
            var assets = string.Join('|', entry.Assets.Select(a => a.Replace('|', '/')));
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(entry.Version),
                Csv(entry.Map),
                entry.TileRow.ToString(CultureInfo.InvariantCulture),
                entry.TileCol.ToString(CultureInfo.InvariantCulture),
                kind,
                entry.MinUniqueId.ToString(CultureInfo.InvariantCulture),
                entry.MaxUniqueId.ToString(CultureInfo.InvariantCulture),
                Csv(entry.FilePath.Replace('\\','/')),
                Csv(assets)
            }));
        }
    }

    private static void WriteMapSummaries(string path, IReadOnlyList<MapVersionSummary> summaries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version,map,range_count,min_unique_id,max_unique_id,distinct_asset_count");
        foreach (var summary in summaries.OrderBy(s => s.Version, StringComparer.OrdinalIgnoreCase)
                                         .ThenBy(s => s.Map, StringComparer.OrdinalIgnoreCase))
        {
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(summary.Version),
                Csv(summary.Map),
                summary.RangeCount.ToString(CultureInfo.InvariantCulture),
                summary.MinUniqueId.ToString(CultureInfo.InvariantCulture),
                summary.MaxUniqueId.ToString(CultureInfo.InvariantCulture),
                summary.DistinctAssetCount.ToString(CultureInfo.InvariantCulture)
            }));
        }
    }

    private static void WriteOverlaps(string path, IReadOnlyList<RangeOverlapEntry> overlaps)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version_a,map_a,tile_row_a,tile_col_a,kind_a,min_unique_id_a,max_unique_id_a,version_b,map_b,tile_row_b,tile_col_b,kind_b,min_unique_id_b,max_unique_id_b,overlap_min,overlap_max");
        foreach (var overlap in overlaps)
        {
            var kindA = overlap.KindA == PlacementKind.M2 ? "M2" : "WMO";
            var kindB = overlap.KindB == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(overlap.VersionA),
                Csv(overlap.MapA),
                overlap.TileRowA.ToString(CultureInfo.InvariantCulture),
                overlap.TileColA.ToString(CultureInfo.InvariantCulture),
                kindA,
                overlap.MinUniqueIdA.ToString(CultureInfo.InvariantCulture),
                overlap.MaxUniqueIdA.ToString(CultureInfo.InvariantCulture),
                Csv(overlap.VersionB),
                Csv(overlap.MapB),
                overlap.TileRowB.ToString(CultureInfo.InvariantCulture),
                overlap.TileColB.ToString(CultureInfo.InvariantCulture),
                kindB,
                overlap.MinUniqueIdB.ToString(CultureInfo.InvariantCulture),
                overlap.MaxUniqueIdB.ToString(CultureInfo.InvariantCulture),
                overlap.OverlapMin.ToString(CultureInfo.InvariantCulture),
                overlap.OverlapMax.ToString(CultureInfo.InvariantCulture)
            }));
        }
    }

    private static void WriteAssetFirstSeen(string path, IReadOnlyList<AssetFirstSeenEntry> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("asset_path,version,map,tile_row,tile_col,kind,min_unique_id,max_unique_id");
        foreach (var entry in entries)
        {
            var kind = entry.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(entry.AssetPath.Replace('\\','/')),
                Csv(entry.Version),
                Csv(entry.Map),
                entry.TileRow.ToString(CultureInfo.InvariantCulture),
                entry.TileCol.ToString(CultureInfo.InvariantCulture),
                kind,
                entry.MinUniqueId.ToString(CultureInfo.InvariantCulture),
                entry.MaxUniqueId.ToString(CultureInfo.InvariantCulture)
            }));
        }
    }

    private static void WriteAssetFolderSummary(string path, IReadOnlyList<AssetFolderSummary> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version,map,tile_row,tile_col,kind,folder,asset_count");
        foreach (var entry in entries)
        {
            var kind = entry.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(entry.Version),
                Csv(entry.Map),
                entry.TileRow.ToString(CultureInfo.InvariantCulture),
                entry.TileCol.ToString(CultureInfo.InvariantCulture),
                kind,
                Csv(entry.Folder),
                entry.AssetCount.ToString(CultureInfo.InvariantCulture)
            }));
        }
    }

    private static void WriteAssetFolderTimeline(string path, IReadOnlyList<AssetFolderTimelineEntry> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version,folder,distinct_asset_count,distinct_map_count,distinct_tile_count,min_unique_id,max_unique_id,maps");
        foreach (var entry in entries)
        {
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(entry.Version),
                Csv(entry.Folder),
                entry.DistinctAssetCount.ToString(CultureInfo.InvariantCulture),
                entry.DistinctMapCount.ToString(CultureInfo.InvariantCulture),
                entry.DistinctTileCount.ToString(CultureInfo.InvariantCulture),
                entry.MinUniqueId.ToString(CultureInfo.InvariantCulture),
                entry.MaxUniqueId.ToString(CultureInfo.InvariantCulture),
                Csv(string.Join('|', entry.Maps))
            }));
        }
    }

    private static void WriteAssetTimeline(string path, IReadOnlyList<AssetTimelineEntry> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version,map,tile_row,tile_col,kind,unique_id,asset_path,folder,category,subcategory");
        foreach (var entry in entries)
        {
            var kind = entry.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(entry.Version),
                Csv(entry.Map),
                entry.TileRow.ToString(CultureInfo.InvariantCulture),
                entry.TileCol.ToString(CultureInfo.InvariantCulture),
                kind,
                entry.UniqueId.ToString(CultureInfo.InvariantCulture),
                Csv(entry.AssetPath.Replace('\\','/')),
                Csv(entry.Folder),
                Csv(entry.Category),
                Csv(entry.Subcategory)
            }));
        }
    }

    private static string Csv(string value)
    {
        if (value.IndexOfAny(new[] { ',', '"', '\n', '\r' }) >= 0)
        {
            return "\"" + value.Replace("\"", "\"\"") + "\"";
        }

        return value;
    }
}
