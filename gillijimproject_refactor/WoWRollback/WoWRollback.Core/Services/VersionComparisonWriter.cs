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
        var designKitAssetsPath = Path.Combine(comparisonDirectory, $"design_kit_assets_{result.ComparisonKey}.csv");
        var designKitRangesPath = Path.Combine(comparisonDirectory, $"design_kit_ranges_{result.ComparisonKey}.csv");
        var designKitSummaryPath = Path.Combine(comparisonDirectory, $"design_kit_summary_{result.ComparisonKey}.csv");
        var designKitTimelinePath = Path.Combine(comparisonDirectory, $"design_kit_timeline_{result.ComparisonKey}.csv");
        var designKitAssetDetailsPath = Path.Combine(comparisonDirectory, $"design_kit_asset_details_{result.ComparisonKey}.csv");
        var uniqueIdAssetsPath = Path.Combine(comparisonDirectory, $"unique_id_assets_{result.ComparisonKey}.csv");
        var assetTimelineDetailedPath = Path.Combine(comparisonDirectory, $"asset_timeline_detailed_{result.ComparisonKey}.csv");

        WriteVersionRanges(versionRangesPath, result.RangeEntries);
        WriteMapSummaries(mapSummaryPath, result.MapSummaries);
        WriteOverlaps(overlapPath, result.Overlaps);
        WriteAssetFirstSeen(assetFirstSeenPath, result.AssetFirstSeen);
        WriteAssetFolderSummary(assetFolderSummaryPath, result.AssetFolderSummaries);
        WriteAssetFolderTimeline(assetFolderTimelinePath, result.AssetFolderTimeline);
        WriteAssetTimeline(assetTimelinePath, result.AssetTimeline);
        WriteDesignKitAssets(designKitAssetsPath, result.DesignKitAssets);
        WriteDesignKitRanges(designKitRangesPath, result.DesignKitRanges);
        WriteDesignKitSummary(designKitSummaryPath, result.DesignKitSummaries);
        WriteDesignKitTimeline(designKitTimelinePath, result.DesignKitTimeline);
        WriteDesignKitAssetDetails(designKitAssetDetailsPath, result.DesignKitAssetDetails);
        WriteUniqueIdAssets(uniqueIdAssetsPath, result.UniqueIdAssets);
        WriteAssetTimelineDetailed(assetTimelineDetailedPath, result.AssetTimelineDetailed);

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
            assetTimelinePath,
            designKitAssetsPath,
            designKitRangesPath,
            designKitSummaryPath,
            designKitTimelinePath,
            designKitAssetDetailsPath,
            uniqueIdAssetsPath,
            assetTimelineDetailedPath);
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

    private static void WriteDesignKitAssets(string path, IReadOnlyList<DesignKitAssetEntry> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version,map,tile_row,tile_col,kind,unique_id,asset_path,design_kit,source_rule");
        foreach (var e in entries)
        {
            var kind = e.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(e.Version),
                Csv(e.Map),
                e.TileRow.ToString(CultureInfo.InvariantCulture),
                e.TileCol.ToString(CultureInfo.InvariantCulture),
                kind,
                e.UniqueId.ToString(CultureInfo.InvariantCulture),
                Csv(e.AssetPath.Replace('\\','/')),
                Csv(e.DesignKit),
                Csv(e.SourceRule)
            }));
        }
    }

    private static void WriteDesignKitRanges(string path, IReadOnlyList<DesignKitRangeEntry> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version,map,tile_row,tile_col,kind,min_unique_id,max_unique_id,design_kit,asset_count,distinct_asset_count,source_rule");
        foreach (var e in entries)
        {
            var kind = e.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(e.Version),
                Csv(e.Map),
                e.TileRow.ToString(CultureInfo.InvariantCulture),
                e.TileCol.ToString(CultureInfo.InvariantCulture),
                kind,
                e.MinUniqueId.ToString(CultureInfo.InvariantCulture),
                e.MaxUniqueId.ToString(CultureInfo.InvariantCulture),
                Csv(e.DesignKit),
                e.AssetCount.ToString(CultureInfo.InvariantCulture),
                e.DistinctAssetCount.ToString(CultureInfo.InvariantCulture),
                Csv(e.SourceRule)
            }));
        }
    }

    private static void WriteDesignKitSummary(string path, IReadOnlyList<DesignKitSummaryEntry> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version,design_kit,distinct_asset_count,asset_count,distinct_map_count,distinct_tile_count,min_unique_id,max_unique_id");
        foreach (var e in entries)
        {
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(e.Version),
                Csv(e.DesignKit),
                e.DistinctAssetCount.ToString(CultureInfo.InvariantCulture),
                e.AssetCount.ToString(CultureInfo.InvariantCulture),
                e.DistinctMapCount.ToString(CultureInfo.InvariantCulture),
                e.DistinctTileCount.ToString(CultureInfo.InvariantCulture),
                e.MinUniqueId.ToString(CultureInfo.InvariantCulture),
                e.MaxUniqueId.ToString(CultureInfo.InvariantCulture)
            }));
        }
    }

    private static void WriteDesignKitTimeline(string path, IReadOnlyList<DesignKitTimelineEntry> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("design_kit,version,asset_count,distinct_map_count,distinct_tile_count,min_unique_id,max_unique_id");
        foreach (var e in entries)
        {
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(e.DesignKit),
                Csv(e.Version),
                e.AssetCount.ToString(CultureInfo.InvariantCulture),
                e.DistinctMapCount.ToString(CultureInfo.InvariantCulture),
                e.DistinctTileCount.ToString(CultureInfo.InvariantCulture),
                e.MinUniqueId.ToString(CultureInfo.InvariantCulture),
                e.MaxUniqueId.ToString(CultureInfo.InvariantCulture)
            }));
        }
    }

    public static string WriteYamlReports(string? rootDirectory, VersionComparisonResult result)
    {
        var baseDirectory = string.IsNullOrWhiteSpace(rootDirectory)
            ? Path.Combine(Directory.GetCurrentDirectory(), "rollback_outputs")
            : rootDirectory!;
        var comparisonDirectory = Path.Combine(baseDirectory, "comparisons", result.ComparisonKey);
        var yamlRoot = Path.Combine(comparisonDirectory, "yaml");
        Directory.CreateDirectory(yamlRoot);

        // Group tiles by map from DesignKitAssetDetails (most complete set)
        var tilesByMap = result.DesignKitAssetDetails
            .GroupBy(d => d.Map, StringComparer.OrdinalIgnoreCase)
            .ToDictionary(
                g => g.Key,
                g => g.Select(x => (x.TileRow, x.TileCol)).Distinct().OrderBy(t => t.TileRow).ThenBy(t => t.TileCol).ToList(),
                StringComparer.OrdinalIgnoreCase);

        // Write per-tile YAMLs under map subfolders
        foreach (var (map, tiles) in tilesByMap)
        {
            var mapDir = Path.Combine(yamlRoot, "map", SanitizeDir(map));
            Directory.CreateDirectory(mapDir);
            foreach (var (row, col) in tiles)
            {
                var tilePath = Path.Combine(mapDir, $"tile_r{row}_c{col}.yaml");
                WriteTileYaml(tilePath, map, row, col, result);
            }
        }

        // Write index.yaml
        var indexPath = Path.Combine(yamlRoot, "index.yaml");
        WriteIndexYaml(indexPath, result, tilesByMap);

        return yamlRoot;
    }

    private static void WriteIndexYaml(string path, VersionComparisonResult result, Dictionary<string, List<(int Row, int Col)>> tilesByMap)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine($"comparison_key: \"{Yaml(result.ComparisonKey)}\"");
        sw.WriteLine("versions:");
        foreach (var v in result.Versions)
            sw.WriteLine($"  - \"{Yaml(v)}\"");
        sw.WriteLine("maps:");
        foreach (var kvp in tilesByMap.OrderBy(k => k.Key, StringComparer.OrdinalIgnoreCase))
        {
            sw.WriteLine($"  - map: \"{Yaml(kvp.Key)}\"");
            sw.WriteLine("    tiles:");
            foreach (var t in kvp.Value)
            {
                var details = result.DesignKitAssetDetails.Where(d => d.Map.Equals(kvp.Key, StringComparison.OrdinalIgnoreCase) && d.TileRow == t.Row && d.TileCol == t.Col);
                var distinctAssets = details.Select(d => d.AssetPath).Distinct(StringComparer.OrdinalIgnoreCase).Count();
                var uidCount = result.UniqueIdAssets.Count(u => u.Map.Equals(kvp.Key, StringComparison.OrdinalIgnoreCase) && u.TileRow == t.Row && u.TileCol == t.Col);
                sw.WriteLine($"      - row: {t.Row}");
                sw.WriteLine($"        col: {t.Col}");
                sw.WriteLine($"        distinct_assets: {distinctAssets}");
                sw.WriteLine($"        uid_count: {uidCount}");
            }
        }
    }

    private static void WriteTileYaml(string path, string map, int row, int col, VersionComparisonResult result)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine($"header:");
        sw.WriteLine($"  comparison_key: \"{Yaml(result.ComparisonKey)}\"");
        sw.WriteLine($"  map: \"{Yaml(map)}\"");
        sw.WriteLine($"  tile: {{ row: {row}, col: {col} }}");
        sw.WriteLine("  versions:");
        foreach (var v in result.Versions)
            sw.WriteLine($"    - \"{Yaml(v)}\"");

        // Sediment layers per version
        sw.WriteLine("sediment_layers:");
        foreach (var v in result.Versions)
        {
            sw.WriteLine($"  - version: \"{Yaml(v)}\"");
            // Ranges
            var ranges = result.RangeEntries.Where(r => r.Version.Equals(v, StringComparison.OrdinalIgnoreCase) && r.Map.Equals(map, StringComparison.OrdinalIgnoreCase) && r.TileRow == row && r.TileCol == col)
                .OrderBy(r => r.MinUniqueId).ToList();
            sw.WriteLine("    ranges:");
            foreach (var r in ranges)
            {
                var kind = r.Kind == PlacementKind.M2 ? "M2" : "WMO";
                sw.WriteLine($"      - kind: {kind}");
                sw.WriteLine($"        min_uid: {r.MinUniqueId}");
                sw.WriteLine($"        max_uid: {r.MaxUniqueId}");
                sw.WriteLine($"        file: \"{Yaml((r.FilePath ?? string.Empty).Replace('\\','/'))}\"");
            }
            // Kits and subkits derived from DesignKitAssetDetails
            var detailsV = result.DesignKitAssetDetails.Where(d => d.Version.Equals(v, StringComparison.OrdinalIgnoreCase) && d.Map.Equals(map, StringComparison.OrdinalIgnoreCase) && d.TileRow == row && d.TileCol == col).ToList();
            var kits = detailsV.GroupBy(d => d.DesignKit, StringComparer.OrdinalIgnoreCase);
            sw.WriteLine("    kits:");
            foreach (var kg in kits.OrderBy(g => g.Key, StringComparer.OrdinalIgnoreCase))
            {
                var sourceRules = kg.Select(k => k.SourceRule).Where(s => !string.IsNullOrWhiteSpace(s)).Distinct(StringComparer.OrdinalIgnoreCase).ToList();
                var distinctAssets = kg.Select(k => k.AssetPath).Distinct(StringComparer.OrdinalIgnoreCase).Count();
                sw.WriteLine($"      - design_kit: \"{Yaml(kg.Key)}\" ");
                sw.WriteLine($"        asset_count: {kg.Count()}");
                sw.WriteLine($"        distinct_assets: {distinctAssets}");
                if (sourceRules.Count > 0)
                {
                    sw.WriteLine("        source_rules:");
                    foreach (var r in sourceRules)
                        sw.WriteLine($"          - \"{Yaml(r)}\"");
                }
                var subkits = kg.GroupBy(k => k.SubkitPath ?? string.Empty, StringComparer.OrdinalIgnoreCase);
                sw.WriteLine("        subkits:");
                foreach (var sg in subkits.OrderBy(s => s.Key, StringComparer.OrdinalIgnoreCase))
                {
                    var subDistinct = sg.Select(x => x.AssetPath).Distinct(StringComparer.OrdinalIgnoreCase).Count();
                    sw.WriteLine($"          - path: \"{Yaml(sg.Key)}\" ");
                    sw.WriteLine($"            asset_count: {sg.Count()}");
                    sw.WriteLine($"            distinct_assets: {subDistinct}");
                }
            }
        }

        // Unique IDs across all versions for this tile
        sw.WriteLine("unique_ids:");
        var uids = result.UniqueIdAssets.Where(u => u.Map.Equals(map, StringComparison.OrdinalIgnoreCase) && u.TileRow == row && u.TileCol == col)
            .OrderBy(u => u.Version, StringComparer.OrdinalIgnoreCase)
            .ThenBy(u => u.Kind)
            .ThenBy(u => u.UniqueId)
            .ToList();
        foreach (var u in uids)
        {
            var kind = u.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine($"  - uid: {u.UniqueId}");
            sw.WriteLine($"    version: \"{Yaml(u.Version)}\"");
            sw.WriteLine($"    kind: {kind}");
            sw.WriteLine($"    asset_path: \"{Yaml(u.AssetPath.Replace('\\','/'))}\"");
            sw.WriteLine($"    design_kit: \"{Yaml(u.DesignKit)}\"");
            sw.WriteLine($"    subkit_path: \"{Yaml(u.SubkitPath)}\"");
            sw.WriteLine($"    source_rule: \"{Yaml(u.SourceRule)}\"");
            sw.WriteLine($"    matched_range: {{ min: {u.MatchedRangeMin}, max: {u.MatchedRangeMax}, file: \"{Yaml((u.MatchedRangeFile ?? string.Empty).Replace('\\','/'))}\", count: {u.MatchedRangeCount} }}");
        }

        // Stats
        var allDetails = result.DesignKitAssetDetails.Where(d => d.Map.Equals(map, StringComparison.OrdinalIgnoreCase) && d.TileRow == row && d.TileCol == col).ToList();
        var totalAssets = allDetails.Count;
        var totalDistinct = allDetails.Select(d => d.AssetPath).Distinct(StringComparer.OrdinalIgnoreCase).Count();
        var uidCount = uids.Count;
        sw.WriteLine("stats:");
        sw.WriteLine($"  total_assets: {totalAssets}");
        sw.WriteLine($"  distinct_assets: {totalDistinct}");
        sw.WriteLine($"  uid_count: {uidCount}");
        var kitTotals = allDetails.GroupBy(d => d.DesignKit, StringComparer.OrdinalIgnoreCase)
            .OrderBy(g => g.Key, StringComparer.OrdinalIgnoreCase);
        sw.WriteLine("  kits:");
        foreach (var kt in kitTotals)
        {
            var dcount = kt.Select(x => x.AssetPath).Distinct(StringComparer.OrdinalIgnoreCase).Count();
            sw.WriteLine($"  - name: \"{Yaml(kt.Key)}\" ");
            sw.WriteLine($"    asset_count: {kt.Count()}");
            sw.WriteLine($"    distinct_assets: {dcount}");
        }
    }

    private static string Yaml(string value)
    {
        if (value is null) return string.Empty;
        var v = value.Replace("\"", "\\\"");
        return v;
    }

    private static string SanitizeDir(string value)
    {
        var invalid = Path.GetInvalidFileNameChars();
        var sb = new StringBuilder(value.Length);
        foreach (var ch in value)
        {
            if (Array.IndexOf(invalid, ch) >= 0 || ch == '/' || ch == '\\') sb.Append('_');
            else sb.Append(ch);
        }
        return sb.ToString();
    }

    private static void WriteDesignKitAssetDetails(string path, IReadOnlyList<DesignKitAssetDetailEntry> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version,map,tile_row,tile_col,kind,unique_id,asset_path,design_kit,source_rule,kit_root,subkit_path,subkit_top,subkit_depth,file_name,file_stem,ext,segment_count");
        foreach (var e in entries)
        {
            var kind = e.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(e.Version),
                Csv(e.Map),
                e.TileRow.ToString(CultureInfo.InvariantCulture),
                e.TileCol.ToString(CultureInfo.InvariantCulture),
                kind,
                e.UniqueId.ToString(CultureInfo.InvariantCulture),
                Csv(e.AssetPath.Replace('\\','/')),
                Csv(e.DesignKit),
                Csv(e.SourceRule),
                Csv(e.KitRoot),
                Csv(e.SubkitPath),
                Csv(e.SubkitTop),
                e.SubkitDepth.ToString(CultureInfo.InvariantCulture),
                Csv(e.FileName),
                Csv(e.FileStem),
                Csv(e.Extension),
                e.SegmentCount.ToString(CultureInfo.InvariantCulture)
            }));
        }
    }

    private static void WriteUniqueIdAssets(string path, IReadOnlyList<UniqueIdAssetEntry> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version,map,tile_row,tile_col,kind,unique_id,asset_path,design_kit,subkit_path,source_rule,matched_range_min,matched_range_max,matched_range_file,matched_range_count");
        foreach (var e in entries)
        {
            var kind = e.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(e.Version),
                Csv(e.Map),
                e.TileRow.ToString(CultureInfo.InvariantCulture),
                e.TileCol.ToString(CultureInfo.InvariantCulture),
                kind,
                e.UniqueId.ToString(CultureInfo.InvariantCulture),
                Csv(e.AssetPath.Replace('\\','/')),
                Csv(e.DesignKit),
                Csv(e.SubkitPath),
                Csv(e.SourceRule),
                e.MatchedRangeMin.ToString(CultureInfo.InvariantCulture),
                e.MatchedRangeMax.ToString(CultureInfo.InvariantCulture),
                Csv((e.MatchedRangeFile ?? string.Empty).Replace('\\','/')),
                e.MatchedRangeCount.ToString(CultureInfo.InvariantCulture)
            }));
        }
    }

    private static void WriteAssetTimelineDetailed(string path, IReadOnlyList<AssetTimelineDetailedEntry> entries)
    {
        using var sw = new StreamWriter(path, false, Encoding.UTF8);
        sw.WriteLine("version,map,tile_row,tile_col,kind,unique_id,asset_path,folder,category,subcategory,design_kit,source_rule,kit_root,subkit_path,subkit_top,subkit_depth,file_name,file_stem,ext");
        foreach (var e in entries)
        {
            var kind = e.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                Csv(e.Version),
                Csv(e.Map),
                e.TileRow.ToString(CultureInfo.InvariantCulture),
                e.TileCol.ToString(CultureInfo.InvariantCulture),
                kind,
                e.UniqueId.ToString(CultureInfo.InvariantCulture),
                Csv(e.AssetPath.Replace('\\','/')),
                Csv(e.Folder),
                Csv(e.Category),
                Csv(e.Subcategory),
                Csv(e.DesignKit),
                Csv(e.SourceRule),
                Csv(e.KitRoot),
                Csv(e.SubkitPath),
                Csv(e.SubkitTop),
                e.SubkitDepth.ToString(CultureInfo.InvariantCulture),
                Csv(e.FileName),
                Csv(e.FileStem),
                Csv(e.Extension)
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
