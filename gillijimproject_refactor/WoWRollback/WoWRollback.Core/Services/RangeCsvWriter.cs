using System.Globalization;
using System.Linq;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

public static class RangeCsvWriter
{
    public static RangeCsvResult WritePerMapCsv(
        string sessionDir,
        string map,
        IEnumerable<PlacementRange> ranges,
        IEnumerable<PlacementAsset>? assets = null)
    {
        Directory.CreateDirectory(sessionDir);
        var outPath = Path.Combine(sessionDir, $"id_ranges_by_map_{map}.csv");
        var ordered = ranges
            .OrderBy(r => r.MinUniqueId)
            .ThenBy(r => r.MaxUniqueId)
            .ThenBy(r => r.TileRow)
            .ThenBy(r => r.TileCol)
            .ThenBy(r => r.Kind)
            .ToList();

        using (var sw = new StreamWriter(outPath, false))
        {
            sw.WriteLine("map,tile_row,tile_col,kind,count,min_unique_id,max_unique_id,file");
            foreach (var r in ordered)
            {
                var kind = r.Kind == PlacementKind.M2 ? "M2" : "WMO";
                sw.WriteLine(string.Join(',', new[]
                {
                    r.Map,
                    r.TileRow.ToString(CultureInfo.InvariantCulture),
                    r.TileCol.ToString(CultureInfo.InvariantCulture),
                    kind,
                    r.Count.ToString(CultureInfo.InvariantCulture),
                    r.MinUniqueId.ToString(CultureInfo.InvariantCulture),
                    r.MaxUniqueId.ToString(CultureInfo.InvariantCulture),
                    r.FilePath.Replace('\\','/')
                }));
            }
        }

        var timelinePath = WriteTimelineCsv(sessionDir, map, ordered);

        string? assetLedgerPath = null;
        string? timelineAssetsPath = null;

        if (assets is not null)
        {
            var assetList = assets.ToList();
            assetLedgerPath = WriteAssetLedgerCsv(sessionDir, map, assetList);
            timelineAssetsPath = WriteTimelineAssetsCsv(sessionDir, map, ordered, assetList);
        }

        return new RangeCsvResult(outPath, timelinePath, timelineAssetsPath, assetLedgerPath);
    }

    private static string WriteTimelineCsv(string sessionDir, string map, IReadOnlyCollection<PlacementRange> ordered)
    {
        var outPath = Path.Combine(sessionDir, $"timeline_{map}.csv");
        using var sw = new StreamWriter(outPath, false);
        sw.WriteLine("map,kind,min_unique_id,max_unique_id,count,tile_row,tile_col,file");
        foreach (var r in ordered)
        {
            var kind = r.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                r.Map,
                kind,
                r.MinUniqueId.ToString(CultureInfo.InvariantCulture),
                r.MaxUniqueId.ToString(CultureInfo.InvariantCulture),
                r.Count.ToString(CultureInfo.InvariantCulture),
                r.TileRow.ToString(CultureInfo.InvariantCulture),
                r.TileCol.ToString(CultureInfo.InvariantCulture),
                r.FilePath.Replace('\\','/')
            }));
        }
        return outPath;
    }

    private static string WriteAssetLedgerCsv(string sessionDir, string map, IReadOnlyList<PlacementAsset> assets)
    {
        var outPath = Path.Combine(sessionDir, $"assets_{map}.csv");
        using var sw = new StreamWriter(outPath, false);
        sw.WriteLine("map,kind,unique_id,asset_path,tile_row,tile_col,file,world_x,world_y,world_z,rot_x,rot_y,rot_z,scale,flags,doodad_set,name_set,folder,category,subcategory,file_name,file_stem,extension");
        foreach (var asset in assets
                     .OrderBy(a => a.Map)
                     .ThenBy(a => a.Kind)
                     .ThenBy(a => a.UniqueId ?? uint.MaxValue)
                     .ThenBy(a => a.TileRow)
                     .ThenBy(a => a.TileCol)
                     .ThenBy(a => a.AssetPath, StringComparer.OrdinalIgnoreCase))
        {
            var kind = asset.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                asset.Map,
                kind,
                asset.UniqueId?.ToString(CultureInfo.InvariantCulture) ?? string.Empty,
                asset.AssetPath.Replace('\\','/'),
                asset.TileRow.ToString(CultureInfo.InvariantCulture),
                asset.TileCol.ToString(CultureInfo.InvariantCulture),
                asset.FilePath.Replace('\\','/'),
                asset.WorldX.ToString(CultureInfo.InvariantCulture),
                asset.WorldY.ToString(CultureInfo.InvariantCulture),
                asset.WorldZ.ToString(CultureInfo.InvariantCulture),
                asset.RotationX.ToString(CultureInfo.InvariantCulture),
                asset.RotationY.ToString(CultureInfo.InvariantCulture),
                asset.RotationZ.ToString(CultureInfo.InvariantCulture),
                asset.Scale.ToString(CultureInfo.InvariantCulture),
                asset.Flags.ToString(CultureInfo.InvariantCulture),
                asset.DoodadSet.ToString(CultureInfo.InvariantCulture),
                asset.NameSet.ToString(CultureInfo.InvariantCulture),
                asset.Folder.Replace('\\','/'),
                asset.Category,
                asset.Subcategory,
                asset.FileName,
                asset.FileStem,
                asset.Extension
            }));
        }

        return outPath;
    }

    private static string? WriteTimelineAssetsCsv(
        string sessionDir,
        string map,
        IReadOnlyCollection<PlacementRange> orderedRanges,
        IReadOnlyList<PlacementAsset> assets)
    {
        if (assets.Count == 0)
        {
            return null;
        }

        var assetsByKey = assets
            .GroupBy(a => (a.Map, a.TileRow, a.TileCol, a.Kind))
            .ToDictionary(g => g.Key, g => g.ToList());

        var outPath = Path.Combine(sessionDir, $"timeline_assets_{map}.csv");
        using var sw = new StreamWriter(outPath, false);
        sw.WriteLine("map,kind,min_unique_id,max_unique_id,asset_count,assets,tile_row,tile_col,file");

        foreach (var range in orderedRanges)
        {
            if (!assetsByKey.TryGetValue((range.Map, range.TileRow, range.TileCol, range.Kind), out var assetCandidates))
            {
                continue;
            }

            var matchingAssets = assetCandidates
                .Where(a => a.UniqueId.HasValue && a.UniqueId.Value >= range.MinUniqueId && a.UniqueId.Value <= range.MaxUniqueId)
                .Select(a => a.AssetPath)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .OrderBy(a => a, StringComparer.OrdinalIgnoreCase)
                .ToList();

            if (matchingAssets.Count == 0)
            {
                // Fallback: include all assets for the layer if unique IDs are missing
                matchingAssets = assetCandidates
                    .Select(a => a.AssetPath)
                    .Distinct(StringComparer.OrdinalIgnoreCase)
                    .OrderBy(a => a, StringComparer.OrdinalIgnoreCase)
                    .ToList();
            }

            var kind = range.Kind == PlacementKind.M2 ? "M2" : "WMO";
            sw.WriteLine(string.Join(',', new[]
            {
                range.Map,
                kind,
                range.MinUniqueId.ToString(CultureInfo.InvariantCulture),
                range.MaxUniqueId.ToString(CultureInfo.InvariantCulture),
                matchingAssets.Count.ToString(CultureInfo.InvariantCulture),
                string.Join('|', matchingAssets.Select(a => a.Replace('\\','/'))),
                range.TileRow.ToString(CultureInfo.InvariantCulture),
                range.TileCol.ToString(CultureInfo.InvariantCulture),
                range.FilePath.Replace('\\','/')
            }));
        }

        return outPath;
    }
}
