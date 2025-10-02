using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

/// <summary>
/// Extracts placement ranges and assets (with world coordinates) from LK-style ADT files for a given map.
/// </summary>
public static class LkAdtAssetExtractor
{
    public static (List<PlacementRange> Ranges, List<PlacementAsset> Assets) Extract(string adtRootDir, string map)
    {
        if (string.IsNullOrWhiteSpace(adtRootDir) || !Directory.Exists(adtRootDir))
            return (new List<PlacementRange>(), new List<PlacementAsset>());

        // 1) Ranges using existing analyzer
        var ranges = RangeScanner.AnalyzeRangesForMap(adtRootDir, map);

        // 2) Assets with world coordinates
        var assets = new List<PlacementAsset>();
        var adtFiles = Directory.EnumerateFiles(adtRootDir, "*.adt", SearchOption.AllDirectories)
            .Where(p => Path.GetFileNameWithoutExtension(p).StartsWith(map + "_", StringComparison.OrdinalIgnoreCase))
            .ToList();

        foreach (var adt in adtFiles)
        {
            if (!TryParseMapTile(adt, out var row, out var col)) continue;

            // Read filename tables per ADT
            var m2Names = LkAdtReader.ReadMmdx(adt);
            var wmoNames = LkAdtReader.ReadMwmo(adt);

            // MDDF (M2/MDX)
            foreach (var p in LkAdtReader.ReadMddf(adt))
            {
                var assetPath = SafeName(m2Names, p.NameIndex);
                BuildNames(assetPath, out var folder, out var category, out var subcategory, out var fn, out var stem, out var ext);
                assets.Add(new PlacementAsset(
                    map,
                    row,
                    col,
                    PlacementKind.M2,
                    unchecked((uint)p.UniqueId),
                    assetPath,
                    Path.GetFileName(adt),
                    p.WorldX, p.WorldY, p.WorldZ,
                    p.RotX, p.RotY, p.RotZ,
                    p.Scale,
                    p.Flags, p.DoodadSet, p.NameSet,
                    folder, category, subcategory,
                    fn, stem, ext));
            }

            // MODF (WMO)
            foreach (var p in LkAdtReader.ReadModf(adt))
            {
                var assetPath = SafeName(wmoNames, p.NameIndex);
                BuildNames(assetPath, out var folder, out var category, out var subcategory, out var fn, out var stem, out var ext);
                assets.Add(new PlacementAsset(
                    map,
                    row,
                    col,
                    PlacementKind.WMO,
                    unchecked((uint)p.UniqueId),
                    assetPath,
                    Path.GetFileName(adt),
                    p.WorldX, p.WorldY, p.WorldZ,
                    p.RotX, p.RotY, p.RotZ,
                    p.Scale,
                    p.Flags, p.DoodadSet, p.NameSet,
                    folder, category, subcategory,
                    fn, stem, ext));
            }
        }

        return (ranges, assets);
    }

    private static bool TryParseMapTile(string adtPath, out int row, out int col)
    {
        row = -1; col = -1;
        var name = Path.GetFileNameWithoutExtension(adtPath);
        var parts = name.Split('_');
        if (parts.Length < 3) return false;
        return int.TryParse(parts[^2], NumberStyles.Integer, CultureInfo.InvariantCulture, out row)
            && int.TryParse(parts[^1], NumberStyles.Integer, CultureInfo.InvariantCulture, out col);
    }

    private static string SafeName(List<string> names, int index)
    {
        if (index >= 0 && index < names.Count)
        {
            var n = names[index]?.Trim() ?? string.Empty;
            return n.Replace('\\', '/');
        }
        return string.Empty;
    }

    private static void BuildNames(string assetPath, out string folder, out string category, out string subcategory, out string fileName, out string fileStem, out string extension)
    {
        var normalized = (assetPath ?? string.Empty).Replace('\\', '/').Trim();
        var segments = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries);
        folder = segments.Length >= 2 ? string.Join('/', segments.Take(2)) : (segments.Length >= 1 ? segments[0] : "(root)");
        category = segments.Length >= 3 ? segments[2] : (segments.Length >= 2 ? segments[1] : (segments.Length >= 1 ? segments[0] : "(root)"));
        subcategory = segments.Length >= 4 ? segments[3] : (segments.Length >= 3 ? segments[2] : "(none)");
        fileName = Path.GetFileName(normalized);
        extension = Path.GetExtension(fileName);
        fileStem = Path.GetFileNameWithoutExtension(fileName);
    }
}
