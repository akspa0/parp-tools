using System.Text.RegularExpressions;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

public static class RangeScanner
{
    public static List<PlacementRange> AnalyzeRangesForMap(string inputDir, string map)
    {
        if (!Directory.Exists(inputDir)) throw new DirectoryNotFoundException(inputDir);
        var results = new List<PlacementRange>();

        // Gather all .adt files under inputDir and filter by map prefix "<map>_"
        var adtFiles = Directory.EnumerateFiles(inputDir, "*.adt", SearchOption.AllDirectories)
            .Where(p => IsMapTile(p, map, out _, out _))
            .ToList();

        foreach (var path in adtFiles)
        {
            foreach (var rec in AdtPlacementAnalyzer.AnalyzeAdt(path))
            {
                if (string.Equals(rec.Map, map, StringComparison.OrdinalIgnoreCase))
                {
                    results.Add(rec);
                }
            }
        }
        return results;
    }

    private static bool IsMapTile(string path, string map, out int row, out int col)
    {
        row = -1; col = -1;
        var name = Path.GetFileNameWithoutExtension(path);
        // Expect pattern: <map>_<row>_<col>
        if (!name.StartsWith(map + "_", StringComparison.OrdinalIgnoreCase)) return false;
        var parts = name.Split('_');
        if (parts.Length < 3) return false;
        return int.TryParse(parts[^2], out row) && int.TryParse(parts[^1], out col);
    }
}
