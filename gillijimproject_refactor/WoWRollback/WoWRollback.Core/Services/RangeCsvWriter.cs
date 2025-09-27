using System.Globalization;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

public static class RangeCsvWriter
{
    public static string WritePerMapCsv(string sessionDir, string map, IEnumerable<PlacementRange> ranges)
    {
        Directory.CreateDirectory(sessionDir);
        var outPath = Path.Combine(sessionDir, $"id_ranges_by_map_{map}.csv");
        using var sw = new StreamWriter(outPath, false);
        sw.WriteLine("map,tile_row,tile_col,kind,count,min_unique_id,max_unique_id,file");
        foreach (var r in ranges.OrderBy(r => r.TileRow).ThenBy(r => r.TileCol).ThenBy(r => r.Kind))
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
        return outPath;
    }
}
