using System.Text;
using GillijimProject.WowFiles;
using GillijimProject.WowFiles.Alpha;
using WoWRollback.Core.Models;

namespace WoWRollback.Core.Services;

/// <summary>
/// Archaeological analyzer for Alpha WDT files - extracts UniqueID ranges representing
/// "volumes of work" performed by ancient developers, treating each range as a sedimentary
/// layer of game development history.
/// 
/// This implementation LEVERAGES the proven approach from AlphaWDTAnalysisTool.
/// </summary>
public static class AlphaWdtAnalyzer
{
    /// <summary>
    /// Analyzes an Alpha WDT file and extracts placement UniqueID ranges from all tiles.
    /// </summary>
    public static IEnumerable<PlacementRange> AnalyzeAlphaWdt(string wdtPath)
    {
        if (!File.Exists(wdtPath))
            throw new FileNotFoundException($"Alpha WDT file not found: {wdtPath}");

        var mapName = Path.GetFileNameWithoutExtension(wdtPath);
        var results = new List<PlacementRange>();

        Console.WriteLine($"[info] Beginning archaeological excavation of {mapName}...");

        try
        {
            // Use the actual working WdtAlphaScanner from AlphaWDTAnalysisTool
            var wdtScanner = new global::AlphaWdtAnalyzer.Core.WdtAlphaScanner(wdtPath);
            
            // Use the actual working AdtScanner from AlphaWDTAnalysisTool
            var adtScanner = new global::AlphaWdtAnalyzer.Core.AdtScanner();
            var adtResult = adtScanner.Scan(wdtScanner);

            // Extract placement ranges grouped by tile and type
            var placementsByTile = adtResult.Placements
                .Where(p => p.UniqueId.HasValue)
                .GroupBy(p => new { p.MapName, p.TileX, p.TileY, p.Type });

            foreach (var tileGroup in placementsByTile)
            {
                var uniqueIds = tileGroup.Select(p => p.UniqueId!.Value).Distinct().OrderBy(x => x).ToList();
                if (!uniqueIds.Any()) continue;

                object? typeValue = tileGroup.Key.Type;
                var placementKind = string.Equals(typeValue?.ToString(), "M2", StringComparison.OrdinalIgnoreCase)
                    ? PlacementKind.M2
                    : PlacementKind.WMO;

                var range = new PlacementRange(
                    tileGroup.Key.MapName,
                    tileGroup.Key.TileY,
                    tileGroup.Key.TileX,
                    placementKind,
                    uniqueIds.Count,
                    (uint)uniqueIds.Min(),
                    (uint)uniqueIds.Max(),
                    $"{tileGroup.Key.MapName}_{tileGroup.Key.TileX}_{tileGroup.Key.TileY}.adt"
                );

                results.Add(range);

                // Archaeological significance logging
                Console.WriteLine($"[artifact] Found {placementKind} sedimentary layer: " +
                    $"tile {tileGroup.Key.TileX},{tileGroup.Key.TileY} " +
                    $"range {uniqueIds.Min()}-{uniqueIds.Max()} " +
                    $"({uniqueIds.Count} artifacts)");
            }

            Console.WriteLine($"[excavation-complete] Unearthed {results.Count} archaeological layers from {mapName}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[excavation-failed] Archaeological dig failed for {mapName}: {ex.Message}");
            // Return partial results if any were found
        }

        return results;
    }

}
