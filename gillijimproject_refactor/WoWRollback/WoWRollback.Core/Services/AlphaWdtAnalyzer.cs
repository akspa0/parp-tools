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
    public static AlphaAnalysisResult AnalyzeAlphaWdt(string wdtPath)
    {
        if (!File.Exists(wdtPath))
            throw new FileNotFoundException($"Alpha WDT file not found: {wdtPath}");

        var mapName = Path.GetFileNameWithoutExtension(wdtPath);
        var ranges = new List<PlacementRange>();
        var assets = new List<PlacementAsset>();

        Console.WriteLine($"[info] Beginning archaeological excavation of {mapName}...");

        try
        {
            // Use the actual working WdtAlphaScanner from AlphaWDTAnalysisTool
            var wdtScanner = new global::AlphaWdtAnalyzer.Core.WdtAlphaScanner(wdtPath);
            
            // Use the actual working AdtScanner from AlphaWDTAnalysisTool
            var adtScanner = new global::AlphaWdtAnalyzer.Core.AdtScanner();
            var adtResult = adtScanner.Scan(wdtScanner);

            foreach (var placement in adtResult.Placements)
            {
                var placementKind = ResolvePlacementKind(placement.Type);
                uint? uniqueId = placement.UniqueId.HasValue
                    ? unchecked((uint)placement.UniqueId.Value)
                    : null;

                assets.Add(new PlacementAsset(
                    placement.MapName,
                    placement.TileY,
                    placement.TileX,
                    placementKind,
                    uniqueId,
                    placement.AssetPath,
                    $"{placement.MapName}_{placement.TileX}_{placement.TileY}.adt",
                    placement.WorldX,
                    placement.WorldY,
                    placement.WorldZ));
            }

            // Extract placement ranges grouped by tile and type (unique IDs only)
            var placementsByTile = adtResult.Placements
                .Where(p => p.UniqueId.HasValue)
                .GroupBy(p => new { p.MapName, p.TileX, p.TileY, p.Type });

            foreach (var tileGroup in placementsByTile)
            {
                var uniqueIds = tileGroup.Select(p => p.UniqueId!.Value).Distinct().OrderBy(x => x).ToList();
                if (!uniqueIds.Any()) continue;

                object? typeValue = tileGroup.Key.Type;
                var placementKind = ResolvePlacementKind(typeValue);

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

                ranges.Add(range);

                // Archaeological significance logging
                Console.WriteLine($"[artifact] Found {placementKind} sedimentary layer: " +
                    $"tile {tileGroup.Key.TileX},{tileGroup.Key.TileY} " +
                    $"range {uniqueIds.Min()}-{uniqueIds.Max()} " +
                    $"({uniqueIds.Count} artifacts)");
            }

            Console.WriteLine($"[excavation-complete] Unearthed {ranges.Count} archaeological layers from {mapName}");
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[excavation-failed] Archaeological dig failed for {mapName}: {ex.Message}");
            // Return partial results if any were found
        }

        return new AlphaAnalysisResult(ranges, assets);
    }

    private static PlacementKind ResolvePlacementKind(object? typeValue)
    {
        static string? NormalizeLabel(object? value)
        {
            if (value is null) return null;

            if (value is Enum enumValue)
            {
                return Enum.GetName(enumValue.GetType(), enumValue);
            }

            return value.ToString();
        }

        var typeLabel = NormalizeLabel(typeValue);
        if (typeLabel is null)
        {
            return PlacementKind.WMO;
        }

        var normalized = typeLabel
            .Replace("_", string.Empty, StringComparison.Ordinal)
            .Replace(" ", string.Empty, StringComparison.Ordinal)
            .ToUpperInvariant();

        return normalized is "M2" or "MDX" or "MDXORM2"
            ? PlacementKind.M2
            : PlacementKind.WMO;
    }
}
