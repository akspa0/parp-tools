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
    /// If convertedAdtDir is provided, uses converted LK ADT files for accurate world coordinates.
    /// </summary>
    public static AlphaAnalysisResult AnalyzeAlphaWdt(string wdtPath, string? convertedAdtDir = null, bool preferRawCoordinates = false)
    {
        if (!File.Exists(wdtPath))
            throw new FileNotFoundException($"Alpha WDT file not found: {wdtPath}");

        var mapName = Path.GetFileNameWithoutExtension(wdtPath);
        var ranges = new List<PlacementRange>();
        var assets = new List<PlacementAsset>();

        Console.WriteLine($"[info] Beginning archaeological excavation of {mapName}...");
        if (!preferRawCoordinates && convertedAdtDir != null)
            Console.WriteLine($"[info] Using converted LK ADT files from: {convertedAdtDir}");

        try
        {
            // Use the actual working WdtAlphaScanner from AlphaWDTAnalysisTool
            var wdtScanner = new global::AlphaWdtAnalyzer.Core.WdtAlphaScanner(wdtPath);
            
            // Use the actual working AdtScanner from AlphaWDTAnalysisTool
            var adtScanner = new global::AlphaWdtAnalyzer.Core.AdtScanner();
            var adtResult = adtScanner.Scan(wdtScanner);

            // Build coordinate lookup from converted LK ADT files if available
            var coordinateLookup = preferRawCoordinates
                ? new Dictionary<uint, (float X, float Y, float Z, float RotX, float RotY, float RotZ, float Scale, ushort Flags, ushort DoodadSet, ushort NameSet)>()
                : BuildCoordinateLookup(mapName, adtResult, convertedAdtDir);

            foreach (var placement in adtResult.Placements)
            {
                var placementKind = ResolvePlacementKind(placement.Type);
                uint? uniqueId = placement.UniqueId.HasValue
                    ? unchecked((uint)placement.UniqueId.Value)
                    : null;

                // Try to get coordinates from converted LK ADT
                float worldX = placement.WorldX;
                float worldY = placement.WorldY;
                float worldZ = placement.WorldZ;
                float rotX = 0f, rotY = 0f, rotZ = 0f, scale = 1f;
                ushort flags = 0, doodadSet = 0, nameSet = 0;
                
                if (!preferRawCoordinates && uniqueId.HasValue && coordinateLookup.TryGetValue(uniqueId.Value, out var coords))
                {
                    worldX = coords.X;
                    worldY = coords.Y;
                    worldZ = coords.Z;
                    rotX = coords.RotX;
                    rotY = coords.RotY;
                    rotZ = coords.RotZ;
                    scale = coords.Scale;
                    flags = coords.Flags;
                    doodadSet = coords.DoodadSet;
                    nameSet = coords.NameSet;
                }

                // Alpha coordinates mirror horizontally relative to LK; flip X once during export.
                worldX = -worldX;

                var folder = ExtractFolder(placement.AssetPath, 2);
                var category = ExtractCategory(placement.AssetPath);
                var subcategory = ExtractSubcategory(placement.AssetPath);
                var (fileName, fileStem, extension) = ExtractFileParts(placement.AssetPath);

                assets.Add(new PlacementAsset(
                    placement.MapName,
                    placement.TileY,
                    placement.TileX,
                    placementKind,
                    uniqueId,
                    placement.AssetPath,
                    $"{placement.MapName}_{placement.TileX}_{placement.TileY}.adt",
                    worldX,
                    worldY,
                    worldZ,
                    rotX,
                    rotY,
                    rotZ,
                    scale,
                    flags,
                    doodadSet,
                    nameSet,
                    folder,
                    category,
                    subcategory,
                    fileName,
                    fileStem,
                    extension));
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

    private static Dictionary<uint, (float X, float Y, float Z, float RotX, float RotY, float RotZ, float Scale, ushort Flags, ushort DoodadSet, ushort NameSet)> BuildCoordinateLookup(
        string mapName,
        global::AlphaWdtAnalyzer.Core.AdtScanner.Result adtResult,
        string? convertedAdtDir)
    {
        var lookup = new Dictionary<uint, (float, float, float, float, float, float, float, ushort, ushort, ushort)>();
        
        if (string.IsNullOrEmpty(convertedAdtDir) || !Directory.Exists(convertedAdtDir))
            return lookup;

        Console.WriteLine($"[info] Building coordinate lookup from converted LK ADT files...");

        // Get all unique tiles from the scan
        var tiles = adtResult.Tiles.Distinct().ToList();
        int coordsFound = 0;

        foreach (var tile in tiles)
        {
            var adtFileName = $"{mapName}_{tile.X}_{tile.Y}.adt";
            var adtPath = Path.Combine(convertedAdtDir, adtFileName);

            if (!File.Exists(adtPath))
                continue;

            // Read MDDF and MODF from converted LK ADT
            var mddfPlacements = LkAdtReader.ReadMddf(adtPath);
            var modfPlacements = LkAdtReader.ReadModf(adtPath);

            foreach (var placement in mddfPlacements.Concat(modfPlacements))
            {
                var uid = unchecked((uint)placement.UniqueId);
                if (!lookup.ContainsKey(uid))
                {
                    lookup[uid] = (
                        placement.WorldX, placement.WorldY, placement.WorldZ,
                        placement.RotX, placement.RotY, placement.RotZ,
                        placement.Scale, placement.Flags, placement.DoodadSet, placement.NameSet);
                    coordsFound++;
                }
            }
        }

        Console.WriteLine($"[info] Found coordinates for {coordsFound} objects in converted LK ADT files");
        return lookup;
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

    // Local helpers duplicated here to avoid cross-class coupling
    private static string NormalizeAssetPath(string value) => value.Replace('\\', '/').Trim();

    private static string ExtractFolder(string assetPath, int depth)
    {
        var normalized = NormalizeAssetPath(assetPath);
        var segments = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length == 0) return "(root)";
        var actualDepth = Math.Clamp(depth, 1, segments.Length);
        return string.Join('/', segments.Take(actualDepth));
    }

    private static string ExtractCategory(string assetPath)
    {
        var normalized = NormalizeAssetPath(assetPath);
        var segments = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length >= 3) return segments[2];
        if (segments.Length >= 2) return segments[1];
        if (segments.Length >= 1) return segments[0];
        return "(root)";
    }

    private static string ExtractSubcategory(string assetPath)
    {
        var normalized = NormalizeAssetPath(assetPath);
        var segments = normalized.Split('/', StringSplitOptions.RemoveEmptyEntries);
        if (segments.Length >= 4) return segments[3];
        if (segments.Length >= 3) return segments[2];
        return "(none)";
    }

    private static (string FileName, string FileStem, string Extension) ExtractFileParts(string assetPath)
    {
        var normalized = NormalizeAssetPath(assetPath);
        var fileName = Path.GetFileName(normalized);
        var ext = Path.GetExtension(fileName);
        var stem = Path.GetFileNameWithoutExtension(fileName);
        return (fileName, stem, string.IsNullOrEmpty(ext) ? string.Empty : ext);
    }
}
