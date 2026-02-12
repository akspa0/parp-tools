using System;
using System.IO;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Detects ADT file format (pre-Cataclysm single file vs Cataclysm+ split files).
/// </summary>
public static class AdtFormatDetector
{
    /// <summary>
    /// ADT file format types.
    /// </summary>
    public enum AdtFormat
    {
        /// <summary>No valid ADT files found.</summary>
        None,
        
        /// <summary>Pre-Cataclysm: single .adt file per tile.</summary>
        PreCataclysm,
        
        /// <summary>Cataclysm+: split into .adt (root), _obj0.adt, _tex0.adt.</summary>
        Cataclysm
    }

    /// <summary>
    /// Detects the ADT format for a specific tile.
    /// </summary>
    /// <param name="mapDirectory">Directory containing map ADT files (e.g., World\Maps\Azeroth).</param>
    /// <param name="mapName">Map name (e.g., "Azeroth").</param>
    /// <param name="tileX">Tile X coordinate.</param>
    /// <param name="tileY">Tile Y coordinate.</param>
    /// <returns>Detected format.</returns>
    public static AdtFormat DetectFormat(string mapDirectory, string mapName, int tileX, int tileY)
    {
        if (string.IsNullOrWhiteSpace(mapDirectory) || !Directory.Exists(mapDirectory))
            return AdtFormat.None;

        var baseName = $"{mapName}_{tileX}_{tileY}";
        var rootPath = Path.Combine(mapDirectory, $"{baseName}.adt");
        var obj0Path = Path.Combine(mapDirectory, $"{baseName}_obj0.adt");

        // Check for Cataclysm+ split files first
        if (File.Exists(obj0Path))
            return AdtFormat.Cataclysm;

        // Check for pre-Cataclysm single file
        if (File.Exists(rootPath))
            return AdtFormat.PreCataclysm;

        return AdtFormat.None;
    }

    /// <summary>
    /// Enumerates all valid ADT tiles in a map directory.
    /// </summary>
    /// <param name="mapDirectory">Directory containing map ADT files.</param>
    /// <param name="mapName">Map name.</param>
    /// <returns>List of (tileX, tileY, format) tuples.</returns>
    public static List<(int TileX, int TileY, AdtFormat Format)> EnumerateMapTiles(string mapDirectory, string mapName)
    {
        var tiles = new List<(int, int, AdtFormat)>();

        if (!Directory.Exists(mapDirectory))
            return tiles;

        // Scan for ADT files
        var adtFiles = Directory.GetFiles(mapDirectory, "*.adt", SearchOption.TopDirectoryOnly);

        var seenTiles = new HashSet<(int, int)>();

        foreach (var filePath in adtFiles)
        {
            var fileName = Path.GetFileNameWithoutExtension(filePath);
            
            // Skip _obj0, _obj1, _tex0, _tex1, _lod files
            if (fileName.Contains("_obj", StringComparison.OrdinalIgnoreCase) ||
                fileName.Contains("_tex", StringComparison.OrdinalIgnoreCase) ||
                fileName.Contains("_lod", StringComparison.OrdinalIgnoreCase))
                continue;

            // Parse tile coordinates from filename: MapName_X_Y.adt
            var parts = fileName.Split('_');
            if (parts.Length >= 3 &&
                int.TryParse(parts[^2], out var tileX) &&
                int.TryParse(parts[^1], out var tileY))
            {
                if (seenTiles.Add((tileX, tileY)))
                {
                    var format = DetectFormat(mapDirectory, mapName, tileX, tileY);
                    if (format != AdtFormat.None)
                    {
                        tiles.Add((tileX, tileY, format));
                    }
                }
            }
        }

        return tiles.OrderBy(t => t.Item2).ThenBy(t => t.Item1).ToList();
    }
}
