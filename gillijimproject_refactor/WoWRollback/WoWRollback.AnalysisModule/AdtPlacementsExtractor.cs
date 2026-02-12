using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using Warcraft.NET.Files.ADT.Terrain.Wotlk;
using Warcraft.NET.Files.ADT.TerrainObject.Zero;

namespace WoWRollback.AnalysisModule;

/// <summary>
/// Extracts object placements (M2 and WMO) from ADT files for UniqueID analysis.
/// Supports both pre-Cataclysm single files and Cataclysm+ split files.
/// </summary>
public sealed class AdtPlacementsExtractor
{
    /// <summary>
    /// Extracts all object placements from a map's ADT tiles.
    /// </summary>
    /// <param name="mapDirectory">Directory containing ADT files (e.g., World\Maps\Azeroth).</param>
    /// <param name="mapName">Map name.</param>
    /// <param name="outputCsvPath">Output CSV file path.</param>
    /// <returns>Result with statistics.</returns>
    public PlacementsExtractionResult Extract(string mapDirectory, string mapName, string outputCsvPath)
    {
        try
        {
            if (!Directory.Exists(mapDirectory))
            {
                return new PlacementsExtractionResult(
                    Success: false,
                    M2Count: 0,
                    WmoCount: 0,
                    TilesProcessed: 0,
                    ErrorMessage: $"Map directory not found: {mapDirectory}");
            }

            var tiles = AdtFormatDetector.EnumerateMapTiles(mapDirectory, mapName);
            if (tiles.Count == 0)
            {
                return new PlacementsExtractionResult(
                    Success: false,
                    M2Count: 0,
                    WmoCount: 0,
                    TilesProcessed: 0,
                    ErrorMessage: $"No valid ADT tiles found in: {mapDirectory}");
            }

            var csv = new StringBuilder();
            csv.AppendLine("map,tile_x,tile_y,type,asset_path,unique_id,world_x,world_y,world_z,rot_x,rot_y,rot_z,scale,doodad_set,name_set,source_file");

            int m2Count = 0;
            int wmoCount = 0;
            int tilesProcessed = 0;
            int tile00Count = 0;

            Console.WriteLine($"[AdtPlacementsExtractor] Processing {tiles.Count} tiles...");

            foreach (var (tileX, tileY, format) in tiles)
            {
                try
                {
                    var sourceFile = format == AdtFormatDetector.AdtFormat.Cataclysm
                        ? $"{mapName}_{tileX}_{tileY}_obj0.adt"
                        : $"{mapName}_{tileX}_{tileY}.adt";
                    
                    var placements = ExtractTilePlacements(mapDirectory, mapName, tileX, tileY, format);
                    
                    if (tileX == 0 && tileY == 0)
                    {
                        Console.WriteLine($"[AdtPlacementsExtractor] Tile (0,0): Found {placements.Count} placements in {sourceFile}");
                        tile00Count = placements.Count;
                    }
                    else if (placements.Count > 0 && tilesProcessed < 5)
                    {
                        Console.WriteLine($"[AdtPlacementsExtractor] Tile ({tileX},{tileY}): Found {placements.Count} placements in {sourceFile}");
                    }
                    
                    foreach (var p in placements)
                    {
                        csv.AppendLine(FormatPlacementCsv(p, sourceFile));
                        
                        if (p.Type == "M2")
                            m2Count++;
                        else
                            wmoCount++;
                    }

                    tilesProcessed++;
                    
                    if (tilesProcessed % 10 == 0)
                    {
                        Console.WriteLine($"[AdtPlacementsExtractor] Progress: {tilesProcessed}/{tiles.Count} tiles, {m2Count + wmoCount} placements");
                    }
                }
                catch (Exception ex)
                {
                    Console.WriteLine($"[AdtPlacementsExtractor] Warning: Failed to process tile ({tileX},{tileY}): {ex.Message}");
                    // Continue processing other tiles
                }
            }
            
            Console.WriteLine($"[AdtPlacementsExtractor] Summary: Tile (0,0) had {tile00Count} placements, Total tiles={tilesProcessed}");

            Directory.CreateDirectory(Path.GetDirectoryName(outputCsvPath)!);
            File.WriteAllText(outputCsvPath, csv.ToString());

            return new PlacementsExtractionResult(
                Success: true,
                M2Count: m2Count,
                WmoCount: wmoCount,
                TilesProcessed: tilesProcessed,
                ErrorMessage: null);
        }
        catch (Exception ex)
        {
            return new PlacementsExtractionResult(
                Success: false,
                M2Count: 0,
                WmoCount: 0,
                TilesProcessed: 0,
                ErrorMessage: $"Extraction failed: {ex.Message}");
        }
    }

    private List<PlacementRecord> ExtractTilePlacements(
        string mapDirectory,
        string mapName,
        int tileX,
        int tileY,
        AdtFormatDetector.AdtFormat format)
    {
        var placements = new List<PlacementRecord>();

        // CRITICAL: MDDF/MODF Position fields use INCONSISTENT coordinate systems!
        // - Some tiles have tile-local coords (0-533 range)
        // - Some tiles have world coords (thousands, can be negative for SE quadrant)
        // - We extract RAW values and trust tile assignment from filename (tileX, tileY)
        // - Coordinate validation is done in OverlayBuilder (currently disabled due to this issue)

        if (format == AdtFormatDetector.AdtFormat.Cataclysm)
        {
            // Cataclysm+ split files: read from _obj0.adt
            var obj0Path = Path.Combine(mapDirectory, $"{mapName}_{tileX}_{tileY}_obj0.adt");
            if (File.Exists(obj0Path))
            {
                var objData = File.ReadAllBytes(obj0Path);
                var objFile = new TerrainObjectZero(objData);

                // Build model path lookup tables
                var m2Paths = BuildM2PathLookup(objFile.Models, objFile.ModelIndices);
                var wmoPaths = BuildWmoPathLookup(objFile.WorldModelObjects, objFile.WorldModelObjectIndices);

                // Extract M2 placements from MDDF
                if (objFile.ModelPlacementInfo?.MDDFEntries != null)
                {
                    foreach (var m2 in objFile.ModelPlacementInfo.MDDFEntries)
                    {
                        var assetPath = m2Paths.TryGetValue(m2.NameId, out var path) 
                            ? path 
                            : $"<NameId:{m2.NameId}>";

                        placements.Add(new PlacementRecord(
                            Map: mapName,
                            TileX: tileX,
                            TileY: tileY,
                            Type: "M2",
                            AssetPath: assetPath,
                            UniqueId: (int)m2.UniqueID,
                            WorldX: m2.Position.X,
                            WorldY: m2.Position.Y,
                            WorldZ: m2.Position.Z,
                            RotX: m2.Rotation.Pitch,
                            RotY: m2.Rotation.Yaw,
                            RotZ: m2.Rotation.Roll,
                            Scale: m2.ScalingFactor,
                            DoodadSet: null,
                            NameSet: null));
                    }
                }

                // Extract WMO placements from MODF
                if (objFile.WorldModelObjectPlacementInfo?.MODFEntries != null)
                {
                    foreach (var wmo in objFile.WorldModelObjectPlacementInfo.MODFEntries)
                    {
                        var assetPath = wmoPaths.TryGetValue(wmo.NameId, out var path)
                            ? path
                            : $"<NameId:{wmo.NameId}>";

                        placements.Add(new PlacementRecord(
                            Map: mapName,
                            TileX: tileX,
                            TileY: tileY,
                            Type: "WMO",
                            AssetPath: assetPath,
                            UniqueId: wmo.UniqueId,
                            WorldX: wmo.Position.X,
                            WorldY: wmo.Position.Y,
                            WorldZ: wmo.Position.Z,
                            RotX: wmo.Rotation.Pitch,
                            RotY: wmo.Rotation.Yaw,
                            RotZ: wmo.Rotation.Roll,
                            Scale: wmo.Scale,
                            DoodadSet: wmo.DoodadSet,
                            NameSet: wmo.NameSet));
                    }
                }
            }
        }
        else if (format == AdtFormatDetector.AdtFormat.PreCataclysm)
        {
            // Pre-Cataclysm single file: read from .adt
            var adtPath = Path.Combine(mapDirectory, $"{mapName}_{tileX}_{tileY}.adt");
            if (File.Exists(adtPath))
            {
                var adtData = File.ReadAllBytes(adtPath);
                var terrain = new Terrain(adtData);

                // Build model path lookup tables
                var m2Paths = BuildM2PathLookup(terrain.Models, terrain.ModelIndices);
                var wmoPaths = BuildWmoPathLookup(terrain.WorldModelObjects, terrain.WorldModelObjectIndices);

                // Extract M2 placements from MDDF
                if (terrain.ModelPlacementInfo?.MDDFEntries != null)
                {
                    foreach (var m2 in terrain.ModelPlacementInfo.MDDFEntries)
                    {
                        var assetPath = m2Paths.TryGetValue(m2.NameId, out var path)
                            ? path
                            : $"<NameId:{m2.NameId}>";

                        placements.Add(new PlacementRecord(
                            Map: mapName,
                            TileX: tileX,
                            TileY: tileY,
                            Type: "M2",
                            AssetPath: assetPath,
                            UniqueId: (int)m2.UniqueID,
                            WorldX: m2.Position.X,
                            WorldY: m2.Position.Y,
                            WorldZ: m2.Position.Z,
                            RotX: m2.Rotation.Pitch,
                            RotY: m2.Rotation.Yaw,
                            RotZ: m2.Rotation.Roll,
                            Scale: m2.ScalingFactor,
                            DoodadSet: null,
                            NameSet: null));
                    }
                }

                // Extract WMO placements from MODF
                if (terrain.WorldModelObjectPlacementInfo?.MODFEntries != null)
                {
                    foreach (var wmo in terrain.WorldModelObjectPlacementInfo.MODFEntries)
                    {
                        var assetPath = wmoPaths.TryGetValue(wmo.NameId, out var path)
                            ? path
                            : $"<NameId:{wmo.NameId}>";

                        placements.Add(new PlacementRecord(
                            Map: mapName,
                            TileX: tileX,
                            TileY: tileY,
                            Type: "WMO",
                            AssetPath: assetPath,
                            UniqueId: wmo.UniqueId,
                            WorldX: wmo.Position.X,
                            WorldY: wmo.Position.Y,
                            WorldZ: wmo.Position.Z,
                            RotX: wmo.Rotation.Pitch,
                            RotY: wmo.Rotation.Yaw,
                            RotZ: wmo.Rotation.Roll,
                            Scale: wmo.Scale,
                            DoodadSet: wmo.DoodadSet,
                            NameSet: wmo.NameSet));
                    }
                }
            }
        }

        return placements;
    }

    /// <summary>
    /// Builds a lookup from MMID indices to actual M2 model paths from MMDX.
    /// </summary>
    private static Dictionary<uint, string> BuildM2PathLookup(Warcraft.NET.Files.ADT.Chunks.MMDX? mmdx, Warcraft.NET.Files.ADT.Chunks.MMID? mmid)
    {
        var lookup = new Dictionary<uint, string>();

        if (mmdx == null || mmid == null || mmdx.Filenames.Count == 0 || mmid.ModelFilenameOffsets.Count == 0)
            return lookup;

        // MMID contains offsets into the MMDX string block
        // For simplicity, we use sequential indices since MDDF.NameId references the index position
        for (int i = 0; i < mmdx.Filenames.Count; i++)
        {
            lookup[(uint)i] = mmdx.Filenames[i];
        }

        return lookup;
    }

    /// <summary>
    /// Builds a lookup from MWID indices to actual WMO paths from MWMO.
    /// </summary>
    private static Dictionary<uint, string> BuildWmoPathLookup(Warcraft.NET.Files.ADT.Chunks.MWMO? mwmo, Warcraft.NET.Files.ADT.Chunks.MWID? mwid)
    {
        var lookup = new Dictionary<uint, string>();

        if (mwmo == null || mwid == null || mwmo.Filenames.Count == 0 || mwid.ModelFilenameOffsets.Count == 0)
            return lookup;

        // MWID contains offsets into the MWMO string block
        // For simplicity, we use sequential indices since MODF.NameId references the index position
        for (int i = 0; i < mwmo.Filenames.Count; i++)
        {
            lookup[(uint)i] = mwmo.Filenames[i];
        }

        return lookup;
    }

    private static string FormatPlacementCsv(PlacementRecord p, string sourceFile)
    {
        return string.Join(",",
            Csv(p.Map),
            p.TileX.ToString(CultureInfo.InvariantCulture),
            p.TileY.ToString(CultureInfo.InvariantCulture),
            Csv(p.Type),
            Csv(p.AssetPath),
            p.UniqueId.ToString(CultureInfo.InvariantCulture),
            p.WorldX.ToString("F3", CultureInfo.InvariantCulture),
            p.WorldY.ToString("F3", CultureInfo.InvariantCulture),
            p.WorldZ.ToString("F3", CultureInfo.InvariantCulture),
            p.RotX.ToString("F6", CultureInfo.InvariantCulture),
            p.RotY.ToString("F6", CultureInfo.InvariantCulture),
            p.RotZ.ToString("F6", CultureInfo.InvariantCulture),
            p.Scale.ToString(CultureInfo.InvariantCulture),
            p.DoodadSet?.ToString(CultureInfo.InvariantCulture) ?? string.Empty,
            p.NameSet?.ToString(CultureInfo.InvariantCulture) ?? string.Empty,
            Csv(sourceFile));
    }

    private static string Csv(string value)
    {
        if (string.IsNullOrEmpty(value))
            return string.Empty;

        if (value.Contains(',') || value.Contains('"') || value.Contains('\n'))
            return $"\"{value.Replace("\"", "\"\"")}\"";

        return value;
    }
}

/// <summary>
/// Placement record for CSV export.
/// </summary>
internal record PlacementRecord(
    string Map,
    int TileX,
    int TileY,
    string Type,
    string AssetPath,
    int UniqueId,
    float WorldX,
    float WorldY,
    float WorldZ,
    float RotX,
    float RotY,
    float RotZ,
    ushort Scale,
    ushort? DoodadSet,
    ushort? NameSet);

/// <summary>
/// Result of placements extraction.
/// </summary>
public record PlacementsExtractionResult(
    bool Success,
    int M2Count,
    int WmoCount,
    int TilesProcessed,
    string? ErrorMessage);
