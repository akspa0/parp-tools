using GillijimProject.WowFiles.Alpha;
using WoWDataPlot.Helpers;
using WoWDataPlot.Models;
using U = GillijimProject.Utilities.Utilities;

namespace WoWDataPlot.Extractors;

/// <summary>
/// Extracts M2/WMO placement data from Alpha v18 WDT files using existing gillijimproject parsers.
/// </summary>
public static class AlphaPlacementExtractor
{
    /// <summary>
    /// Extract all M2 and WMO placements from an Alpha WDT file.
    /// Uses existing gillijimproject WdtAlpha/AdtAlpha parsers.
    /// </summary>
    /// <param name="wdtPath">Path to Alpha .wdt file</param>
    /// <param name="progress">Optional progress callback</param>
    /// <returns>List of placement records</returns>
    public static List<PlacementRecord> Extract(string wdtPath, IProgress<string>? progress = null)
    {
        if (!File.Exists(wdtPath))
        {
            throw new FileNotFoundException($"WDT file not found: {wdtPath}");
        }

        progress?.Report($"Reading Alpha WDT: {Path.GetFileName(wdtPath)}");
        
        var records = new List<PlacementRecord>();
        
        try
        {
            // Parse Alpha WDT using existing gillijimproject code
            var wdtAlpha = new WdtAlpha(wdtPath);
            
            // Get M2 and WMO name lists
            var mdnmFiles = wdtAlpha.GetAdtOffsetsInMain(); // This gets tile info
            var existingAdts = wdtAlpha.GetExistingAdtsNumbers();
            
            progress?.Report($"Found {existingAdts.Count} existing ADT tiles to process");
            
            // Extract placements from each tile
            foreach (var adtNum in existingAdts)
            {
                try
                {
                    var offsets = wdtAlpha.GetAdtOffsetsInMain();
                    int offset = offsets[adtNum];
                    
                    if (offset == 0) continue;
                    
                    // Parse this ADT tile
                    var adtAlpha = new AdtAlpha(wdtPath, offset, adtNum);
                    int x = adtAlpha.GetXCoord();
                    int y = adtAlpha.GetYCoord();
                    
                    // Extract M2 placements from MDDF (36-byte entries)
                    var m2Placements = ExtractMddfPlacements(adtAlpha);
                    foreach (var placement in m2Placements)
                    {
                        records.Add(new PlacementRecord
                        {
                            X = placement.x,
                            Y = placement.y,
                            Z = placement.z,
                            UniqueId = placement.uniqueId,
                            Type = "M2",
                            Name = "",
                            AreaId = 0,
                            TileX = x,
                            TileY = y,
                            Version = "Alpha"
                        });
                    }
                    
                    // Extract WMO placements from MODF (64-byte entries)
                    var wmoPlacement = ExtractModfPlacements(adtAlpha);
                    foreach (var placement in wmoPlacement)
                    {
                        records.Add(new PlacementRecord
                        {
                            X = placement.x,
                            Y = placement.y,
                            Z = placement.z,
                            UniqueId = placement.uniqueId,
                            Type = "WMO",
                            Name = "",
                            AreaId = 0,
                            TileX = x,
                            TileY = y,
                            Version = "Alpha"
                        });
                    }
                    
                    if ((records.Count % 1000) == 0 && records.Count > 0)
                    {
                        progress?.Report($"Extracted {records.Count} placements so far...");
                    }
                }
                catch (Exception ex)
                {
                    progress?.Report($"Warning: Failed to process ADT tile {adtNum}: {ex.Message}");
                }
            }
            
            progress?.Report($"Extracted {records.Count} total placements");
            
            if (records.Count == 0)
            {
                progress?.Report("WARNING: No placements found. This WDT might be empty or terrain-only.");
            }
            
            return records;
        }
        catch (Exception ex)
        {
            throw new InvalidOperationException($"Failed to extract placements from {wdtPath}: {ex.Message}", ex);
        }
    }
    
    private static List<(float x, float y, float z, uint uniqueId)> ExtractMddfPlacements(AdtAlpha adtAlpha)
    {
        // Use reflection to access private _mddf field
        var mddfField = typeof(AdtAlpha).GetField("_mddf", 
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        
        if (mddfField == null)
            return new List<(float, float, float, uint)>();
        
        var mddf = mddfField.GetValue(adtAlpha) as GillijimProject.WowFiles.Mddf;
        if (mddf == null || mddf.Data == null || mddf.Data.Length == 0)
            return new List<(float, float, float, uint)>();
        
        const int entrySize = 36;
        var placements = new List<(float x, float y, float z, uint uniqueId)>();
        
        for (int start = 0; start + entrySize <= mddf.Data.Length; start += entrySize)
        {
            // MDDF format: nameId(4), uniqueId(4), pos(12=X,Z,Y), rot(12), scale(2), flags(2)
            // NOTE: Position is stored as X, Z, Y (not X, Y, Z!) - verified 4 weeks ago
            uint uniqueId = BitConverter.ToUInt32(mddf.Data, start + 4);
            float x = BitConverter.ToSingle(mddf.Data, start + 8);   // offset +8: X
            float z = BitConverter.ToSingle(mddf.Data, start + 12);  // offset +12: Z (height)
            float y = BitConverter.ToSingle(mddf.Data, start + 16);  // offset +16: Y
            
            placements.Add((x, y, z, uniqueId));
        }
        
        return placements;
    }
    
    private static List<(float x, float y, float z, uint uniqueId)> ExtractModfPlacements(AdtAlpha adtAlpha)
    {
        // Use reflection to access private _modf field
        var modfField = typeof(AdtAlpha).GetField("_modf", 
            System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Instance);
        
        if (modfField == null)
            return new List<(float, float, float, uint)>();
        
        var modf = modfField.GetValue(adtAlpha) as GillijimProject.WowFiles.Modf;
        if (modf == null || modf.Data == null || modf.Data.Length == 0)
            return new List<(float, float, float, uint)>();
        
        const int entrySize = 64;
        var placements = new List<(float x, float y, float z, uint uniqueId)>();
        
        for (int start = 0; start + entrySize <= modf.Data.Length; start += entrySize)
        {
            // MODF format: nameId(4), uniqueId(4), pos(12=X,Z,Y), rot(12), bounds(24), flags(2), doodadSet(2), nameSet(2), scale(2)
            // NOTE: Position is stored as X, Z, Y (not X, Y, Z!) - verified 4 weeks ago
            uint uniqueId = BitConverter.ToUInt32(modf.Data, start + 4);
            float x = BitConverter.ToSingle(modf.Data, start + 8);   // offset +8: X
            float z = BitConverter.ToSingle(modf.Data, start + 12);  // offset +12: Z (height)
            float y = BitConverter.ToSingle(modf.Data, start + 16);  // offset +16: Y
            
            placements.Add((x, y, z, uniqueId));
        }
        
        return placements;
    }
    
    /// <summary>
    /// Calculate tile X/Y indices from world coordinates using proper WoW coordinate system.
    /// Formula from wowdev wiki: floor((32 - (axis / 533.33333)))
    /// </summary>
    private static PlacementRecord CalculateTileIndices(PlacementRecord record)
    {
        // Use the proper WoW coordinate transformation
        var (tileX, tileY) = CoordinateTransform.WorldToTile(record.X, record.Y);
        
        // Return new record with updated tile indices
        return record with { TileX = tileX, TileY = tileY };
    }
}
