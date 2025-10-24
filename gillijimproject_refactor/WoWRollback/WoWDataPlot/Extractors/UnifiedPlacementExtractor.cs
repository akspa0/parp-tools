using GillijimProject.WowFiles.Alpha;
using WoWDataPlot.Models;
using WoWRollback.Core.Services;

namespace WoWDataPlot.Extractors;

/// <summary>
/// Unified extractor that auto-detects file format and extracts placements.
/// Supports: Alpha WDT, LK WDT, Alpha ADT, LK ADT
/// </summary>
public static class UnifiedPlacementExtractor
{
    public enum FileFormat
    {
        AlphaWdt,
        LkWdt,
        AlphaAdt,
        LkAdt,
        Unknown
    }
    
    public static FileFormat DetectFormat(string filePath)
    {
        var fileName = Path.GetFileName(filePath).ToLowerInvariant();
        var bytes = File.ReadAllBytes(filePath);
        
        // Check if it's a WDT or ADT
        bool isWdt = fileName.EndsWith(".wdt");
        bool isAdt = fileName.EndsWith(".adt") || fileName.EndsWith("_obj0.adt");
        
        if (!isWdt && !isAdt)
            return FileFormat.Unknown;
        
        // Detect Alpha vs LK using reliable WDT marker:
        // - LK WDT contains MPHD (reversed on disk as 'DPHM') and MAIN
        // - Alpha WDT does NOT have MPHD; MAIN may still be present
        if (isWdt)
        {
            // If MPHD exists -> LK WDT; otherwise assume Alpha WDT
            if (HasChunk(bytes, "DPHM"))
                return FileFormat.LkWdt;
            return FileFormat.AlphaWdt;
        }
        else if (isAdt)
        {
            // LK ADT has MHDR, MCIN, MCNK structure
            // Alpha ADT is similar but different versions
            if (HasChunk(bytes, "RDHM")) // MHDR reversed
                return FileFormat.LkAdt;
            // Alpha ADT also has MHDR, so check version or other markers
            return FileFormat.AlphaAdt;
        }
        
        return FileFormat.Unknown;
    }
    
    private static bool HasChunk(byte[] data, string fourCC)
    {
        var search = System.Text.Encoding.ASCII.GetBytes(fourCC);
        for (int i = 0; i < data.Length - 4; i++)
        {
            if (data[i] == search[0] && 
                data[i+1] == search[1] && 
                data[i+2] == search[2] && 
                data[i+3] == search[3])
                return true;
        }
        return false;
    }
    
    public static List<PlacementRecord> Extract(string filePath, IProgress<string>? progress = null)
    {
        var format = DetectFormat(filePath);
        progress?.Report($"Detected format: {format}");
        
        return format switch
        {
            FileFormat.AlphaWdt => ExtractAlphaWdt(filePath, progress),
            FileFormat.LkWdt => ExtractLkWdt(filePath, progress),
            FileFormat.AlphaAdt => ExtractAlphaAdt(filePath, progress),
            FileFormat.LkAdt => ExtractLkAdt(filePath, progress),
            _ => throw new NotSupportedException($"Unknown or unsupported file format: {filePath}")
        };
    }
    
    private static List<PlacementRecord> ExtractAlphaWdt(string wdtPath, IProgress<string>? progress)
    {
        // Use existing AlphaPlacementExtractor
        return AlphaPlacementExtractor.Extract(wdtPath, progress);
    }
    
    private static List<PlacementRecord> ExtractLkWdt(string wdtPath, IProgress<string>? progress)
    {
        progress?.Report("LK WDT support - scanning for ADT files...");
        
        // LK WDT points to ADT files, we need to read them
        var wdtDir = Path.GetDirectoryName(wdtPath) ?? "";
        var mapName = Path.GetFileNameWithoutExtension(wdtPath);
        
        var allPlacements = new List<PlacementRecord>();
        
        // Check for split ADT format (Cata 4.0+): MapName_XX_YY_obj0.adt
        var objAdtFiles = Directory.GetFiles(wdtDir, $"{mapName}_*_obj0.adt", SearchOption.TopDirectoryOnly);
        
        if (objAdtFiles.Length > 0)
        {
            // Split ADT format - read from _obj0.adt files
            progress?.Report($"Found {objAdtFiles.Length} split ADT files (_obj0.adt)");
            
            foreach (var adtFile in objAdtFiles)
            {
                var adtPlacements = ExtractLkAdt(adtFile, null);
                allPlacements.AddRange(adtPlacements);
            }
        }
        else
        {
            // Classic LK format - scan for regular ADT files (exclude _lgt, _occ, etc.)
            var adtFiles = Directory.GetFiles(wdtDir, $"{mapName}_*.adt", SearchOption.TopDirectoryOnly)
                .Where(f => !f.Contains("_obj") && !f.Contains("_tex") && !f.Contains("_lgt") && !f.Contains("_occ"))
                .ToArray();
            
            progress?.Report($"Found {adtFiles.Length} ADT files");
            
            foreach (var adtFile in adtFiles)
            {
                var adtPlacements = ExtractLkAdt(adtFile, null);
                allPlacements.AddRange(adtPlacements);
            }
        }
        
        progress?.Report($"Extracted {allPlacements.Count} placements from ADT files");
        return allPlacements;
    }
    
    private static List<PlacementRecord> ExtractAlphaAdt(string adtPath, IProgress<string>? progress)
    {
        progress?.Report("Alpha ADT single-tile extraction not yet implemented");
        // TODO: Implement Alpha ADT parsing if needed
        return new List<PlacementRecord>();
    }
    
    private static List<PlacementRecord> ExtractLkAdt(string adtPath, IProgress<string>? progress)
    {
        var records = new List<PlacementRecord>();
        
        // Extract tile coordinates from filename: MapName_XX_YY.adt
        var fileName = Path.GetFileNameWithoutExtension(adtPath);
        var parts = fileName.Split('_');
        
        int tileX = -1, tileY = -1;
        if (parts.Length >= 3 && int.TryParse(parts[^2], out tileX) && int.TryParse(parts[^1], out tileY))
        {
            // Valid tile coordinates
        }
        
        // Read M2 placements
        var m2Placements = LkAdtReader.ReadMddf(adtPath);
        foreach (var p in m2Placements)
        {
            records.Add(new PlacementRecord
            {
                Type = "M2",
                UniqueId = (uint)p.UniqueId,
                X = p.WorldX,
                Y = p.WorldY,
                Z = p.WorldZ,
                TileX = tileX,
                TileY = tileY,
                Version = "LK"
            });
        }
        
        // Read WMO placements
        var wmoPlacements = LkAdtReader.ReadModf(adtPath);
        foreach (var p in wmoPlacements)
        {
            records.Add(new PlacementRecord
            {
                Type = "WMO",
                UniqueId = (uint)p.UniqueId,
                X = p.WorldX,
                Y = p.WorldY,
                Z = p.WorldZ,
                TileX = tileX,
                TileY = tileY,
                Version = "LK"
            });
        }
        
        return records;
    }
}
