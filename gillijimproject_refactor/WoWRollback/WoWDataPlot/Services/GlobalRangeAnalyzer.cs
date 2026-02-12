using System.Text.Json;
using WoWDataPlot.Extractors;

namespace WoWDataPlot.Services;

/// <summary>
/// Analyzes UniqueID ranges across all maps in a WoW client.
/// </summary>
public static class GlobalRangeAnalyzer
{
    public static GlobalRangeResult AnalyzeClient(string clientRoot, IProgress<string>? progress = null)
    {
        progress?.Report("Scanning for WDT files...");
        
        var mapsDir = Path.Combine(clientRoot, "World", "Maps");
        if (!Directory.Exists(mapsDir))
        {
            throw new DirectoryNotFoundException($"Maps directory not found: {mapsDir}");
        }
        
        var wdtFiles = Directory.GetFiles(mapsDir, "*.wdt", SearchOption.AllDirectories)
            .Where(f => !f.Contains("_occ.wdt") && !f.Contains("_lgt.wdt"))
            .ToList();
        
        progress?.Report($"Found {wdtFiles.Count} WDT files");
        
        var mapRanges = new Dictionary<string, MapRangeInfo>();
        uint globalMin = uint.MaxValue;
        uint globalMax = uint.MinValue;
        int totalPlacements = 0;
        
        foreach (var wdtPath in wdtFiles)
        {
            try
            {
                var mapName = Path.GetFileNameWithoutExtension(wdtPath);
                progress?.Report($"Analyzing {mapName}...");
                
                var records = UnifiedPlacementExtractor.Extract(wdtPath, null);
                
                if (records.Count > 0)
                {
                    uint mapMin = records.Min(r => r.UniqueId);
                    uint mapMax = records.Max(r => r.UniqueId);
                    
                    mapRanges[mapName] = new MapRangeInfo
                    {
                        MinUniqueId = mapMin,
                        MaxUniqueId = mapMax,
                        PlacementCount = records.Count,
                        M2Count = records.Count(r => r.Type == "M2"),
                        WmoCount = records.Count(r => r.Type == "WMO"),
                        WdtPath = wdtPath
                    };
                    
                    globalMin = Math.Min(globalMin, mapMin);
                    globalMax = Math.Max(globalMax, mapMax);
                    totalPlacements += records.Count;
                    
                    progress?.Report($"  {mapName}: {records.Count:N0} placements, UniqueID {mapMin:N0}-{mapMax:N0}");
                }
                else
                {
                    progress?.Report($"  {mapName}: No placements found");
                }
            }
            catch (Exception ex)
            {
                progress?.Report($"  ERROR processing {Path.GetFileName(wdtPath)}: {ex.Message}");
            }
        }
        
        if (mapRanges.Count == 0)
        {
            throw new InvalidOperationException("No maps with placements found!");
        }
        
        return new GlobalRangeResult
        {
            GlobalMinUniqueId = globalMin,
            GlobalMaxUniqueId = globalMax,
            TotalPlacementCount = totalPlacements,
            Maps = mapRanges,
            AnalysisDate = DateTime.UtcNow,
            ClientRoot = clientRoot
        };
    }
    
    public static void SaveToJson(GlobalRangeResult result, string outputPath)
    {
        var options = new JsonSerializerOptions
        {
            WriteIndented = true,
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };
        
        var json = JsonSerializer.Serialize(result, options);
        File.WriteAllText(outputPath, json);
    }
    
    public static GlobalRangeResult LoadFromJson(string jsonPath)
    {
        var json = File.ReadAllText(jsonPath);
        var options = new JsonSerializerOptions
        {
            PropertyNamingPolicy = JsonNamingPolicy.CamelCase
        };
        
        return JsonSerializer.Deserialize<GlobalRangeResult>(json, options)
            ?? throw new InvalidDataException("Failed to deserialize global range data");
    }
}

public class GlobalRangeResult
{
    public uint GlobalMinUniqueId { get; set; }
    public uint GlobalMaxUniqueId { get; set; }
    public int TotalPlacementCount { get; set; }
    public Dictionary<string, MapRangeInfo> Maps { get; set; } = new();
    public DateTime AnalysisDate { get; set; }
    public string ClientRoot { get; set; } = "";
}

public class MapRangeInfo
{
    public uint MinUniqueId { get; set; }
    public uint MaxUniqueId { get; set; }
    public int PlacementCount { get; set; }
    public int M2Count { get; set; }
    public int WmoCount { get; set; }
    public string WdtPath { get; set; } = "";
}
