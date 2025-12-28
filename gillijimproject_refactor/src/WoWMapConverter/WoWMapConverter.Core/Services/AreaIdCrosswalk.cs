namespace WoWMapConverter.Core.Services;

/// <summary>
/// Handles AreaID mapping between Alpha WDT and LK 3.3.5 ADT formats.
/// Consumes DBCTool.V2 crosswalk CSVs for strict per-map numeric mapping.
/// </summary>
public class AreaIdCrosswalk
{
    private readonly Dictionary<string, Dictionary<int, int>> _mapCrosswalks = new();
    private readonly Dictionary<int, string> _areaNames = new();

    /// <summary>
    /// Load crosswalk data from DBCTool.V2 compare/v2/ directory.
    /// Each CSV file represents a map's Alpha→LK AreaID mappings.
    /// </summary>
    public void LoadFromDirectory(string crosswalkDir)
    {
        if (!Directory.Exists(crosswalkDir))
            throw new DirectoryNotFoundException($"Crosswalk directory not found: {crosswalkDir}");

        foreach (var csvFile in Directory.GetFiles(crosswalkDir, "*.csv"))
        {
            var mapName = Path.GetFileNameWithoutExtension(csvFile);
            LoadMapCrosswalk(mapName, csvFile);
        }
    }

    /// <summary>
    /// Load a single map's crosswalk CSV.
    /// Expected format: src_areaNumber,tgt_areaID,src_name,tgt_name
    /// </summary>
    public void LoadMapCrosswalk(string mapName, string csvPath)
    {
        if (!File.Exists(csvPath))
            return;

        var mappings = new Dictionary<int, int>();
        var lines = File.ReadAllLines(csvPath);
        
        // Skip header
        foreach (var line in lines.Skip(1))
        {
            if (string.IsNullOrWhiteSpace(line))
                continue;

            var parts = line.Split(',');
            if (parts.Length >= 2 &&
                int.TryParse(parts[0].Trim(), out var srcArea) &&
                int.TryParse(parts[1].Trim(), out var tgtArea))
            {
                mappings[srcArea] = tgtArea;
                
                // Cache area name if available
                if (parts.Length >= 4 && !string.IsNullOrWhiteSpace(parts[3]))
                {
                    _areaNames[tgtArea] = parts[3].Trim().Trim('"');
                }
            }
        }

        _mapCrosswalks[mapName.ToLowerInvariant()] = mappings;
    }

    /// <summary>
    /// Map an Alpha AreaID to LK AreaID for a specific map.
    /// Returns 0 if no mapping exists (strict, no heuristics).
    /// </summary>
    public int MapAreaId(string mapName, int alphaAreaId)
    {
        var key = mapName.ToLowerInvariant();
        if (_mapCrosswalks.TryGetValue(key, out var mappings) &&
            mappings.TryGetValue(alphaAreaId, out var lkAreaId))
        {
            return lkAreaId;
        }
        return 0; // No mapping found
    }

    /// <summary>
    /// Get the area name for an LK AreaID.
    /// </summary>
    public string? GetAreaName(int lkAreaId)
    {
        return _areaNames.TryGetValue(lkAreaId, out var name) ? name : null;
    }

    /// <summary>
    /// Check if a map has crosswalk data loaded.
    /// </summary>
    public bool HasMapData(string mapName)
    {
        return _mapCrosswalks.ContainsKey(mapName.ToLowerInvariant());
    }

    /// <summary>
    /// Get all loaded map names.
    /// </summary>
    public IEnumerable<string> GetLoadedMaps() => _mapCrosswalks.Keys;

    /// <summary>
    /// Get mapping count for a specific map.
    /// </summary>
    public int GetMappingCount(string mapName)
    {
        var key = mapName.ToLowerInvariant();
        return _mapCrosswalks.TryGetValue(key, out var mappings) ? mappings.Count : 0;
    }
}
