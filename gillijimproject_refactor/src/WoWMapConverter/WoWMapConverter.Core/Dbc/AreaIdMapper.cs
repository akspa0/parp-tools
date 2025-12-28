namespace WoWMapConverter.Core.Dbc;

/// <summary>
/// Maps Alpha AreaTable IDs to LK 3.3.5 AreaTable IDs.
/// Integrated from DBCTool.V2 - no external tool needed.
/// </summary>
public class AreaIdMapper
{
    private readonly Dictionary<int, AreaEntry> _alphaAreas = new();
    private readonly Dictionary<int, AreaEntry> _lkAreas = new();
    private readonly Dictionary<int, int> _mapCrosswalk = new();

    public AreaIdMapper()
    {
        // Auto-load embedded defaults
        LoadEmbeddedDefaults();
    }

    // Lookup indices
    private readonly Dictionary<string, int> _lkByName = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<(int map, string name), int> _lkByMapAndName = new();

    /// <summary>
    /// Load Alpha and LK AreaTable DBCs for mapping.
    /// </summary>
    public void LoadDbcs(string alphaAreaTablePath, string lkAreaTablePath, 
                         string? alphaMapPath = null, string? lkMapPath = null)
    {
        // Load Alpha AreaTable
        if (File.Exists(alphaAreaTablePath))
        {
            var alphaDbc = DbcReader.Load(alphaAreaTablePath);
            ParseAlphaAreaTable(alphaDbc);
        }

        // Load LK AreaTable
        if (File.Exists(lkAreaTablePath))
        {
            var lkDbc = DbcReader.Load(lkAreaTablePath);
            ParseLkAreaTable(lkDbc);
            BuildLkIndices();
        }

        // Load Map crosswalks if available
        if (!string.IsNullOrEmpty(alphaMapPath) && File.Exists(alphaMapPath) &&
            !string.IsNullOrEmpty(lkMapPath) && File.Exists(lkMapPath))
        {
            BuildMapCrosswalk(alphaMapPath, lkMapPath);
        }
    }

    /// <summary>
    /// Load from pre-generated crosswalk CSV files.
    /// </summary>
    public void LoadCrosswalkCsv(string csvDir)
    {
        foreach (var csvFile in Directory.GetFiles(csvDir, "*.csv"))
        {
            var lines = File.ReadAllLines(csvFile);
            foreach (var line in lines.Skip(1)) // Skip header
            {
                var parts = line.Split(',');
                if (parts.Length >= 2 &&
                    int.TryParse(parts[0].Trim(), out var alphaId) &&
                    int.TryParse(parts[1].Trim(), out var lkId))
                {
                    _mapCrosswalk[alphaId] = lkId;
                }
            }
        }
    }

    /// <summary>
    /// Map an Alpha area ID to LK area ID.
    /// </summary>
    public int MapAreaId(int alphaAreaId, int? continentHint = null)
    {
        // Direct crosswalk lookup
        if (_mapCrosswalk.TryGetValue(alphaAreaId, out var lkId))
            return lkId;

        // Try by name matching
        if (_alphaAreas.TryGetValue(alphaAreaId, out var alphaArea))
        {
            // Exact name match
            if (_lkByName.TryGetValue(alphaArea.Name, out lkId))
                return lkId;

            // Name + continent match
            if (continentHint.HasValue)
            {
                var lkMap = MapContinent(continentHint.Value);
                if (_lkByMapAndName.TryGetValue((lkMap, NormalizeName(alphaArea.Name)), out lkId))
                    return lkId;
            }

            // Fuzzy match
            lkId = FuzzyMatch(alphaArea.Name);
            if (lkId > 0)
                return lkId;
        }

        return 0; // No mapping found
    }

    /// <summary>
    /// Map an Alpha continent ID to LK map ID.
    /// </summary>
    public int MapContinent(int alphaContinentId)
    {
        // Common mappings
        return alphaContinentId switch
        {
            0 => 0,   // Azeroth (Eastern Kingdoms)
            1 => 1,   // Kalimdor
            _ => alphaContinentId
        };
    }

    /// <summary>
    /// Get the LK area name for an area ID.
    /// </summary>
    public string? GetAreaName(int lkAreaId)
    {
        return _lkAreas.TryGetValue(lkAreaId, out var area) ? area.Name : null;
    }

    private void ParseAlphaAreaTable(DbcReader dbc)
    {
        // Alpha AreaTable structure (0.5.3):
        // Field 0: ID
        // Field 1: ContinentID
        // Field 2: ParentAreaID
        // Field 3: AreaBit
        // Field 4: Flags
        // Field 5: SoundProviderPref
        // Field 6: SoundProviderPrefUnderwater
        // Field 7: MIDIAmbience
        // Field 8: MIDIAmbienceUnderwater
        // Field 9: ZoneIntroMusicTable
        // Field 10: IntroSound
        // Field 11: ExplorationLevel
        // Field 12: AreaName (string offset)
        // Field 13: FactionGroupMask
        // Field 14-17: Liquid stuff

        for (int i = 0; i < dbc.Rows.Count; i++)
        {
            var entry = new AreaEntry
            {
                Id = dbc.GetInt(i, 0),
                ContinentId = dbc.GetInt(i, 1),
                ParentAreaId = dbc.GetInt(i, 2),
                Name = dbc.GetString(i, 12)
            };

            if (entry.Id > 0)
                _alphaAreas[entry.Id] = entry;
        }
    }

    private void ParseLkAreaTable(DbcReader dbc)
    {
        // LK AreaTable structure (3.3.5):
        // Field 0: ID
        // Field 1: ContinentID
        // Field 2: ParentAreaID
        // Field 3: AreaBit
        // Field 4: Flags
        // Field 5-10: Various fields
        // Field 11: AreaName_lang (string offset)

        for (int i = 0; i < dbc.Rows.Count; i++)
        {
            var entry = new AreaEntry
            {
                Id = dbc.GetInt(i, 0),
                ContinentId = dbc.GetInt(i, 1),
                ParentAreaId = dbc.GetInt(i, 2),
                Name = dbc.GetString(i, 11)
            };

            if (entry.Id > 0)
                _lkAreas[entry.Id] = entry;
        }
    }

    private void BuildLkIndices()
    {
        foreach (var (id, area) in _lkAreas)
        {
            var normName = NormalizeName(area.Name);
            if (!string.IsNullOrEmpty(normName))
            {
                _lkByName.TryAdd(normName, id);
                _lkByMapAndName.TryAdd((area.ContinentId, normName), id);
            }
        }
    }

    private void BuildMapCrosswalk(string alphaMapPath, string lkMapPath)
    {
        var alphaMap = DbcReader.Load(alphaMapPath);
        var lkMap = DbcReader.Load(lkMapPath);

        // Build LK map index by directory name
        var lkByDir = new Dictionary<string, int>(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < lkMap.Rows.Count; i++)
        {
            var dir = lkMap.GetString(i, 1); // Directory field
            var id = lkMap.GetInt(i, 0);
            if (!string.IsNullOrEmpty(dir))
                lkByDir.TryAdd(NormalizeName(dir), id);
        }

        // Match Alpha maps to LK
        for (int i = 0; i < alphaMap.Rows.Count; i++)
        {
            var dir = alphaMap.GetString(i, 1);
            var alphaId = alphaMap.GetInt(i, 0);
            var normDir = NormalizeName(dir);
            
            if (!string.IsNullOrEmpty(normDir) && lkByDir.TryGetValue(normDir, out var lkId))
                _mapCrosswalk[alphaId] = lkId;
        }
    }

    private int FuzzyMatch(string name)
    {
        if (string.IsNullOrEmpty(name))
            return 0;

        var normName = NormalizeName(name);
        int bestId = 0;
        int bestDist = int.MaxValue;

        foreach (var (id, area) in _lkAreas)
        {
            var normAreaName = NormalizeName(area.Name);
            var dist = EditDistance(normName, normAreaName);
            
            if (dist < bestDist && dist <= 2) // Max 2 edits
            {
                bestDist = dist;
                bestId = id;
            }
        }

        return bestId;
    }

    private static string NormalizeName(string name)
    {
        if (string.IsNullOrEmpty(name))
            return string.Empty;
        return name.ToLowerInvariant().Replace(" ", "").Replace("'", "").Replace("-", "");
    }

    private static int EditDistance(string a, string b)
    {
        if (string.IsNullOrEmpty(a)) return b?.Length ?? 0;
        if (string.IsNullOrEmpty(b)) return a.Length;

        int n = a.Length, m = b.Length;
        var dp = new int[n + 1, m + 1];
        
        for (int i = 0; i <= n; i++) dp[i, 0] = i;
        for (int j = 0; j <= m; j++) dp[0, j] = j;
        
        for (int i = 1; i <= n; i++)
        {
            for (int j = 1; j <= m; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                dp[i, j] = Math.Min(Math.Min(dp[i - 1, j] + 1, dp[i, j - 1] + 1), dp[i - 1, j - 1] + cost);
            }
        }
        
        return dp[n, m];
    }

    public int AlphaAreaCount => _alphaAreas.Count;
    public int LkAreaCount => _lkAreas.Count;
    public int CrosswalkCount => _mapCrosswalk.Count;

    private void LoadEmbeddedDefaults()
    {
        try
        {
            var assembly = System.Reflection.Assembly.GetExecutingAssembly();
            var resourceName = "WoWMapConverter.Core.Resources.area_crosswalk.csv";
            
            using var stream = assembly.GetManifestResourceStream(resourceName);
            if (stream == null) return;

            using var reader = new StreamReader(stream);
            var headerLine = reader.ReadLine();
            if (string.IsNullOrEmpty(headerLine)) return;

            var headers = headerLine.Split(',');
            int colSrc = Array.IndexOf(headers, "src_areaNumber");
            int colTgt = Array.IndexOf(headers, "tgt_id_335");

            if (colSrc < 0 || colTgt < 0) return;

            while (!reader.EndOfStream)
            {
                var line = reader.ReadLine();
                if (string.IsNullOrEmpty(line)) continue;

                var parts = line.Split(',');
                if (parts.Length > Math.Max(colSrc, colTgt) &&
                    int.TryParse(parts[colSrc], out var srcId) &&
                    int.TryParse(parts[colTgt], out var tgtId) &&
                    srcId > 0 && tgtId > 0)
                {
                    _mapCrosswalk[srcId] = tgtId;
                }
            }
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] Failed to load embedded crosswalk: {ex.Message}");
        }
    }
}

public struct AreaEntry
{
    public int Id;
    public int ContinentId;
    public int ParentAreaId;
    public string Name;
}
