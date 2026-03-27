namespace WowViewer.Core.IO.Dbc;

public sealed class AreaIdMapper
{
    private const string EmbeddedCrosswalkResourceName = "WowViewer.Core.IO.Resources.area_crosswalk.csv";

    private readonly Dictionary<int, AreaEntry> _alphaAreas = new();
    private readonly Dictionary<int, AreaEntry> _lkAreas = new();
    private readonly Dictionary<int, int> _directAreaCrosswalk = new();
    private readonly Dictionary<int, int> _mapIdCrosswalk = new();
    private readonly Dictionary<string, int> _lkByName = new(StringComparer.OrdinalIgnoreCase);
    private readonly Dictionary<(int map, string name), int> _lkByMapAndName = new();

    public AreaIdMapper()
    {
        LoadEmbeddedDefaults();
    }

    public void LoadDbcs(string alphaAreaTablePath, string lkAreaTablePath, string? alphaMapPath = null, string? lkMapPath = null)
    {
        _alphaAreas.Clear();
        _lkAreas.Clear();
        _lkByName.Clear();
        _lkByMapAndName.Clear();
        _mapIdCrosswalk.Clear();

        if (File.Exists(alphaAreaTablePath))
        {
            DbcReader alphaDbc = DbcReader.Load(alphaAreaTablePath);
            ParseAlphaAreaTable(alphaDbc);
        }

        if (File.Exists(lkAreaTablePath))
        {
            DbcReader lkDbc = DbcReader.Load(lkAreaTablePath);
            ParseLkAreaTable(lkDbc);
            BuildLkIndices();
        }

        if (!string.IsNullOrEmpty(alphaMapPath) && File.Exists(alphaMapPath) &&
            !string.IsNullOrEmpty(lkMapPath) && File.Exists(lkMapPath))
        {
            BuildMapCrosswalk(alphaMapPath, lkMapPath);
        }
    }

    public bool TryAutoLoadFromTestData()
    {
        const string alphaDbcPath = "test_data/0.5.3/tree/DBFilesClient/AreaTable.dbc";
        const string lkDbcPath = "test_data/3.3.5/tree/DBFilesClient/AreaTable.dbc";
        const string alphaMapPath = "test_data/0.5.3/tree/DBFilesClient/Map.dbc";
        const string lkMapPath = "test_data/3.3.5/tree/DBFilesClient/Map.dbc";

        List<string> searchDirectories = [Directory.GetCurrentDirectory()];

        string? assemblyDirectory = Path.GetDirectoryName(typeof(AreaIdMapper).Assembly.Location);
        if (!string.IsNullOrEmpty(assemblyDirectory))
        {
            searchDirectories.Add(assemblyDirectory);
        }

        foreach (string startDirectory in searchDirectories)
        {
            DirectoryInfo? current = new(startDirectory);
            for (int i = 0; i < 6 && current != null; i++)
            {
                string alphaPath = Path.Combine(current.FullName, alphaDbcPath);
                string lkPath = Path.Combine(current.FullName, lkDbcPath);

                if (File.Exists(alphaPath) && File.Exists(lkPath))
                {
                    string alphaMap = Path.Combine(current.FullName, alphaMapPath);
                    string lkMap = Path.Combine(current.FullName, lkMapPath);

                    LoadDbcs(
                        alphaPath,
                        lkPath,
                        File.Exists(alphaMap) ? alphaMap : null,
                        File.Exists(lkMap) ? lkMap : null);

                    Console.WriteLine($"[INFO] Auto-loaded AreaTable from test_data: {AlphaAreaCount} Alpha, {LkAreaCount} LK areas");
                    return true;
                }

                current = current.Parent;
            }
        }

        return false;
    }

    public void LoadCrosswalkCsv(string csvDirectory)
    {
        foreach (string csvFile in Directory.GetFiles(csvDirectory, "*.csv"))
        {
            using StreamReader reader = new(csvFile);
            LoadCrosswalk(reader);
        }
    }

    public int MapAreaId(int alphaAreaId, int? continentHint = null)
    {
        if (_directAreaCrosswalk.TryGetValue(alphaAreaId, out int lkAreaId))
        {
            return lkAreaId;
        }

        if (!_alphaAreas.TryGetValue(alphaAreaId, out AreaEntry alphaArea))
        {
            return 0;
        }

        string normalizedName = NormalizeName(alphaArea.Name);
        if (string.IsNullOrEmpty(normalizedName))
        {
            return 0;
        }

        if (continentHint.HasValue)
        {
            int lkMapId = MapContinent(continentHint.Value);
            if (_lkByMapAndName.TryGetValue((lkMapId, normalizedName), out lkAreaId))
            {
                return lkAreaId;
            }
        }

        if (_lkByName.TryGetValue(normalizedName, out lkAreaId))
        {
            return lkAreaId;
        }

        return FuzzyMatch(normalizedName, continentHint.HasValue ? MapContinent(continentHint.Value) : null);
    }

    public int MapContinent(int alphaContinentId)
    {
        if (_mapIdCrosswalk.TryGetValue(alphaContinentId, out int lkMapId))
        {
            return lkMapId;
        }

        return alphaContinentId switch
        {
            0 => 0,
            1 => 1,
            _ => alphaContinentId,
        };
    }

    public string? GetAreaName(int lkAreaId)
    {
        return _lkAreas.TryGetValue(lkAreaId, out AreaEntry area) ? area.Name : null;
    }

    public int AlphaAreaCount => _alphaAreas.Count;

    public int LkAreaCount => _lkAreas.Count;

    public int CrosswalkCount => _directAreaCrosswalk.Count;

    private void ParseAlphaAreaTable(DbcReader dbc)
    {
        for (int i = 0; i < dbc.Rows.Count; i++)
        {
            AreaEntry entry = new(
                dbc.GetInt(i, 0),
                dbc.GetInt(i, 1),
                dbc.GetInt(i, 2),
                dbc.GetString(i, 12));

            if (entry.Id > 0)
            {
                _alphaAreas[entry.Id] = entry;
            }
        }
    }

    private void ParseLkAreaTable(DbcReader dbc)
    {
        for (int i = 0; i < dbc.Rows.Count; i++)
        {
            AreaEntry entry = new(
                dbc.GetInt(i, 0),
                dbc.GetInt(i, 1),
                dbc.GetInt(i, 2),
                dbc.GetString(i, 11));

            if (entry.Id > 0)
            {
                _lkAreas[entry.Id] = entry;
            }
        }
    }

    private void BuildLkIndices()
    {
        foreach ((int id, AreaEntry area) in _lkAreas)
        {
            string normalizedName = NormalizeName(area.Name);
            if (string.IsNullOrEmpty(normalizedName))
            {
                continue;
            }

            _lkByName.TryAdd(normalizedName, id);
            _lkByMapAndName.TryAdd((area.ContinentId, normalizedName), id);
        }
    }

    private void BuildMapCrosswalk(string alphaMapPath, string lkMapPath)
    {
        DbcReader alphaMap = DbcReader.Load(alphaMapPath);
        DbcReader lkMap = DbcReader.Load(lkMapPath);

        Dictionary<string, int> lkByDirectory = new(StringComparer.OrdinalIgnoreCase);
        for (int i = 0; i < lkMap.Rows.Count; i++)
        {
            string directory = NormalizeName(lkMap.GetString(i, 1));
            int id = lkMap.GetInt(i, 0);
            if (!string.IsNullOrEmpty(directory))
            {
                lkByDirectory.TryAdd(directory, id);
            }
        }

        for (int i = 0; i < alphaMap.Rows.Count; i++)
        {
            string directory = NormalizeName(alphaMap.GetString(i, 1));
            int alphaId = alphaMap.GetInt(i, 0);
            if (!string.IsNullOrEmpty(directory) && lkByDirectory.TryGetValue(directory, out int lkId))
            {
                _mapIdCrosswalk[alphaId] = lkId;
            }
        }
    }

    private int FuzzyMatch(string normalizedName, int? continentHint)
    {
        int bestId = 0;
        int bestDistance = int.MaxValue;

        foreach ((int id, AreaEntry area) in _lkAreas)
        {
            if (continentHint.HasValue && area.ContinentId != continentHint.Value)
            {
                continue;
            }

            int distance = EditDistance(normalizedName, NormalizeName(area.Name));
            if (distance < bestDistance && distance <= 2)
            {
                bestDistance = distance;
                bestId = id;
            }
        }

        return bestId;
    }

    private void LoadEmbeddedDefaults()
    {
        try
        {
            using Stream? stream = typeof(AreaIdMapper).Assembly.GetManifestResourceStream(EmbeddedCrosswalkResourceName);
            if (stream == null)
            {
                return;
            }

            using StreamReader reader = new(stream);
            LoadCrosswalk(reader);
        }
        catch (Exception ex)
        {
            Console.WriteLine($"[WARN] Failed to load embedded crosswalk: {ex.Message}");
        }
    }

    private void LoadCrosswalk(TextReader reader)
    {
        string? headerLine = reader.ReadLine();
        if (string.IsNullOrWhiteSpace(headerLine))
        {
            return;
        }

        string[] headers = headerLine.Split(',');
        int sourceColumn = FindColumn(headers, "src_areaId", "src_areaNumber");
        int targetColumn = FindColumn(headers, "tgt_id_335", "tgt_areaID");
        int matchesColumn = FindColumn(headers, "matches");

        while (reader.ReadLine() is { } line)
        {
            if (string.IsNullOrWhiteSpace(line))
            {
                continue;
            }

            string[] parts = line.Split(',');
            if (!TryReadSourceId(parts, sourceColumn, out int sourceId) || sourceId <= 0)
            {
                continue;
            }

            int targetId = ReadTargetId(parts, targetColumn, matchesColumn);
            if (targetId > 0)
            {
                _directAreaCrosswalk[sourceId] = targetId;
            }
        }
    }

    private static bool TryReadSourceId(string[] parts, int sourceColumn, out int sourceId)
    {
        if (sourceColumn >= 0 && sourceColumn < parts.Length)
        {
            return int.TryParse(parts[sourceColumn].Trim(), out sourceId);
        }

        if (parts.Length >= 2)
        {
            return int.TryParse(parts[0].Trim(), out sourceId);
        }

        sourceId = 0;
        return false;
    }

    private static int ReadTargetId(string[] parts, int targetColumn, int matchesColumn)
    {
        if (matchesColumn >= 0 && matchesColumn < parts.Length)
        {
            int targetId = ReadTargetIdFromMatches(parts[matchesColumn]);
            if (targetId > 0)
            {
                return targetId;
            }
        }

        if (targetColumn >= 0 && targetColumn < parts.Length && int.TryParse(parts[targetColumn].Trim(), out int parsedTargetId))
        {
            return parsedTargetId;
        }

        return parts.Length >= 2 && int.TryParse(parts[1].Trim(), out parsedTargetId) ? parsedTargetId : 0;
    }

    private static int ReadTargetIdFromMatches(string matches)
    {
        if (string.IsNullOrWhiteSpace(matches))
        {
            return 0;
        }

        string firstMatch = matches.Split('|', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).FirstOrDefault() ?? string.Empty;
        if (string.IsNullOrWhiteSpace(firstMatch))
        {
            return 0;
        }

        string[] matchParts = firstMatch.Split(':');
        return matchParts.Length >= 2 && int.TryParse(matchParts[1], out int targetId) ? targetId : 0;
    }

    private static int FindColumn(string[] headers, params string[] names)
    {
        for (int i = 0; i < headers.Length; i++)
        {
            string header = headers[i].Trim();
            if (names.Any(name => header.Equals(name, StringComparison.OrdinalIgnoreCase)))
            {
                return i;
            }
        }

        return -1;
    }

    private static string NormalizeName(string name)
    {
        if (string.IsNullOrEmpty(name))
        {
            return string.Empty;
        }

        return name.ToLowerInvariant().Replace(" ", string.Empty).Replace("'", string.Empty).Replace("-", string.Empty);
    }

    private static int EditDistance(string a, string b)
    {
        if (string.IsNullOrEmpty(a))
        {
            return b?.Length ?? 0;
        }

        if (string.IsNullOrEmpty(b))
        {
            return a.Length;
        }

        int[,] distances = new int[a.Length + 1, b.Length + 1];
        for (int i = 0; i <= a.Length; i++)
        {
            distances[i, 0] = i;
        }

        for (int j = 0; j <= b.Length; j++)
        {
            distances[0, j] = j;
        }

        for (int i = 1; i <= a.Length; i++)
        {
            for (int j = 1; j <= b.Length; j++)
            {
                int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                distances[i, j] = Math.Min(
                    Math.Min(distances[i - 1, j] + 1, distances[i, j - 1] + 1),
                    distances[i - 1, j - 1] + cost);
            }
        }

        return distances[a.Length, b.Length];
    }

    private readonly record struct AreaEntry(int Id, int ContinentId, int ParentAreaId, string Name);
}