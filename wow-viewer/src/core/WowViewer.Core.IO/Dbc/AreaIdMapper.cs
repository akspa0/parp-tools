using DBCD;
using DBCD.Providers;
using WowViewer.Core.IO.Files;

namespace WowViewer.Core.IO.Dbc;

public sealed class AreaIdMapper
{
    private const string EmbeddedCrosswalkResourceName = "WowViewer.Core.IO.Resources.area_crosswalk.csv";
    private static readonly string[] AlphaTestDataAreaPaths =
    [
        Path.Combine("gillijimproject_refactor", "test_data", "0.5.3", "tree", "DBFilesClient", "AreaTable.dbc"),
        Path.Combine("test_data", "0.5.3", "tree", "DBFilesClient", "AreaTable.dbc"),
    ];
    private static readonly string[] LkTestDataAreaPaths =
    [
        Path.Combine("gillijimproject_refactor", "test_data", "3.3.5", "tree", "DBFilesClient", "AreaTable.dbc"),
        Path.Combine("test_data", "3.3.5", "tree", "DBFilesClient", "AreaTable.dbc"),
    ];
    private static readonly string[] AlphaTestDataMapPaths =
    [
        Path.Combine("gillijimproject_refactor", "test_data", "0.5.3", "tree", "DBFilesClient", "Map.dbc"),
        Path.Combine("test_data", "0.5.3", "tree", "DBFilesClient", "Map.dbc"),
    ];
    private static readonly string[] LkTestDataMapPaths =
    [
        Path.Combine("gillijimproject_refactor", "test_data", "3.3.5", "tree", "DBFilesClient", "Map.dbc"),
        Path.Combine("test_data", "3.3.5", "tree", "DBFilesClient", "Map.dbc"),
    ];
    private const string AlphaBuildVersion053 = "0.5.3.3368";
    private const string LkBuildVersion335 = "3.3.5.12340";

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

    public string? LastLoadMessage { get; private set; }

    public bool LastLoadUsedSchemaDefinitions { get; private set; }

    public bool TryLoadFromArchives(IArchiveReader alphaArchiveReader, string alphaBuildVersion, IArchiveReader lkArchiveReader, string lkBuildVersion)
    {
        ArgumentNullException.ThrowIfNull(alphaArchiveReader);
        ArgumentNullException.ThrowIfNull(lkArchiveReader);
        ArgumentException.ThrowIfNullOrWhiteSpace(alphaBuildVersion);
        ArgumentException.ThrowIfNullOrWhiteSpace(lkBuildVersion);

        alphaBuildVersion = NormalizeBuildVersion(alphaBuildVersion);
        lkBuildVersion = NormalizeBuildVersion(lkBuildVersion);

        ResetLoadedTables();

        byte[]? alphaAreaTableData = DbClientFileReader.TryReadTable(alphaArchiveReader, "AreaTable");
        byte[]? lkAreaTableData = DbClientFileReader.TryReadTable(lkArchiveReader, "AreaTable");
        byte[]? alphaMapData = DbClientFileReader.TryReadTable(alphaArchiveReader, "Map");
        byte[]? lkMapData = DbClientFileReader.TryReadTable(lkArchiveReader, "Map");

        if (alphaAreaTableData is null || lkAreaTableData is null)
        {
            LastLoadMessage = BuildMissingArchiveMessage(
                alphaAreaTableData is not null,
                lkAreaTableData is not null,
                alphaMapData is not null,
                lkMapData is not null);
            LastLoadUsedSchemaDefinitions = false;
            return false;
        }

        bool loadedAlphaWithDbcd = TryLoadAreaTableWithDbcd(new InMemoryDbcProvider(("AreaTable", alphaAreaTableData)), alphaBuildVersion, isAlpha: true);
        bool loadedLkWithDbcd = TryLoadAreaTableWithDbcd(new InMemoryDbcProvider(("AreaTable", lkAreaTableData)), lkBuildVersion, isAlpha: false);
        bool loadedMapsWithDbcd = false;

        if (alphaMapData is not null && lkMapData is not null)
        {
            loadedMapsWithDbcd = TryBuildMapCrosswalkWithDbcd(
                new InMemoryDbcProvider(("Map", alphaMapData)),
                alphaBuildVersion,
                new InMemoryDbcProvider(("Map", lkMapData)),
                lkBuildVersion);
        }

        LastLoadUsedSchemaDefinitions = loadedAlphaWithDbcd || loadedLkWithDbcd || loadedMapsWithDbcd;

        if (!loadedAlphaWithDbcd)
        {
            ParseAlphaAreaTable(DbcReader.Load(alphaAreaTableData));
        }

        if (!loadedLkWithDbcd)
        {
            ParseLkAreaTable(DbcReader.Load(lkAreaTableData));
            BuildLkIndices();
        }

        if (!loadedMapsWithDbcd && alphaMapData is not null && lkMapData is not null)
        {
            BuildMapCrosswalk(alphaMapData, lkMapData);
        }

        string mode = LastLoadUsedSchemaDefinitions ? "DBCD+WoWDBDefs" : "raw DbcReader fallback";
        LastLoadMessage = $"Loaded AreaTable/Map data from archive-backed sources using {mode}: {AlphaAreaCount} Alpha, {LkAreaCount} LK, {_mapIdCrosswalk.Count} map links";
        return true;
    }

    public void LoadDbcs(string alphaAreaTablePath, string lkAreaTablePath, string? alphaMapPath = null, string? lkMapPath = null)
    {
        ResetLoadedTables();

        string? dbdDirectory = TryFindDefinitionsDirectory(alphaAreaTablePath, lkAreaTablePath, alphaMapPath, lkMapPath);
        string? alphaBuildVersion = InferBuildVersion(alphaAreaTablePath);
        string? lkBuildVersion = InferBuildVersion(lkAreaTablePath);

        bool loadedAlphaWithDbcd = false;
        bool loadedLkWithDbcd = false;
        bool loadedMapsWithDbcd = false;

        if (!string.IsNullOrEmpty(dbdDirectory))
        {
            loadedAlphaWithDbcd = TryLoadAreaTableWithDbcd(alphaAreaTablePath, alphaBuildVersion, isAlpha: true);
            loadedLkWithDbcd = TryLoadAreaTableWithDbcd(lkAreaTablePath, lkBuildVersion, isAlpha: false);

            if (!string.IsNullOrEmpty(alphaMapPath) && !string.IsNullOrEmpty(lkMapPath))
            {
                loadedMapsWithDbcd = TryBuildMapCrosswalkWithDbcd(alphaMapPath, alphaBuildVersion, lkMapPath, lkBuildVersion);
            }

            LastLoadUsedSchemaDefinitions = loadedAlphaWithDbcd || loadedLkWithDbcd || loadedMapsWithDbcd;
        }

        if (!loadedAlphaWithDbcd && File.Exists(alphaAreaTablePath))
        {
            DbcReader alphaDbc = DbcReader.Load(alphaAreaTablePath);
            ParseAlphaAreaTable(alphaDbc);
        }

        if (!loadedLkWithDbcd && File.Exists(lkAreaTablePath))
        {
            DbcReader lkDbc = DbcReader.Load(lkAreaTablePath);
            ParseLkAreaTable(lkDbc);
            BuildLkIndices();
        }

        if (!loadedMapsWithDbcd && !string.IsNullOrEmpty(alphaMapPath) && File.Exists(alphaMapPath) &&
            !string.IsNullOrEmpty(lkMapPath) && File.Exists(lkMapPath))
        {
            BuildMapCrosswalk(alphaMapPath, lkMapPath);
        }

        string mode = LastLoadUsedSchemaDefinitions ? "DBCD+WoWDBDefs" : "raw DbcReader";
        LastLoadMessage = $"Loaded AreaTable/Map data using {mode}: {AlphaAreaCount} Alpha, {LkAreaCount} LK, {_mapIdCrosswalk.Count} map links";
    }

    public bool TryAutoLoadFromTestData()
    {
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
                if (TryLoadKnownTestDataFromRoot(current.FullName))
                {
                    return true;
                }

                current = current.Parent;
            }
        }

        LastLoadMessage = BuildMissingTestDataMessage(searchDirectories);
        return false;
    }

    public bool TryLoadKnownTestDataFromRoot(string rootDirectory)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(rootDirectory);

        string? alphaPath = FindFirstExistingPath(rootDirectory, AlphaTestDataAreaPaths);
        string? lkPath = FindFirstExistingPath(rootDirectory, LkTestDataAreaPaths);
        if (alphaPath == null || lkPath == null)
        {
            LastLoadMessage = BuildMissingTestDataMessage([rootDirectory]);
            LastLoadUsedSchemaDefinitions = false;
            return false;
        }

        string? alphaMap = FindFirstExistingPath(rootDirectory, AlphaTestDataMapPaths);
        string? lkMap = FindFirstExistingPath(rootDirectory, LkTestDataMapPaths);

        LoadDbcs(
            alphaPath,
            lkPath,
            alphaMap,
            lkMap);

        string mode = LastLoadUsedSchemaDefinitions ? "DBCD+WoWDBDefs" : "raw DbcReader fallback";
        LastLoadMessage = $"Auto-loaded AreaTable from known test-data roots using {mode}: {AlphaAreaCount} Alpha, {LkAreaCount} LK areas";
        return true;
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

    private void ResetLoadedTables()
    {
        _alphaAreas.Clear();
        _lkAreas.Clear();
        _lkByName.Clear();
        _lkByMapAndName.Clear();
        _mapIdCrosswalk.Clear();

        LastLoadMessage = null;
        LastLoadUsedSchemaDefinitions = false;
    }

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

        BuildMapCrosswalk(alphaMap, lkMap);
    }

    private void BuildMapCrosswalk(byte[] alphaMapData, byte[] lkMapData)
    {
        DbcReader alphaMap = DbcReader.Load(alphaMapData);
        DbcReader lkMap = DbcReader.Load(lkMapData);

        BuildMapCrosswalk(alphaMap, lkMap);
    }

    private void BuildMapCrosswalk(DbcReader alphaMap, DbcReader lkMap)
    {

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

    private bool TryLoadAreaTableWithDbcd(string tablePath, string? buildVersion, bool isAlpha)
    {
        if (string.IsNullOrEmpty(buildVersion) || !File.Exists(tablePath))
        {
            return false;
        }

        string? dbdDirectory = TryFindDefinitionsDirectory(tablePath);
        if (string.IsNullOrEmpty(dbdDirectory))
        {
            return false;
        }

        try
        {
            IDBCDStorage storage = LoadTableWithDbcd(new FilesystemDBCProvider(Path.GetDirectoryName(tablePath)!, useCache: true), "AreaTable", buildVersion);
            if (isAlpha)
            {
                ParseAlphaAreaTable(storage);
            }
            else
            {
                ParseLkAreaTable(storage);
                BuildLkIndices();
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    private bool TryLoadAreaTableWithDbcd(IDBCProvider dbcProvider, string? buildVersion, bool isAlpha)
    {
        if (string.IsNullOrEmpty(buildVersion) || string.IsNullOrEmpty(TryFindDefinitionsDirectory()))
        {
            return false;
        }

        try
        {
            IDBCDStorage storage = LoadTableWithDbcd(dbcProvider, "AreaTable", buildVersion);
            if (isAlpha)
            {
                ParseAlphaAreaTable(storage);
            }
            else
            {
                ParseLkAreaTable(storage);
                BuildLkIndices();
            }

            return true;
        }
        catch
        {
            return false;
        }
    }

    private bool TryBuildMapCrosswalkWithDbcd(string alphaMapPath, string? alphaBuildVersion, string lkMapPath, string? lkBuildVersion)
    {
        if (string.IsNullOrEmpty(alphaBuildVersion) || string.IsNullOrEmpty(lkBuildVersion))
        {
            return false;
        }

        string? dbdDirectory = TryFindDefinitionsDirectory(alphaMapPath, lkMapPath);
        if (string.IsNullOrEmpty(dbdDirectory) || !File.Exists(alphaMapPath) || !File.Exists(lkMapPath))
        {
            return false;
        }

        try
        {
            IDBCDStorage alphaMap = LoadTableWithDbcd(new FilesystemDBCProvider(Path.GetDirectoryName(alphaMapPath)!, useCache: true), "Map", alphaBuildVersion);
            IDBCDStorage lkMap = LoadTableWithDbcd(new FilesystemDBCProvider(Path.GetDirectoryName(lkMapPath)!, useCache: true), "Map", lkBuildVersion);

            BuildMapCrosswalk(alphaMap, lkMap);

            return true;
        }
        catch
        {
            return false;
        }
    }

    private bool TryBuildMapCrosswalkWithDbcd(IDBCProvider alphaMapProvider, string? alphaBuildVersion, IDBCProvider lkMapProvider, string? lkBuildVersion)
    {
        if (string.IsNullOrEmpty(alphaBuildVersion) || string.IsNullOrEmpty(lkBuildVersion) || string.IsNullOrEmpty(TryFindDefinitionsDirectory()))
        {
            return false;
        }

        try
        {
            IDBCDStorage alphaMap = LoadTableWithDbcd(alphaMapProvider, "Map", alphaBuildVersion);
            IDBCDStorage lkMap = LoadTableWithDbcd(lkMapProvider, "Map", lkBuildVersion);

            BuildMapCrosswalk(alphaMap, lkMap);

            return true;
        }
        catch
        {
            return false;
        }
    }

    private void BuildMapCrosswalk(IDBCDStorage alphaMap, IDBCDStorage lkMap)
    {

        Dictionary<string, int> lkByDirectory = new(StringComparer.OrdinalIgnoreCase);
        foreach (DBCDRow row in lkMap.Values)
        {
            string directory = NormalizeName(GetStringField(row, "Directory", "Folder", "FolderName") ?? string.Empty);
            int id = GetIntField(row, "ID") ?? 0;
            if (!string.IsNullOrEmpty(directory) && id > 0)
            {
                lkByDirectory.TryAdd(directory, id);
            }
        }

        foreach (DBCDRow row in alphaMap.Values)
        {
            string directory = NormalizeName(GetStringField(row, "Directory", "Folder", "FolderName") ?? string.Empty);
            int alphaId = GetIntField(row, "ID") ?? 0;
            if (!string.IsNullOrEmpty(directory) && alphaId > 0 && lkByDirectory.TryGetValue(directory, out int lkId))
            {
                _mapIdCrosswalk[alphaId] = lkId;
            }
        }
    }

    private IDBCDStorage LoadTableWithDbcd(IDBCProvider dbcProvider, string tableName, string buildVersion)
    {
        string? dbdDirectory = TryFindDefinitionsDirectory();
        if (string.IsNullOrEmpty(dbdDirectory))
        {
            throw new DirectoryNotFoundException("WoWDBDefs definitions directory was not found.");
        }

        FilesystemDBDProvider dbdProvider = new(dbdDirectory);
        DBCD.DBCD dbcd = new(dbcProvider, dbdProvider);

        try
        {
            return dbcd.Load(tableName, buildVersion, Locale.EnUS);
        }
        catch
        {
            return dbcd.Load(tableName, buildVersion, Locale.None);
        }
    }

    private void ParseAlphaAreaTable(IDBCDStorage storage)
    {
        foreach (DBCDRow row in storage.Values)
        {
            int id = GetIntField(row, "ID") ?? 0;
            if (id <= 0)
            {
                continue;
            }

            AreaEntry entry = new(
                id,
                GetIntField(row, "ContinentID") ?? 0,
                GetIntField(row, "ParentAreaID") ?? GetIntField(row, "ParentAreaNum") ?? 0,
                GetStringField(row, "AreaName_lang", "AreaName", "ZoneName") ?? string.Empty);

            _alphaAreas[id] = entry;
        }
    }

    private void ParseLkAreaTable(IDBCDStorage storage)
    {
        foreach (DBCDRow row in storage.Values)
        {
            int id = GetIntField(row, "ID") ?? 0;
            if (id <= 0)
            {
                continue;
            }

            AreaEntry entry = new(
                id,
                GetIntField(row, "ContinentID") ?? 0,
                GetIntField(row, "ParentAreaID") ?? GetIntField(row, "ParentAreaNum") ?? 0,
                GetStringField(row, "AreaName_lang", "AreaName", "ZoneName") ?? string.Empty);

            _lkAreas[id] = entry;
        }
    }

    private static int? GetIntField(DBCDRow row, params string[] fieldNames)
    {
        foreach (string fieldName in fieldNames)
        {
            try
            {
                object value = row[fieldName];
                if (value is int intValue)
                {
                    return intValue;
                }

                if (value is uint uintValue)
                {
                    return checked((int)uintValue);
                }

                if (value is long longValue)
                {
                    return checked((int)longValue);
                }

                if (value is short shortValue)
                {
                    return shortValue;
                }

                if (value is ushort ushortValue)
                {
                    return ushortValue;
                }

                if (value is byte byteValue)
                {
                    return byteValue;
                }

                if (value is sbyte sbyteValue)
                {
                    return sbyteValue;
                }

                if (value != null)
                {
                    return Convert.ToInt32(value, System.Globalization.CultureInfo.InvariantCulture);
                }
            }
            catch
            {
            }
        }

        return null;
    }

    private static string? GetStringField(DBCDRow row, params string[] fieldNames)
    {
        foreach (string fieldName in fieldNames)
        {
            try
            {
                object value = row[fieldName];
                if (value is string stringValue)
                {
                    return stringValue;
                }

                if (value != null)
                {
                    return value.ToString();
                }
            }
            catch
            {
            }
        }

        return null;
    }

    private static string? InferBuildVersion(string path)
    {
        string normalizedPath = path.Replace('\\', '/');
        if (normalizedPath.Contains("/0.5.3/", StringComparison.OrdinalIgnoreCase))
        {
            return AlphaBuildVersion053;
        }

        if (normalizedPath.Contains("/3.3.5/", StringComparison.OrdinalIgnoreCase))
        {
            return LkBuildVersion335;
        }

        return null;
    }

    private static string NormalizeBuildVersion(string buildVersion)
    {
        return buildVersion switch
        {
            "0.5.3" => AlphaBuildVersion053,
            "3.3.5" => LkBuildVersion335,
            _ => buildVersion,
        };
    }

    private static string? TryFindDefinitionsDirectory(params string?[] paths)
    {
        List<string> startDirectories = [];
        foreach (string? path in paths)
        {
            if (string.IsNullOrWhiteSpace(path))
            {
                continue;
            }

            string? directory = Directory.Exists(path) ? path : Path.GetDirectoryName(path);
            if (!string.IsNullOrEmpty(directory))
            {
                startDirectories.Add(directory);
            }
        }

        startDirectories.Add(Directory.GetCurrentDirectory());

        string? assemblyDirectory = Path.GetDirectoryName(typeof(AreaIdMapper).Assembly.Location);
        if (!string.IsNullOrEmpty(assemblyDirectory))
        {
            startDirectories.Add(assemblyDirectory);
        }

        foreach (string startDirectory in startDirectories.Distinct(StringComparer.OrdinalIgnoreCase))
        {
            DirectoryInfo? current = new(startDirectory);
            for (int i = 0; i < 8 && current != null; i++)
            {
                string[] candidates =
                [
                    Path.Combine(current.FullName, "definitions"),
                    Path.Combine(current.FullName, "gillijimproject_refactor", "lib", "WoWDBDefs", "definitions"),
                    Path.Combine(current.FullName, "gillijimproject_refactor", "src", "MdxViewer", "bin", "Debug", "net10.0-windows", "definitions"),
                    Path.Combine(current.FullName, "wow-viewer", "libs", "wowdev", "WoWDBDefs", "definitions"),
                    Path.Combine(current.FullName, "libs", "wowdev", "WoWDBDefs", "definitions"),
                    Path.Combine(current.FullName, "lib", "WoWDBDefs", "definitions"),
                ];

                foreach (string candidate in candidates)
                {
                    if (Directory.Exists(candidate))
                    {
                        return candidate;
                    }
                }

                current = current.Parent;
            }
        }

        return null;
    }

    private static string BuildMissingTestDataMessage(IEnumerable<string> searchRoots)
    {
        string searchedRoots = string.Join(", ", searchRoots.Select(root => Path.GetFullPath(root)).Distinct(StringComparer.OrdinalIgnoreCase));
        string expectedPaths = string.Join(", ",
            AlphaTestDataAreaPaths
                .Concat(LkTestDataAreaPaths)
                .Concat(AlphaTestDataMapPaths)
                .Concat(LkTestDataMapPaths)
                .Distinct(StringComparer.OrdinalIgnoreCase)
                .Select(path => $"'{path.Replace('\\', '/')}'"));

        return $"Missing extracted AreaTable/Map DBC trees. Expected one of: {expectedPaths}. Searched roots: {searchedRoots}. Falling back to crosswalk-only behavior.";
    }

    private static string BuildMissingArchiveMessage(bool hasAlphaArea, bool hasLkArea, bool hasAlphaMap, bool hasLkMap)
    {
        List<string> missing = [];

        if (!hasAlphaArea)
            missing.Add("Alpha AreaTable");
        if (!hasLkArea)
            missing.Add("LK AreaTable");
        if (!hasAlphaMap)
            missing.Add("Alpha Map");
        if (!hasLkMap)
            missing.Add("LK Map");

        return $"Missing {string.Join(", ", missing)} in archive-backed DBC sources. Expected DBFilesClient or DBC table entries inside the configured Alpha and LK archives. Falling back to crosswalk-only behavior.";
    }

    private static string? FindFirstExistingPath(string rootDirectory, IEnumerable<string> relativePaths)
    {
        foreach (string relativePath in relativePaths)
        {
            string candidate = Path.Combine(rootDirectory, relativePath);
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }

        return null;
    }

    private sealed class InMemoryDbcProvider(params (string TableName, byte[] Data)[] tables) : IDBCProvider
    {
        private readonly Dictionary<string, byte[]> _tables = tables.ToDictionary(
            static entry => entry.TableName,
            static entry => entry.Data,
            StringComparer.OrdinalIgnoreCase);

        public Stream StreamForTableName(string tableName, string build)
        {
            if (!_tables.TryGetValue(tableName, out byte[]? data))
            {
                throw new FileNotFoundException($"Table {tableName} was not supplied to the in-memory DBC provider.");
            }

            return new MemoryStream(data, writable: false);
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