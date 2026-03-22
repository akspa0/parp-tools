using DBCD;
using DBCD.Providers;
using MdxViewer.Logging;

namespace MdxViewer.Terrain;

/// <summary>
/// Loads AreaTable.dbc via DBCD and provides AreaID → area name lookups.
/// Used to display the current area name in the status bar based on the camera's chunk position.
/// </summary>
public class AreaTableService
{
    private readonly Dictionary<int, AreaEntry> _areas = new();
    private int _rowCount;
    private int _primaryKeyCount;
    private int _fallbackAliasCount;
    private int _fallbackAliasCollisions;

    public record AreaEntry(int Id, string Name, int ParentAreaId, int MapId, int Flags);

    public int Count => _primaryKeyCount;
    public string? LoadedBuild { get; private set; }
    public string? LoadedLocale { get; private set; }
    public string? NameColumn { get; private set; }
    public string? IdColumn { get; private set; }
    public string? ParentColumn { get; private set; }
    public string? MapColumn { get; private set; }
    public string? FlagsColumn { get; private set; }

    /// <summary>
    /// Load AreaTable.dbc from the given DBC provider.
    /// </summary>
    public void Load(IDBCProvider dbcProvider, string dbdDir, string build)
    {
        _areas.Clear();
        _rowCount = 0;
        _primaryKeyCount = 0;
        _fallbackAliasCount = 0;
        _fallbackAliasCollisions = 0;
        LoadedBuild = build;

        var dbdProvider = new FilesystemDBDProvider(dbdDir);
        var dbcd = new DBCD.DBCD(dbcProvider, dbdProvider);

        IDBCDStorage storage;
        Locale localeUsed;
        try
        {
            try
            {
                storage = dbcd.Load("AreaTable", build, Locale.EnUS);
                localeUsed = Locale.EnUS;
            }
            catch (Exception enUsEx)
            {
                ViewerLog.Important(ViewerLog.Category.General,
                    $"[AreaTable] Locale.EnUS load failed for build {build}: {enUsEx.Message}. Retrying Locale.None.");
                storage = dbcd.Load("AreaTable", build, Locale.None);
                localeUsed = Locale.None;
            }
        }
        catch (Exception ex)
        {
            ViewerLog.Error(ViewerLog.Category.Dbc,
                $"[AreaTable] Failed to load AreaTable.dbc for build {build}: {ex.Message}");
            return;
        }

        LoadedLocale = localeUsed.ToString();

        var availableColumns = new HashSet<string>(storage.AvailableColumns, StringComparer.OrdinalIgnoreCase);

        // Detect column names from the active DBD-backed layout rather than probing a row.
        string? nameCol = DetectColumn(availableColumns, "AreaName_lang", "AreaName", "Name");
        string? idCol = DetectColumn(availableColumns, "ID", "AreaID", "AreaNumber");
        string? areaNumberCol = DetectColumn(availableColumns, "AreaNumber");
        string? parentCol = DetectColumn(availableColumns, "ParentAreaID", "ParentAreaNum");
        string? mapCol = DetectColumn(availableColumns, "ContinentID", "MapID", "Continent");
        string? flagsCol = DetectColumn(availableColumns, "Flags", "AreaFlags");
        NameColumn = nameCol;
        IdColumn = idCol;
        ParentColumn = parentCol;
        MapColumn = mapCol;
        FlagsColumn = flagsCol;

        // MCNK AreaId should resolve against the canonical AreaTable ID for the active layout.
        // Older tables also expose AreaNumber-style aliases, so keep those as fallbacks instead
        // of treating them as the primary key for every build.
        foreach (var key in storage.Keys)
        {
            _rowCount++;
            var row = storage[key];
            int areaId = SafeField<int>(row, idCol, key);
            if (areaId == 0 && key != 0)
                areaId = key;

            string name = Sanitize(SafeField<string>(row, nameCol, string.Empty) ?? string.Empty);
            int parentId = SafeField<int>(row, parentCol, 0);
            int mapId = SafeField<int>(row, mapCol, 0);
            int flags = SafeField<int>(row, flagsCol, 0);
            int areaNumber = SafeField<int>(row, areaNumberCol, 0);

            var entry = new AreaEntry(areaId, name, parentId, mapId, flags);
            RegisterPrimary(areaId, entry);
            RegisterAlias(key, entry);
            RegisterAlias(areaNumber, entry);
            RegisterLegacyPackedAreaNumberAliases(build, areaNumber, entry);
        }

        ViewerLog.Important(ViewerLog.Category.General,
            $"[AreaTable] Loaded build={LoadedBuild} locale={LoadedLocale} rows={_rowCount} indexed={_areas.Count} primaryKeys={_primaryKeyCount} fallbackAliases={_fallbackAliasCount} aliasCollisions={_fallbackAliasCollisions} nameCol='{FormatColumn(nameCol)}' idCol='{FormatColumn(idCol)}' parentCol='{FormatColumn(parentCol)}' mapCol='{FormatColumn(mapCol)}' flagsCol='{FormatColumn(flagsCol)}'");
    }

    /// <summary>
    /// Look up an area name by AreaID. Returns null if not found.
    /// </summary>
    public string? GetAreaName(int areaId)
    {
        return _areas.TryGetValue(areaId, out var entry) ? entry.Name : null;
    }

    /// <summary>
    /// Get full area entry by ID.
    /// </summary>
    public AreaEntry? GetArea(int areaId)
    {
        return _areas.TryGetValue(areaId, out var entry) ? entry : null;
    }

    /// <summary>
    /// Get area name with parent context (e.g. "Durotar > Razor Hill").
    /// </summary>
    public string GetAreaDisplayName(int areaId)
    {
        if (!_areas.TryGetValue(areaId, out var entry))
            return $"Unknown ({areaId})";

        if (entry.ParentAreaId != 0 && _areas.TryGetValue(entry.ParentAreaId, out var parent))
            return $"{parent.Name} > {entry.Name}";

        return entry.Name;
    }

    /// <summary>
    /// Get area name with parent context, but only if the area belongs to the given MapID.
    /// Returns null if the area belongs to a different map or AreaID is not found.
    /// MCNK AreaID maps directly to AreaTable.dbc ID — no byte packing.
    /// </summary>
    public string? GetAreaDisplayNameForMap(int areaId, int mapId)
    {
        if (!_areas.TryGetValue(areaId, out var entry) || entry.MapId != mapId)
            return null;

        if (entry.ParentAreaId != 0 && _areas.TryGetValue(entry.ParentAreaId, out var parent))
            return $"{parent.Name} > {entry.Name}";
        return entry.Name;
    }

    public string DescribeLoadContext()
    {
        return $"build={LoadedBuild ?? "unknown"}, locale={LoadedLocale ?? "unknown"}, rows={_rowCount}, indexed={_areas.Count}, primaryKeys={_primaryKeyCount}, idCol={IdColumn ?? "?"}, mapCol={MapColumn ?? "?"}";
    }

    public string DescribeLookup(int areaId, int mapId)
    {
        if (!_areas.TryGetValue(areaId, out var entry))
            return $"[AreaTable] Lookup miss: AreaId={areaId}, MapId={mapId}, {DescribeLoadContext()}";

        if (entry.MapId != mapId)
        {
            return $"[AreaTable] Map mismatch: AreaId={areaId} resolved='{entry.Name}' entryMapId={entry.MapId} requestedMapId={mapId} parentAreaId={entry.ParentAreaId} flags=0x{entry.Flags:X} {DescribeLoadContext()}";
        }

        return $"[AreaTable] Lookup resolved: AreaId={areaId} -> '{entry.Name}' on MapId={mapId} {DescribeLoadContext()}";
    }

    private void RegisterPrimary(int areaId, AreaEntry entry)
    {
        if (_areas.TryAdd(areaId, entry))
        {
            _primaryKeyCount++;
            return;
        }

        _areas[areaId] = entry;
    }

    private void RegisterAlias(int aliasId, AreaEntry entry)
    {
        if (aliasId == 0 || aliasId == entry.Id)
            return;

        if (_areas.TryGetValue(aliasId, out var existing))
        {
            if (existing.Id != entry.Id)
                _fallbackAliasCollisions++;
            return;
        }

        _areas[aliasId] = entry;
        _fallbackAliasCount++;
    }

    private void RegisterLegacyPackedAreaNumberAliases(string? build, int areaNumber, AreaEntry entry)
    {
        if (areaNumber == 0 || string.IsNullOrWhiteSpace(build) || !build.StartsWith("0.5.", StringComparison.OrdinalIgnoreCase))
            return;

        int lowWord = areaNumber & 0xFFFF;
        int highWord = (int)((uint)areaNumber >> 16);
        RegisterAlias(lowWord, entry);
        RegisterAlias(highWord, entry);
    }

    private static string? DetectColumn(ISet<string> availableColumns, params string[] candidates)
    {
        foreach (var col in candidates)
        {
            if (availableColumns.Contains(col))
                return col;
        }

        return null;
    }

    private static T SafeField<T>(dynamic row, string? col, T fallback)
    {
        if (string.IsNullOrWhiteSpace(col))
            return fallback;

        try { return (T)row[col]; }
        catch { return fallback; }
    }

    private static string FormatColumn(string? col)
    {
        return string.IsNullOrWhiteSpace(col) ? "n/a" : col;
    }

    private static string Sanitize(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        int nullIdx = s.IndexOf('\0');
        if (nullIdx >= 0) s = s[..nullIdx];
        return new string(s.Where(c => !char.IsControl(c) || c == '\n').ToArray());
    }
}
