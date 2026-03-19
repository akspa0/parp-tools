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

    public int Count => _areas.Count;
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

        // Detect column names (varies by build)
        string nameCol = DetectColumn(storage, "AreaName_lang", "AreaName", "Name");
        string idCol = DetectColumn(storage, "AreaNumber", "ID", "AreaID");
        string parentCol = DetectColumn(storage, "ParentAreaNum", "ParentAreaID", "ParentAreaNum");
        string mapCol = DetectColumn(storage, "ContinentID", "MapID", "Continent");
        string flagsCol = DetectColumn(storage, "Flags", "AreaFlags");
        NameColumn = nameCol;
        IdColumn = idCol;
        ParentColumn = parentCol;
        MapColumn = mapCol;
        FlagsColumn = flagsCol;

        // MCNK AreaId uses the DBC row key (implicit ID), NOT the AreaNumber field.
        // AreaNumber is a packed value (e.g. 1048576) unrelated to MCNK placement.
        // Index primarily by row key. Also index by AreaNumber as fallback.
        foreach (var key in storage.Keys)
        {
            _rowCount++;
            var row = storage[key];
            int areaNumber = SafeField<int>(row, idCol, key);
            string name = Sanitize(SafeField<string>(row, nameCol, "") ?? "");
            int parentId = SafeField<int>(row, parentCol, 0);
            int mapId = SafeField<int>(row, mapCol, 0);
            int flags = SafeField<int>(row, flagsCol, 0);

            var entry = new AreaEntry(key, name, parentId, mapId, flags);
            _areas[key] = entry;
            _primaryKeyCount++;

            if (areaNumber != key)
            {
                if (_areas.TryAdd(areaNumber, entry))
                    _fallbackAliasCount++;
                else
                    _fallbackAliasCollisions++;
            }
        }

        ViewerLog.Important(ViewerLog.Category.General,
            $"[AreaTable] Loaded build={LoadedBuild} locale={LoadedLocale} rows={_rowCount} indexed={_areas.Count} primaryKeys={_primaryKeyCount} fallbackAliases={_fallbackAliasCount} aliasCollisions={_fallbackAliasCollisions} nameCol='{nameCol}' idCol='{idCol}' parentCol='{parentCol}' mapCol='{mapCol}' flagsCol='{flagsCol}'");
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
        return $"build={LoadedBuild ?? "unknown"}, locale={LoadedLocale ?? "unknown"}, rows={_rowCount}, indexed={_areas.Count}, idCol={IdColumn ?? "?"}, mapCol={MapColumn ?? "?"}";
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

    private static string DetectColumn(IDBCDStorage storage, params string[] candidates)
    {
        if (storage.Values.Count == 0) return candidates[0];
        var row = storage.Values.First();
        foreach (var col in candidates)
        {
            try { _ = row[col]; return col; }
            catch { }
        }
        return candidates[0];
    }

    private static T SafeField<T>(dynamic row, string col, T fallback)
    {
        try { return (T)row[col]; }
        catch { return fallback; }
    }

    private static string Sanitize(string s)
    {
        if (string.IsNullOrEmpty(s)) return s;
        int nullIdx = s.IndexOf('\0');
        if (nullIdx >= 0) s = s[..nullIdx];
        return new string(s.Where(c => !char.IsControl(c) || c == '\n').ToArray());
    }
}
